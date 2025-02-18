# Copyright 2025 Janik Haitz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
from typing import Optional, Union, Literal
from math import ceil

import cv2
import numpy as np
import rich_click as click
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from scipy.spatial import distance_matrix
from pypxml import PageXML, PageType
from rich import print as rprint
from rich.progress import (Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn,
                           TimeElapsedColumn, SpinnerColumn)


# Config
__version__ = "1.2"
__prog__ = "pagemerge"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.MAX_WIDTH = 96
click.rich_click.RANGE_STRING = ""
click.rich_click.OPTION_GROUPS = {
    "*": [
        {
            "name": "Input",
            "options": ["xmls", "--glob", "--image-suffix"]
        },
        {
            "name": "Options",
            "options": ["--output", "--padding", "--vertical", "--horizontal", "--mode"]
        },
        {
            "name": "Help",
            "options": ["--help", "--version"]
        }
    ],
}
progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
    TextColumn("• {task.fields[filename]}")
)
SHRINK_MODE = Literal["merge", "largest"]


# Callbacks
def paths_callback(ctx, param, value: Optional[list[str]]) -> Optional[list[Path]]:
    """Parse a list of click paths to a list of pathlib Path objects."""
    return [] if value is None else list([Path(p) for p in value])

def path_callback(ctx, param, value: str) -> Optional[Path]:
    """ Parse a click path to a pathlib Path object. """
    return None if value is None else Path(value)

def suffix_callback(ctx, param, value: Optional[str]) -> str:
    """Parses a string to a valid suffix."""
    return None if value is None else (value if value.startswith('.') else f".{value}")


# Utils
def expand_paths(paths: Union[Path, list[Path]], glob: str = '*') -> list[Path]:
    """ Expands a list of paths by unpacking directories. """
    result = []
    if isinstance(paths, list):
        for path in paths:
            if path.is_dir():
                result.extend([p for p in path.glob(glob) if p.is_file()])
            else:
                result.append(path)
    elif isinstance(paths, Path):
        if paths.is_dir():
            result.extend([p for p in paths.glob(glob) if p.is_file()])
        else:
            result.append(paths)
    return sorted(result)

def find_image(pagexml: Path, image_suffix: Optional[str] = None) -> Optional[Path]:
    """
    Finds corresponding image files for a PageXML file.
    Args:
        pagexml: PageXML file path.
        image_suffix: If provided, search for image with this suffix.
            Else use filename provided in PageXMLs imageFilename attribute.
    Returns:
        The Path to the image file if found, else None.
    """
    if image_suffix:
        image = pagexml.parent.joinpath(pagexml.name.split('.')[0] + image_suffix)
        if image.exists():
            return image
    else:
        try:
            pxml = PageXML.from_xml(pagexml)
        except ValueError as e:
            rprint(f"[bold red]ERROR:[/bold red] Could not parse file {pagexml}: {e}")
            return None
        image = pagexml.parent.joinpath(pxml.image_filename)
        if image.exists():
            return image
    return None


# Logic
class Pageshrink:
    def __init__(self, xml: Path, image: Path):
        self.xml = xml
        self.image = image

    def shrink(self, p: int = 5, h: int = 1, v: int = 1, mode: SHRINK_MODE = "merge",
               whitelist: Optional[list[PageType]] = None, logger: Optional[Progress] = None) -> PageXML:
        """
        Shrinks the region polygons of a PageXML file to its content.
        Args:
            p: Padding between region borders and its content.
            h: horizontal smoothing, calculated as the factor of the average glyph width.
            v: vertical smoothing, calculated as the factor of the average glyph height.
            mode: The shrinking mode to use:
                - merge: Merges all resulting polygons of each region after shrinking.
                - largest: Keeps only the largest resulting polygon of each region after shrinking.
            whitelist: A list of valid region types to shrink. If nothing is provided, all regions are shrunk.
            logger: A rich Progress object to log progress. If not provided, errors are printed.
        Returns:
            A modified PageXML object.
        """
        pxml = PageXML.from_xml(self.xml)
        img_inv = ~cv2.imread(str(self.image), cv2.IMREAD_GRAYSCALE)
        if not self.is_bitonal(img_inv):
            raise ValueError("Image is not bitonal.")
        if not self.validate_files(pxml, img_inv):
            raise ValueError("Image and PageXML do not match in size.")

        for region in pxml.regions:
            if whitelist and region.type not in whitelist:
                continue  # skip region if not in whitelist
            coords_element = region.get_coords()
            if coords_element is None or coords_element["points"] is None:
                self.log(f"Could not find valid Coords element in region {region.id} ({self.xml})", logger)
                continue
            region_arr = self.string_to_polygon(coords_element["points"])
            im_masked = self.mask_image(img_inv, region_arr)
            if region.type in [PageType.TextRegion, PageType.SeparatorRegion]:
                glyph_width, glyph_height = self.estimate_scales(im_masked)
            else:
                glyph_width, glyph_height = 25, 25

            # dilate text considering the average glyph width and height
            dilation_kernel = np.ones((glyph_width, glyph_width), np.uint8)
            im_dil = cv2.dilate(im_masked, dilation_kernel, iterations=2)

            # close gaps between symbols by provided h and v factors
            smooth_kernel = np.ones((glyph_height * v, glyph_width * h), np.uint8)
            im_smooth = cv2.morphologyEx(im_dil, cv2.MORPH_CLOSE, smooth_kernel)

            # find contours of the smoothed image
            contours = cv2.findContours(im_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours[0] if len(contours) == 2 else contours[1], key=cv2.contourArea, reverse=True)
            if not sum([cv2.contourArea(contour) for contour in contours]):
                self.log(f"Shrunk region area is 0 in region {region.id} ({self.xml})", logger)
                continue

            # postprocess resulting regions
            if mode == "largest":
                poly = Polygon([tuple(pt[0]) for pt in contours[0]])
                poly = poly.buffer(p - glyph_width, join_style=2)
                if poly.area > 0:
                    coords_element["points"] = self.polygon_to_string(np.array(poly.exterior.coords, dtype=np.int32))
                else:
                    self.log(f"Could not shrink region {region.id} ({self.xml})", logger)
            elif mode == "merge":
                polys = [Polygon([tuple(pt[0]) for pt in contour]) for contour in contours]
                polys = [poly.buffer(p - glyph_width, join_style=2) for poly in polys]
                polys = [poly for poly in polys if poly.area > 0]
                if polys:
                    try:
                        poly = self.merge_polygons(polys, p)
                        if poly is None:
                            self.log(f"Could not merge multiple resulting polygons in region {region.id} ({self.xml})", logger)
                            continue
                        arr = np.array(poly.exterior.coords, dtype=np.int32)
                        coords_element['points'] = self.polygon_to_string(arr)
                    except Exception as e:
                        self.log(f"Could not merge multiple resulting polygons in region {region.id} ({self.xml}): {e}", logger)
                        continue
                else:
                    self.log(f"Could not shrink region {region.id} ({self.xml})", logger)
        return pxml

    @staticmethod
    def log(message: str, logger: Optional[Progress] = None):
        if logger:
            logger.log(message)
        else:
            rprint(message)
            
    @staticmethod
    def is_bitonal(image: np.ndarray) -> bool:
        """
        Check and image if it is bitonal.
        Args:
            image: The image to check
        Returns:
            True if the image contains only two different color values (black and white).
        """
        return np.array_equal(np.unique(image), [0, 255])

    @staticmethod
    def validate_files(xml: PageXML, image: np.ndarray) -> bool:
        """
        Validates the PageXML and image file.
        Args:
            xml: The PageXML object matching the image file.
            image: The image object to check.
        Returns:
            Checks if the image and PageXML have the same size.
        """
        h, w = image.shape
        return xml.width == w and xml.height == h

    @staticmethod
    def string_to_polygon(points: str) -> np.ndarray:
        """
        Converts a string of pagexml points to a numpy array.
        Args:
            points: PageXML points string.
        Returns:
            Numpy array containing the polygon points.
        """
        return np.array([tuple(map(int, xy.split(','))) for xy in points.split()], dtype=np.int32)

    @staticmethod
    def polygon_to_string(polygon: np.ndarray) -> str:
        """
        Converts a numpy array to a string of PageXML points.
        Args:
            polygon: Numpy array containing the polygon points.
        Returns:
            PageXML points string.
        """
        return ' '.join([f"{xy[0]},{xy[1]}" for xy in polygon])

    @staticmethod
    def mask_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Masks an image with a binary mask.
        Args:
            image: The image to mask.
            mask: The binary mask to apply.
        Returns:
            The masked image.
        """
        zero_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zero_mask, [mask], (255, 255, 255))
        return image & zero_mask

    @staticmethod
    def estimate_scales(image: np.ndarray) -> tuple[int, int]:
        """
        Estimates the average glyph width and height from an image or region.
        Args:
            image: The binary image to calculate scales from. Can be a whole image or a region.
        Returns:
            A tuple containing the average width and height in pixels.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        if len(stats) > 0:
            avg_width = np.median(stats[1:, 2])  # Average width of the bounding boxes (ignoring the background)
            avg_height = np.median(stats[1:, 3])  # Average height of the bounding boxes
            return max(1, round(avg_width)), max(1, round(avg_height))
        return 25, 25

    @staticmethod
    def merge_polygons(polygons: list[Polygon], b: int = 1) -> Optional[Polygon]:
        """
        Merges a list of polygons.
        Args:
            polygons: List of polygons to merge.
            b: Buffer size in pixels for connecting multiple polygons.
        Returns:
            A single polygon.
        """
        merged_geom = unary_union(polygons)  # merge overlapping geometries
        if merged_geom.geom_type == "Polygon":  # if already a single polygon, return it
            return merged_geom
        else:  # else calculate minimal connections between polygons
            polygons = list(merged_geom.geoms)
            centroids = np.array([poly.centroid.coords[0] for poly in polygons])
            dist_matrix = distance_matrix(centroids, centroids)  # pairwise distances between polygon centroids
            np.fill_diagonal(dist_matrix, np.inf)  # avoid self-connections

            # compute minimum spanning tree (Prim's algorithm)
            n = len(polygons)
            in_tree = [False] * n
            in_tree[0] = True
            mst_edges = []
            for _ in range(n - 1):
                min_dist = np.inf
                min_edge = None
                for i in range(n):
                    if in_tree[i]:
                        for j in range(n):
                            if not in_tree[j] and dist_matrix[i, j] < min_dist:
                                min_dist = dist_matrix[i, j]
                                min_edge = (i, j)
                if min_edge:
                    mst_edges.append(min_edge)
                    in_tree[min_edge[1]] = True

            # connect polygons
            connections = [LineString([centroids[i], centroids[j]]).buffer(ceil(b / 2)) for i, j in mst_edges]
            combined_geometry = unary_union(polygons + connections) # merge polygons/connections into a single geometry

            # validate resulting polygon
            if combined_geometry.geom_type == "Polygon":
                return combined_geometry
            elif combined_geometry.geom_type == "GeometryCollection":
                all_exteriors = []
                for poly in combined_geometry.geoms:
                    all_exteriors.extend(poly.exterior.coords)
                return Polygon(all_exteriors)
            else:
                return None


# CLI
@click.command()
@click.help_option("--help")
@click.version_option(__version__,
                      "--version",
                      prog_name=__prog__,
                      message=f"{__prog__} v{__version__} - Developed at Centre for Philology and Digitality (ZPD), "
                              f"University of Würzburg")
@click.argument("xmls",
                type=click.Path(exists=True, file_okay=True, dir_okay=True, resolve_path=True),
                callback=paths_callback, required=True, nargs=-1)
@click.option("-g", "--glob", "glob",
              help="Glob pattern for matching PageXML files within directories. "
                   "Only applicable when directories are passed in XMLS.",
              type=click.STRING,
              default="*.xml", required=False, show_default=True)
@click.option("-i", "--image-suffix", "image_suffix",
              help="Suffix of the image files to search for. "
                   "If not provided, the image filename from the PageXML is used.",
              type=click.STRING, callback=suffix_callback, required=False, show_default=True)
@click.option("-o", "--output", "output",
              help="Specify output directory for shrunk files. If not set, overwrite input files.",
              type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True),
              callback=path_callback, required=False)
@click.option("-p", "--padding", "padding",
              help="Padding between region borders and its content in pixels.",
              type=click.INT, default=5, show_default=5)
@click.option("-v", "--vertical", "vertical", 
              help="Vertical smoothing factor as a multiple of the average glyph height.",
              type=click.INT, default=1, show_default=True)
@click.option("-h", "--horizontal", "horizontal",
              help="Horizontal smoothing factor as a multiple of the average glyph width.",
              type=click.INT, default=1, show_default=True)
@click.option("-m", "--mode", "mode",
              help="Shrinking mode to use for regions. "
                   "_Merge_ merges all resulting polygons of each region after shrinking. "
                   "`Largest` keeps only the largest resulting polygon of each region after shrinking.",
              type=click.Choice(["merge", "largest"], case_sensitive=False),
              default="merge", show_default=True)
def pageshrink_cli(xmls: list[Path], glob: str = "*.xml", image_suffix: Optional[str] = None, 
                   output: Optional[Path] = None,  padding: int = 5, vertical: int = 1, horizontal: int = 1, 
                   mode: SHRINK_MODE = "merge"):
    """
    Shrinks the region polygons of PageXML files to its content.
    
    XMLS: List of PageXML files or directories containing PageXML files. 
    Accepts individual files, wildcards, or directories (with -g option for pattern matching).
    """
    xmls = expand_paths(xmls, glob)
    if not xmls:
        rprint("[bold red]ERROR:[/bold red] No valid PageXML files found.")
        return
    if output and not output.exists():
        output.mkdir(parents=True, exist_ok=True)
    
    with progress as p:
        task = p.add_task("Shrinking regions...", total=len(xmls), filename="")
        for fp in xmls:
            p.update(task, filename=fp)
            image = find_image(fp, image_suffix)
            shrinker = Pageshrink(fp, image)
            try:
                pxml = shrinker.shrink(padding, horizontal, vertical, mode, logger=p)
                if output:
                    pxml.to_xml(output.joinpath(fp.name))
                else:
                    pxml.to_xml(fp)
            except Exception as e:
                p.log(f"Could not shrink {fp}: {e}")
    

if __name__ == "__main__":
    pageshrink_cli()
