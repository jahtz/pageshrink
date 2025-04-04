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

from typing import Literal, Optional
from pathlib import Path
from math import ceil

import cv2
import numpy as np
from pypxml import PageXML, PageType
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from scipy.spatial import distance_matrix
from rich.progress import Progress
from rich import print as rprint


__all__ = ["SHRINK_MODE", "shrink"]
SHRINK_MODE = Literal["merge", "largest"]
GLYPH_ESITMATION = [PageType.TextRegion]


def log(message: str = "", logger: Optional[Progress] = None):
    if logger:
        logger.log(message)
    else:
        rprint(message)


def is_bitonal(image: np.ndarray) -> bool:
    """
    Check and image if it is bitonal.
    Args:
        image: The image to check
    Returns:
        True if the image contains only two different color values (black and white).
    """
    return np.array_equal(np.unique(image), [0, 255])


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


def string_to_polygon(points: str) -> np.ndarray:
    """
    Converts a string of pagexml points to a numpy array.
    Args:
        points: PageXML points string.
    Returns:
        Numpy array containing the polygon points.
    """
    return np.array([tuple(map(int, xy.split(','))) for xy in points.split()], dtype=np.int32)

    
def polygon_to_string(polygon: np.ndarray) -> str:
    """
    Converts a numpy array to a string of PageXML points.
    Args:
        polygon: Numpy array containing the polygon points.
    Returns:
        PageXML points string.
    """
    return ' '.join([f"{xy[0]},{xy[1]}" for xy in polygon])


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
    

def estimate_glyphs(image: np.ndarray, smallest: bool = False) -> int:
    """
    Estimates the median glyph size from an image or region.
    Args:
        image: The binary image to calculate scale from. Can be a whole image or a region.
        smallest: Wheter to use smallest or biggest value of height/width
    Returns:
        The median glyph size, calculated from the biggest 50% of the glyphs (noise removal).
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    if len(stats) > 1:
        glyph_widths = sorted(stats[1:, 2], reverse=True)
        glyph_heights = sorted(stats[1:, 3], reverse=True)
        avg_width = np.median(glyph_widths[:ceil(len(glyph_widths) * 0.5)])
        avg_height = np.median(glyph_heights[:ceil(len(glyph_heights) * 0.5)])
        if smallest:
            return max(1, min(round(avg_width), round(avg_height)))
        return max(1, max(round(avg_width), round(avg_height)))
    return 25


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
        if not polygons:
            return None
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


def shrink(xml: Path, image: Path, p: int = 5, s: float = 1.0, n: Optional[int] = None, m: SHRINK_MODE = "merge", 
           w: Optional[list[PageType]] = None, logger: Optional[Progress] = None) -> PageXML:
    """
    Shrinks the region polygons of a PageXML file to its content.
    Args:
        xml: PageXML file to modify.
        image: Related image file for region shrinking. Binary image is required.
        p: Padding between region borders and its content. Defaults to 5.
        s: Smoothing, calculated as the factor of the average glyph size. Defaults to 1.0.
        n: Remove noise from the input image with a kernel of this size. 1 equals to no noise removal.
        m: The shrinking mode to use:
            - merge: Merges all resulting polygons of each region after shrinking.
            - largest: Keeps only the largest resulting polygon of each region after shrinking. 
            Defaults to "merge".
        w: A list of PageTypes to shrink. If not provided, all regions are shrunk.
        logger: A rich Progress object to log progress. If not provided, errors are thrown and warnings are printed.
    Returns:
        A modified PageXML object.
    """
    pxml = PageXML.from_xml(xml)
    img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    
    if not is_bitonal(img):
        raise ValueError("Image is not bitonal.")
    if not validate_files(pxml, img):
        raise ValueError("Image and PageXML do not match in size.")
    
    if n and n >= 3:  # denoise image if a kernel size is provided
        img = cv2.medianBlur(img, n)
    img_inv = ~img  # invert image for further processing
    
    for region in pxml.regions:
        if w and region.type not in w:
            continue
        
        coords = region.get_coords()
        if coords is None or coords["points"] is None:
            log(f"Could not find valid Coords element in region {region.id} ({region.type}): {xml}", logger)
            continue
        region_arr = string_to_polygon(coords["points"])
        
        img_masked = mask_image(img_inv, region_arr)
        
        if region.type in GLYPH_ESITMATION:
            glyph_scale = estimate_glyphs(img_masked)
        elif region.type == PageType.SeparatorRegion:
            glyph_scale = estimate_glyphs(img_masked, smallest=True)
        else:
            glyph_scale = 25
        
        # dilate text considering the median glyph scale
        dilation_kernel = np.ones((glyph_scale, glyph_scale), np.uint8)
        img_dil = cv2.dilate(img_masked, dilation_kernel, iterations=2)

        # close gaps between symbols by provided scale factor
        smooth_kernel = np.ones((round(glyph_scale * s), round(glyph_scale * s)), np.uint8)
        img_smooth = cv2.morphologyEx(img_dil, cv2.MORPH_CLOSE, smooth_kernel)
        
        # find contours of the smoothed image
        contours = cv2.findContours(img_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours[0] if len(contours) == 2 else contours[1], key=cv2.contourArea, reverse=True)
        if not sum([cv2.contourArea(contour) for contour in contours]):
            log(f"Sum of resulting polygons equals 0 in region {region.id} ({region.type}): {xml}", logger)
            continue
        
        # postprocess contours
        if m == "largest":
            poly = Polygon([tuple(pt[0]) for pt in contours[0]])
            poly = poly.buffer(p - glyph_scale, join_style=2)
            if poly.area > 0:
                coords["points"] = polygon_to_string(np.array(poly.exterior.coords, dtype=np.int32))
            else:
                log(f"Area of resulting polygon equals 0 in region {region.id} ({region.type}): {xml}", logger)
                continue
        elif m == "merge":
            polys = [Polygon([tuple(pt[0]) for pt in contour]) for contour in contours if len(contour) > 3]  # create shapely polygons from contours
            polys = [poly.buffer(p - glyph_scale, join_style=2) for poly in polys]  # add padding to final polygons
            if polys:
                try:
                    poly = merge_polygons(polys, max(3, p))
                    if poly is None:
                        log(f"No remaining polygons after merging in region {region.id} ({region.type}): {xml}", logger)
                        continue
                    coords['points'] = polygon_to_string(np.array(poly.exterior.coords, dtype=np.int32))
                except Exception as e:
                    log(f"Error merging polygons in region {region.id} ({region.type}): {xml}\n{e}", logger)
                    continue
            else:
                log(f"No valid polygons found in region {region.id} ({region.type}): {xml}", logger)
                continue
    return pxml
