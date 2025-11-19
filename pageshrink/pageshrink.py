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

import logging
from math import ceil
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pypxml import PageXML, PageType
from scipy.spatial import distance_matrix
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union


log = logging.getLogger(__name__)


# When estimating the region scales, a bounding box is calculated around each contour.
# For some content types, such as text, using the larger value results in better results 
# (e.g. the height of the glyph 'i' is more important than its width).
# For other types, such as separator lines, using the larger value would result in huge polygons. 
# Here, using the smaller value should result in better results
ESTIMATE_MAX = [PageType.TextRegion]
ESTIMATE_MIN = [PageType.SeparatorRegion]


def is_binary(image: np.ndarray) -> bool:
    """ Checks if an image is bitonal """
    unique = np.unique(image)
    return any(np.array_equal(unique, eq) for eq in [[0, 255], [0], [255]])


def validate_image(pagexml: PageXML, image: np.ndarray):
    """ Checks if the size of an image matches a PageXML file """
    height, width = image.shape
    return int(pagexml["imageHeight"]) == height and int(pagexml["imageWidth"]) == width


def points_to_array(points: str) -> np.ndarray:
    """ Converts a PageXML points string to a numpy array """
    return np.array([tuple(map(int, xy.split(','))) for xy in points.split()], dtype=np.int32)


def array_to_points(array: np.ndarray) -> str:
    """ Converts a numpy array to a PageXML points string """
    return ' '.join([f"{max(0, round(xy[0]))},{max(0, round(xy[1]))}" for xy in array])


def estimate_scales(image: np.ndarray, pagetype: PageType | None = None) -> int:
    """
    Estimates the median contour size from an image or region.
    Args:
        image: The binary image (may be masked) to calculate the scale from.
        pagetype: If present, the type of the region to choose between the smallest or largest value. 
                  By default, 25 is returned. Defaults to None.
    Returns:
        The median contour size, calculated from the biggest 50% of the contours (for noise removal).
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    if len(stats) > 1:  # stats[0] is the background
        widths = sorted(stats[1:, 2], reverse=True)
        heights = sorted(stats[1:, 3], reverse=True)
        avg_width = np.median(widths[:ceil(len(widths) * 0.5)])
        avg_height = np.median(heights[:ceil(len(heights) * 0.5)])
        if pagetype and pagetype in ESTIMATE_MAX:
            return max(1, max(round(avg_width), round(avg_height)))
        elif pagetype and pagetype in ESTIMATE_MIN:
            return max(1, min(round(avg_width), round(avg_height)))
    return 25


def merge_polygons(polygons: list[Polygon], b: int = 1) -> Polygon | None:
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
        # this could probably be done using some library or in a more elegant way...
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

        # connect polygons using mst
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
    

def shrink(
    pagexml: Path,
    binaryimage: Path,
    padding: int = 5,
    smoothing: float = 1.0,
    mode: Literal["merge", "largest"] = "merge",
    bbox: list[str] | None = None,
    exclude: list[str] | None = None
) -> PageXML:
    """
    Shrink the region polygons of a PageXML file.
    Args:
        pagexml: The PageXML file to process.
        image: The related binary image for region shrinking.
        padding: Padding between region borders and its content. Defaults to 5.
        smoothing: Smoothing of the region, calculated as the factor of the average glyph size. Defaults to 1.0.
        mode: The shrinking mode to use. 'merge' merges all resulting polygons of each region after shrinking. 
              'largest' only keeps the largest resulting polygon of each region after shrinking. Defaults to "merge".
        bbox: A list of regions to draw a minimal bounding box after shrinking. 
              They should be in format PageType or PageType.subtype. Defaults to None.
        exclude: A list of regions to exclude from shrinking. They should be in format PageType or PageType.subtype. 
                 Defaults to None.
    Returns:
        The processed PageXML object.
    """
    log.info(f"Processing {pagexml} with {binaryimage}")
    xml = PageXML.from_file(pagexml, raise_on_error=False)
    im = ~cv2.imread(binaryimage.as_posix(), cv2.IMREAD_GRAYSCALE)  # invert
    
    if not is_binary(im):
        raise ValueError("Image is not binary")
    if not validate_image(xml, im):
        raise ValueError("Image and PageXML do not match in size")
    
    for region in xml.regions:
        if exclude is not None:
            if region.pagetype.value in exclude or f"{region.pagetype.value}.{str(region['type'])}" in exclude:
                continue
        log.debug(f"Processing region {region}")
        coords = region.find_coords()
        if coords is None or "points" not in coords:
            log.warning(f"Could not find a valid Coords element in {region} ({pagexml})")
            continue
        
        log.debug("Masking region")
        region_array = points_to_array(coords["points"])
        region_mask = np.zeros(im.shape[:2], dtype=np.uint8)
        cv2.fillPoly(region_mask, [region_array], (255, 255, 255))
        im_masked = im & region_mask
                
        log.debug("Calculating region contour scales")
        scale = estimate_scales(im_masked, region.pagetype)
        log.debug(f"Estimated scale: {scale}")
        
        log.debug("Dilate text considering the median contour scale")
        dilation_kernel = np.ones((scale, scale), np.uint8)
        im_dilated = cv2.dilate(im_masked, dilation_kernel, iterations=2)
        
        log.debug("Close gaps between symbols using provided smoothing factor")
        smooth_kernel = np.ones((round(scale * smoothing), round(scale * smoothing)), np.uint8)
        im_smoothed = cv2.morphologyEx(im_dilated, cv2.MORPH_CLOSE, smooth_kernel)
        
        log.debug("Calculting content contours")
        contours = cv2.findContours(im_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours[0] if len(contours) == 2 else contours[1], key=cv2.contourArea, reverse=True)
        if not sum([cv2.contourArea(contour) for contour in contours]):
            log.warning(f"Sum of resulting polygons equals 0 in region {region} ({pagexml})")
            continue
        
        log.debug("Converting found contours to polygons")
        if bbox is not None:
            if region.pagetype.value in bbox or f"{region.pagetype.value}.{str(region['type'])}" in bbox:
                contours = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(contours)
                polygon = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
                polygon = polygon.buffer(padding - scale, join_style="mitre")
                if polygon.area > 0:
                    coords["points"] = array_to_points(np.array(polygon.exterior.coords, dtype=np.int32))
                else:
                    log.warning(f"Area of resulting bounding box equals 0 in region {region} ({pagexml})")
                continue
        
        if mode == "largest":
            polygon = Polygon([tuple(pt[0] for pt in contours[0])])
            polygon = polygon.buffer(padding - scale, join_style="mitre")
            if polygon.area > 0:
                coords["points"] = array_to_points(np.array(polygon.exterior.coords, dtype=np.int32))
            else:
                log.warning(f"Area of resulting polygon equals 0 in region {region} ({pagexml})")
                
        elif mode == "merge":
            polygons = [Polygon([tuple(pt[0]) for pt in contour]) for contour in contours if len(contour) > 3]  # create shapely polygons from contours
            polygons = [poly.buffer(padding - scale, join_style="mitre") for poly in polygons]  # add padding to final polygons
            if not polygons:
                log.warning(f"No valid polygons remaining in region {region} ({pagexml})")
                continue
            log.debug(f"Merging {len(polygons)} resulting polygons")
            polygon = merge_polygons(polygons, max(3, padding))
            if polygon is None:  # if merging fails, keep original polygon
                log.warning(f"No remaining polygons after merging in region {region} ({pagexml})")
                continue
            coords["points"] = array_to_points(np.array(polygon.exterior.coords, dtype=np.int32))
            
    return xml
