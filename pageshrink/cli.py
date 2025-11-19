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
from pathlib import Path
from typing import Literal

import click
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
from pypxml import PageXML

from . import __version__, __prog__, __footer__
from .pageshrink import shrink


logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)]
)
log = logging.getLogger(__name__)

PROGRESS = Progress(
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


def find_image(pagexml: Path, image_suffix: str | None = None) -> Path | None:
    """ Finds corresponding image files for a PageXML file """
    if image_suffix:
        image = pagexml.parent.joinpath(pagexml.name.split('.')[0] + image_suffix)
        if image.exists():
            return image
    pxml = PageXML.from_file(pagexml, skip_unknown=True)
    image = pagexml.parent.joinpath(pxml["imageFilename"])
    if image.exists():
        return image
    return None


@click.command(epilog=__footer__)
@click.help_option("--help")
@click.version_option(
    __version__, "--version",
    prog_name=__prog__,
    message=f"{__prog__} v{__version__}\n{__footer__}"
)
@click.argument(
    "files",
    type=click.Path(exists=False, dir_okay=False, file_okay=True, resolve_path=True, path_type=Path),
    required=True,
    nargs=-1
)
@click.option(
    "-i", "--images", "image_suffix",
    help="The full suffix of the binary image files to search for. If not provided, "
         "the imageFilename attribute is used. Example: '.bin.png'",
    metavar="SUFFIX",
    type=click.STRING
)
@click.option(
    "-o", "--output", "output",
    help="Specify output directory for processed files. If not set, overwrite input files.",
    type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True, path_type=Path)
)
@click.option(
    "-p", "--padding", "padding",
    help="Padding between region borders and its content in pixels.",
    type=click.INT, 
    default=5, 
    show_default=True
)
@click.option(
    "-s", "--smoothing", "smoothing",
    help="Smoothing, calculated as the factor of the average glyph size. Prevents regions cutting between text.",
    type=click.FLOAT, 
    default=1.0, 
    show_default=True
)
@click.option(
    "-m", "--mode", "mode",
    help="Shrinking mode to use for regions. 'merge' merges all resulting polygons of each region after shrinking. "
         "'largest' keeps only the largest resulting polygon of each region after shrinking.",
    type=click.Choice(["merge", "largest"]), 
    default="merge", 
    show_default=True
)
@click.option(
    "-b", "--bbox", "bbox",
    help="Draw a minimal bounding box for a specific region after shrinking. "
         "Should be of format 'PageType' or 'PageType.subtype'. Multiple regions can be specified. "
         "Examples: 'ImageRegion', 'TextRegion.paragraph'",
    metavar="PageType",
    type=click.STRING,
    multiple=True
)
@click.option(
    "-e", "--exclude", "exclude",
    help="Exclude a specific region from shrinking. Should be of format PageType or PageType.subtype. "
         "Multiple excludes can be specified. Examples: 'ImageRegion', 'TextRegion.paragraph'",
    metavar="PageType",
    type=click.STRING,
    multiple=True,
)
@click.option(
     "--logging", "logging_level",
     help="Set logging level.", 
     type=click.Choice(["ERROR", "WARNING", "INFO", "DEBUG"]),
     default="WARNING",
     show_default=True
)
def main(
    files: list[Path],
    image_suffix: str | None = None,
    output: Path | None = None,
    padding: int = 5,
    smoothing: float = 1.0,
    mode: Literal["merge", "largest"] = "merge",
    bbox: list[str] | None = None,
    exclude: list[str] | None = None,
    logging_level: Literal["ERROR", "WARNING", "INFO", "DEBUG"] = "ERROR"
) -> None:
    """
    Shrinks the region polygons of PageXML files.
    
    FILES: List of PageXML file paths to process.
    """
    log.setLevel(logging_level)
    log.info("Loading files")

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        
    with PROGRESS as progressbar:
        task = progressbar.add_task("Processing...", total=len(files), filename="")
        for fp in files:
            progressbar.update(task, filename=Path("/", *fp.parts[-min(len(fp.parts), 4):]))
            image = find_image(fp, image_suffix if image_suffix.startswith('.') else '.' + image_suffix)
            if not image:
                log.error(f"Could not find corresponding image for {fp}")
                progressbar.advance(task)
            try:
                pagexml = shrink(fp, image, padding, smoothing, mode, bbox, exclude)
                pagexml.to_file(fp if output is None else output.joinpath(fp.name))
            except Exception as e:
                log.error(f"Error while processing {fp}:\n{e}")
            progressbar.advance(task)
        progressbar.update(task, filename="Done")
