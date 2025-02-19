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
from typing import Optional, Union

import rich_click as click
from rich import print as rprint
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn, TimeElapsedColumn
from pypxml import PageXML

from pageshrink import shrink, SHRINK_MODE


__version__ = "2.0.0"
__prog__ = "pageshrink"

click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.MAX_WIDTH = 96
click.rich_click.RANGE_STRING = ""
click.rich_click.OPTION_GROUPS = {
    "*": [
        {
            "name": "Input",
            "options": ["pagexml", "--glob", "--images"]
        },
        {
            "name": "Options",
            "options": ["--output", "--padding", "--smoothing", "--noise", "--mode"]
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


@click.command()
@click.help_option("--help")
@click.version_option(__version__,
                      "--version",
                      prog_name=__prog__,
                      message=f"{__prog__} v{__version__} - Developed at Centre for Philology and Digitality (ZPD), "
                              f"University of Würzburg")
@click.argument("pagexml",
                type=click.Path(exists=True, file_okay=True, dir_okay=True, resolve_path=True),
                callback=paths_callback, required=True, nargs=-1)
@click.option("-g", "--glob", "glob",
              type=click.STRING, default="*.xml", required=False, show_default=True,
              help="Glob pattern for matching PageXML files within directories. "
                   "Only applicable when directories are passed in PAGEXML.")
@click.option("-i", "--images", "image_suffix",
              type=click.STRING, callback=suffix_callback, required=False, show_default=True,
              help="Suffix of the image files to search for. "
                   "If not provided, the imageFilename attribute from the PageXML is used.")
@click.option("-o", "--output", "output",
              type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True),
              callback=path_callback, required=False,
              help="Specify output directory for shrunk files. If not set, overwrite input files.")
@click.option("-p", "--padding", "padding",
              type=click.INT, default=5, show_default=True,
              help="Padding between region borders and its content in pixels.")
@click.option("-s", "--smoothing", "smoothing",
              type=click.FLOAT, default=1.0, show_default=True,
              help="Smoothing, calculated as the factor of the average glyph size. "
                   "Prevents regions eating between text.")
@click.option("-n", "--noise", "noise",
              type=click.IntRange(3),
              help="Noise reduction, by applying a kernel of this size to the image. Should be at least 3")
@click.option("-m", "--mode", "mode",
              type=click.Choice(["merge", "largest"], case_sensitive=False), default="merge", show_default=True,
              help="Shrinking mode to use for regions. "
                   "`Merge` merges all resulting polygons of each region after shrinking. "
                   "`Largest` keeps only the largest resulting polygon of each region after shrinking.")
def pageshrink_cli(pagexml: list[Path], glob: str = "*.xml", image_suffix: Optional[str] = None,
                   output: Optional[Path] = None, mode: SHRINK_MODE = "merge",
                   padding: int = 5, smoothing: float = 1.0, noise: int = 1):
    pagexml = expand_paths(pagexml, glob)
    if not pagexml:
        rprint("[bold red]ERROR:[/bold red] No valid PageXML files found.")
        return
    if output and not output.exists():
        output.mkdir(parents=True, exist_ok=True)
    with progress as p:
        task = p.add_task("Shrinking regions...", total=len(pagexml), filename="")
        for fp in pagexml:
            p.update(task, filename=fp)
            image = find_image(fp, image_suffix)
            if not image:
                p.log(f"Could not find corresponding image for {fp}")
                p.update(task, advance=1)
                continue
            try:
                pxml = shrink(fp, image, padding, smoothing, noise, mode, logger=p)
                if output:
                    pxml.to_xml(output.joinpath(fp.name))
                else:
                    pxml.to_xml(fp)
            except Exception as e:
                p.log(f"Error processing {fp}: {e}")
            p.update(task, advance=1)
        p.update(task, filename="Done!")


if __name__ == "__main__":
    pageshrink_cli()