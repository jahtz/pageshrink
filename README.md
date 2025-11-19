# pageshrink

Shrinks the region polygons of PageXML files to its content.

## Setup
> [!NOTE]
> Python: `>=3.13`

1. Clone repository
	```shell
	git clone https://github.com/jahtz/pageshrink
	```

2. Install dependencies
	```shell
	pip install ./pageshrink
	```

## Usage
```
$ pageshrink --help
Usage: pageshrink [OPTIONS] FILES...

  Shrinks the region polygons of PageXML files.

  FILES: List of PageXML file paths to process.

Options:
  --help                          Show this message and exit.
  --version                       Show the version and exit.
  -i, --images SUFFIX             The full suffix of the binary image files to
                                  search for. If not provided, the
                                  imageFilename attribute is used. Example:
                                  '.bin.png'
  -o, --output DIRECTORY          Specify output directory for processed
                                  files. If not set, overwrite input files.
  -p, --padding INTEGER           Padding between region borders and its
                                  content in pixels.  [default: 5]
  -s, --smoothing FLOAT           Smoothing, calculated as the factor of the
                                  average glyph size. Prevents regions cutting
                                  between text.  [default: 1.0]
  -m, --mode [merge|largest]      Shrinking mode to use for regions. 'merge'
                                  merges all resulting polygons of each region
                                  after shrinking. 'largest' keeps only the
                                  largest resulting polygon of each region
                                  after shrinking.  [default: merge]
  -b, --bbox PageType             Draw a minimal bounding box for a specific
                                  region after shrinking. Should be of format
                                  'PageType' or 'PageType.subtype'. Multiple
                                  regions can be specified. Examples:
                                  'ImageRegion', 'TextRegion.paragraph'
  -e, --exclude PageType          Exclude a specific region from shrinking.
                                  Should be of format PageType or
                                  PageType.subtype. Multiple excludes can be
                                  specified. Examples: 'ImageRegion',
                                  'TextRegion.paragraph'
  --logging [ERROR|WARNING|INFO|DEBUG]
                                  Set logging level.  [default: WARNING]
```

### Example
```shell
$ pageshrink ./samples/*.xml -i .sbb.bin.png -o ./samples_shrinked -b TableRegion -b ImageRegion
```

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
