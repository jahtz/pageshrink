# pageshrink

Shrinks the region polygons of PageXML files to its content.

## Setup
>[!NOTE]
> Tested Python versions: `3.13.1`

>[!IMPORTANT]
>The following setup process uses [PyEnv](https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix)

1. Clone repository
	```shell
	git clone https://github.com/jahtz/pageshrink
	```

2. Create Virtual Environment
	```shell
	pyenv install 3.13.1
	pyenv virtualenv 3.13.1 pageshrink
	pyenv activate pageshrink
	```

3. Install dependencies
	```shell
	pip install -r pageshrink/requirements.txt
	```

## Usage
```
> python pageshrink --help
                                                                                          
 Usage: pageshrink [OPTIONS] PAGEXML...                                                   
                                                                                          
 Shrinks the region polygons of PageXML files to its content.                             
 PAGEXML: List of PageXML files or directories containing PageXML files.  Accepts         
 individual files, wildcards, or directories (with -g option for pattern matching).       
                                                                                          
╭─ Input ────────────────────────────────────────────────────────────────────────────────╮
│ *  PAGEXML       PATH  [required]                                                      │
│    --glob    -g  TEXT  Glob pattern for matching PageXML files within directories.     │
│                        Only applicable when directories are passed in PAGEXML.         │
│                        [default: *.xml]                                                │
│    --images  -i  TEXT  Suffix of the image files to search for. If not provided, the   │
│                        imageFilename attribute from the PageXML is used.               │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────╮
│ --output     -o  DIRECTORY        Specify output directory for shrunk files. If not    │
│                                   set, overwrite input files.                          │
│ --padding    -p  INTEGER          Padding between region borders and its content in    │
│                                   pixels.                                              │
│                                   [default: 5]                                         │
│ --smoothing  -s  FLOAT            Smoothing, calculated as the factor of the average   │
│                                   glyph size. Prevents regions eating between text.    │
│                                   [default: 1.0]                                       │
│ --noise      -n  INTEGER RANGE    Noise reduction, by applying a kernel of this size   │
│                                   to the image. Should be at least 3                   │
│ --mode       -m  [merge|largest]  Shrinking mode to use for regions. `Merge` merges    │
│                                   all resulting polygons of each region after          │
│                                   shrinking. `Largest` keeps only the largest          │
│                                   resulting polygon of each region after shrinking.    │
│                                   [default: merge]                                     │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Help ─────────────────────────────────────────────────────────────────────────────────╮
│ --help         Show this message and exit.                                             │
│ --version      Show the version and exit.                                              │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
