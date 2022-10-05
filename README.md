# Codes for XX

## Previous Deforestation Map Generator
usage: previous-def-gen.py [-h] [-b BASE_IMAGE] [-d DEFORESTATION_SHAPE] [-p PREVIOUS_DEFORESTATION_SHAPE] [-y YEAR] [-o OUTPUT_PATH]

Generate .tif previous deforestation temporal distance map. As older is the deforestation, the value is close to 0. As recent is the deforestation, the value is      
close to 1

optional arguments:
  -h, --help            show this help message and exit
  -b BASE_IMAGE, --base-image BASE_IMAGE
                        Path to optical tiff file as base to generate aligned labels
  -d DEFORESTATION_SHAPE, --deforestation-shape DEFORESTATION_SHAPE
                        Path to PRODES yearly deforestation shapefile (.shp)
  -p PREVIOUS_DEFORESTATION_SHAPE, --previous-deforestation-shape PREVIOUS_DEFORESTATION_SHAPE
                        Path to PRODES previous deforestation shapefile (.shp)
  -y YEAR, --year YEAR  Reference year to generate the temporal distance map. The higher value is 'year'-1.
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path to output label .tif folder
                        
 
 ## Label Generation
 Generate .tif label file from PRODES deforestation shapefile.

optional arguments:
  -h, --help            show this help message and exit
  -b BASE_IMAGE, --base-image BASE_IMAGE
                        Path to optical tiff file as base to generate aligned labels
  -d DEFORESTATION_SHAPE, --deforestation-shape DEFORESTATION_SHAPE
                        Path to PRODES yearly deforestation shapefile (.shp)
  -p PREVIOUS_DEFORESTATION_SHAPE, --previous-deforestation-shape PREVIOUS_DEFORESTATION_SHAPE
                        Path to PRODES previous deforestation shapefile (.shp)
  -w HYDROGRAPHY_SHAPE, --hydrography-shape HYDROGRAPHY_SHAPE
                        Path to PRODES hydrography shapefile (.shp)
  -n NO_FOREST_SHAPE, --no-forest-shape NO_FOREST_SHAPE
                        Path to PRODES no forest shapefile (.shp)
  -y YEAR, --year YEAR  Reference year to generate the labels
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path to output label .tif folder
  -i INNER_BUFFER, --inner-buffer INNER_BUFFER
                        Inner buffer between deforestation and no deforestation to be ignored
  -u OUTER_BUFFER, --outer-buffer OUTER_BUFFER
                        Outer buffer between deforestation and no deforestation to be ignored
                        
<h1 align="center">Hi 👋, I'm Felipe</h1>
<h3 align="center">A passionate frontend developer from India</h3>

<h3 align="left">Connect with me:</h3>
<p align="left">
</p>

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> </p>
