import argparse
import pathlib
from conf import paths, default
import numpy as np
from osgeo import gdal, gdalconst, ogr
import os

parser = argparse.ArgumentParser(
    description='Generate .tif previous deforestation temporal distance map. As older is the deforestation, the value is close to 0. As recent is the deforestation, the value is close to 1'
)

parser.add_argument( # base image 
    '-b', '--base-image',
    type = pathlib.Path,
    default = paths.OPT_TIFF_FILE_0,
    help = 'Path to optical tiff file as base to generate aligned labels'
)

parser.add_argument( # PRODES yearly deforestation shapefile 
    '-d', '--deforestation-shape',
    type = pathlib.Path,
    default = paths.PRODES_YEAR_DEF_SHP,
    help = 'Path to PRODES yearly deforestation shapefile (.shp)'
)

parser.add_argument( # PRODES previous deforestation shapefile 
    '-p', '--previous-deforestation-shape',
    type = pathlib.Path,
    default = paths.PRODES_PREV_DEF_SHP,
    help = 'Path to PRODES previous deforestation shapefile (.shp)'
)

parser.add_argument( # referent year 
    '-y', '--year',
    type = int,
    default = 2018,
    help = "Reference year to generate the temporal distance map. The higher value is 'year'-1."
)

parser.add_argument( # output 
    '-o', '--output-path',
    type = pathlib.Path,
    default = paths.LABELS_PATH,
    help = 'Path to output label .tif folder'
)

args = parser.parse_args()

v_yearly_def = ogr.Open(str(args.deforestation_shape))
l_yearly_def = v_yearly_def.GetLayer()

v_previous_def = ogr.Open(str(args.previous_deforestation_shape))
l_previous_def = v_previous_def.GetLayer()

base_data = gdal.Open(str(args.base_image), gdalconst.GA_ReadOnly)
geo_transform = base_data.GetGeoTransform()
x_res = base_data.RasterXSize
y_res = base_data.RasterYSize
crs = base_data.GetSpatialRef()
proj = base_data.GetProjection()

output = os.path.join(args.output_path, f'{default.PREVIOUS_PREFIX}_{args.year}.tif')

target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Float32)
target_ds.SetGeoTransform(geo_transform)
target_ds.SetSpatialRef(crs)
target_ds.SetProjection(proj)

last_year = args.year - 1
b_year = 2007
years = np.arange(b_year, args.year)
vals = np.linspace(0,1, len(years)+1)


gdal.RasterizeLayer(target_ds, [1], l_previous_def, burn_values=[vals[1]])
print('prev', vals[1])

for i, t_year in enumerate(years[1:]):
    v = vals[i+2]
    print(t_year, v)
    where = f'"year"={t_year}'
    l_yearly_def.SetAttributeFilter(where)
    gdal.RasterizeLayer(target_ds, [1], l_yearly_def, burn_values=[v])

target_ds = None