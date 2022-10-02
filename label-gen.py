import argparse
import pathlib
from conf import paths, default
from osgeo import ogr, gdal, gdalconst
from skimage.morphology import disk, dilation, erosion
import os
import numpy as np

parser = argparse.ArgumentParser(
    description='Generate .tif label file from PRODES deforestation shapefile.'
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

parser.add_argument( # PRODES hydrography shapefile
    '-w', '--hydrography-shape',
    type = pathlib.Path,
    default = paths.PRODES_HYDRO_SHP,
    help = 'Path to PRODES hydrography shapefile (.shp)'
)

parser.add_argument( # PRODES no forest shapefile
    '-n', '--no-forest-shape',
    type = pathlib.Path,
    default = paths.PRODES_NO_FOREST_DEF_SHP,
    help = 'Path to PRODES no forest shapefile (.shp)'
)

parser.add_argument( # referent year 
    '-y', '--year',
    type = int,
    default = 2018,
    help = 'Reference year to generate the labels'
)

parser.add_argument( # output 
    '-o', '--output-path',
    type = pathlib.Path,
    default = paths.LABELS_PATH,
    help = 'Path to output label .tif folder'
)

parser.add_argument( # inner buffer size 
    '-i', '--inner-buffer',
    type = int,
    default = default.LABEL_INNER_BUFFER,
    help = 'Inner buffer between deforestation and no deforestation to be ignored'
)

parser.add_argument( # outer buffer size 
    '-u', '--outer-buffer',
    type = int,
    default = default.LABEL_OUTER_BUFFER,
    help = 'Outer buffer between deforestation and no deforestation to be ignored'
)

args = parser.parse_args()

v_yearly_def = ogr.Open(str(args.deforestation_shape))
l_yearly_def = v_yearly_def.GetLayer()

v_previous_def = ogr.Open(str(args.previous_deforestation_shape))
l_previous_def = v_previous_def.GetLayer()

v_no_forest = ogr.Open(str(args.no_forest_shape))
l_no_forest = v_no_forest.GetLayer()

v_hydrography = ogr.Open(str(args.hydrography_shape))
l_hydrography = v_hydrography.GetLayer()

base_data = gdal.Open(str(args.base_image), gdalconst.GA_ReadOnly)

geo_transform = base_data.GetGeoTransform()
x_res = base_data.RasterXSize
y_res = base_data.RasterYSize
crs = base_data.GetSpatialRef()
proj = base_data.GetProjection()

output = os.path.join(args.output_path, f'{default.LABEL_PREFIX}_{args.year}.tif')

target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Byte)
target_ds.SetGeoTransform(geo_transform)
target_ds.SetSpatialRef(crs)
target_ds.SetProjection(proj)

band = target_ds.GetRasterBand(1)
band.FlushCache()
where_past = f'"year"<={args.year-1}'
where_ref = f'"year"={args.year}'

gdal.RasterizeLayer(target_ds, [1], l_previous_def, burn_values=[2])
gdal.RasterizeLayer(target_ds, [1], l_no_forest, burn_values=[2])
gdal.RasterizeLayer(target_ds, [1], l_hydrography, burn_values=[2])

l_yearly_def.SetAttributeFilter(where_past)
gdal.RasterizeLayer(target_ds, [1], l_yearly_def, burn_values=[2])

l_yearly_def.SetAttributeFilter(where_ref)
gdal.RasterizeLayer(target_ds, [1], l_yearly_def, burn_values=[1])

rasterized_data = target_ds.ReadAsArray() 

defor_data = rasterized_data == 1
defor_data = defor_data.astype(np.uint8)
border_data = dilation(defor_data, disk(args.inner_buffer)) - erosion(defor_data, disk(args.outer_buffer))

rasterized_data[border_data==1] = 2

target_ds.GetRasterBand(1).WriteArray(rasterized_data)
target_ds = None