import argparse
import pathlib
from conf import paths, default
from ops.ops import load_opt_image
import numpy as np
from osgeo import gdal, gdalconst

parser = argparse.ArgumentParser(
    description='Generate .tif with training (0) and validation (1) areas.'
)

parser.add_argument( # base image 
    '-b', '--base-image',
    type = pathlib.Path,
    default = paths.OPT_TIFF_FILE_0,
    help = 'Path to optical tiff file as base to generate aligned data'
)

parser.add_argument( # number of tiles lines
    '-l', '--lines',
    type = int,
    default = default.TILES_LIN,
    help = 'Number of lines to split the image'
)

parser.add_argument( # number of tiles columns
    '-c', '--columns',
    type = int,
    default = default.TILES_COL,
    help = 'Number of columns to split the image'
)

parser.add_argument( # index the tiles to be marked as validation
    '-v', '--validation-tiles',
    default = default.TILES_VALIDATION,
    help = 'Index of the tiles to be marked as validation'
)

parser.add_argument( # output file
    '-o', '--output',
    type = pathlib.Path,
    default = paths.TILES_PATH,
    help = 'Path to the output tiles file (.tif)'
)

args = parser.parse_args()

shape = load_opt_image(str(args.base_image)).shape[0:2]

tiles = np.zeros(shape, dtype=np.uint8).reshape((-1,1))
idx_matrix = np.arange(shape[0]*shape[1], dtype=np.uint32).reshape(shape)

tiles_idx = []
for hor in np.array_split(idx_matrix, args.lines, axis=0):
    for tile in np.array_split(hor, args.columns, axis=1):
        tiles_idx.append(tile)

   
for i, tile in enumerate(tiles_idx):
    if i in args.validation_tiles:
        tiles[tile] = 1

tiles = tiles.reshape(shape)

base_data = gdal.Open(str(args.base_image), gdalconst.GA_ReadOnly)
geo_transform = base_data.GetGeoTransform()
x_min = geo_transform[0]
y_max = geo_transform[3]
x_max = x_min + geo_transform[1] * base_data.RasterXSize
y_min = y_max + geo_transform[5] * base_data.RasterYSize
x_res = base_data.RasterXSize
y_res = base_data.RasterYSize

crs = base_data.GetSpatialRef()
proj = base_data.GetProjection()

pixel_width = geo_transform[1]

target_ds = gdal.GetDriverByName('GTiff').Create(str(args.output), x_res, y_res, 1, gdal.GDT_Byte)

target_ds.SetGeoTransform(geo_transform)
target_ds.SetSpatialRef(crs)
target_ds.SetProjection(proj)
band = target_ds.GetRasterBand(1)
band.FlushCache()

target_ds.GetRasterBand(1).WriteArray(tiles)
target_ds = None