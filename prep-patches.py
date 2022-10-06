import argparse
import pathlib
from conf import paths, default, general
import numpy as np
from osgeo import gdal, gdalconst, ogr
import os
from ops.ops import load_opt_image, filter_outliers, load_label_image
from matplotlib import pyplot as plt
from skimage.util import view_as_windows


parser = argparse.ArgumentParser(
    description='prepare the files to be used in the training/testing steps'
)

parser.add_argument( # optical image 0 image 
    '--image-0',
    type = pathlib.Path,
    default = paths.OPT_TIFF_FILE_0,
    help = 'Path to optical image (.tif) file of year 0'
)

parser.add_argument( # optical image 1 image 
    '--image-1',
    type = pathlib.Path,
    default = paths.OPT_TIFF_FILE_1,
    help = 'Path to optical image (.tif) file of year 1'
)

parser.add_argument( # optical image 2 image 
    '--image-2',
    type = pathlib.Path,
    default = paths.OPT_TIFF_FILE_2,
    help = 'Path to optical image (.tif) file of year 2'
)

parser.add_argument( # tiles file path 
    '-t', '--tiles',
    type = pathlib.Path,
    default = paths.TILES_PATH,
    help = 'Path to the tiles file (.tif)'
)

parser.add_argument( # filter outliers?
    '-f', '--filter-outliers',
    type = bool,
    default = False,
    help = 'Filter image outliers option'
)

parser.add_argument( # min deforestation
    '-m', '--min-deforestation',
    type = float,
    default = default.MIN_DEFORESTATION,
    help = 'Minimum deforestation'
)

args = parser.parse_args()

if not os.path.exists(paths.PREPARED_PATH):
    os.mkdir(paths.PREPARED_PATH)

img_0 = load_opt_image(str(args.image_0))
img_1 = load_opt_image(str(args.image_1))
img_2 = load_opt_image(str(args.image_2))

print('img_0:', img_0.shape, img_0.dtype)
print('img_1:', img_1.shape, img_1.dtype)
print('img_2:', img_2.shape, img_2.dtype)

if args.filter_outliers:
    img_0 = filter_outliers(img_0)
    img_1 = filter_outliers(img_1)
    img_2 = filter_outliers(img_2)


mean = []
std = []

mean.append(img_0.mean())
mean.append(img_1.mean())
mean.append(img_2.mean())

std.append(img_0.std())
std.append(img_1.std())
std.append(img_2.std())

mean = np.array(mean).mean()
std = np.array(std).mean()

print('mean:', mean, 'std:', std)

statistics = np.array([
    mean,
    std
])
np.save(os.path.join(paths.PREPARED_PATH, 'statistics.npy'), statistics)

img_0 = (img_0 - mean) / std
img_1 = (img_1 - mean) / std
img_2 = (img_2 - mean) / std

np.save(os.path.join(paths.PREPARED_PATH, f'{general.YEAR_0}.npy'), img_0.astype(np.float16))
np.save(os.path.join(paths.PREPARED_PATH, f'{general.YEAR_1}.npy'), img_1.astype(np.float16))
np.save(os.path.join(paths.PREPARED_PATH, f'{general.YEAR_2}.npy'), img_2.astype(np.float16))

print(f'Image 0 Mean: {img_0.mean():.4f} | Std: {img_0.std():.4f}')
print(f'Image 1 Mean: {img_1.mean():.4f} | Std: {img_1.std():.4f}')
print(f'Image 2 Mean: {img_2.mean():.4f} | Std: {img_2.std():.4f}')

del img_0, img_1, img_2

#tiles file open
tiles = load_label_image(str(args.tiles))

#label preparation
label_1 = os.path.join(paths.LABELS_PATH, f'{general.LABEL_PREFIX}_{general.YEAR_1}.tif')
label_2 = os.path.join(paths.LABELS_PATH, f'{general.LABEL_PREFIX}_{general.YEAR_2}.tif')

label_1 = load_label_image(label_1)
label_2 = load_label_image(label_2)

np.save(os.path.join(paths.PREPARED_PATH, f'{general.LABEL_PREFIX}_{general.YEAR_1}.npy'), label_1)
np.save(os.path.join(paths.PREPARED_PATH, f'{general.LABEL_PREFIX}_{general.YEAR_2}.npy'), label_2)


#previous deforestation map preparation
previous_1 = os.path.join(paths.LABELS_PATH, f'{general.PREVIOUS_PREFIX}_{general.YEAR_1}.tif')
previous_2 = os.path.join(paths.LABELS_PATH, f'{general.PREVIOUS_PREFIX}_{general.YEAR_2}.tif')

previous_1 = load_label_image(previous_1)
previous_2 = load_label_image(previous_2)

np.save(os.path.join(paths.PREPARED_PATH, f'{general.PREVIOUS_PREFIX}_{general.YEAR_1}.npy'), previous_1)
np.save(os.path.join(paths.PREPARED_PATH, f'{general.PREVIOUS_PREFIX}_{general.YEAR_2}.npy'), previous_2)

plt.imshow(tiles)

shape = label_1.shape[0:2]
idx_matrix = np.arange(shape[0]*shape[1], dtype=np.uint32).reshape(shape)

patch_size = general.PATCH_SIZE
train_step = int((1-general.PATCH_OVERLAP)*patch_size)

label_patches_1 = view_as_windows(label_1, (patch_size, patch_size), train_step).reshape((-1, patch_size, patch_size))
label_patches_2 = view_as_windows(label_2, (patch_size, patch_size), train_step).reshape((-1, patch_size, patch_size))
tiles_patches = view_as_windows(tiles, (patch_size, patch_size), train_step).reshape((-1, patch_size, patch_size))
idx_patches = view_as_windows(idx_matrix, (patch_size, patch_size), train_step).reshape((-1, patch_size, patch_size))

keep_patches_1 = np.mean((label_patches_1 == 1), axis=(1,2)) >= args.min_deforestation
keep_patches_2 = np.mean((label_patches_2 == 1), axis=(1,2)) >= args.min_deforestation

tiles_patches_1 = tiles_patches[keep_patches_1]
tiles_patches_2 = tiles_patches[keep_patches_2]
print(tiles_patches_1.shape)
print(tiles_patches_2.shape)

keep_patches_1_train = np.all(tiles_patches_1 == 0, axis=(1,2))
keep_patches_1_val = np.all(tiles_patches_1 == 1, axis=(1,2))

keep_patches_2_train = np.all(tiles_patches_2 == 0, axis=(1,2))
keep_patches_2_val = np.all(tiles_patches_2 == 1, axis=(1,2))

idx_patches_1_train = idx_patches[keep_patches_1][keep_patches_1_train]
idx_patches_1_val = idx_patches[keep_patches_1][keep_patches_1_val]
idx_patches_2_train = idx_patches[keep_patches_2][keep_patches_2_train]
idx_patches_2_val = idx_patches[keep_patches_2][keep_patches_2_val]

print('Train: train tiles patches shape:', idx_patches_1_train.shape)
print('Train: validation tiles patches shape:', idx_patches_1_val.shape)
print('Test: train tiles patches patches:', idx_patches_2_train.shape)
print('Test: train tiles patches patches:', idx_patches_2_val.shape)

np.save(os.path.join(paths.PREPARED_PATH, f'{general.DEFORESTATION_PATCHES_PREFIX}_{general.YEAR_1}_train.npy'), idx_patches_1_train)
np.save(os.path.join(paths.PREPARED_PATH, f'{general.DEFORESTATION_PATCHES_PREFIX}_{general.YEAR_1}_val.npy'), idx_patches_1_val)

np.save(os.path.join(paths.PREPARED_PATH, f'{general.DEFORESTATION_PATCHES_PREFIX}_{general.YEAR_2}_train.npy'), idx_patches_2_train)
np.save(os.path.join(paths.PREPARED_PATH, f'{general.DEFORESTATION_PATCHES_PREFIX}_{general.YEAR_2}_val.npy'), idx_patches_2_val)

del idx_patches, label_patches_1, label_patches_2, idx_patches_1_train, idx_patches_1_val, idx_patches_2_train, idx_patches_2_val

label_patches_1 = view_as_windows(label_1, (patch_size, patch_size), train_step).reshape((-1, patch_size, patch_size))
label_patches_2 = view_as_windows(label_2, (patch_size, patch_size), train_step).reshape((-1, patch_size, patch_size))
tiles_patches = view_as_windows(tiles, (patch_size, patch_size), train_step).reshape((-1, patch_size, patch_size))
idx_patches = view_as_windows(idx_matrix, (patch_size, patch_size), train_step).reshape((-1, patch_size, patch_size))

keep_patches_1 = np.mean((label_patches_1 == 1), axis=(1,2)) < args.min_deforestation
keep_patches_2 = np.mean((label_patches_2 == 1), axis=(1,2)) < args.min_deforestation

tiles_patches_1 = tiles_patches[keep_patches_1]
tiles_patches_2 = tiles_patches[keep_patches_2]
print(tiles_patches_1.shape)
print(tiles_patches_2.shape)

keep_patches_1_train = np.all(tiles_patches_1 == 0, axis=(1,2))
keep_patches_1_val = np.all(tiles_patches_1 == 1, axis=(1,2))

keep_patches_2_train = np.all(tiles_patches_2 == 0, axis=(1,2))
keep_patches_2_val = np.all(tiles_patches_2 == 1, axis=(1,2))

idx_patches_1_train = idx_patches[keep_patches_1][keep_patches_1_train]
idx_patches_1_val = idx_patches[keep_patches_1][keep_patches_1_val]
idx_patches_2_train = idx_patches[keep_patches_2][keep_patches_2_train]
idx_patches_2_val = idx_patches[keep_patches_2][keep_patches_2_val]

print('Train: complementary train tiles patches shape:', idx_patches_1_train.shape)
print('Train: complementary validation tiles patches shape:', idx_patches_1_val.shape)
print('Test: complementary train tiles patches patches:', idx_patches_2_train.shape)
print('Test: complementary train tiles patches patches:', idx_patches_2_val.shape)

np.save(os.path.join(paths.PREPARED_PATH, f'{general.NO_DEFORESTATION_PATCHES_PREFIX}_{general.YEAR_1}_train.npy'), idx_patches_1_train)
np.save(os.path.join(paths.PREPARED_PATH, f'{general.NO_DEFORESTATION_PATCHES_PREFIX}_{general.YEAR_1}_val.npy'), idx_patches_1_val)

np.save(os.path.join(paths.PREPARED_PATH, f'{general.NO_DEFORESTATION_PATCHES_PREFIX}_{general.YEAR_2}_train.npy'), idx_patches_2_train)
np.save(os.path.join(paths.PREPARED_PATH, f'{general.NO_DEFORESTATION_PATCHES_PREFIX}_{general.YEAR_2}_val.npy'), idx_patches_2_val)
