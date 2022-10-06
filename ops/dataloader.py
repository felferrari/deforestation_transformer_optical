from email.policy import default
import numpy as np
import tensorflow as tf
import os
from conf import general, paths
from skimage.util import view_as_windows
import random
from keras.utils import to_categorical


def train_data_gen(year):
    prep_path = paths.PREPARED_PATH

    n_img_layers = general.N_IMG_LAYERS
    patch_size = general.PATCH_SIZE
    n_classes = general.N_CLASSES

    patches_idxs_train = np.load(os.path.join(prep_path, f'{general.DEFORESTATION_PATCHES_PREFIX}_{year}_train.npy' ))
    patches_idxs_val = np.load(os.path.join(prep_path, f'{general.DEFORESTATION_PATCHES_PREFIX}_{year}_val.npy' ))

    c_patches_idxs_train = np.load(os.path.join(prep_path, f'{general.NO_DEFORESTATION_PATCHES_PREFIX}_{year}_train.npy' ))
    c_patches_idxs_val = np.load(os.path.join(prep_path, f'{general.NO_DEFORESTATION_PATCHES_PREFIX}_{year}_val.npy' ))

    n_patches_idxs_train = c_patches_idxs_train.shape[0]
    n_patches_idxs_val = c_patches_idxs_val.shape[0]
    
    t_0 = f'{year-1}'
    t_1 = f'{year}'

    labels = np.load(os.path.join(prep_path, f'{general.LABEL_PREFIX}_{year}.npy'), allow_pickle=True).reshape((-1,1))
    previous = np.load(os.path.join(prep_path, f'{general.PREVIOUS_PREFIX}_{year}.npy'), allow_pickle=True).reshape((-1,1))

    img_0 = np.load(os.path.join(prep_path, f'{t_0}.npy'), allow_pickle=True).reshape((-1,n_img_layers))
    img_1 = np.load(os.path.join(prep_path, f'{t_1}.npy'), allow_pickle=True).reshape((-1,n_img_layers))

    def func_train():
        while True:
            np.random.shuffle(patches_idxs_train)
            for patch_idx in patches_idxs_train:
                yield ( #yield a patch wiht at least 2% of deforestation
                    img_0[patch_idx].reshape((patch_size,patch_size,n_img_layers)).astype(np.float32),
                    img_1[patch_idx].reshape((patch_size,patch_size,n_img_layers)).astype(np.float32),
                    previous[patch_idx].reshape((patch_size,patch_size,1)).astype(np.float32),
                    to_categorical(labels[patch_idx].reshape((patch_size, patch_size, 1)), n_classes).astype(np.float32)
                )
                c_patch_idx = c_patches_idxs_train[random.randrange(n_patches_idxs_train)]
                yield ( #yield a patch with less than 2% of deforestation
                    img_0[c_patch_idx].reshape((patch_size,patch_size,n_img_layers)).astype(np.float32),
                    img_1[c_patch_idx].reshape((patch_size,patch_size,n_img_layers)).astype(np.float32),
                    previous[c_patch_idx].reshape((patch_size,patch_size,1)).astype(np.float32),
                    to_categorical(labels[c_patch_idx].reshape((patch_size, patch_size, 1)), n_classes).astype(np.float32)
                )
    def func_val():
        while True:
            for patch_idx in patches_idxs_val:
                yield (
                    img_0[patch_idx].reshape((patch_size,patch_size,n_img_layers)).astype(np.float32),
                    img_1[patch_idx].reshape((patch_size,patch_size,n_img_layers)).astype(np.float32),
                    previous[patch_idx].reshape((patch_size,patch_size,1)).astype(np.float32),
                    to_categorical(labels[patch_idx].reshape((patch_size, patch_size, 1)), n_classes).astype(np.float32)
                    )
                c_patch_idx = c_patches_idxs_val[random.randrange(n_patches_idxs_val)]
                yield (
                    img_0[c_patch_idx].reshape((patch_size,patch_size,n_img_layers)).astype(np.float32),
                    img_1[c_patch_idx].reshape((patch_size,patch_size,n_img_layers)).astype(np.float32),
                    previous[c_patch_idx].reshape((patch_size,patch_size,1)).astype(np.float32),
                    to_categorical(labels[c_patch_idx].reshape((patch_size, patch_size, 1)), n_classes).astype(np.float32)
                )
                    

    return func_train, func_val, 2*patches_idxs_train.shape[0], 2*patches_idxs_val.shape[0]

def get_train_val_dataset(year):
    patch_size = general.PATCH_SIZE
    n_img_layers = general.N_IMG_LAYERS
    n_classes = general.N_CLASSES

    output_signature = (
        tf.TensorSpec(shape=(patch_size , patch_size , n_img_layers), dtype=tf.float32),
        tf.TensorSpec(shape=(patch_size , patch_size , n_img_layers), dtype=tf.float32),
        tf.TensorSpec(shape=(patch_size , patch_size , 1), dtype=tf.float32),
        tf.TensorSpec(shape=(patch_size , patch_size , n_classes), dtype=tf.float32),
        )
    data_gen_train, data_gen_val, n_patches_train, n_patches_val = train_data_gen(year)
    ds_train = tf.data.Dataset.from_generator(
        generator = data_gen_train, 
        output_signature = output_signature
        )

    ds_val = tf.data.Dataset.from_generator(
        generator = data_gen_val, 
        output_signature = output_signature
        )
    return ds_train, ds_val, n_patches_train, n_patches_val

def data_augmentation(*data):
    x_0 = data[0] 
    x_1 = data[1] 
    x_2 = data[2] 
    x_3 = data[3] 
    if tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1), tf.constant(0.5)):
        x_0 = tf.image.flip_left_right(x_0)
        x_1 = tf.image.flip_left_right(x_1)
        x_2 = tf.image.flip_left_right(x_2)
        x_3 = tf.image.flip_left_right(x_3)
    if tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1), tf.constant(0.5)):
        x_0 = tf.image.flip_up_down(x_0)
        x_1 = tf.image.flip_up_down(x_1)
        x_2 = tf.image.flip_up_down(x_2)
        x_3 = tf.image.flip_up_down(x_3)

    k = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
    x_0 = tf.image.rot90(x_0, k)
    x_1 = tf.image.rot90(x_1, k)
    x_2 = tf.image.rot90(x_2, k)
    x_3 = tf.image.rot90(x_3, k)

    return x_0, x_1, x_2, x_3

def prep_data(*data):
    return (
        (data[0], data[1], data[2]), data[3]
    )


class PredictDataGen_opt(tf.keras.utils.Sequence):
    def __init__(self, year, batch_size):
        prep_path = paths.PREPARED_PATH

        n_img_layers = general.N_IMG_LAYERS
        patch_size = general.PATCH_SIZE
        test_crop = general.TEST_CROP
        crop_size = patch_size - 2*test_crop

        self.batch_size = batch_size

        t_0 = f'{year-1}'
        t_1 = f'{year}'

        shape = np.load(os.path.join(prep_path, f'{general.LABEL_PREFIX}_{year}.npy'), allow_pickle=True).shape

        #n_shape = (shape[0]+2*test_crop , shape[1]+2*test_crop)

        pad_0 = crop_size - (shape[0] % crop_size)
        pad_1 = crop_size - (shape[1] % crop_size)

        pad_matrix = (
            (test_crop, test_crop + pad_0),
            (test_crop, test_crop + pad_1),
            (0,0)
        )

        n_shape = (shape[0] + pad_0 + 2*test_crop, shape[1] + pad_1 + 2*test_crop)

        self.opt_0 = np.pad(np.load(os.path.join(prep_path, f'{t_0}.npy'), allow_pickle=True), pad_matrix, mode='reflect').reshape((-1,n_img_layers))
        self.opt_1 = np.pad(np.load(os.path.join(prep_path, f'{t_1}.npy'), allow_pickle=True), pad_matrix, mode='reflect').reshape((-1,n_img_layers))

        self.previous = np.pad(np.expand_dims(np.load(os.path.join(prep_path, f'{general.PREVIOUS_PREFIX}_{year}.npy'), allow_pickle=True), axis=-1), pad_matrix, mode='reflect').reshape((-1,1))

        idx_matrix = np.arange(n_shape[0]*n_shape[1]).reshape(n_shape)

        idx_patches = view_as_windows(idx_matrix, (patch_size, patch_size), crop_size)

        self.blocks_shape = idx_patches.shape[0:2]
        self.shape = shape

        self.idx_patches = idx_patches.reshape((-1, patch_size, patch_size))

    def __len__(self):
        return 1 + (self.idx_patches.shape[0] // self.batch_size)

    def __getitem__(self, index):
        sel_idx_patches = self.idx_patches[index*self.batch_size:(index+1)*self.batch_size, :,:]
        return (
            self.opt_0[sel_idx_patches], 
            self.opt_1[sel_idx_patches], 
            self.previous[sel_idx_patches]
        )

class SampleData(tf.keras.utils.Sequence):
    def __init__(self, year, size, batch_size):
        prep_path = paths.PREPARED_PATH

        n_img_layers = general.N_IMG_LAYERS

        t_0 = f'{year-1}'
        t_1 = f'{year}'

        label = np.load(os.path.join(prep_path, f'{general.LABEL_PREFIX}_{year}.npy'), allow_pickle=True)
        shape = label.shape
        label = label.reshape((-1,1))

        img_0 = np.load(os.path.join(prep_path, f'{t_0}.npy'), allow_pickle=True).reshape((-1,n_img_layers))
        img_1 = np.load(os.path.join(prep_path, f'{t_1}.npy'), allow_pickle=True).reshape((-1,n_img_layers))

        previous = np.expand_dims(np.load(os.path.join(prep_path, f'{general.PREVIOUS_PREFIX}_{year}.npy'), allow_pickle=True), axis=-1).reshape((-1,1))

        patches_idxs = np.load(os.path.join(prep_path, f'{general.DEFORESTATION_PATCHES_PREFIX}_{year}_val.npy' ), allow_pickle=True)
        np.random.seed(123)
        np.random.shuffle(patches_idxs)
        self.patches_idxs = patches_idxs[:size,:,:]

        self.img_0 = img_0#[patches_idxs]
        self.img_1 = img_1#[patches_idxs]
        self.previous = previous#[patches_idxs]
        self.label = label#[patches_idx]

        self.shape = shape
        self.batch_size = batch_size


    def __len__(self):
        return 1 + (self.patches_idxs.shape[0] // self.batch_size)


    def get(self, index):
        sel_idx_patches = self.patches_idxs[index*self.batch_size:(index+1)*self.batch_size, :,:]
        return (
            (
                self.img_0[sel_idx_patches],
                self.img_1[sel_idx_patches],
                self.previous[sel_idx_patches]
            ),
            to_categorical(self.label[sel_idx_patches], general.N_CLASSES)
        )