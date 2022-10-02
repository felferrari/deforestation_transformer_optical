import argparse
import pathlib
import importlib
from conf import default, general, paths
from ops.dataloader import get_train_val_dataset, data_augmentation, prep_data, SampleData
import tensorflow as tf
import os
import time
from multiprocessing import Process
import sys
from models.callbacks import ImageSampleLogger

parser = argparse.ArgumentParser(
    description='Train N models based in the same parameters'
)

parser.add_argument( # Experiment number
    '-e', '--experiment',
    type = int,
    default = 2,
    help = 'The number of the experiment'
)

parser.add_argument( # batch size
    '-b', '--batch-size',
    type = int,
    default = default.BATCH_SIZE,
    help = 'The number of samples of each batch'
)

parser.add_argument( # Number of models to be trained
    '-n', '--number-models',
    type = int,
    default = default.N_TRAIN_MODELS,
    help = 'The number models to be trained from the scratch'
)

parser.add_argument( # Experiment path
    '-x', '--experiments-path',
    type = pathlib.Path,
    default = paths.EXPERIMENTS_PATH,
    help = 'The patch to data generated by all experiments'
)

args = parser.parse_args()

exp_path = os.path.join(str(args.experiments_path), f'exp_{args.experiment}')
if not os.path.exists(exp_path):
    os.mkdir(exp_path)

logs_path = os.path.join(exp_path, f'logs')
if not os.path.exists(logs_path):
    os.mkdir(logs_path)

models_path = os.path.join(exp_path, f'models')
if not os.path.exists(models_path):
    os.mkdir(models_path)

visual_path = os.path.join(exp_path, f'visual')
if not os.path.exists(visual_path):
    os.mkdir(visual_path)

predicted_path = os.path.join(exp_path, f'predicted')
if not os.path.exists(predicted_path):
    os.mkdir(predicted_path)

results_path = os.path.join(exp_path, f'results')
if not os.path.exists(results_path):
    os.mkdir(results_path)

def run(model_idx):
    outfile = os.path.join(logs_path, f'train_{args.experiment}_{model_idx}.txt')
    with open(outfile, 'w') as sys.stdout:
        ds_train, ds_val, n_patches_train, n_patches_val = get_train_val_dataset(general.YEAR_1)

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        ds_train = ds_train.map(data_augmentation, num_parallel_calls=AUTOTUNE)
        ds_train = ds_train.map(prep_data, num_parallel_calls=AUTOTUNE)
        ds_train = ds_train.shuffle(50*args.batch_size)
        ds_train = ds_train.batch(args.batch_size)
        ds_train = ds_train.prefetch(AUTOTUNE)

        ds_val = ds_val.map(data_augmentation, num_parallel_calls=AUTOTUNE)
        ds_val = ds_val.map(prep_data, num_parallel_calls=AUTOTUNE)
        ds_val = ds_val.batch(args.batch_size)
        ds_val = ds_val.prefetch(AUTOTUNE)

        ds_sample = SampleData(general.YEAR_1, 24, args.batch_size)

        train_steps = (n_patches_train // args.batch_size)
        val_steps = (n_patches_val // args.batch_size)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            restore_best_weights = True,
            patience=general.EARLY_STOP_PATIENCE
        )

        model_log_path = os.path.join(logs_path, f'log_tb_{model_idx}')

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir = model_log_path,
            histogram_freq = 10
        )

        visual_log_path = os.path.join(visual_path, f'log_model_{model_idx}')
        if not os.path.exists(visual_log_path):
            os.mkdir(visual_log_path)

        imgLogger = ImageSampleLogger(ds_sample, visual_log_path)

        callbacks = [
            early_stop,
            tensorboard,
            imgLogger
            ]

        model_m =importlib.import_module(f'conf.model_{args.experiment}')
        model = model_m.get_model()

        print('Loss: ', model.loss)
        print('Weights: ',model.loss.weights)
        print('Optimizer: ', model.optimizer)

        t0 = time.perf_counter()
        history = model.fit(
            x=ds_train,
            validation_data=ds_val,
            epochs = general.N_MAX_EPOCHS,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            verbose=2, 
            callbacks = callbacks
        )
        print(f'Training time: {(time.perf_counter() - t0)/60} mins')
        model.summary()
        model.save_weights(os.path.join(models_path, f'model_{model_idx}'))

if __name__=="__main__":
    
    for model_idx in range(args.number_models):
        p = Process(target=run, args=(model_idx,))
        p.start()
        p.join()