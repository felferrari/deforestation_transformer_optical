from xml.sax.xmlreader import InputSource
import tensorflow as tf
from keras.callbacks import Callback
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from conf import paths, general
import os

class ImageSampleLogger(Callback):
    def __init__(self, ds, path):
        super().__init__()
        self.ds = ds
        self.path = path
        self.stats = np.load(os.path.join(paths.PREPARED_PATH, 'statistics.npy'))

    def on_epoch_end(self, epoch, logs=None):
        for batch_i in range(len(self.ds)):
                
            inputs, labels = self.ds.get(batch_i)
            predicted = self.model.predict_on_batch(inputs)

            for sample_i in range(labels.shape[0]):
                figure, ax = plt.subplots(nrows=1, ncols=5, figsize = (16,5))
                ax[0].imshow((inputs[0][sample_i][:,:,[3,2,1]]*self.stats[1]+self.stats[0])/2000)
                ax[0].title.set_text(str(general.YEAR_0))
                ax[1].imshow((inputs[1][sample_i][:,:,[3,2,1]]*self.stats[1]+self.stats[0])/2000)
                ax[1].title.set_text(str(general.YEAR_1))
                ax[2].imshow(inputs[2][sample_i], cmap = 'gray')
                ax[2].title.set_text('Previous Def.')
                ax[3].imshow(labels[sample_i, :,:, 1], cmap = 'gray')
                ax[3].title.set_text('Ground Truth')
                ax[4].imshow(predicted[sample_i, :,:, 1], cmap = 'gray')
                ax[4].title.set_text(f'Prediction epoch {epoch+1}')

                for i in range(5):
                    ax[i].patch.set_edgecolor('black')  
                    ax[i].patch.set_linewidth('1') 
                    ax[i].set_xticks([], []) 
                    ax[i].set_yticks([], [])
                    #ax[i].axis('off')

                plt.subplots_adjust(wspace=0.05, hspace=0)

                figure.savefig(os.path.join(self.path, f'sample_{sample_i+batch_i*self.ds.batch_size}_epoch_{epoch}.png'), bbox_inches='tight')
                figure.clf()
                plt.close()



