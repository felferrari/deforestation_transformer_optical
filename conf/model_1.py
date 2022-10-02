from models.models import build_resunet
from models.losses import WBCE
from conf import default, general
import tensorflow as tf

def get_model():
    patch_size = general.PATCH_SIZE
    learning_rate = default.LEARNING_RATE
    class_weights = default.CLASS_WEIGHTS

    shape_img = (patch_size, patch_size, general.N_IMG_LAYERS)
    shape_previous = (patch_size, patch_size, 1)

    model_size = [64, 128, 128]

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss = WBCE(class_weights)
    metrics = ['accuracy']

    model =  build_resunet(shape_img, shape_previous, model_size, general.N_CLASSES)

    model.compile(
        loss=loss,
        optimizer = optimizer,
        run_eagerly = True,
        metrics = metrics
    )

    return model