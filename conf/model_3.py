from models.model_t import SM_Transformer
from models.losses import WBCE
from conf import default, general
import tensorflow as tf

def get_model():
    learning_rate = default.LEARNING_RATE
    class_weights = default.CLASS_WEIGHTS
    n_classes = general.N_CLASSES


    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss = WBCE(class_weights)
    metrics = ['accuracy']

    model =  SM_Transformer(n_classes)

    model.compile(
        loss=loss,
        optimizer = optimizer,
        run_eagerly = True,
        metrics = metrics
    )

    return model