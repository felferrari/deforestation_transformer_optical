import tensorflow as tf
from keras.models import Model, load_model, Sequential
from keras.layers import Conv2D, Dropout, Add, MaxPool2D, Input, UpSampling2D

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, size, strides, name = ''):
        super().__init__()

        self.idt_conv =  tf.keras.layers.Conv2D(size, (1,1), padding='same', strides=strides, name = f'{name}_conv_idt')
        self.bn_relu_0 = BN_Relu(name = f'{name}_bn_relu_0')
        self.conv_0 = tf.keras.layers.Conv2D(size, (3,3), padding='same', strides = strides, name = f'{name}_conv_0')
        #self.dropout = tf.keras.layers.Dropout(0.5)

        self.bn_relu_1 = BN_Relu(name = f'{name}_bn_relu_1')
        self.conv_1 = tf.keras.layers.Conv2D(size, (3,3), padding='same', name = f'{name}_conv_1')

        self.add = tf.keras.layers.Add(name = f'{name}_add')

    def call(self, inputs, training = None):
        idt = self.idt_conv(inputs)
        x = self.bn_relu_0(inputs, training)
        x = self.conv_0(x)

        x = self.bn_relu_1(x, training)
        #x = self.dropout(x, training)
        x = self.conv_1(x)

        return self.add([idt, x])

class BN_Relu(tf.keras.layers.Layer):
    def __init__(self, name = ''):
        super().__init__()
        self.bn = tf.keras.layers.BatchNormalization(name = f'{name}_bn')

    def call(self, x, training = None):
        x = self.bn(x, training)
        return tf.keras.activations.relu(x)

class ResUnetEncoder(tf.keras.layers.Layer):
    def __init__(self, model_size, name = ''):
        super().__init__()

        self.conv_0 = tf.keras.layers.Conv2D(model_size[0], (3,3), padding='same', name = f'{name}_e0_conv_0')
        self.bn = BN_Relu(name = f'{name}_e0_bn')

        self.conv_1 = tf.keras.layers.Conv2D(model_size[0], (3,3), padding='same', name = f'{name}_e0_conv_1')
        self.conv_idt = tf.keras.layers.Conv2D(model_size[0], (1,1), padding='same', name = f'{name}_e0_conv_idt')
        self.add = tf.keras.layers.Add(name = f'{name}_add')

        self.res_block_1 = ResBlock(model_size[1], 2, name = f'{name}_e1')
        self.res_block_2 = ResBlock(model_size[2], 2, name = f'{name}_e2')
        self.res_block_3 = ResBlock(model_size[3], 2, name = f'{name}_e3')

    def call(self, inputs, training = None):
        idt = self.conv_idt(inputs)
        x = self.conv_0(inputs)
        x = self.bn(x, training)
        x = self.conv_1(x)
        e0 = self.add([idt, x])

        e1 = self.res_block_1(e0, training)
        e2 = self.res_block_2(e1, training)
        e3 = self.res_block_3(e2, training)

        return e0, e1, e2, e3

class ResUnetDecoder(tf.keras.layers.Layer):
    def __init__(self, model_size, name = ''):
        super().__init__()

        self.upsample_3 = tf.keras.layers.UpSampling2D(size=2, name = f'{name}_upsample_3')
        self.upsample_2 = tf.keras.layers.UpSampling2D(size=2, name = f'{name}_upsample_2')
        self.upsample_1 = tf.keras.layers.UpSampling2D(size=2, name = f'{name}_upsample_1')

        self.res_block_0 = ResBlock(model_size[0], 1, name = f'{name}_d1')
        self.res_block_1 = ResBlock(model_size[1], 1, name = f'{name}_d2')
        self.res_block_2 = ResBlock(model_size[2], 1, name = f'{name}_d3')


    def call(self, inputs, training = None):
        e0, e1, e2, e3 = inputs

        d2 = self.upsample_3(e3)
        d2 = tf.concat([d2, e2], axis=-1)
        d2 = self.res_block_2(d2, training)

        d1 = self.upsample_2(d2)
        d1 = tf.concat([d1, e1], axis=-1)
        d1 = self.res_block_1(d1, training)

        d0 = self.upsample_2(d1)
        d0 = tf.concat([d0, e0], axis=-1)
        d0 = self.res_block_0(d0, training)

        return d0


class ResUnetPM(tf.keras.Model):
    def __init__(self, model_size, n_output, name = ''):
        super().__init__()
        self.encoder = ResUnetEncoder(model_size, name = f'{name}_encoder')
        self.decoder = ResUnetDecoder(model_size, name = f'{name}_decoder')

        self.classifier = tf.keras.layers.Conv2D(n_output, (1,1), padding='same', name = f'{name}_classifier')

    def call(self, *inputs, training = None):
        x_0, x_1, x_prev = inputs[0]
        input_concat = tf.concat([x_0, x_1, x_prev], axis=-1)
        x = self.encoder(input_concat, training)
        x = self.decoder(x, training)

        x = self.classifier(x)
        return tf.keras.activations.softmax(x)

def resnet_block(x, n_filter, ind):
    x_init = x
    ## Conv 1
    x =  Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res1_net'+str(ind))(x)
    x = Dropout(0.5, name = 'drop_net'+str(ind))(x)
    
    ## Conv 2
    x =  Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res2_net'+str(ind))(x)
    
    ## Shortcut
    s  = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res3_net'+str(ind))(x_init)
    ## Add
    x = Add()([x, s])
    return x

# Residual U-Net model
def build_resunet(shape, shape_previous, nb_filters, n_classes):

    input_0= tf.keras.Input(shape) 
    input_1= tf.keras.Input(shape) 
    previous_input= tf.keras.Input(shape_previous) 
    input_layer = tf.concat([input_0,  input_1, previous_input], axis=-1)
    '''Base network to be shared (eq. to feature extraction)'''
    
    res_block1 = resnet_block(input_layer, nb_filters[0], 1)
    pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
    
    res_block2 = resnet_block(pool1, nb_filters[1], 2) 
    pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
    
    res_block3 = resnet_block(pool2, nb_filters[2], 3) 
    pool3 = MaxPool2D((2 , 2), name='pool_net3')(res_block3)
    
    res_block4 = resnet_block(pool3, nb_filters[2], 4)
    
    res_block5 = resnet_block(res_block4, nb_filters[2], 5)
    
    res_block6 = resnet_block(res_block5, nb_filters[2], 6)
    
    upsample3 = Conv2D(nb_filters[2], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net3')(UpSampling2D(size = (2,2))(res_block6))
    
    merged3 = tf.concat([res_block3, upsample3], axis=-1)

    upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net2')(UpSampling2D(size = (2,2))(merged3))
                                                 
    merged2 = tf.concat([res_block2, upsample2], axis=-1)
                                                                                          
    upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))
    merged1 = tf.concat([res_block1, upsample1], axis=-1)

    output = Conv2D(n_classes,(1,1), activation = 'softmax', padding = 'same', name = 'output')(merged1)
                                                                                                           
    return Model([input_0, input_1, previous_input], output)
    