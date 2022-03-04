from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Activation
from keras.models import Model
from blocks import Transpose2D_block
from blocks import Upsample2D_block
from blocks import bilinear_upsample_weights
from utils import get_layer_number, to_tuple
from keras.initializers import Constant
def build_unet(backbone, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=True):

    input = backbone.input
    x = backbone.output
    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])

    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model

def build_FCN32(backbone, classes):
    input = backbone.input
    x = backbone.output
    x = Conv2D(filters=classes, 
               kernel_size=(1, 1))(x)
    x = Conv2DTranspose(filters=classes, 
                        kernel_size=(64, 64),
                        strides=(32, 32),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=Constant(bilinear_upsample_weights(32, classes)))(x)   

    model = Model(input, x)

    return model
def build_FCN8(backbone, classes):
    input = backbone.input
    x = backbone.output
    x = Conv2D(filters=classes, 
               kernel_size=(1, 1))(x)
    x = Conv2DTranspose(filters=classes, 
                        kernel_size=(2, 2),
                        strides=(2, 2),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=Constant(bilinear_upsample_weights(32, classes)))(x)   
    x = Conv2D(filters=classes, 
               kernel_size=(1, 1))(x)
    x = Conv2DTranspose(filters=classes, 
                        kernel_size=(4, 4),
                        strides=(4, 4),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=Constant(bilinear_upsample_weights(32, classes)))(x)  
    

    return model

    