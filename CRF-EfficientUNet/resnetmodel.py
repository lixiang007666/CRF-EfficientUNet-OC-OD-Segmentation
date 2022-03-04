from keras.utils import get_file

import keras.backend as K
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Add
from keras.layers import Dense
from keras.models import Model
from keras.engine import get_source_inputs


# params
def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'he_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    default_bn_params = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params


def get_model_params(name):

    params = {

        'resnet18': {
            'repetitions': (2, 2, 2, 2),
            'block_type': 'conv',
            'attention': None,
        },

        'resnet34': {
            'repetitions': (3, 4, 6, 3),
            'block_type': 'conv',
            'attention': None,
        },

        'resnet50': {
            'repetitions': (3, 4, 6, 3),
            'block_type': 'bottleneck',
            'attention': None,
        },

        'resnet101': {
            'repetitions': (3, 4, 23, 3),
            'block_type': 'bottleneck',
            'attention': None,
        },

        'resnet152': {
            'repetitions': (3, 8, 36, 3),
            'block_type': 'bottleneck',
            'attention': None,
        },

        'seresnet18': {
            'repetitions': (2, 2, 2, 2),
            'block_type': 'conv',
            'attention': 'cse',
        },

        'seresnet34': {
            'repetitions': (3, 4, 6, 3),
            'block_type': 'conv',
            'attention': 'cse',
        },

        'seresnet50': {
            'repetitions': (3, 4, 6, 3),
            'block_type': 'bottleneck',
            'attention': 'cse',
        },

        'seresnet101': {
            'repetitions': (3, 4, 23, 3),
            'block_type': 'bottleneck',
            'attention': 'cse',
        },

        'seresnet152': {
            'repetitions': (3, 8, 36, 3),
            'block_type': 'bottleneck',
            'attention': 'cse',
        },

        'csseresnet18': {
            'repetitions': (2, 2, 2, 2),
            'block_type': 'conv',
            'attention': 'csse',
        },

        'csseresnet34': {
            'repetitions': (3, 4, 6, 3),
            'block_type': 'conv',
            'attention': 'csse',
        },

        'csseresnet50': {
            'repetitions': (3, 4, 6, 3),
            'block_type': 'bottleneck',
            'attention': 'csse',
        },

        'csseresnet101': {
            'repetitions': (3, 4, 23, 3),
            'block_type': 'bottleneck',
            'attention': 'csse',
        },

        'csseresnet152': {
            'repetitions': (3, 8, 36, 3),
            'block_type': 'bottleneck',
            'attention': 'csse',
        },

    }

    return params[name]
# end params

# block
def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def residual_conv_block(filters, stage, block, strides=(1, 1), attention=None, cut='pre'):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = Add()([x, shortcut])
        return x

    return layer


def residual_bottleneck_block(filters, stage, block, strides=None, attention=None, cut='pre'):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = Conv2D(filters*4, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '2', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = Activation('relu', name=relu_name + '3')(x)
        x = Conv2D(filters*4, (1, 1), name=conv_name + '3', **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = Add()([x, shortcut])

        return x

    return layer
#end block
# common block
import keras.layers as kl
from keras.utils.generic_utils import get_custom_objects
def ChannelSE(reduction=16):
    """
    Squeeze and Excitation block, reimplementation inspired by
        https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py

    Args:
        reduction: channels squeeze factor

    """

    def layer(input_tensor):
        # get number of channels/filters
        channels = K.int_shape(input_tensor)[-1]

        x = input_tensor

        # squeeze and excitation block in PyTorch style with
        # custom global average pooling where keepdims=True
        x = kl.Lambda(lambda a: K.mean(a, axis=[1, 2], keepdims=True))(x)
        x = kl.Conv2D(channels // reduction, (1, 1), kernel_initializer='he_uniform')(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv2D(channels, (1, 1), kernel_initializer='he_uniform')(x)
        x = kl.Activation('sigmoid')(x)

        # apply attention
        x = kl.Multiply()([input_tensor, x])

        return x

    return layer


def SpatialSE():
    """
    Spatial squeeze and excitation block (applied across spatial dimensions)
    """

    def layer(input_tensor):
        x = kl.Conv2D(1, (1, 1), kernel_initializer="he_normal", activation='sigmoid', use_bias=False)(input_tensor)
        x = kl.Multiply()([input_tensor, x])
        return x

    return layer


def ChannelSpatialSE(reduction=2):
    """
    Spatial and Channel Squeeze & Excitation Block (scSE)
        https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66568

    Implementation of Concurrent Spatial and Channel `Squeeze & Excitation` in Fully Convolutional Networks
        https://arxiv.org/abs/1803.02579
    """

    def layer(input_tensor):
        cse = ChannelSE(reduction=reduction)(input_tensor)
        sse = SpatialSE()(input_tensor)
        x = kl.Add()([cse, sse])

        return x

    return layer
#end common block

# builder
def build_resnet(
     repetitions=(2, 2, 2, 2),
     include_top=False,
     input_tensor=None,
     input_shape=None,
     classes=1,
     block_type='conv',
     attention=None):
    
    """
    TODO
    """

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # choose residual block type
    if block_type == 'conv':
        residual_block = residual_conv_block
    elif block_type == 'bottleneck':
        residual_block = residual_bottleneck_block
    else:
        raise ValueError('Block type "{}" not in ["conv", "bottleneck"]'.format(block_type))

    # choose attention block type
    if attention == 'sse':
        attention_block = SpatialSE()
    elif attention == 'cse':
        attention_block = ChannelSE(reduction=16)
    elif attention == 'csse':
        attention_block = ChannelSpatialSE(reduction=2)
    elif attention is None:
        attention_block = None
    else:
        raise ValueError('Supported attention blocks are: sse, cse, csse. Got "{}".'.format(attention))

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = 64

    # resnet bottom
    x = BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = BatchNormalization(name='bn0', **bn_params)(x)
    x = Activation('relu', name='relu0')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)
    
    # resnet body
    for stage, rep in enumerate(repetitions):
        for block in range(rep):
            
            filters = init_filters * (2**stage)
            
            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = residual_block(filters, stage, block, strides=(1, 1),
                                   cut='post', attention=attention_block)(x)
                
            elif block == 0:
                x = residual_block(filters, stage, block, strides=(2, 2),
                                   cut='post', attention=attention_block)(x)
                
            else:
                x = residual_block(filters, stage, block, strides=(1, 1),
                                   cut='pre', attention=attention_block)(x)
                
    x = BatchNormalization(name='bn1', **bn_params)(x)
    x = Activation('relu', name='relu1')(x)

    # resnet top
    if include_top:
        x = GlobalAveragePooling2D(name='pool1')(x)
        x = Dense(classes, name='fc1')(x)
        x = Activation('softmax', name='softmax')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create model.
    model = Model(inputs, x)

    return model
#end builder
# utils
def find_weights(weights_collection, model_name, dataset, include_top):
    w = list(filter(lambda x: x['model'] == model_name, weights_collection))
    w = list(filter(lambda x: x['dataset'] == dataset, w))
    w = list(filter(lambda x: x['include_top'] == include_top, w))
    return w
def load_model_weights(weights_collection, model, dataset, classes, include_top):
    weights = find_weights(weights_collection, model.name, dataset, include_top)

    if weights:
        weights = weights[0]

        if include_top and weights['classes'] != classes:
            raise ValueError('If using `weights` and `include_top`'
                             ' as true, `classes` should be {}'.format(weights['classes']))

        weights_path = get_file(weights['name'],
                                weights['url'],
                                cache_subdir='models',
                                md5_hash=weights['md5'])

        model.load_weights(weights_path)

    else:
        raise ValueError('There is no weights for such configuration: ' +
                         'model = {}, dataset = {}, '.format(model.name, dataset) +
                         'classes = {}, include_top = {}.'.format(classes, include_top))
#end utils

weights_collection = [

    # ResNet18
    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000.h5',
        'name': 'resnet18_imagenet_1000.h5',
        'md5': '64da73012bb70e16c901316c201d9803',
    },

    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5',
        'name': 'resnet18_imagenet_1000.h5',
        'md5': '318e3ac0cd98d51e917526c9f62f0b50',
    },

    # ResNet34
    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000.h5',
        'name': 'resnet34_imagenet_1000.h5',
        'md5': '2ac8277412f65e5d047f255bcbd10383',
    },

    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5',
        'name': 'resnet34_imagenet_1000_no_top.h5',
        'md5': '8caaa0ad39d927cb8ba5385bf945d582',
    },

    # ResNet50
    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet_1000.h5',
        'name': 'resnet50_imagenet_1000.h5',
        'md5': 'd0feba4fc650e68ac8c19166ee1ba87f',
    },

    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet_1000_no_top.h5',
        'name': 'resnet50_imagenet_1000_no_top.h5',
        'md5': 'db3b217156506944570ac220086f09b6',
    },

    {
        'model': 'resnet50',
        'dataset': 'imagenet11k-places365ch',
        'classes': 11586,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet11k-places365ch_11586.h5',
        'name': 'resnet50_imagenet11k-places365ch_11586.h5',
        'md5': 'bb8963db145bc9906452b3d9c9917275',
    },

    {
        'model': 'resnet50',
        'dataset': 'imagenet11k-places365ch',
        'classes': 11586,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet11k-places365ch_11586_no_top.h5',
        'name': 'resnet50_imagenet11k-places365ch_11586_no_top.h5',
        'md5': 'd8bf4e7ea082d9d43e37644da217324a',
    },

    # ResNet101
    {
        'model': 'resnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet101_imagenet_1000.h5',
        'name': 'resnet101_imagenet_1000.h5',
        'md5': '9489ed2d5d0037538134c880167622ad',
    },

    {
        'model': 'resnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet101_imagenet_1000_no_top.h5',
        'name': 'resnet101_imagenet_1000_no_top.h5',
        'md5': '1016e7663980d5597a4e224d915c342d',
    },


    # ResNet152
    {
        'model': 'resnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet_1000.h5',
        'name': 'resnet152_imagenet_1000.h5',
        'md5': '1efffbcc0708fb0d46a9d096ae14f905',
    },

    {
        'model': 'resnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet_1000_no_top.h5',
        'name': 'resnet152_imagenet_1000_no_top.h5',
        'md5': '5867b94098df4640918941115db93734',
    },

    {
        'model': 'resnet152',
        'dataset': 'imagenet11k',
        'classes': 11221,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet11k_11221.h5',
        'name': 'resnet152_imagenet11k_11221.h5',
        'md5': '24791790f6ef32f274430ce4a2ffee5d',
    },

    {
        'model': 'resnet152',
        'dataset': 'imagenet11k',
        'classes': 11221,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet11k_11221_no_top.h5',
        'name': 'resnet152_imagenet11k_11221_no_top.h5',
        'md5': '25ab66dec217cb774a27d0f3659cafb3',
    },


    # ResNeXt50
    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext50_imagenet_1000.h5',
        'name': 'resnext50_imagenet_1000.h5',
        'md5': '7c5c40381efb044a8dea5287ab2c83db',
    },

    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext50_imagenet_1000_no_top.h5',
        'name': 'resnext50_imagenet_1000_no_top.h5',
        'md5': '7ade5c8aac9194af79b1724229bdaa50',
    },


    # ResNeXt101
    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext101_imagenet_1000.h5',
        'name': 'resnext101_imagenet_1000.h5',
        'md5': '432536e85ee811568a0851c328182735',
    },

    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext101_imagenet_1000_no_top.h5',
        'name': 'resnext101_imagenet_1000_no_top.h5',
        'md5': '91fe0126320e49f6ee607a0719828c7e',
    },

    # SE models
    {
        'model': 'seresnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet50_imagenet_1000.h5',
        'name': 'seresnet50_imagenet_1000.h5',
        'md5': 'ff0ce1ed5accaad05d113ecef2d29149',
    },

    {
        'model': 'seresnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet50_imagenet_1000_no_top.h5',
        'name': 'seresnet50_imagenet_1000_no_top.h5',
        'md5': '043777781b0d5ca756474d60bf115ef1',
    },

    {
        'model': 'seresnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet101_imagenet_1000.h5',
        'name': 'seresnet101_imagenet_1000.h5',
        'md5': '5c31adee48c82a66a32dee3d442f5be8',
    },

    {
        'model': 'seresnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet101_imagenet_1000_no_top.h5',
        'name': 'seresnet101_imagenet_1000_no_top.h5',
        'md5': '1c373b0c196918713da86951d1239007',
    },

    {
        'model': 'seresnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet152_imagenet_1000.h5',
        'name': 'seresnet152_imagenet_1000.h5',
        'md5': '96fc14e3a939d4627b0174a0e80c7371',
    },

    {
        'model': 'seresnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet152_imagenet_1000_no_top.h5',
        'name': 'seresnet152_imagenet_1000_no_top.h5',
        'md5': 'f58d4c1a511c7445ab9a2c2b83ee4e7b',
    },

    {
        'model': 'seresnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnext50_imagenet_1000.h5',
        'name': 'seresnext50_imagenet_1000.h5',
        'md5': '5310dcd58ed573aecdab99f8df1121d5',
    },

    {
        'model': 'seresnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnext50_imagenet_1000_no_top.h5',
        'name': 'seresnext50_imagenet_1000_no_top.h5',
        'md5': 'b0f23d2e1cd406d67335fb92d85cc279',
    },

    {
        'model': 'seresnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnext101_imagenet_1000.h5',
        'name': 'seresnext101_imagenet_1000.h5',
        'md5': 'be5b26b697a0f7f11efaa1bb6272fc84',
    },

    {
        'model': 'seresnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnext101_imagenet_1000_no_top.h5',
        'name': 'seresnext101_imagenet_1000_no_top.h5',
        'md5': 'e48708cbe40071cc3356016c37f6c9c7',
    },

    {
        'model': 'senet154',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/senet154_imagenet_1000.h5',
        'name': 'senet154_imagenet_1000.h5',
        'md5': 'c8eac0e1940ea4d8a2e0b2eb0cdf4e75',
    },

    {
        'model': 'senet154',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/senet154_imagenet_1000_no_top.h5',
        'name': 'senet154_imagenet_1000_no_top.h5',
        'md5': 'd854ff2cd7e6a87b05a8124cd283e0f2',
    },

    {
        'model': 'seresnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet18_imagenet_1000.h5',
        'name': 'seresnet18_imagenet_1000.h5',
        'md5': '9a925fd96d050dbf7cc4c54aabfcf749',
    },

    {
        'model': 'seresnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet18_imagenet_1000_no_top.h5',
        'name': 'seresnet18_imagenet_1000_no_top.h5',
        'md5': 'a46e5cd4114ac946ecdc58741e8d92ea',
    },

    {
        'model': 'seresnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet34_imagenet_1000.h5',
        'name': 'seresnet34_imagenet_1000.h5',
        'md5': '863976b3bd439ff0cc05c91821218a6b',
    },

    {
        'model': 'seresnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet34_imagenet_1000_no_top.h5',
        'name': 'seresnet34_imagenet_1000_no_top.h5',
        'md5': '3348fd049f1f9ad307c070ff2b6ec4cb',
    },

]



__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
           'SEResNet18', 'SEResNet34', 'preprocess_input']

# preprocessing function
preprocess_input = lambda x: x


def _get_resnet(name):
    def classifier(input_shape, input_tensor=None, weights=False, classes=1, include_top=False):
        model_params = get_model_params(name)
        model = build_resnet(input_tensor=input_tensor,
                             input_shape=input_shape,
                             classes=classes,
                             include_top=include_top,
                             **model_params)

        model.name = name

        if weights:
            load_model_weights(weights_collection, model, weights, classes, include_top)

        return model
    return classifier


# classic resnet models
ResNet18 = _get_resnet('resnet18')
ResNet34 = _get_resnet('resnet34')
ResNet50 = _get_resnet('resnet50')
ResNet101 = _get_resnet('resnet101')
ResNet152 = _get_resnet('resnet152')

# resnets with squeeze and excitation attention block
SEResNet18 = _get_resnet('seresnet18')
SEResNet34 = _get_resnet('seresnet34')
# SEResNet50 = _get_resnet('seresnet50')
# SEResNet101 = _get_resnet('seresnet101')
# SEResNet152 = _get_resnet('seresnet152')
#
# # resnets with concurrent squeeze and excitation attention block
# CSSEResNet18 = _get_resnet('csseresnet18')
# CSSEResNet34 = _get_resnet('csseresnet34')
# CSSEResNet50 = _get_resnet('csseresnet50')
# CSSEResNet101 = _get_resnet('csseresnet101')
# CSSEResNet152 = _get_resnet('csseresnet152')
