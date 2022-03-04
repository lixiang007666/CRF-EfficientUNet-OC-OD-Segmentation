
from utils import freeze_model
from utils import legacy_support
from model_efficent import *
from builder import build_unet
old_args_map = {
    'freeze_encoder': 'encoder_freeze',
    'skip_connections': 'encoder_features',
    'upsample_rates': None,  # removed
    'input_tensor': None,  # removed
}


@legacy_support(old_args_map)
def Unet(backbone_name='vgg16',
         input_shape=(None, None, 3),
         classes=1,
         activation='sigmoid',
         encoder_weights='imagenet',
         encoder_freeze=True,
         encoder_features='default',
         decoder_block_type='transpose',
         decoder_filters=(256, 128, 64, 32,16),
         decoder_use_batchnorm=True,
         **kwargs):
    """ Unet_ is a fully convolution neural network for image semantic segmentation

        Args:
            backbone_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
            input_shape: shape of input data/image ``(H, W, C)``, in general
                case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
                able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
            classes: a number of classes for output (output shape - ``(h, w, classes)``).
            activation: name of one of ``keras.activations`` for last model layer
                (e.g. ``sigmoid``, ``softmax``, ``linear``).
            encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
            encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
            encoder_features: a list of layer numbers or names starting from top of the model.
                Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
                layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
            decoder_block_type: one of blocks with following layers structure:

                - `upsampling`:  ``Upsampling2D`` -> ``Conv2D`` -> ``Conv2D``
                - `transpose`:   ``Transpose2D`` -> ``Conv2D``

            decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
            decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used.

        Returns:
            ``keras.models.Model``: **Unet**

        .. _Unet:
            https://arxiv.org/pdf/1505.04597

    """

    img_size_target = 224   
    if backbone_name=='efficientnet-b0':
        #img_size_target = 224
        backbone = EfficientNetB0(include_top=False,weights='imagenet',  pooling=None)
    if backbone_name=='efficientnet-b1':
        #img_size_target = 240
        backbone = EfficientNetB1(include_top=False,input_shape=(img_size_target, img_size_target,3),weights='imagenet',  pooling=None)
    if backbone_name=='efficientnet-b2':
        #img_size_target = 260
        backbone = EfficientNetB2(include_top=False,weights='imagenet',  pooling=None)                                  
    if backbone_name=='efficientnet-b3':
        #img_size_target = 300
        backbone = EfficientNetB3(include_top=False,weights='imagenet',  pooling=None)
    if backbone_name=='efficientnet-b4':
        #img_size_target = 380
        backbone = EfficientNetB4(include_top=False,weights='imagenet',  pooling=None)
    if backbone_name=='efficientnet-b5':
        #img_size_target = 456
        backbone = EfficientNetB5(include_top=False,weights='imagenet',  pooling=None)
    if encoder_features == 'default':
        if backbone_name=='efficientnet-b0':
            encoder_features = list([169, 77, 47, 17])
        if backbone_name=='efficientnet-b1':
            #encoder_features = list([246, 122, 76,30])
            encoder_features = list([246, 122, 76])
        if backbone_name=='efficientnet-b2':
            encoder_features = list([246, 122, 76, 30])
        if backbone_name=='efficientnet-b3':
            encoder_features = list([278, 122, 76, 30])
        if backbone_name=='efficientnet-b4':
            encoder_features = list([342, 154, 92, 30])
        if backbone_name=='efficientnet-b5':
            encoder_features = list([419, 199, 121, 43])

    model = build_unet(backbone,
                       classes,
                       encoder_features,
                       decoder_filters=decoder_filters,
                       block_type=decoder_block_type,
                       activation=activation,
                       n_upsample_blocks=len(decoder_filters),
                       upsample_rates=(2, 2, 2, 2,2),
                       use_batchnorm=decoder_use_batchnorm)

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone)

    model.name = 'u-{}'.format(backbone_name)

    return model
