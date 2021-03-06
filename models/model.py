import keras,os
from keras.models import Model
from keras.layers.merge import add,multiply
from keras.layers import Lambda,Input, Conv2D,Conv2DTranspose, MaxPooling2D, UpSampling2D,Cropping2D, core, Dropout,BatchNormalization,concatenate,Activation
from keras import backend as K
from keras.layers import Layer, InputSpec
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
import tensorflow_addons as tfa
#from keras_radam import RAdam
from perception.bases.model_base import ModelBase
from .efficientunet import *
from .efficientunet.efficientunet import _get_efficient_unet

from crfrnn_layer import CrfRnnLayer


def dice(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice(y_true, y_pred)

def focal_loss(y_true, y_pred):
    gamma = 2
    alpha = 0.25
    '''tf.where(tensor,a,b):将tensor中true位置元素替换为ａ中对应位置元素,false的替换为ｂ中对应位置元素'''
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)
    return K.mean(-alpha*K.pow(1.-pt_1, gamma)*K.log(pt_1)-(1-alpha)*K.pow(pt_0, gamma)*K.log(1.-pt_0))

def my_dice(y_true,y_pred):
    return (dice_loss(y_true, y_pred)+focal_loss(y_true, y_pred))/2


class SegmentionModel(ModelBase):
    def __init__(self,config=None):
        super(SegmentionModel, self).__init__(config)

        self.patch_height=config.patch_height
        self.patch_width = config.patch_width
        self.num_seg_class=config.seg_num

        self.build_model()
        self.save()

    def _MiniUnet(self,input,shape):
        x1 = Conv2D(shape, (3, 3), strides=(1, 1), padding="same",activation="relu")(input)
        pool1=MaxPooling2D(pool_size=(2, 2))(x1)

        x2 = Conv2D(shape*2, (3, 3), strides=(1, 1), padding="same",activation="relu")(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(x2)

        x3 = Conv2D(shape * 3, (3, 3), strides=(1, 1), padding="same",activation="relu")(pool2)

        x=concatenate([UpSampling2D(size=(2,2))(x3),x2],axis=3)
        x = Conv2D(shape*2, (3, 3), strides=(1, 1), padding="same",activation="relu")(x)

        x = concatenate([UpSampling2D(size=(2, 2))(x),x1],axis=3)
        x = Conv2D(shape, (3, 3), strides=(1, 1), padding="same", activation="sigmoid")(x)
        return x

    def expend_as(self,tensor, rep):
        my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
        return my_repeat

    def AttnGatingBlock(self,x, g, inter_shape):
        shape_x = K.int_shape(x)  # 32
        shape_g = K.int_shape(g)  # 16

        theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
        shape_theta_x = K.int_shape(theta_x)

        phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
        upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16

        concat_xg = add([upsample_g, theta_x])
        act_xg = Activation('relu')(concat_xg)
        psi = Conv2D(1, (1, 1), padding='same')(act_xg)
        sigmoid_xg = Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

        # my_repeat=Lambda(lambda xinput:K.repeat_elements(xinput[0],shape_x[1],axis=1))
        # upsample_psi=my_repeat([upsample_psi])
        upsample_psi = self.expend_as(upsample_psi, shape_x[3])

        y = multiply([upsample_psi, x])

        # print(K.is_keras_tensor(upsample_psi))

        result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
        result_bn = BatchNormalization()(result)
        return result_bn

    def UnetGatingSignal(self,input, is_batchnorm=False):
        shape = K.int_shape(input)
        x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
        if is_batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def UnetConv2D(self,input, outdim, is_batchnorm=False):
        shape = K.int_shape(input)
        x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
        if is_batchnorm:
            x =BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
        if is_batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    def UnetConv2DPro(self,input, outdim):
        x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        attn_shortcut=self._MiniUnet(input,outdim)

        merge=multiply([attn_shortcut,x])
        result=add([merge,x])
        return result


    def build_model(self):
        inputs = Input((self.patch_height, self.patch_width,1))
        conv = Conv2D(16, (3, 3), padding='same')(inputs)  # 'valid'
        conv = LeakyReLU(alpha=0.3)(conv)

        conv1 = self.UnetConv2D(conv, 32,is_batchnorm=True)  # 32 128
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.UnetConv2D(pool1, 32,is_batchnorm=True)  # 32 64
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.UnetConv2D(pool2, 64,is_batchnorm=True)  # 64 32
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.UnetConv2D(pool3, 64,is_batchnorm=True)  # 64 16
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        center = self.UnetConv2D(pool4, 128,is_batchnorm=True)  # 128 8

        gating = self.UnetGatingSignal(center, is_batchnorm=True)
        attn_1 = self.AttnGatingBlock(conv4, gating, 128)
        up1 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu")(center), attn_1], axis=3)

        gating = self.UnetGatingSignal(up1, is_batchnorm=True)
        attn_2 = self.AttnGatingBlock(conv3, gating, 64)
        up2 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu")(up1), attn_2], axis=3)

        gating = self.UnetGatingSignal(up2, is_batchnorm=True)
        attn_3 = self.AttnGatingBlock(conv2, gating, 32)
        up3 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up2), attn_3], axis=3)

        up4 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up3), conv1], axis=3)


        conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(up4)
        conv8 = core.Reshape((self.patch_height * self.patch_width,(self.num_seg_class + 1)))(conv8)
        ############
        act = Activation('softmax')(conv8)


        
        model_lx= get_efficient_unet_b4((self.patch_height, self.patch_width, 1), block_type='transpose', concat_input=True)
        conv8_lx = Conv2D(self.num_seg_class + 2, (1, 1), activation='relu', padding='same')(model_lx.layers[-2].output)
        conv8_lx = core.Reshape((self.patch_height * self.patch_width,(self.num_seg_class + 2)))(conv8_lx)
        act = Activation('softmax')(conv8_lx)
        output = CrfRnnLayer(image_dims=(self.patch_height, self.patch_width, 1),
                             num_classes=3,
                             theta_alpha=160.,
                             theta_beta=3.,
                             theta_gamma=3.,
                             num_iterations=10,
                             name='crfrnn')([act, model_lx.input])


        model = Model(inputs=model_lx.input, outputs=output)
        model.summary()
        #model.compile(optimizer='adam',
        #			  loss=my_dice, metrics=['categorical_accuracy'])
        #model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5),
        #			  loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.compile(optimizer= tfa.optimizers.Lookahead(tfa.optimizers.RectifiedAdam(learning_rate=0.001)), loss=my_dice, metrics=['categorical_accuracy'])
#		plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
        self.model = model