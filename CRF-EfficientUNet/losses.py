from __future__ import absolute_import
import keras.backend as K
from keras.losses import binary_crossentropy
from keras.losses import mean_squared_error
from keras.losses import categorical_crossentropy
from keras.utils.generic_utils import get_custom_objects
from metrics1 import jaccard_score, f_score
import keras_contrib.backend as KC
SMOOTH = 1.

__all__ = [
    'jaccard_loss', 'bce_jaccard_loss', 'cce_jaccard_loss',
    'dice_loss', 'bce_dice_loss', 'cce_dice_loss',
]


# ============================== Dice Losses ================================

def dice_loss(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True, beta=1.6):
    r"""Dice loss function for imbalanced datasets:

    .. math:: L(precision, recall) = 1 - (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        Dice loss in range [0, 1]

    """
    return 1 - f_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image, beta=beta)

def bce_dice_loss(gt, pr, bce_weight=0.4, smooth=SMOOTH, per_image=True, beta=1.6):
    r"""Sum of binary crossentropy and dice losses:
    
    .. math:: L(A, B) = bce_weight * binary_crossentropy(A, B) + dice_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for dice loss, len(weights) = C 
        smooth: value to avoid division by zero
        per_image: if ``True``, dice loss is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        loss
    
    """
    bce = K.mean(binary_crossentropy(gt, pr))
    loss = bce_weight * bce + dice_loss(gt, pr, smooth=smooth, per_image=per_image, beta=beta)
    return loss
def Asymmetric_loss(gt, pr,  smooth=SMOOTH, per_image=True, beta=1.6):
    r"""Sum of binary crossentropy and dice losses:
    
    .. math:: L(A, B) = bce_weight * binary_crossentropy(A, B) + dice_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for dice loss, len(weights) = C 
        smooth: value to avoid division by zero
        per_image: if ``True``, dice loss is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        loss
    
    """
    loss = dice_loss(gt, pr, smooth=smooth, per_image=per_image, beta=1.6)
    return loss
def bce_dice_loss_gm(gt, pr, gama=1.1, smooth=SMOOTH, per_image=True, beta=1.):
    r"""Sum of binary crossentropy and dice losses:
    
    .. math:: L(A, B) = bce_weight * binary_crossentropy(A, B) + dice_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for dice loss, len(weights) = C 
        smooth: value to avoid division by zero
        per_image: if ``True``, dice loss is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        loss
    
    """
    bce = K.mean(binary_crossentropy(gt, pr))
    loss = -(1-gama) * bce + gama * dice_loss(gt, pr, smooth=smooth, per_image=per_image, beta=beta)
    return loss

def cce_dice_loss(gt, pr, cce_weight=1., class_weights=1., smooth=SMOOTH, per_image=True, beta=1.):
    r"""Sum of categorical crossentropy and dice losses:
    
    .. math:: L(A, B) = cce_weight * categorical_crossentropy(A, B) + dice_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for dice loss, len(weights) = C 
        smooth: value to avoid division by zero
        per_image: if ``True``, dice loss is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        loss
    
    """
    cce = categorical_crossentropy(gt, pr) * class_weights
    cce = K.mean(cce)
    return cce_weight * cce + dice_loss(gt, pr, smooth=smooth, class_weights=class_weights, per_image=per_image, beta=beta)

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed
# Update custom objects
get_custom_objects().update({
    'dice_loss': dice_loss,
    'bce_dice_loss': bce_dice_loss,
    'bce_dice_loss_gm': bce_dice_loss_gm,
    'cce_dice_loss': cce_dice_loss,
})
def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tversky loss function.
    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return 1-answer

def ssim_loss(y_true, y_pred):
    k1=0.01
    k2=0.03
    kernel_size=3
    max_value=1.0
    c1 = (k1 * max_value) ** 2
    c2 = (k2 * max_value) ** 2    
    dim_ordering = K.image_data_format()
    bakend='tensorflow'
    kernel = [kernel_size, kernel_size]
    y_true = K.reshape(y_true, [-1] + list(K.int_shape(y_pred)[1:]))
    y_pred = K.reshape(y_pred, [-1] + list(K.int_shape(y_pred)[1:]))
    patches_pred = KC.extract_image_patches(y_pred, kernel, kernel, 'valid',dim_ordering)
    patches_true = KC.extract_image_patches(y_true, kernel, kernel, 'valid',dim_ordering)
    bs, w, h, c1, c2, c3 = K.int_shape(patches_pred)
    patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
    patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    # Get variance
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    # Get std dev
    covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred
    ssim = (2 * u_true * u_pred + c1) * (2 * covar_true_pred + c2)
    denom = ((K.square(u_true)+ K.square(u_pred) + c1) * (var_pred + var_true + c2))
    ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
    return K.mean((1.0 - ssim) / 2.0)

def ssim_mse_loss(gt, pr, mse_weight=0.4):
    mse = K.mean(mean_squared_error(gt, pr))
    loss = mse_weight * mse + ssim_loss(gt, pr)
    return loss