
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.layers import Conv3DTranspose,Conv3D,UpSampling3D, MaxPool3D,AveragePooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#put your own neural network backbones here

def dice_coef(y_true, y_pred,smooth=0.001):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dice_coef_3D(y_true,y_pred,depth):
    score=0.0
    for i in range(depth):
        score+=dice_coef(y_true[i],y_pred[i])
    return score/depth

def DICE_LOSS_3D(depth):
    def dice_coef_loss_3D(y_true,y_pred):
        return -dice_coef_3D(y_true,y_pred)
    return dice_coef_loss_3D

def iou(y_true, y_pred,smooth=0.001):

    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)


def unet2(input_size=(256,256,3)):
    inputs =tf.keras.layers.Input(input_size)
    
    conv1 = Conv2D(32, kernel_size=(2, 2), strides=(2, 2),padding='valid')(inputs)
    bn1 = BatchNormalization(axis=3)(conv1)
    bn1 = Activation('relu')(conv1)
    bn1=Dropout(0.1)(bn1)


    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    bn2 = Activation('relu')(conv2)
    bn2=Dropout(0.1)(bn2)
    conv2 = Conv2D(64, (3, 3), padding='same')(bn2)
    bn2 = BatchNormalization(axis=3)(conv2)
    bn2 = Activation('relu')(bn2)
    bn2=Dropout(0.1)(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    bn3 = Activation('relu')(conv3)
    bn3=Dropout(0.1)(bn3)
    conv3 = Conv2D(128, (3, 3), padding='same')(bn3)
    bn3 = BatchNormalization(axis=3)(conv3)
    bn3 = Activation('relu')(bn3)
    bn3=Dropout(0.1)(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    bn4 = Activation('relu')(conv4)
    bn4=Dropout(0.1)(bn4)
    conv4 = Conv2D(256, (3, 3), padding='same')(bn4)
    bn4 = BatchNormalization(axis=3)(conv4)
    bn4 = Activation('relu')(bn4)
    bn4=Dropout(0.1)(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
    bn5 = Activation('relu')(conv5)
    bn5=Dropout(0.1)(bn5)
    conv5 = Conv2D(512, (3, 3), padding='same')(bn5)
    bn5 = BatchNormalization(axis=3)(conv5)
    bn5 = Activation('relu')(bn5)
    bn5=Dropout(0.1)(bn5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), padding='same')(up6)
    bn6 = Activation('relu')(conv6)
    bn6=Dropout(0.1)(bn6)
    conv6 = Conv2D(256, (3, 3), padding='same')(bn6)
    bn6 = BatchNormalization(axis=3)(conv6)
    bn6 = Activation('relu')(bn6)
    bn6=Dropout(0.1)(bn6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), padding='same')(up7)
    bn7 = Activation('relu')(conv7)
    bn7=Dropout(0.1)(bn7)
    conv7 = Conv2D(128, (3, 3), padding='same')(bn7)
    bn7 = BatchNormalization(axis=3)(conv7)
    bn7 = Activation('relu')(bn7)
    bn7=Dropout(0.1)(bn7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), padding='same')(up8)
    bn8 = Activation('relu')(conv8)
    bn8=Dropout(0.1)(bn8)
    conv8 = Conv2D(64, (3, 3), padding='same')(bn8)
    bn8 = BatchNormalization(axis=3)(conv8)
    bn8 = Activation('relu')(bn8)
    bn8=Dropout(0.1)(bn8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(bn8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), padding='same')(up9)
    bn9 = Activation('relu')(conv9)
    bn9=Dropout(0.1)(bn9)
    conv9 = Conv2D(32, (3, 3), padding='same')(bn9)
    bn9 = BatchNormalization(axis=3)(conv9)
    bn9 = Activation('relu')(bn9)
    bn9=Dropout(0.1)(bn9)

    up10=Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(bn9)
    conv10 = Conv2D(16, (3, 3), padding='same')(up10)
    bn10 = Activation('relu')(conv10)
    bn10=Dropout(0.1)(bn10)
    conv10 = Conv2D(16, (3, 3), padding='same')(bn10)
    bn10 = BatchNormalization(axis=3)(conv10)
    bn10 = Activation('relu')(bn10)
    bn10=Dropout(0.1)(bn10)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn10)

    return Model(inputs=[inputs], outputs=[conv10])

