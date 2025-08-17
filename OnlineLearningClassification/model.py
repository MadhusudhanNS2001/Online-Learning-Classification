from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.nadam import Nadam
from tensorflow.python.keras.regularizers import l2

from utils import CLASSES

ENCODER_IN_DIM = 91


def sparse_regularizer(activation_matrix):
    p = 0.01
    beta = 3
    p_hat = K.mean(activation_matrix)
    KL_divergence = p * (K.log(p / p_hat)) + (1 - p) * (K.log(1 - p / 1 - p_hat))
    sum = K.sum(KL_divergence)
    return beta * sum


def encoder():
    print('[INFO] Building Encoder')
    model = Sequential(name='encoder')
    model.add(Dense(128, input_dim=ENCODER_IN_DIM))
    model.add(Dense(64))
    model.add(Dense(len(CLASSES)))
    return model


def decoder():
    KR = l2(0.001 / 2)
    AR = sparse_regularizer
    print('[INFO] Building Decoder')
    model = Sequential(name='decoder')
    model.add(Dense(64, input_dim=len(CLASSES)))
    model.add(Dense(128))
    model.add(Dense(len(CLASSES), activation='softmax', kernel_regularizer=KR, activity_regularizer=AR))
    return model


def buildDSSAE():
    enc = encoder()
    dec = decoder()
    print('[INFO] Building Deep Stacked Sparse AutoEncoder')
    model = Sequential([enc, dec], name='DSSAE')
    print('[INFO] Compiling DSSAE Using NADAM Optimizer')
    opt = Nadam(lr=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
