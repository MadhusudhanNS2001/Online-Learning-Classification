if True:
    from reset_random import reset_random

    reset_random()
import os
import shutil
import time

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from model import buildDSSAE
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils import CLASSES, plot, TrainingCallback

matplotlib.use('Qt5Agg')
matplotlib.style.use('seaborn-darkgrid')
plt.rcParams['font.family'] = 'JetBrains Mono'

ACC_PLOT = plt.figure(num=2)
LOSS_PLOT = plt.figure(num=3)
RESULTS_PLOT = {
    'Train': {
        'CONF_MAT': plt.figure(num=4),
        'PR_CURVE': plt.figure(num=5),
        'ROC_CURVE': plt.figure(num=6),
    },
    'Test': {
        'CONF_MAT': plt.figure(num=7),
        'PR_CURVE': plt.figure(num=8),
        'ROC_CURVE': plt.figure(num=9),
    }
}


def get_data():
    dp = 'Data/preprocessed.csv'
    df = pd.read_csv(dp)
    x_, y_ = df.values[:, :-1], df.values[:, -1]
    return x_, y_


def train():
    reset_random()

    x, y = get_data()

    y_cat = to_categorical(y, len(CLASSES))
    print('[INFO] Splitting Data Into Training|Testing')
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=1)
    test_y_cat = to_categorical(test_y, len(CLASSES))
    print('[INFO] X Shape :: {0}'.format(x.shape))
    print('[INFO] Train X Shape :: {0}'.format(train_x.shape))
    print('[INFO] Test X Shape :: {0}'.format(test_x.shape))

    model_dir = 'models'
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    acc_loss_csv_path = os.path.join(model_dir, 'acc_loss.csv')
    model_path = os.path.join(model_dir, 'model.h5')

    training_cb = TrainingCallback(acc_loss_csv_path, ACC_PLOT, LOSS_PLOT)
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True,
                                 monitor='val_accuracy', mode='max', verbose=False)

    model = buildDSSAE()

    initial_epoch = 0
    if os.path.isfile(model_path) and os.path.isfile(acc_loss_csv_path):
        print('[INFO] Loading Pre-Trained Model :: {0}'.format(model_path))
        model.load_weights(model_path)
        initial_epoch = len(pd.read_csv(acc_loss_csv_path))

    t1 = time.time()
    print('[INFO] Fitting Data')
    model.fit(x, y_cat,
              validation_data=(test_x, test_y_cat), epochs=50,
              verbose=0, initial_epoch=initial_epoch, callbacks=[training_cb, checkpoint])
    t2 = time.time()
    print('[INFO] Computational Time :: {0} secs'.format(int(t2 - t1)))

    model.load_weights(model_path)

    train_prob = model.predict(train_x)
    train_pred = np.argmax(train_prob, axis=1).ravel().astype(int)
    plot(train_y.astype(int), train_pred, train_prob, RESULTS_PLOT, 'results/Train')

    test_prob = model.predict(test_x)
    test_pred = np.argmax(test_prob, axis=1).ravel().astype(int)
    plot(test_y.astype(int), test_pred, test_prob, RESULTS_PLOT, 'results/Test')


if __name__ == '__main__':
    train()
