from utils import WAV_DICT
import numpy as np
import pandas as pd
import gc
np.random.seed(42)
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Bidirectional, concatenate, SpatialDropout1D
from keras.layers import CuDNNGRU, CuDNNLSTM, GlobalAvgPool1D, GlobalMaxPool1D, BatchNormalization

def id_model(compile=True, lstm=True, verif=False):
    inp = Input(shape=(299, 26))
    x = SpatialDropout1D(.1)(inp)
    if(lstm):
        x = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
    else:
        x = Bidirectional(CuDNNGRU(256, return_sequences=True))(x)
    gmp = GlobalMaxPool1D()(x)
    gap = GlobalAvgPool1D()(x)
    x = concatenate([gmp, gap])
    x = Dropout(.5)(x)
    x = BatchNormalization()(x)
    if(verif):
        x = Dense(1211, activation='softmax')(x)
    else:
        x = Dense(1251, activation='softmax')(x)
    model = Model(inp, x)
    if(compile):
        model.compile(loss='categorical_crossentropy',
                      metrics=['acc', 'top_k_categorical_accuracy'], optimizer='adam')
    return model


if __name__ == '__main__':
    features = np.load('data/cmvn_features.npy')
    train = pd.read_csv('meta/train_id.csv')
    val = pd.read_csv('meta/val_id.csv')
    test = pd.read_csv('meta/test_id.csv')

    X_train = np.array([features[WAV_DICT[p]] for p in train.path])
    y_train = pd.get_dummies(train.label)
    X_val = np.array([features[WAV_DICT[p]] for p in val.path])
    y_val = pd.get_dummies(val.label)
    X_test = np.array([features[WAV_DICT[p]] for p in test.path])
    y_test = pd.get_dummies(test.label)

    del features
    gc.collect()
    K.clear_session()
    model = id_model()
    model.summary()
    hist = model.fit(X_train, y_train, batch_size=512, epochs=50, validation_data=(X_val, y_val))
