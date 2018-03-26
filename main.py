from utils import EER, WAV_DICT
from identification_model import id_model
import numpy as np
import pandas as pd

np.random.seed(42)
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, concatenate, Lambda, Activation
from keras.layers import BatchNormalization


def load_data():
    train = pd.read_csv(
        'meta/train_trials.csv').sample(frac=.2, random_state=42)
    test = pd.read_csv('meta/test_trials.csv')

    validation = train.sample(frac=.3, random_state=42)
    train = train.drop(validation.index)

    features = np.load('data/cmvn_features.npy')

    POI_train = np.array([features[WAV_DICT[p]] for p in train.POI_wav])
    QUERY_train = np.array([features[WAV_DICT[p]] for p in train.Query_wav])
    y_train = train['is_match'].values

    test = pd.read_csv('meta/test_trials.csv')
    POI_test = np.array([features[WAV_DICT[p]] for p in test.POI_wav])
    QUERY_test = np.array([features[WAV_DICT[p]] for p in test.Query_wav])
    y_test = test['is_match'].values

    POI_val = np.array([features[WAV_DICT[p]] for p in validation.POI_wav])
    QUERY_val = np.array([features[WAV_DICT[p]] for p in validation.Query_wav])
    y_val = validation['is_match'].values

    return {'train': (POI_train, QUERY_train, y_train),
            'val': (POI_val, QUERY_val, y_val),
            'test': (POI_test, QUERY_test, y_test)}
#%%


def verif_model(lstm=True, verif=True, w_path='meta/models/lstm_identification_model.h5'):
    """
    returns the verification model

    w_path is the path of the trained weights of the base identification model
    """
    input_POI = Input(shape=(299, 26))
    input_Query = Input(shape=(299, 26))

    base_model = id_model(compile=False, verif=verif, lstm=lstm)
    base_model.trainable = False
    base_model.name = 'siamese_id_path'
    base_model.layers[-1].activation = Activation('sigmoid', name='activation')
    base_model.load_weights(w_path)
    POI = base_model(input_POI)
    Query = base_model(input_Query)
    out = concatenate([POI, Query])
    out = Dense(1024, activation='relu')(out)
    out = Dense(1024, activation='relu')(out)
    out = Dense(1, activation='sigmoid')(out)
    model = Model([input_POI, input_Query], out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adamax', metrics=['acc'])
    return model


#%%
if __name__ == '__main__':

    data = load_data()
    POI_train, QUERY_train, y_train = data['train']
    POI_val, QUERY_val, y_val = data['val']
    POI_test, QUERY_test, y_test = data['test']

    K.clear_session()
    model = verif_model()
    model.summary()

    BATCH_SIZE = 512
    es = EarlyStopping(patience=20)
    cp = ModelCheckpoint('meta/models/verification_model_v2_weights.h5',
                         monitor='val_acc',
                         save_best_only=True,
                         save_weights_only=True)


    hist = model.fit([POI_train, QUERY_train],
                     y_train,
                     validation_data=([POI_val, QUERY_val], y_val),
                     epochs=100,
                     batch_size=BATCH_SIZE,
                     callbacks=[es, cp])

    model.load_weights('meta/models/verification_model_v2_weights.h5')
    y_pred = model.predict([POI_test, QUERY_test],
                           batch_size=BATCH_SIZE,
                           verbose=1)

    y_pred = y_pred.flatten()
    print(EER(y_test, y_pred))
