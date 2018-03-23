from utils import EER, WAV_DICT
from identification_model import id_model
import gc
import numpy as np
import pandas as pd

np.random.seed(42)
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, concatenate, Lambda, Activation
from keras.layers import BatchNormalization

train = pd.read_csv('meta/train_trials.csv').sample(frac=.2, random_state=42)
test = pd.read_csv('meta/test_trials.csv')

validation = train.sample(frac=.3, random_state=42)
train = train.drop(validation.index)

features = features = np.load('data/cmvn_features.npy')

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

del features
gc.collect()
#%%


def verif_model():
    input_POI = Input(shape=(299, 26))
    input_Query = Input(shape=(299, 26))

    base_model = id_model(compile=False, verif=True)
    base_model.trainable = False
    base_model.name = 'siamese_id_path'
    base_model.layers[-1].activation = Activation('sigmoid', name='activation')
    base_model.load_weights('meta/models/lstm_identification_model.h5')
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


def build_model():
    id_model_1 = load_model('meta/models/identification_model.h5')
    id_model_1.layers[-1].activation = 'sigmoid'
    for l in id_model_1.layers:
        l.name = l.name + '_POI'

    id_model_2 = load_model('meta/models/identification_model.h5')
    id_model_2.layers[-1].activation = 'sigmoid'
    for l in id_model_2.layers:
        l.name = l.name + '_Query'

    POI = id_model_1.get_layer(index=-1).output
    QUERY = id_model_2.get_layer(index=-1).output
    out = concatenate([POI, QUERY])
    out = Dense(1024, activation='relu')(out)
    out = Dense(1024, activation='relu')(out)
    out = Dense(1, activation='sigmoid')(out)
    verification_model = Model([id_model_1.input, id_model_2.input], out)
    verification_model.compile(
        loss='binary_crossentropy', optimizer='adamax', metrics=['acc'])
    return verification_model


K.clear_session()
model = verif_model()
model.summary()
#%%
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
#%%
model.load_weights('meta/models/verification_model_v2_weights.h5')
y_pred = model.predict([POI_test, QUERY_test],
                       batch_size=BATCH_SIZE,
                       verbose=1)

y_pred = y_pred.flatten()
print(EER(y_test, y_pred))
