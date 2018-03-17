import numpy as np
from sklearn.metrics import roc_curve
from keras.utils import Sequence

with open('data/wav_list') as f:
    WAV_LIST = [line.strip() for line in f.readlines()]

WAV_DICT = {path: index for index, path in enumerate(WAV_LIST)}
# mmap mode keeps most of the file on disk
audio_ts = np.load('data/audio_ts.npy', mmap_mode='r')
spectogram_features = np.memmap('data/spectogram_features.npy',
                                dtype='float64',
                                shape=(152799, 513, 301),
                                mode='r')
mfcc_features = np.load('data/mfcc_features.npy', mmap_mode='r')


class Generator(Sequence):
    """loads features per-batch"""

    def __init__(self, inputs, output, batch_size, feature='mfcc'):
        self.x_poi, self.x_query = inputs
        self.y = output
        self.batch_size = batch_size
        self.feature = feature

    def __len__(self):
        return round(len(self.y) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        x_poi_f = np.array([get_features(x, self.feature)
                            for x in self.x_poi[start_idx:end_idx]])
        x_query_f = np.array([get_features(x, self.feature)
                              for x in self.x_query[start_idx:end_idx]])
        batch_x = [x_poi_f, x_query_f]
        batch_y = self.y[start_idx:end_idx]

        return batch_x, batch_y


class TestGenerator(Sequence):
    """loads features per-batch"""

    def __init__(self, inputs, batch_size, feature='mfcc'):
        self.x_poi, self.x_query = inputs
        self.batch_size = batch_size
        self.feature = feature

    def __len__(self):
        return round(len(self.x_poi) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        x_poi_f = np.array([get_features(x, self.feature)
                            for x in self.x_poi[start_idx:end_idx]])
        x_query_f = np.array([get_features(x, self.feature)
                              for x in self.x_query[start_idx:end_idx]])
        batch_x = [x_poi_f, x_query_f]

        return batch_x


def get_features(path, feature):
    index = WAV_DICT[path]

    if(feature == 'mfcc'):
        return mfcc_features[index]
    elif(feature == 'spectogram'):
        return spectogram_features[index]
    elif(feature == 'time_series'):
        return audio_ts[index]
    else:
        raise KeyError(f'unknown feature name: {feature}')


def EER(y, y_pred, return_threshold=False):
    FAR, TAR, threshold = roc_curve(y, y_pred)
    FRR = 1 - TAR
    eer_threshold = threshold[np.nanargmin(np.absolute((FRR - FAR)))]
    diff = np.abs(FRR - FAR)
    EER1 = FAR[np.argmin(diff)]
    EER2 = FRR[np.argmin(diff)]
    EER = (EER1 + EER2) / 2
    if(return_threshold):
        return EER, eer_threshold
    else:
        return EER
