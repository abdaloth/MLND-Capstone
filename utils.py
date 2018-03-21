import numpy as np
from sklearn.metrics import roc_curve

with open('data/wav_list') as f:
    WAV_LIST = [line.strip() for line in f.readlines()]

WAV_DICT = {path: index for index, path in enumerate(WAV_LIST)}

def EER(y, y_pred, return_threshold=False):
    FAR, TAR, threshold = roc_curve(y, y_pred)
    FRR = 1 - TAR
    eer_threshold = threshold[np.nanargmin(np.absolute((FRR - FAR)))]
    diff = np.abs(FRR - FAR)
    EER = np.mean([FAR[np.argmin(diff)], FRR[np.argmin(diff)]])
    if(return_threshold):
        return EER, eer_threshold
    else:
        return EER
