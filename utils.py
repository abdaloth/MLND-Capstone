import numpy as np
from sklearn.metrics import roc_curve

def EER(y, y_pred, return_threshold=False):
    FAR, TAR, threshold = roc_curve(y, y_pred)
    FRR = 1 - TAR
    eer_threshold = threshold(np.nanargmin(np.absolute((FRR - FAR))))
    diff = np.abs(FRR - FAR)
    EER1 = FAR[np.argmin(diff)]
    EER2 = FRR[np.argmin(diff)]
    EER = (EER1 + EER2)/2
    return EER, eer_threshold
