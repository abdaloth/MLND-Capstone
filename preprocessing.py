import numpy as np
import python_speech_features as psf
from scipy.io.wavfile import read as read_wav

SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 26
N_FFT = 1024
STEP = .01
WINDOW_LENGTH = .025
with open('data/wav_list') as f:
    WAV_LIST = [line.strip() for line in f.readlines()]


# Takes 80+ minutes to run on the full dataset
def create_mfcc_array(output_path='data/mfcc_features.npy',
                      wav_list=WAV_LIST,
                      n_mfcc=N_MFCC,
                      sample_rate=SAMPLE_RATE,
                      duration=DURATION,
                      window_length=WINDOW_LENGTH,
                      step=STEP,
                      n_fft=N_FFT):
    n_frames = int(duration / step) - 1
    mfcc_features = np.zeros((len(wav_list), n_frames, n_mfcc))
    for i, wav_path in enumerate(wav_list):
        rate, signal = read_wav(wav_path)
        mfcc_features[i] = psf.mfcc(signal[:(duration * sample_rate)],
                                    samplerate=sample_rate,
                                    winlen=window_length,
                                    winstep=step,
                                    numcep=n_mfcc,
                                    nfft=n_fft)
    np.save(output_path, mfcc_features)
    return mfcc_features



def normalize_array(array, mean=True, variance=True, output_path='data/mcmvn_features.npy'):
    normalized = array
    mu = 0
    std = 1
    for i in range(array.shape[0]):
        if(mean):
            mu = np.mean(normalized[i], axis=0)
        if(variance):
            std = np.std(normalized[i], axis=0)
        normalized[i] = (normalized[i] - mu) / std
    np.save(output_path, normalized)
    return normalized
