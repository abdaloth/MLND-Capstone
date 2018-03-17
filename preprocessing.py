import numpy as np
from scipy.signal import hamming, spectrogram

from librosa.feature import mfcc
from librosa.core import load, stft

SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 26
N_FFT = 1024
STEP = int(10 * SAMPLE_RATE / 1000)
WINDOW_LENGTH = int(25 * SAMPLE_RATE / 1000)

with open('data/wav_list') as f:
    WAV_LIST = [line.strip() for line in f.readlines()]

# mmap mode keeps most of the file on disk
audio_ts = np.load('data/audio_ts.npy', mmap_mode='r')

def create_audio_matrix(path, wav_list=WAV_LIST):
    """ reads the wav files and returns the audio time series
        for the specified duration and sample rate
        * NOTE this takes 40+ minutes to run on the full dataset
    """
    audio_ts = np.zeros((len(wav_list), SAMPLE_RATE * DURATION))
    for i, p in enumerate(wav_list):
        audio_ts[i] = load_audio(p)
    np.save(path, audio_ts)


def create_mfcc_matrix(path, wav_list=WAV_LIST):
    """ reads the audio arrays and returns the mfcc features
        * NOTE this takes 30+ minutes to run on the full dataset
    """
    mfccs = np.zeros((len(wav_list), N_MFCC, 94))
    for i, p in enumerate(wav_list):
        mfccs[i] = create_mfcc(p)
    np.save(path, mfccs)


def create_spectogram_matrix(path, wav_list=WAV_LIST):
    """ reads the audio arrays and returns the spectogram features
        * NOTE this takes 3+ hours to run on the full dataset
    """
    spectogram_shape_0 = int(N_FFT / 2) + 1
    spectogram_shape_1 = int(DURATION * SAMPLE_RATE / STEP) + 1
    spectograms = np.zeros((int(len(WAV_LIST) / 3),
                       spectogram_shape_0,
                       spectogram_shape_1))

    idx = [i for i in range(0, len(WAV_LIST) + 1, int(len(WAV_LIST) / 3))]

    for i, (start, end) in enumerate(zip(idx[:-1], idx[1:])):
        for j, p in enumerate(WAV_LIST[start:end]):
            spectograms[j] = create_spectogram(p)
        np.save(f'{path}_{i}', spectograms)



def merge_spectogram_files():
    """ merge the three files to one single file in such a way where we can avoid memory error"""
    part_1 = np.load('spectogram_features_0.npy', mmap_mode='r')
    part_2 = np.load('spectogram_features_1.npy', mmap_mode='r')
    part_3 = np.load('spectogram_features_2.npy', mmap_mode='r')

    spectogram_features = np.memmap('spectogram_features_0.npy', dtype='float64', mode='r+', shape=(152799,513,301))
    spectogram_features[:50933,:] = part_1
    spectogram_features[50933:101866,:] = part_2
    spectogram_features[101866:,:] = part_3



def load_audio(path):
    audio, _ = load(path, sr=SAMPLE_RATE, duration=DURATION)
    return audio


def create_mfcc(path, wav_list=WAV_LIST):
    audio = audio_ts[wav_list.index(path)]
    return mfcc(audio, SAMPLE_RATE, n_mfcc=N_MFCC)


def create_spectogram(path, wav_list=WAV_LIST):
    audio = audio_ts[wav_list.index(path)]
    return np.abs(stft(audio,
                       n_fft=N_FFT, window=hamming,
                       hop_length=STEP, win_length=WINDOW_LENGTH
                       )
                  )


#%%
# def get_spectogram_scipy(path):
#     audio = audio_ts[WAV_LIST.index(path)]
#     _, _, spect = spectrogram(audio, nfft=1024, window=hamming(160, False), mode='magnitude')
#     return spec
