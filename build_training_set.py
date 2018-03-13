# -*- coding: utf-8 -*-
"""Placeholder docstrings.

Todo:
    * Write Docstring
    * Write Comments
"""

from glob import glob
from random import choice
from warnings import warn
import pandas as pd
TRAIN_UTTERANCES = [u for u in glob('data\\*\\*.wav') if not u.split('\\')[1].startswith('E')]
TRAIN_POIS = sorted(set([u.split('\\')[1] for u in TRAIN_UTTERANCES]))
TRAIN_PAIRS = []
POSITIVE_TRIALS_PER_UTTERANCE = 4 # same as negative trials to keep it balanced

def make_trial(utterance, utterance_list, same=False, max_tries=500):
    for i in range(max_tries):
        pair = choice(utterance_list)
        if(same):
            if(utterance[:-12] != pair[:-12]):
                return (1, utterance, pair)
        else:
            if(utterance[:pair.rindex('\\')] != pair[:pair.rindex('\\')]):
                return (0, utterance, pair)
    warn('WARNING max tries reached, returning None')
    return None

if __name__ == '__main__':
    for person in TRAIN_POIS:
        person_utterances = glob('data\\'+person+'\\*.wav')
        for utterance in person_utterances:
            for _ in range(TRIALS_PER_UTTERANCE):
                TRAIN_PAIRS.append(make_trial(utterance, person_utterances, same=True))
                TRAIN_PAIRS.append(make_trial(utterance, TRAIN_UTTERANCES, same=False))


    df = pd.DataFrame(TRAIN_PAIRS, columns=['same_person', 'first', 'second'])
    df = df.drop_duplicates()
    df.to_csv('meta/train_trials.csv', index=False)
