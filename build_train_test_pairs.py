# -*- coding: utf-8 -*-
"""
This module is responsible for creating the training dataset for the verification model
"""

import re
import random
import pandas as pd
from glob import glob
from warnings import warn

random.seed('42') # for reproducibility
TRAIN_UTTERANCES = [u for u in glob('data\\*\\*.wav')
                    if not u.split('\\')[1].startswith('E')]
TRAIN_POIS = sorted(set([u.split('\\')[1] for u in TRAIN_UTTERANCES]))
TRAIN_PAIRS = []
POSITIVE_TRIALS_PER_UTTERANCE = 2  # same as negative trials to keep it balanced

columns = ['is_match', 'POI_wav', 'Query_wav']


def make_trial(utterance, utterance_list, same=False, max_tries=500):
    """
    this function takes as input an utterance and a list of utterances and creates
    a labeled verification pair.
    if same == True it will create pairs that are a speaker match. ensuring the
    two utterances are from different video sources.
    else it will make a pair where the speakers do not match.
    """
    for i in range(max_tries): # to keep the function from running too long
        pair = random.choice(utterance_list)
        if(same):
            if(utterance[:-12] != pair[:-12]):
                return sorted(['1', utterance, pair])
        else:
            if(utterance[:pair.rindex('\\')] != pair[:pair.rindex('\\')]):
                return sorted(['0', utterance, pair])
    warn('WARNING max tries reached, returning None')
    return None


if __name__ == '__main__':
    for person in TRAIN_POIS:
        person_utterances = glob('data\\' + person + '\\*.wav')
        for utterance in person_utterances:
            for _ in range(POSITIVE_TRIALS_PER_UTTERANCE):
                TRAIN_PAIRS.append(make_trial(
                    utterance, person_utterances, same=True))
                TRAIN_PAIRS.append(make_trial(
                    utterance, TRAIN_UTTERANCES, same=False))

    df = pd.DataFrame(TRAIN_PAIRS, columns=columns)
    df = df.drop_duplicates()
    df.to_csv('meta/train_trials.csv', index=False)

    with open('meta/verification_test_pairs.txt') as f:
        test_pairs = [line.strip() for line in f.readlines()]
    test_pairs = [sorted(re.sub(r'/', r'\\', p).split()) for p in test_pairs]

    df = pd.DataFrame(test_pairs, columns=columns)
    # The video Emily_Atack\\d2Lasybvo7s is a duplicate
    df.loc[:, 'POI_wav'] = df.POI_wav.apply(lambda p: None
                                            if 'Emily_Atack\\d2Lasybvo7s' in p
                                            else 'data\\' + p)
    df.loc[:, 'Query_wav'] = df.Query_wav.apply(lambda p: None
                                                if 'Emily_Atack\\d2Lasybvo7s' in p
                                                else 'data\\' + p)
    df = df.dropna()
    df = df.drop_duplicates()
    df.to_csv('meta/test_trials.csv', index=False)
