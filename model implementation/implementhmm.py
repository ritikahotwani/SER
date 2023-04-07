import os
import argparse
import warnings
from scipy.io import wavfile
import numpy as np
from hmmlearn import hmm
import pandas
from python_speech_features import mfcc
import parse
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_TYPE='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components =n_components
        self.cov_TYPE = cov_TYPE
        self.n_iter = n_iter
        if self.model_name == 'GaussianHMM': 
            self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type, n_iter = self.n_iter)

        else:

            raise TypeError('Invalid model type')


#X is a 20 numpy array where each row is 130

def train(self, x):
    np.seterr(all='ignore') 
    self.models.append(self.model.fit(x))

# Run the model on input data

def get_score(self, input_data):

    return self.model.score (input_data)
