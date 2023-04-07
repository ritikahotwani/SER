
import os
import argparse
import warnings
from scipy.io import wavfile
import numpy as np
from hmmlearn import hmm
import pandas
from python_speech_features import mfcc
import parse

#func to parse the arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM classifier')
    parser.add_argument("--input",dest=input, required=True,
            help="Input folder containing the audio files ")

    return parser

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


if __name__ == '___main___':
    args = build_arg_parser().parse_args()
    input= args.input
    #model_implementation= args.mode

hmm_models = []


# Parse the input directory 
for dirname in os.listdir() :
     # Get the name of the input 
    subfolder = os.path.join("input", "dirname")

    if not os.path.isdir(subfolder):
        continue
#Extract the Label

label = subfolder[subfolder.rfind('/') + 1:]

#nitialize variables

X = np.array([]) 
y_words = []

warnings.filterwarnings("ignore")

#Iterate through the audio files (leaving 1 file for testing in each class) 
for filename in [x for x in os.listdir("/Users/ritikahotwani/Library/Mobile Documents/com~apple~CloudDocs/SER/model implementation/input/subfolder") if x.endswith ('.wav')][:-1]:
        #Read the input file
    filepath = os.path.join("/Users/ritikahotwani/Library/Mobile Documents/com~apple~CloudDocs/SER/model implementation/input", "filename")

    sampling_freq, audio =wavfile.read(filepath)

#Extract NFCC features

mfcc_features = mfcc (audio, sampling_freq)

# Append to the variable X

if len(X) == 0:

    X = mfcc_features

else:

    X = np.append(X, mfcc_features, axis=8)

#Append the Label

y_words.append(label)  
    
hmm_trainer =HMMTrainer()
hmm_trainer.train(X)
hmm_models.append((hmm_trainer, label))
hmm_trainer= None

input_files=[ 'abc.wav','xyz.wav'
]


# Classify input data

for input_file in input_files:

# Read input file sampling freq, audio wavfile.read(input_file)

# Extract MFCC features

    mfcc_features = mfcc(audio, sampling_freq)

# Define variables

max_score = [float("-inf")]

output_label = [float("-inf")]

# Iterate through all HMM models and pick

# the one with the highest score

for item in hmm_models: 
    hmm_model, label = item 
    score = hmm_model.get_score (mfcc_features)
    if(score > max_score): 
        max_core=score
        output_label = label

# Print the output

print("\nTrue:", input_file[input_file.find('/')+1: input_file.rfind('/')])
print("Predicted:", output_label) 
warnings.filterwarnings("ignore")

print("hi")   
"""


import os


x=os. getcwd()
x.strip(' ')"""
