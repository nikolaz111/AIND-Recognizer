import numpy as np
import pandas as pd
from asl_data import AslDb


asl = AslDb() # initializes the database
asl.df.head() # displays the first five rows of the asl database, indexed by video and frame


asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df.head()  # the new feature 'grnd-ry' is now in the frames dictionary


asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df.head()

asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df.head()

asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
asl.df.head()


# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']


df_means = asl.df.groupby('speaker').mean()
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df.head()


df_std = asl.df.groupby('speaker').std()

asl.df['right-x-mean']= asl.df['speaker'].map(df_means['right-x'])
asl.df['right-x-std']= asl.df['speaker'].map(df_std['right-x'])
asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean']) / asl.df['right-x-std']

asl.df['right-y-mean']= asl.df['speaker'].map(df_means['right-y'])
asl.df['right-y-std']= asl.df['speaker'].map(df_std['right-y'])
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean']) / asl.df['right-y-std']

asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df['left-x-std']= asl.df['speaker'].map(df_std['left-x'])
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean']) / asl.df['left-x-std']

asl.df['left-y-mean']= asl.df['speaker'].map(df_means['left-y'])
asl.df['left-y-std']= asl.df['speaker'].map(df_std['left-y'])
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean']) / asl.df['left-y-std']

asl.df.head()

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']

asl.df['polar-rr'] = np.sqrt( np.power(asl.df['grnd-rx'], 2) + np.power(asl.df['grnd-ry'], 2) )
asl.df['polar-rtheta'] = np.arctan2( asl.df['grnd-rx'], asl.df['grnd-ry'] )

asl.df['polar-lr'] = np.sqrt( np.power(asl.df['grnd-lx'], 2) + np.power(asl.df['grnd-ly'], 2) ) 
asl.df['polar-ltheta'] = np.arctan2( asl.df['grnd-lx'], asl.df['grnd-ly'] )

asl.df.head()

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']


asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(value=0)
asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(value=0)
asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(value=0)
asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(value=0)

asl.df.head()

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']


asl.df['hand-dist-x'] = np.fabs(asl.df['right-x'] - asl.df['left-x'])
asl.df['hand-dist-y'] = np.fabs(asl.df['right-y'] - asl.df['left-y'])
asl.df['hand-dist'] = np.sqrt( np.power(asl.df['right-x'] - asl.df['left-x'], 2) + np.power(asl.df['right-y'] - asl.df['left-y'], 2) )
asl.df['hand-angle'] = np.arctan2( asl.df['grnd-rx'] - asl.df['grnd-lx'], asl.df['grnd-ry'] - asl.df['grnd-ly'] )

# TODO define a list named 'features_custom' for building the training set
features_custom = ['hand-dist-x', 'hand-dist-y', 'hand-dist', 'hand-angle']


# TODO implement the recognize method in my_recognizer
from my_recognizer import recognize
from asl_utils import show_errors

from my_model_selectors import *


def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

# TODO Choose a feature set and model selector
features = features_ground
# features = features_norm
# features = features_polar
#features = features_delta
#features = features_custom
print(features)
model_selector = SelectorBIC # change as needed

# TODO Recognize the test set and display the result with the show_errors method
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)