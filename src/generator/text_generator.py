#tutorial from: https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
import os, sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
text=(open("{0}/test/pg1041.txt".format(root_folder,)).read())
text=text.lower()

characters = sorted(list(set(text)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}