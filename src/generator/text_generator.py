#tutorial from: https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
import os, sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

#load data
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
text=(open("{0}/test/pg1041.txt".format(root_folder,)).read())
text=text.lower()

#generate character mapping (could use word mapping too but requires bigger data set)
characters = sorted(list(set(text)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

#generate feature arrays
X = []
Y = []
length = len(text)
seq_length = 100
for i in range(0, length-seq_length, 1):
    sequence = text[i:i + seq_length]
    label = text[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])

#reshape feature arrays
X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)

#model
#sequential LSTM model with two LSTM layers having 400 units each
model = Sequential() 
#The first layer needs to be fed in with the input shape. 
#In order for the next LSTM layer to be able to process the same sequences, 
#we enter the return_sequences parameter as True.
model.add(LSTM(700, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
#Also, dropout layers with a 20% dropout have been added to check for over-fitting.
model.add(Dropout(0.2))
model.add(LSTM(700, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700))
model.add(Dropout(0.2))
#The last layer outputs a one hot encoded vector which gives the character output.
model.add(Dense(Y_modified.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#fit model, save and load weights
model.fit(X_modified, Y_modified, epochs=100, batch_size=50)
model.save_weights('{0}/test/text_generator_100_50_baseline.h5'.format(root_folder,))
model.load_weights('{0}/test/text_generator_100_50_baseline.h5'.format(root_folder,))

#generate characters
#start with a random element of X
string_mapped = X[99]
full_string = [n_to_char[value] for value in string_mapped]
print("initial string: ", string_mapped)
#replace random element with array of predicted characters of same length
for i in range(400):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(characters))
    #predict next character
    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    #add predicted character to random element
    full_string.append(n_to_char[pred_index])
    string_mapped.append(pred_index)
    #remove first character from random element
    string_mapped = string_mapped[1:len(string_mapped)]

#combining text
print("full string: ", "".join(full_string))