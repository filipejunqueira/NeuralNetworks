#Library for NN
from keras.models import Sequential
#Library for Fully conected NN
from keras.layers import Dense
#Library for gathering information and then ploting
from keras.utils import plot_model
#Library for plots
import matplotlib.pyplot as plt
#Standard numerical library
import numpy
#library for recording time
import time
#Library for doing some math
import math

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(128, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model

starttime = time.time()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.1, epochs=2000, batch_size=50, verbose=1)
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#print the model
plot_model(model, to_file='model.png',show_shapes =True,show_layer_names = True)

elapsedtime = 'The time elapsed was ' + repr(time.time() - starttime) + 's'

# list all data in history
print(history.history.keys())
print(elapsedtime)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
