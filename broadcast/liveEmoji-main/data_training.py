import os  
import numpy as np 
from tensorflow.keras.utils import to_categorical 
from keras.layers import Input, Dense 
from keras.models import Model 

is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
        if not is_init:
            is_init = True 
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]] * np.load(i).shape[0]).reshape(-1, 1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c += 1

# Convert labels to integers
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]

# Convert labels to categorical format
y = to_categorical(y)

# Shuffle data
permutation = np.random.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]

# Define input shape
input_shape = X.shape[1:]

# Define input layer
ip = Input(shape=input_shape)

# Define model architecture
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m) 

# Create model
model = Model(inputs=ip, outputs=op)

# Compile model
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train model
model.fit(X, y, epochs=50)

# Save model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
