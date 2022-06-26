
# CPU
import tensorflow as tf

# GPU
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Data analysis
import pandas as pd

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Numerical computation
import numpy as np

# Use pandas to read in the dataset
data = pd.read_csv('dataset.csv')

# Split the data into features and labels
X = data.drop('label', axis=1)
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Define the model
model = tf.keras.models.Sequential()

# Add the first layer
model.add(tf.keras.layers.Dense(4, activation='relu', input_shape=(4,)))

# Add the second layer
model.add(tf.keras.layers.Dense(4, activation='relu'))

# Add the output layer
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test set accuracy: {:.3f}'.format(accuracy))

# Make predictions
predictions = model.predict(X_test)
print('First 10 predictions: {}'.format(predictions[:10]))

# Use the CPU
with tf.device('/cpu:0'):
   model = tf.keras.models.Sequential()

# Use the GPU
with tf.device('/gpu:0'):
   model = tf.keras.models.Sequential()

# Train the model on the GPU
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model on the GPU
loss, accuracy = model.evaluate(X_test, y_test)
print('Test set accuracy: {:.3f}'.format(accuracy))

# Make predictions on the GPU
predictions = model.predict(X_test)
print('First 10 predictions: {}'.format(predictions[:10]))


# Save the model
model.save('my_model.h5')

# Load the model
model = tf.keras.models.load_model('my_model.h5')

# Use the model to make predictions
predictions = model.predict(X_test)
print('First 10 predictions: {}'.format(predictions[:10]))

# Use the model to make a prediction on a single input
print(model.predict(np.array([[0.5, 0.4, 0.3, 0.2]])))
