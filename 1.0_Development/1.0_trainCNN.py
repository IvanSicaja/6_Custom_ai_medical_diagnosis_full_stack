import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
CSVDataFileName = 'Br35H'
file_path = '0_Datasets/' + CSVDataFileName + '.csv'
df = pd.read_csv(file_path)

# Extracting features (X) and labels (y)
X = df.iloc[:, 1:].values  # Assuming pixel values are from column 1 to 4096
y = df.iloc[:, 0].values   # Assuming labels are in the first column

# Normalize pixel values
X = X / 255.0

x0=X[0]

# Reshape X to (num_samples, height, width, channels) for convolutional layer
X = X.reshape(-1, 64, 64, 1)

# Convert labels to numerical format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(y)
print(type(y))
print(y.shape)

# Convert labels to one-hot encoding
y = to_categorical(y)
print(y)
print(type(y))
print(y.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Convolutional Neural Network (CNN) model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))  # Adding a dropout layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Adding another dropout layer
model.add(Dense(2, activation='softmax'))  # Output layer with two classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model_name = f'{CSVDataFileName}_model_{accuracy * 100:.2f}.h5'
model.save('2_savedModels/' + model_name)

# Plot the learning curve
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(f'2_savedModels/{model_name.replace(".h5", "_acc.png")}')
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'2_savedModels/{model_name.replace(".h5", "_loss.png")}')
plt.show()

