import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def predictClassProbabilityFromCSV(model, df, new_input):

    # Extracting features (X) and labels (y)
    X = df.iloc[:, 1:].values  # Assuming pixel values are from column 1 to 4096
    y = df.iloc[:, 0].values   # Assuming labels are in the first column

    # Normalize pixel values
    X = X / 255.0

    # Reshape X to (num_samples, height, width, channels) for convolutional layer
    X = X.reshape(-1, 64, 64, 1)

    # Make predictions on a new input
    new_input = int(new_input)  # Convert the string input to an integer
    input_data = X[new_input:new_input + 1]  # Take the specified sample
    predictions = model.predict(input_data)

    # Convert the predicted probabilities to class labels
    predicted_class = np.argmax(predictions, axis=1)

    # Print the predicted class and its probability
    predicted_probability = predictions[0, predicted_class[0]]

    # Type-shape prints
    # print(f"Predicted class type: {type(predicted_class)}")
    # print(f"Predicted probability type: {type(predicted_probability)} ")
    # print(f"Predicted class shape: {predicted_class.shape}")
    # print(f"Predicted probability shape: {predicted_probability.shape} ")

    # Get output in the wanted shape
    predicted_probability *= 100  # Scale to percentage
    formatted_probability = round(predicted_probability, 1)
    output_probability = str(formatted_probability) + "%"

    return predicted_class[0], output_probability



# MODEL DATASET AND WANTED INPUT SELECTION
model_path = '2_savedModels/brain_tumor_model.h5'
csv_dataset_path = '0_Datasets/1_BrainTumorDataset_Br35H.csv'
new_input_sample = 123

# LOADING WANTED TRAINED MODEL AND CSV DATASET
# Load the trained model
model = load_model(model_path)
# Load the CSV dataset
df = pd.read_csv(csv_dataset_path)

# EXAMPLE USAGE:
predicted_class, predicted_probability = predictClassProbabilityFromCSV(model, df, new_input_sample)
print(f"Predicted class: {predicted_class}")
print(f"Predicted probability: {predicted_probability}")
