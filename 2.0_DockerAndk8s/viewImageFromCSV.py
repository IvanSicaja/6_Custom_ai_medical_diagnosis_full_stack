import pandas as pd
import numpy as np
import cv2


def viewImageFromCSV(df, new_input):
    # Extracting features (X) and labels (y)
    X = df.iloc[:, 1:].values  # Assuming pixel values are from column 1 to 4096
    y = df.iloc[:, 0].values   # Assuming labels are in the first column

    # Reshape X to (num_samples, height, width, channels) for convolutional layer
    X = X.reshape(-1, 64, 64, 1)

    # Make predictions on a new input
    new_input = int(new_input)  # Convert the string input to an integer
    input_data = X[new_input:new_input + 1]  # Take the specified sample

    # Convert the image data to uint8
    image = input_data[0].reshape(64, 64).astype(np.uint8)  # Reshape to 2D array for display

    # Display the image using OpenCV
    cv2.imshow('Input Image', image)
    while True:
        if cv2.waitKey(20) & 0xFF == ord('x'):
            break
    cv2.destroyAllWindows()


# DATASET AND WANTED INPUT SELECTION
csv_dataset_path = '0_Datasets/Br35H.csv'
new_input_sample = 100

# LOADING WANTED CSV DATASET
# Load the CSV dataset
df = pd.read_csv(csv_dataset_path)

# Call the function to display the specified sample
viewImageFromCSV(df, new_input_sample)
