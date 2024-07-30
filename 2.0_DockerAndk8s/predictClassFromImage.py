import cv2
import numpy as np
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def unifyImageSize(image_path, desired_width, desired_height, offsetInPixelNumber):

    # Load another image from the folder
    folder_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if the image is grayscale
    if len(folder_image.shape) == 2:
        folder_image = cv2.cvtColor(folder_image, cv2.COLOR_GRAY2BGR)

    # Get the dimensions of the folder image
    folder_height, folder_width, number_of_channels = folder_image.shape

    # Create a black image with the desired dimensions
    black_image = np.zeros((desired_height, desired_width, number_of_channels), dtype=np.uint8)

    # Calculate the scaling factor to fit the image into the desired dimensions
    scaling_factor = min(desired_width / folder_width, desired_height / folder_height)

    # Resize the folder image while maintaining aspect ratio
    scaled_width = int(folder_width * scaling_factor)
    scaled_height = int(folder_height * scaling_factor)
    resized_folder_image = cv2.resize(folder_image, (scaled_width, scaled_height))

    # Calculate the position to center the folder image in the black image
    x_offset = (desired_width - scaled_width) // 2
    y_offset = (desired_height - scaled_height) // 2

    # Place the resized folder image onto the black image
    black_image[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = resized_folder_image

    # Create a black border around the result image with the specified offsetInPixelNumber
    result_height, result_width, _ = black_image.shape
    bordered_image = np.zeros((result_height + 2 * offsetInPixelNumber, result_width + 2 * offsetInPixelNumber, number_of_channels), dtype=np.uint8)
    bordered_image[offsetInPixelNumber:offsetInPixelNumber + result_height, offsetInPixelNumber:offsetInPixelNumber + result_width] = black_image

    # Save the edited image to the output folder
    cv2.imwrite(image_path, bordered_image)

    # Display the result using OpenCV
    # cv2.imshow("Result Image with Border", bordered_image)
    print("Result Image with Border type:", type(bordered_image))
    print("Result Image with Border     :", bordered_image.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return bordered_image


def predictClassProbabilityFromImage(model, image_path):

    # Load the image in grayscale
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Flatten the 2D array to a 1D array
    pixel_values = gray_image.flatten()

    # Normalize pixel values
    pixel_values = pixel_values / 255.0

    # Reshape X to (num_samples, height, width, channels) for convolutional layer
    pixel_values=pixel_values.reshape(-1,64,64,1)

    # Make predictions on a new input
    predictions = model.predict(pixel_values)

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



# # MODEL DATASET AND WANTED INPUT SELECTION
# model_path = '2_savedModels/brain_tumor_model.h5'
# image_path = 'uploads/no0.jpg'
#
# # ADAPT IMAGE SHAPE
# # unifyImageSize("uploads/no0.jpg", 60, 60, 2)
#
# # LOAD THE TRAINED MODEL
# model = load_model(model_path)
#
# # EXAMPLE USAGE:
# predicted_class, predicted_probability = predictClassProbabilityFromImage(model, image_path)
# print(f"Predicted class: {predicted_class}")
# print(f"Predicted probability: {predicted_probability}")

