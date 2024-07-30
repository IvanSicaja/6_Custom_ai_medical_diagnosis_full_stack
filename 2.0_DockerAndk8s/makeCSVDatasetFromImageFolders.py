import os
import re
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

def makeCSVDatasetFromImageFolders(folder_paths, output_csv_path, label_values):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_csv_path = output_csv_path.replace("rsna-mammography", f"rsna-mammography_{timestamp}")

    # Check if the output CSV file already exists
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)
        print(f"Deleted existing CSV file: {output_csv_path}")

    # Create an empty DataFrame to store pixel values
    df_pixels = pd.DataFrame()

    # Define a function to extract numerical values for sorting
    def numerical_sort(value):
        parts = re.split(r'(\d+)', value)
        return [int(part) if part.isdigit() else part for part in parts]

    # Calculate the total number of images
    total_images = 0
    for folder_path in folder_paths:
        total_images += len([f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))])

    print(f"Total number of images: {total_images}")

    current_image_index = 0

    # Iterate over all folder paths and corresponding label values
    for folder_path, label_value in zip(folder_paths, label_values):
        # Get all image filenames in the specified folder and sort them using the numerical sort
        filenames = sorted(
            [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))],
            key=numerical_sort
        )

        # Iterate over all sorted filenames
        for filename in filenames:
            # Construct the full path to the image
            image_path = os.path.join(folder_path, filename)

            # Load the image in grayscale
            gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Flatten the 2D array to a 1D array
            pixel_values = gray_image.flatten()

            # Create a dictionary with column names and values
            row_data = {"label": int(label_value)}
            for i, pixel_value in enumerate(pixel_values):
                row_data[f"pixel{i}"] = pixel_value

            # Append the row to the DataFrame
            df_pixels = df_pixels._append(row_data, ignore_index=True)

            # Print the progress
            current_image_index += 1
            print(f"Writing row {current_image_index}/{total_images} ({(current_image_index / total_images) * 100:.2f}%)")

    # Save the DataFrame to a CSV file
    df_pixels.to_csv(output_csv_path, index=False)

    print("CSV file saved successfully.")

# Example usage with multiple folder paths and corresponding label values

#All paths
#0_Datasets/Br35H/no/1_AdaptedAndUnifiedImagesSize
#0_Datasets/Br35H/yes/1_AdaptedAndUnifiedImagesSize
#0_Datasets/chest_xray/NORMAL/1_AdaptedAndUnifiedImagesSize
#0_Datasets/chest_xray/PNEUMONIA/1_AdaptedAndUnifiedImagesSize
#0_Datasets/melanoma_cancer_dataset/benign/1_AdaptedAndUnifiedImagesSize
#0_Datasets/melanoma_cancer_dataset/malignant/1_AdaptedAndUnifiedImagesSize
#0_Datasets/rsna-mammography-768-vl-perlabel/0/1_AdaptedAndUnifiedImagesSize
#0_Datasets/rsna-mammography-768-vl-perlabel/1/1_AdaptedAndUnifiedImagesSize_64x64
#NOTE:0_Datasets/rsna-mammography-768-vl-perlabel/0/1_AdaptedAndUnifiedImagesSize_ReducedNumberOfInstances_64x64
folder_paths = [
    "0_Datasets/melanoma_cancer_dataset/benign/1_AdaptedAndUnifiedImagesSize",
    "0_Datasets/melanoma_cancer_dataset/malignant/1_AdaptedAndUnifiedImagesSize"
]
label_values = [0, 1]  # label value for each folder path

makeCSVDatasetFromImageFolders(folder_paths, "0_Datasets/melanoma_cancer.csv", label_values)
