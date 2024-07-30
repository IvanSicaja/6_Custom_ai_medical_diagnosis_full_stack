import os
import cv2
import shutil
import numpy as np

def unifyImageSizeFromFolderAndSaveAsImage_blackBackground(folder_path, desired_width, desired_height, offsetInPixelNumber, output_format=".jpg"):
    # Create the output folder within the specified folder_path
    output_folder = os.path.join(folder_path, "1_AdaptedAndUnifiedImagesSize")

    # Check if the output folder already exists
    if os.path.exists(output_folder):
        # If it exists, delete the folder and its contents
        shutil.rmtree(output_folder)

    # Create a new output folder
    os.makedirs(output_folder)

    # Initialize the counter
    counter = 0

    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            # Construct the full path to the image
            image_path = os.path.join(folder_path, filename)

            # Load the image
            folder_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            print(folder_image.shape)

            # Check if the image is grayscale
            if len(folder_image.shape) == 2:
                folder_image = cv2.cvtColor(folder_image, cv2.COLOR_GRAY2BGR)

            # Get the dimensions of the folder image
            folder_height, folder_width, number_of_channels = folder_image.shape

            # Calculate the scaling factor to fit the image into the desired dimensions
            scaling_factor = min(desired_width / folder_width, desired_height / folder_height)

            # Resize the folder image while maintaining aspect ratio
            scaled_width = int(folder_width * scaling_factor)
            scaled_height = int(folder_height * scaling_factor)
            resized_folder_image = cv2.resize(folder_image, (scaled_width, scaled_height))

            # Calculate the position to center the folder image in the black image
            x_offset = (desired_width - scaled_width) // 2
            y_offset = (desired_height - scaled_height) // 2

            # Create a black image with the desired dimensions
            black_image = np.zeros((desired_height, desired_width, folder_image.shape[2]), dtype=np.uint8)

            # Place the resized folder image onto the black image
            black_image[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = resized_folder_image

            # Create a black border around the result image with the specified offsetInPixelNumber
            result_height, result_width, _ = black_image.shape
            bordered_image = np.zeros((result_height + 2 * offsetInPixelNumber, result_width + 2 * offsetInPixelNumber, folder_image.shape[2]), dtype=np.uint8)
            bordered_image[offsetInPixelNumber:offsetInPixelNumber + result_height, offsetInPixelNumber:offsetInPixelNumber + result_width] = black_image

            # Save the edited image to the output folder with the counter
            output_path = os.path.join(output_folder, f"{counter}_edited_{os.path.splitext(filename)[0]}{output_format}")
            print("Saving:", output_path)
            cv2.imwrite(output_path, bordered_image)

            # Increment the counter
            counter += 1

    print("Editing and saving complete.")


#All paths
#0_Datasets/Br35H/no
#0_Datasets/Br35H/yes
#0_Datasets/chest_xray/NORMAL
#0_Datasets/chest_xray/PNEUMONIA
#0_Datasets/melanoma_cancer_dataset/benign
#0_Datasets/melanoma_cancer_dataset/malignant
#0_Datasets/rsna-mammography-768-vl-perlabel/0
#0_Datasets/rsna-mammography-768-vl-perlabel/1
#NOTE:0_Datasets/rsna-mammography-768-vl-perlabel/0/1_AdaptedAndUnifiedImagesSize_ReducedSize

# Call the function with desired values and folder path
unifyImageSizeFromFolderAndSaveAsImage_blackBackground("0_Datasets/melanoma_cancer_dataset/malignant", 60, 60, 2)
