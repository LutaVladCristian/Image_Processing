import cv2
import numpy as np
import os


def process(image_name, file_name):

    # Read the image
    img = cv2.imread(image_name)

    # Parse the text file
    f = open(file_name,'r')

    try:
        os.makedirs(f'output_processed')
    except:
        pass

    lines = f.readlines()

    index = 0

    for line in lines:

        center_x = float(line.split(' ')[1]) * img.shape[0]
        center_y = float(line.split(' ')[2]) * img.shape[1]
        width = float(line.split(' ')[3]) * img.shape[0]
        height = float(line.split(' ')[4]) * img.shape[1]

        f.close()

        # Crop the image
        img_cropped = img[int(center_y - height/2):int(center_y + height/2), int(center_x - width/2):int(center_x + width/2)]
        cv2.imwrite(f'output_processed/cropped_{index}.bmp',img_cropped)

        # Change the color space
        img_HLS = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HLS)
        cv2.imwrite(f'output_processed/img_HLS_{index}.bmp',img_HLS)

        # Create a mask 
        lower = np.array([0,0,0])
        higher = np.array([140,140,140])

        mask = cv2.inRange(img_cropped, lower, higher)
        mask = cv2.bitwise_not(mask)
        cv2.imwrite(f'output_processed/mask_{index}.bmp',mask)

        # Apply the mask
        img_cropped_masked = cv2.bitwise_and(img_cropped, img_cropped, mask=mask)
        cv2.imwrite(f'output_processed/img_masked_{index}.bmp',img_cropped_masked)

        # Make the background white
        for i in range(img_cropped.shape[0]):
            for j in range(img_cropped.shape[1]):
                if img_cropped_masked[i][j][0] == 0 and img_cropped_masked[i][j][1] == 0 and img_cropped_masked[i][j][2] == 0:
                    img_cropped_masked[i][j][0] = 255
                    img_cropped_masked[i][j][1] = 255
                    img_cropped_masked[i][j][2] = 255

        cv2.imwrite(f'output_processed/img_transparent_{index}.bmp',img_cropped_masked)

        index += 1

# Test the function
process('input_process/poza.bmp', 'input_process/poza.txt')
