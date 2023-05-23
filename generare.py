# Stable diffusion model
from tensorflow import keras
import keras_cv

keras.mixed_precision.set_global_policy("mixed_float16")  #every layer will be float16 (only works for NVIDIA GPU's)

# Visualization
import matplotlib.pyplot as plt

# Save the image
from PIL import Image

# Create the model
model = keras_cv.models.StableDiffusion(img_height=512, img_width=512, jit_compile=True)

# Function to plot images
def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i+1)
        plt.imshow(images[i])
        plt.axis("off")

# Generate images
images = model.text_to_image(prompt="A bowl of vegetables that contains a brown carrot, highly detailed", 
                            batch_size=12,
                            num_steps=30,
                            seed=42)

# Plot the images
plot_images(images)

# Save the images
index = 0
for image in images:
    image = Image.fromarray(image)
    image = image.convert('RGB')
    image.save(f'output_keras/poza_{index}.bmp')
    index+=1
