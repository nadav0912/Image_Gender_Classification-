# ----------------- Import Libraries ----------------- #
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import plot_image_size_distribution, pad_to_100x128, plot_images_examples

# Check if torch is installed
print(f"Using torch version {torch.__version__}")

# Set device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} deivce")


# ----------------- Hyperparameters ----------------- #
CLASS_NAMES = ["male", "female"]

BATCH_SIZE = 32



# ----------------- Dataset Information ----------------- #
"""
Dataset Explanation:
This dataset contains face images labeled as male or female. 
It includes 1747 male and 1747 female images for training, and 100 male + 100 female images for both testing and validation. 
The training and test are balanced, each containing roughly 50% male and 50% female images.
All images were automatically cropped to show only the face using a tool called MTCNN.
The dataset is designed to train a model to predict gender from a face image.

Link: https://www.kaggle.com/datasets/gpiosenka/gender-classification-from-an-image

Note: You need to download the folder named gender_rev2 and then rename its folder to data.
"""

# Check images sizes
plot_image_size_distribution('data/train')


"""
Based on the results from plot_image_size_distribution,
this transform first resizes each image to fit within a 100 by 128 pixel box while preserving its original aspect ratio without stretching or distortion.
If the image is smaller than these dimensions, it remains unchanged. 
Then, it pads the resized image with zeros (black borders) evenly on all sides to ensure the final image size is exactly 100 pixels wide and 128 pixels high. 
This approach standardizes all images to the same size required by the model while maintaining their original proportions.
"""

transform = transforms.Compose([
    transforms.Lambda(pad_to_100x128),
    transforms.ToTensor()
])

# Load datasets
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
val_dataset   = datasets.ImageFolder(root='data/valid', transform=transform)
test_dataset  = datasets.ImageFolder(root='data/test', transform=transform)

plot_images_examples(datset=train_dataset, class_names=CLASS_NAMES)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"\nLength of Train dataloader: {len(train_loader)}, batches of {BATCH_SIZE}.")
print(f"Length of Test dataloader: {len(test_loader)}, batches of {BATCH_SIZE}.")
print("The dataset is balanced with approximately 50% of images from each class (male and female).")






