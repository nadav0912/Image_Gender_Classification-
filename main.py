# ----------------- Import Libraries ----------------- #
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm  # For progress bar

from utils import plot_image_size_distribution, pad_to_100x128, plot_images_examples

# Check if torch is installed
print(f"Using torch version {torch.__version__}")

# Set device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} deivce")


# ----------------- Hyperparameters ----------------- #
CLASS_NAMES = ["male", "female"]

RANDOM_SEED = 142  # Seed for reproducibility

BATCH_SIZE = 32
EPOCHS = 10
HIDDEN_UNITS = 64  # Number of filters in the convolutional layers
KERNEL_SIZE = 3  # Size of the convolutional filters (3x3)
STEP_SIZE = 1  # Stride of the convolutional filters (1 pixel step)
POOL_KERNEL_SIZE = 2  # Size of the max pooling window (2x2)

LERRNING_RATE = 0.001  # Learning rate for the optimizer

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

print(f"Label map of datset: {train_dataset.class_to_idx}")

plot_images_examples(datset=train_dataset, class_names=CLASS_NAMES)


# Create DataLoaders
torch.manual_seed(RANDOM_SEED)  # Set random seed for reproducibility
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nLength of Train dataloader: {len(train_loader)}, batches of {BATCH_SIZE}.")
print(f"Length of Test dataloader: {len(test_loader)}, batches of {BATCH_SIZE}.")
print("The dataset is balanced with approximately 50% of images from each class (male and female).")


# ----------------- Building the Model ----------------- #
class GenderClassifier(nn.Module):
    """
    Simple CNN model with 3 convolutional layers for feature extraction 
    and 1 fully connected layer for binary classification (e.g., male vs female).
    """
    def __init__(self, hidden_units: int, kernel_size: int, step_size: int, pool_kernel_size: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,  # Number of input channels, 3 for RGB images.
                      out_channels=hidden_units,  # Number of filters (feature maps) the layer will produce. ("num neurons")
                      kernel_size=kernel_size,  # Size of each filter (e.g., 3 means 3×3 kernels).
                      stride=step_size,  # How many pixels the filter moves at each step (1 = move one pixel at a time)
                      padding=0   # No padding here since images were already padded during dataset preparation to fit 100×128.
            ),  
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=pool_kernel_size)  # Window size, Take only the max vaue from each window step    
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units, 
                      kernel_size=kernel_size, 
                      stride=step_size,  
                      padding=3  
            ),  
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=pool_kernel_size)    
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units, 
                      kernel_size=kernel_size, 
                      stride=step_size,  
                      padding=3  
            ),  
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=pool_kernel_size)    
        )

        self.fully_connected_1 = nn.Sequential(
            nn.Flatten(),  # Flatten the output from the convolutional layers to feed into the fully connected layer
            nn.Linear(in_features=15*18*hidden_units,  # Number of input features, calculated based on the output size of conv layers
                      out_features=hidden_units // 2),  
            nn.ReLU()  
        )

        self.fully_connected_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units // 2, 
                      out_features=1  # Binary classification
            )
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        #print(f"Shape after conv layers: {x.shape}")
        x = self.fully_connected_1(x)
        x = self.fully_connected_2(x)
        return x
    

# Create model instance
model = GenderClassifier(hidden_units=HIDDEN_UNITS,
                        kernel_size=KERNEL_SIZE,
                        step_size=STEP_SIZE,
                        pool_kernel_size=POOL_KERNEL_SIZE).to(device)


# Test the model with a random input, and check shape needed for the first fully connected layer
model.eval() 
with torch.no_grad():
    sample_input = torch.randn(1, 3, 100, 128).to(device)  # Random input tensor with shape (batch_size, channels, height, width)
    sample_logits = model(sample_input)  # Forward pass
    print(f"Sample logits shape: {sample_logits.shape}")  # Should be (1, 1) for binary classification

# We get print from the forword func of torch.Size([1, 64, 15, 18]) wich means each image is reduced to 15x18 feature maps after the convolutional layers.


# Binary Cross Entropy with Logits Loss For binary classification
loss_fn = nn.BCEWithLogitsLoss()

# Function to calculate accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,  y_pred).sum().item()
    acc = correct / len(y_pred) * 100
    return acc

# Adam optimizer for training the model
optimizer = torch.optim.Adam(params=model.parameters(), lr=LERRNING_RATE)


# Training loop
def train():
    model.train()
    total_loss, total_acc = 0, 0

    for batch, labels in tqdm(train_loader):
        print(batch.shape, labels.shape)  # Debugging: Check batch and labels shapes
        # Move data to device
        labels = labels.to(device)
        batch = batch.to(device)

        # Forward pass
        logits = model(batch)
        print(f"Logits shape: {logits.shape}")  # Should be [batch_size, 1] for binary classification
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        y_pred = torch.round(torch.sigmoid(logits))
        total_acc += accuracy_fn(y_true=labels, y_pred=y_pred)

    return total_loss / len(train_loader), total_acc / len(train_loader)


def evaluate():
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for batch, labels in tqdm(val_loader):
            # Move data to device
            labels = labels.to(device)
            batch = batch.to(device)

            # Forward pass
            logits = model(batch)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # Calculate accuracy
            y_pred = torch.round(torch.sigmoid(logits))
            total_acc += accuracy_fn(y_true=labels, y_pred=y_pred)

    return total_loss / len(val_loader), total_acc / len(val_loader)

train_loss, train_acc = train()
print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")





