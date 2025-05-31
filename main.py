import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
Dataset Explanation:
This dataset contains face images labeled as male or female. 
It includes 1747 male and 1747 female images for training, and 100 male + 100 female images for both testing and validation. 
All images were automatically cropped to show only the face using a tool called MTCNN.
The dataset is designed to train a model to predict gender from a face image.

Link: https://www.kaggle.com/datasets/gpiosenka/gender-classification-from-an-image

Note: You need to download the folder named gender_rev2 and then rename its folder to data.
"""

