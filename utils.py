import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as F

def plot_image_size_distribution(root_dir):
    widths = []
    heights = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg')):
                path = os.path.join(subdir, file)
                try:
                    img = Image.open(path)
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                except Exception as e:
                    print(f"Error loading {path}: {e}")


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=30, color='skyblue')
    plt.title('Image Width Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Number of images')

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=30, color='lightgreen')
    plt.title('Image Height Distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Number of images')

    plt.tight_layout()
    plt.show()



def pad_to_100x128(img):
    # Resize to fit inside 100x128, keep aspect ratio
    img.thumbnail((100, 128))  # modifies img in-place

    # Get new size
    w, h = img.size

    # Compute padding amounts (left, top, right, bottom)
    pad_left = (100 - w) // 2
    pad_top = (128 - h) // 2
    pad_right = 100 - w - pad_left
    pad_bottom = 128 - h - pad_top

    # Pad the image
    return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)



def plot_images_examples(datset, class_names):
    torch.manual_seed(142)
    fig = plt.figure(figsize=(9,9))
    rows, cols = 4, 4

    for i in range(1, rows*cols + 1):
        random_idx = torch.randint(0, len(datset), size=[1]).item()
        img, label = datset[random_idx]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        plt.title(class_names[label])
        plt.axis(False)

    plt.show()
