import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def show_image(image, grayscale=True, ax=None, title=""):
    if ax is None:
        plt.figure()
    plt.axis("off")

    if len(image.shape) == 2 or grayscale:
        if len(image.shape) == 3:
            image = np.sum(np.abs(image), axis=2)

        vmax = np.percentile(image, 99)
        vmin = np.min(image)

        plt.imshow(image, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        plt.title(title)
    else:
        image = image + 127.5
        image = image.astype("uint8")

        plt.imshow(image)
        plt.title(title)


def load_image(file_path):
    im = Image.open(file_path)  # Use PIL.Image
    im = np.asarray(im)

    return im - 127.5
