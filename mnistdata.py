# SPDX: gps-v3-or-later
"""Handles the MNIST handwritten digits as an in-memory  dataset."""

import matplotlib.pyplot as plot  # pylint: disable=import-error
import torch  # pylint: disable=import-error
import pandas  # pylint: disable=import-error

from idxreader import IDXFile


class MNISTDataset(torch.utils.data.Dataset):
    """Allows access to MNIST database as an iterator."""

    def __init__(self, image_filename, labels_filename):
        """Initialize dataset given image and label filename."""
        with open(image_filename, "rb") as idxfile:
            idxdata = IDXFile(idxfile)
            mask = (1 << idxdata.word_size) - 1
            images = [[(x/mask)*0.9+0.05 for x in img] for img in idxdata]
        with open(labels_filename, "rb") as idxfile:
            labels = [lbl[0] for lbl in IDXFile(idxfile)]
        self.data = list(zip(labels, images))

    def __len__(self):
        """Return the number of itens in the dataset"""
        return len(self.data)

    def __getitem__(self, index):
        """Retrun the label, image and output vector for the given index."""
        # label
        label, img = self.data[index]
        # target
        target = torch.zeros(10)
        target[label] = 1.0
        # image
        img = torch.FloatTensor(img)
        return label, img, target

    def plot_image(self, index):
        """Plot the image for the given index."""
        label, img, _ = self[index]
        img = pandas.DataFrame(img).values.reshape(28, 28)
        plot.title(f"label = {label:d}")
        plot.imshow(img, interpolation='none', cmap='Blues')
        plot.show()
