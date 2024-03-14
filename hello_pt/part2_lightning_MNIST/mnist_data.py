

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Resize
import matplotlib.pyplot as plt

from utils import MNIST_CST


class MNISTdataset(Dataset):
    def __init__(self, path_to_mnist, img_hw, train=True):
        MNISThw = MNIST_CST().MNISThw
        self.data = datasets.MNIST(root=path_to_mnist, train=train, download=True)
        print(f"\nloading MNIST {'train' if train else 'test'} set of size {len(self)}")
        img_transforms = [ToTensor()]
        if img_hw!=MNISThw:
            print(f"rescaling MNIST images from size {MNISThw} to {img_hw}")
            img_transforms.append(Resize((img_hw, img_hw), antialias=True))
        self.img_transforms = Compose(img_transforms)
        # TODO: could add e.g. data augmentations for the training
        # TODO: could automatically check for counts per digit
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image = self.img_transforms(self.data[idx][0])
        label = self.data[idx][1]
        return image, label

if __name__ == "__main__":
    img_hw = 32  # e.g. we set a %8 size for input data, original MNIST data is (28, 28)

    train_data = MNISTdataset(MNIST_CST().path_to_mnist, img_hw, train=True)  # MNIST train set of size 60000
    test_data = MNISTdataset(MNIST_CST().path_to_mnist, img_hw, train=False)  # MNIST test set of size 10000
    train_sample, train_label = train_data[np.random.choice(len(train_data))]
    test_sample, test_label = test_data[np.random.choice(len(test_data))]

    print(f"train sample of shape {train_sample.shape}, "
          f"values in range {[torch.min(train_sample), torch.max(train_sample)]} and label {train_label}")
    print(f"test sample of shape {test_sample.shape}, "
          f"values in range {[torch.min(test_sample), torch.max(test_sample)]} and label {test_label}")
    # train sample of shape torch.Size([1, 32, 32]), values in range [tensor(0.), tensor(0.9978)] and label 0
    # --> minibatches will then be as (batch, 1, 32, 32) and (batch)

    plt.figure()
    plt.title(f"training sample with label {train_label}")
    plt.imshow(train_sample[0], cmap="gray")
    plt.savefig(os.path.join(Path(__file__).parent, "artefacts", "MNIST_train_img.jpg"))

