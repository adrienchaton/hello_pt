# examples to show different ways to handle loading and batching data during training


from functools import partial
import numpy as np
from sklearn import datasets
import torch
from torch.utils.data import Dataset, DataLoader


iris_dataset = datasets.load_iris()
batch_size = 8
num_workers = 2

########################################################################################################################
## 0) load data as torch.tensor and cast to usual dtypes for classification task

x_features = torch.from_numpy(iris_dataset["data"]).float()
x_min, x_max = torch.min(x_features), torch.max(x_features)
print(f"\nloaded IRIS features with shape {x_features.shape} and type {x_features.dtype}, in range {[x_min, x_max]}")
# loaded IRIS features with shape torch.Size([150, 4]) and type torch.float32, in range [tensor(0.1000), tensor(7.9000)]
# TODO: we may want to e.g. scale the input values in [0, 1]

y_labels = torch.from_numpy(iris_dataset["target"]).long()
y_classes, y_counts = np.unique(y_labels.numpy(), return_counts=True)
print(f"\nloaded IRIS labels with shape {y_labels.shape} and type {y_labels.dtype}")
print(f"classes and counts are {dict(zip(y_classes, y_counts))}")
# loaded IRIS labels with shape torch.Size([150]) and type torch.int64
# classes and counts are {0: 50, 1: 50, 2: 50}
# TODO: the "ideal" dataset, but watch out e.g. for class imbalance
#   all usual data explorations, statistical checks and pre-processing shall be used for "real-world" data

########################################################################################################################
## 1) the "simplest" dataloader, since our inputs are already stacked with fixed dimension

dataset1 = torch.utils.data.TensorDataset(x_features/x_max, y_labels)
dataloader1 = DataLoader(dataset1, batch_size=batch_size, num_workers=num_workers,
                         shuffle=True, drop_last=True, pin_memory=True)
# the default collate fn. just does stacking the random minibatch
x1, y1 = next(iter(dataloader1))
print(f"\ndataloader1: sampling minibatch with shape {x1.shape} {y1.shape}")
# dataloader1: sampling minibatch with shape torch.Size([8, 4]) torch.Size([8])

########################################################################################################################
## 2) a dataloader with custom collate fn. applied on-the-fly

def some_collate_fn(data, scale_value=1.):
    x, y = zip(*data)
    minibatch = {"x": torch.stack(x)/scale_value, "y": torch.stack(y)}
    return minibatch
# TODO: can be used to handle inputs of varying shapes, e.g. padding to the largest element when needed

dataset2 = torch.utils.data.TensorDataset(x_features, y_labels)  # here we will scale values on-the-fly
dataloader2 = DataLoader(dataset2, batch_size=batch_size, num_workers=num_workers,
                         shuffle=True, drop_last=True, pin_memory=True,
                         collate_fn=partial(some_collate_fn, scale_value=x_max))
minibatch2 = next(iter(dataloader2))
print(f"\ndataloader2: sampling minibatch with shape {minibatch2['x'].shape} {minibatch2['y'].shape}")
# dataloader2: sampling minibatch with shape torch.Size([8, 4]) torch.Size([8])

########################################################################################################################
## 3) a custom dataset to handle loading the data and applying transformations

class IrisDataset(Dataset):
    def __init__(self, x, y, scale_value=1.):
        self.x = x
        self.y = y
        self.scale_value = scale_value
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx]/self.scale_value, self.y[idx]
# TODO: can be used to handle input of various formats
#   e.g. a pandas dataframe, a long text which isn't chunked, streaming data from local files or from the web

dataset3 = IrisDataset(x_features, y_labels, scale_value=x_max)
dataloader3 = DataLoader(dataset3, batch_size=batch_size, num_workers=num_workers,
                         shuffle=True, drop_last=True, pin_memory=True)
x3, y3 = next(iter(dataloader3))
print(f"\ndataloader3: sampling minibatch with shape {x3.shape} {y3.shape}")
# dataloader3: sampling minibatch with shape torch.Size([8, 4]) torch.Size([8])
