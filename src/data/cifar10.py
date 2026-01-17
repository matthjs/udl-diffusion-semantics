from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np


def dses_to_dls(train_ds, val_ds, test_ds, batch_size, n_cpus):
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=n_cpus,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        persistent_workers=True,
        num_workers=n_cpus,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
        num_workers=n_cpus,
    )
    return train_dl, val_dl, test_dl


def get_dls(data_dir, batch_size, n_cpus, val_ratio, seed):
    cifar10_mean = [0.491, 0.482, 0.447]
    cifar10_std = [0.247, 0.244, 0.262]

    # Training augmentation: standard CIFAR-10 crop + flip
    train_transformer = T.Compose(
        [
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=cifar10_mean, std=cifar10_std),
        ],
    )
    
    val_transformer = T.Compose(
        [T.ToTensor(), T.Normalize(mean=cifar10_mean, std=cifar10_std)],
    )
    
    train_val_ds = CIFAR10(root=data_dir, train=True, download=True, transform=train_transformer)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_ds)),
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=[sample[1] for sample in train_val_ds],
    )
    train_ds = Subset(train_val_ds, train_idx)
    val_ds_raw = CIFAR10(root=data_dir, train=True, download=True, transform=val_transformer)
    val_ds = Subset(val_ds_raw, val_idx)
    test_ds = CIFAR10(root=data_dir, train=False, download=True, transform=val_transformer)
    return dses_to_dls(
        train_ds, val_ds, test_ds, batch_size=batch_size, n_cpus=n_cpus,
    )
