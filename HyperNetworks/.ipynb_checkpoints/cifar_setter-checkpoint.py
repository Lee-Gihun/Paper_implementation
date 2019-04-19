import torch
import torch.nn as nn
from torchvision import datasets, models
import torchvision.transforms as transforms


def cifar_10_setter(batch_size=128, valid_size=5000):
    # Data augmentation and normalization for training
    # Just normalization for validation
    mean = [0.4914, 0.4822, 0.4465]
    stdv = [0.2023, 0.1994, 0.2010]
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    # Datasets
    cifar10_train_set = datasets.CIFAR10('./data/cifar-10', train=True, transform=train_transforms, download=True)
    cifar10_test_set = datasets.CIFAR10('./data/cifar-10', train=False, transform=test_transforms, download=True)
    
    batch_size = batch_size
    valid_size = valid_size


    train_indices = []
    valid_indices = []

    for elem in range(len(cifar10_train_set)):
        if elem % 10 == 0:
            valid_indices.append(elem)
        else:
            train_indices.append(elem)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(cifar10_train_set, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(cifar10_train_set, batch_size=batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(cifar10_test_set, batch_size=batch_size)

    dataloaders = {'train' : train_loader,
                   'valid' : valid_loader,
                   'test' : test_loader,}

    dataset_sizes = {'train': len(cifar10_train_set)-valid_size, 'valid' : valid_size, 'test' : len(cifar10_test_set)}
    
    return dataloaders, dataset_sizes


def cifar_100_setter(batch_size=128, valid_size=5000):
    # Data augmentation and normalization for training
    # Just normalization for validation
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    # Datasets
    cifar100_train_set = datasets.CIFAR100('./data/cifar-100', train=True, transform=train_transforms, download=True)
    cifar100_test_set = datasets.CIFAR100('./data/cifar-100', train=False, transform=test_transforms, download=True)
    
    batch_size = batch_size
    valid_size = valid_size


    train_indices = []
    valid_indices = []

    for elem in range(len(cifar100_train_set)):
        if elem % 10 == 0:
            valid_indices.append(elem)
        else:
            train_indices.append(elem)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(cifar100_train_set, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(cifar100_train_set, batch_size=batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(cifar100_test_set, batch_size=batch_size)

    dataloaders = {'train' : train_loader,
                   'valid' : valid_loader,
                   'test' : test_loader,}

    dataset_sizes = {'train': len(cifar100_train_set)-valid_size, 'valid' : valid_size, 'test' : len(cifar100_test_set)}
    
    return dataloaders, dataset_sizes
