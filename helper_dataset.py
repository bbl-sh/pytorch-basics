import torch
from torch.utils.data import sampler
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms

def helper_dataloader_mnist( train_transform = None, test_transform = None, num_workers =0, batch_size=8):
  if train_transform is None:
        train_transform = transforms.ToTensor()

  if test_transform is None:
        test_transform = transforms.ToTensor()
  
  train_dataset = datasets.MNIST(
    root= 'data',
    train = True,
    transform = train_transform,
    download = True
  )
  valid_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=test_transform
  )

  test_dataset = datasets.MNIST(root='data',
    train=False,
    transform=test_transform
  )

  train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)

  test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=False)
  validation_fraction = 0.1
  num = int(validation_fraction * 60000)
  train_indices = torch.arange(0, 60000 - num)
  valid_indices = torch.arange(60000 - num, 60000)
  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(valid_indices)

  valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            sampler=valid_sampler)

  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=True,
                            sampler=train_sampler)
  
  return train_loader, test_loader, valid_loader