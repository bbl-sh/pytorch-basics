import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from torchvision import transforms

# Defining class of the dataset 

class MyDataset(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_names = df['File Name']
        self.y = df['Class Label']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]

# loading the datatset class and assigning it to train dataset variable

custom_transform = transforms.Compose([
                                       transforms.ToTensor()
                                      ])

train_dataset = MyDataset(csv_path='mnist_train.csv',
                          img_dir='mnist_train',
                          transform=custom_transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          drop_last=True,
                          shuffle=True,
                          num_workers=0)
valid_dataset = MyDataset(csv_path='mnist_valid.csv',
                          img_dir='mnist_valid',
                          transform=custom_transform)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=100,
                          shuffle=False,
                          num_workers=0)


# loading the datatset class and assigning it to test dataset variable


test_dataset = MyDataset(csv_path='mnist_test.csv',
                         img_dir='mnist_test',
                         transform=custom_transform)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=100,
                         shuffle=False,
                         num_workers=0)



# Iterating through the dataset of the custom dataloader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# in case of google colab use this 
#device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 2
for epoch in range(num_epochs):

    for batch_idx, (x, y) in enumerate(train_loader):
        
        print('Epoch:', epoch+1, end='')
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])
        
        x = x.to(device)
        y = y.to(device)