import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ETL952Dataset(Dataset):
    def __init__(self, root_dir, folder_name, transform=None):
        self.root_dir = os.path.join(root_dir, 'data', 'etl_952_singlechar_size_64', 'etl_952_singlechar_size_64', folder_name)
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),  # Resize to match SqueezeNet input size
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        
        self.classes = [str(i) for i in range(952)]
        self.data = []  # Will store image data
        self.labels = []  # Will store labels

        # Load all images into memory
        for class_idx, class_name in enumerate(self.classes):
            print("start to load", class_name)
            class_path = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', )):
                    img_path = os.path.join(class_path, img_name)
                    image = Image.open(img_path).convert('RGB')
                    image = image.resize((32, 32))  # Resize to 32x32
                    image_array = np.array(image)
                    self.data.append(image_array)
                    self.labels.append(class_idx)

        # Convert to numpy arrays
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

class ETL952Train(ETL952Dataset):
    def __init__(self, root_dir, transform=None):
        super(ETL952Train, self).__init__(root_dir, folder_name="952_train", transform=transform)
class ETL952Test(ETL952Dataset):
    def __init__(self, root_dir, transform=None):
        super(ETL952Test, self).__init__(root_dir, folder_name="952_test", transform=transform)
class ETL952Val(ETL952Dataset):
    def __init__(self, root_dir, transform=None):
        super(ETL952Val, self).__init__(root_dir, folder_name="952_val", transform=transform)

import pandas as pd
class ETL952Labels():
    
    def __init__(self,path="pytorch-cifar100\data\etl_952_singlechar_size_64\etl_952_singlechar_size_64\952_labels.txt"):
        self.path=path
        self.data=pd.read_csv(path,sep="\s+",header=0,names=['label', 'character', 'JISx0208', 'UTF8', 'Cangjie'])


from torch.utils.data import DataLoader
# train_set = ETL952Train(root_dir="pytorch-cifar100", transform=transforms.ToTensor())

train_set = ETL952Train(root_dir="", transform=transforms.ToTensor())
train_loader = DataLoader(train_set, batch_size=64, shuffle=True,num_workers=8)
# train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# print("total number of train data: ", len(train_set))

for images, labels in train_loader:
    print("images shape: ", images.shape)
    print("labels shape: ", labels.shape)
    break