import PIL
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

class ImageFilelist(data.Dataset):
    """Custom data generator from list of paths"""
    def __init__(self, image_list, label_list, char_dict, transform=None):
        self.imlist = image_list
        self.imglabels = label_list
        self.transform = transform
        self.char_dict = char_dict

    def __getitem__(self, index):
        imgpath = self.imlist[index]
        label = self.imglabels[index]
        img = PIL.Image.open(imgpath).convert("L")
        label_int = torch.tensor(list(map(self.char_dict.get, list(label))))
        if self.transform:
            img = self.transform(img)   
        return img, label_int

    def __len__(self):
        return len(self.imlist)

def split_data(images, labels, train_size=0.9, shuffle=True):
    """Funtion to split train and test"""
    size = len(images)
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    train_samples = int(size * train_size)
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid