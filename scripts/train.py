import os
import glob
import torch
import numpy as np
from model import CaptchaNet
from dataset import split_data, ImageFilelist
from torchvision import transforms

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    iter = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        iter.append(result)
    return iter

# Dataset preprocessing
data_folder = '../../dataset'
images = sorted(glob.glob(os.path.join(data_folder,'*.png')))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
chars = set(char for label in labels for char in label)

char_dict = {x: i for i, x in enumerate(chars)}
num_dict = {i: x for i, x in enumerate(chars)}

x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torch.utils.data.DataLoader(ImageFilelist(x_train, y_train, char_dict, transform),
                                            batch_size=32, shuffle=True,
                                            num_workers=4, pin_memory=True)
valid_dataset = torch.utils.data.DataLoader(ImageFilelist(x_valid, y_valid, transform),
                                            batch_size=4, shuffle=True,
                                            num_workers=4, pin_memory=True)

# Model
model = CaptchaNet()

# Copying dataloader and model from cpu to gpu
device = get_default_device()
train_dl = DeviceDataLoader(train_dataset, device)
val_dl = DeviceDataLoader(valid_dataset, device)
to_device(model, device)

model = to_device(CaptchaNet(), device)

# optimization
num_epochs = 200
opt_func = torch.optim.Adam
lr = 0.001
iter = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

# Saving the model
torch.save(model, 'captcha_model.pth')