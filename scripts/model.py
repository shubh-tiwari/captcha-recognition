import torch
import torch.nn as nn
from train import num_dict

class CaptchaBase(nn.Module):
    def training_step(self, batch):
        ctc_loss = nn.CTCLoss()
        images, y_true = batch
        y_pred = self(images)                  # Generate predictions
        batch_size = list(y_pred.size())[0]
        input_size = list(y_pred.size())[1]
        label_size = list(y_pred.size())[1]
        input_length = torch.full(size=(batch_size,), fill_value=input_size, dtype="int64")
        target_length = torch.full(size=(batch_size,), fill_value=50, dtype="int64")
        loss = ctc_loss(y_true, y_pred, input_length, target_length)
        return loss
    
    def validation_step(self, batch):
        ctc_loss = nn.CTCLoss()
        images, y_true = batch
        y_pred = self(images)                    # Generate predictions
        batch_size = list(y_pred.size())[0]
        input_size = list(y_pred.size())[1]
        label_size = list(y_pred.size())[1]
        input_length = torch.full(size=(batch_size,), fill_value=input_size, dtype="int64")
        target_length = torch.full(size=(batch_size,), fill_value=50, dtype="int64")
        loss = ctc_loss(y_true, y_pred, input_length, target_length)
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

class CaptchaNet(CaptchaBase):
    def __init__(self):
        super().__init__()
        self.cnn_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.rnn_net = nn.LSTM(input_size=524,hidden_size=64,num_layers=4,bidirectional=True,dropout=0.2)
        self.linear1 = nn.Linear(128*6,524)
        self.linear2 = nn.Linear(128,len(num_dict)+1)
        self.softmax = nn.Softmax(dim=2)
    
    def reshape(self, y):
        return torch.reshape(y, (y.shape[0], 25, 6*128))
        
    def forward(self, xb):
        y = self.cnn_net(xb)
        y = self.reshape(y)
        y = self.linear1(y)
        y = self.rnn_net(y)
        y = self.linear2(y[0])
        y = self.softmax(y)
        return y