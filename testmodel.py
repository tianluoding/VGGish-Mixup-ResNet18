from torch import nn, optim
import numpy as np
import h5py
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from model import ResNet_18, ResNetBlock


def read_train_data(path):
    data = h5py.File(path, 'r')
    return data['x_train'],data['y_train']

def read_test_data(path):
    data = h5py.File(path, 'r')
    return data['x_test'],data['y_test']

class DealtrainDataset(Dataset):
    def __init__(self, h5_path):
        x, y = read_train_data(h5_path)
        x = np.array(x)
        y = np.array(y)
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.len = y.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class DealtestDataset(Dataset):
    def __init__(self, h5_path):
        x, y = read_test_data(h5_path)
        x = np.array(x)
        y = np.array(y)
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.len = y.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

model = torch.load('mixup_resnet_model.pkl')

criterion = nn.BCEWithLogitsLoss()

test_set = DealtestDataset('test_set.h5')
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)
model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    inputs, labels = data
    if torch.cuda.is_available():
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
    else:
        inputs, labels = Variable(inputs), Variable(labels)
    labels = labels.squeeze(1)
    inputs = inputs.float()
    out = model(inputs)
    #print(out)
    loss = criterion(out, labels.float())
    eval_loss += loss.item()*labels.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == labels).sum()
    eval_acc += num_correct.item()

print(eval_acc)
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss/(len(test_set)), eval_acc/(len(test_set))))
