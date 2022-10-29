from model import ResNet_18, DealtrainDataset, DealtestDataset
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from torch.autograd import Variable
from mixup import mixup_data, mixup_criterion


# hyperparameters
batch_size = 64
learning_rate = 1e-3
epoches = 300


train_set = DealtrainDataset('train_set.h5')
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)


model = ResNet_18()
if torch.cuda.is_available():
    print()
    model = model.cuda()

criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()
train_loss = []

for epoch in range(epoches):
    train_acc = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data

        if torch.cuda.is_available():
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        #inputs = Variable(torch.unsqueeze(inputs, dim=1).float(), requires_grad=False)

        inputs = inputs.float()
        labels = labels.float()

        '''
        optimizer.zero_grad()
        outputs = model(inputs)
        #print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        '''

        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, 0.5)
        optimizer.zero_grad()
        inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
        outputs = model(inputs)

        loss_func = mixup_criterion(targets_a, targets_b, lam)
        loss = loss_func(criterion, outputs)
        loss.backward()
        optimizer.step()


        logsoftmax = nn.LogSoftmax(dim=1)
        out = logsoftmax(outputs)
        _, pred = torch.max(out, 1)
        _,label_index = torch.max(labels, 1)
        num_correct = (pred == label_index).sum()
        train_acc += num_correct.item()
        print_loss = loss.data
        train_loss.append((i, print_loss))
    #if i%5 == 4:

    print('Epoch[{}/{}], loss: {:.5f}, train_acc {:.5f}'.format(epoch+1, epoches, print_loss, train_acc/len(train_set)))


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
    # inputs = Variable(torch.unsqueeze(inputs, dim=1).float(), requires_grad=False)
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


if(eval_acc/(len(test_set)) > 0.7):
   # torch.save(model, 'resnet_model.pkl')
    torch.save(model, 'mixup_resnet_model.pkl')