#data
import torch
import torchvision
import torchvision.transforms as T
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
df = torch.load("train_dataset.pt")
print("dataset loaded")
#model
model_w = torch.hub.load("facebookresearch/swag", model="vit_h14_in1k")
model_w.head.layers.head = nn.Linear(in_features=1280, out_features=5, bias=True)
print(model_w)
#optimizer
lr = 0.001
for name,param in model_w.named_parameters():
    if 'head' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
optimizer_ft = optim.SGD(model_w.parameters(), lr=lr, momentum=0.9)
print("optimizer done")
#loss
criterion = nn.CrossEntropyLoss()
#Train
train_loader= torch.utils.data.DataLoader(df, batch_size=12, shuffle=True)
for epoch in range(1):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    
    model_w.train()
    for inputs, targets in tqdm(train_loader):
        # forward + backward + optimize
        optimizer_ft.zero_grad()
        outputs = model_w(inputs)
        
        targets = targets.squeeze().long()
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer_ft.step()
torch.save(model_w.state_dict(), "model_w1.pt")
