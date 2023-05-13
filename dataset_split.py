#data
import torch
import torchvision
import torchvision.transforms as T
#import os
#import torch.nn as nn
#import torch.optim as optim
#from tqdm import tqdm
trans = T.Compose([T.Resize(518),T.CenterCrop(518),T.ToTensor(),
                 T.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
total_dataset = torchvision.datasets.ImageFolder("./Data", transform = trans)
generator = torch.Generator().manual_seed(42)
train_dataset,test_dataset,nullds=torch.utils.data.random_split(total_dataset, [0.04, 0.01,0.95], generator=generator)
torch.save(train_dataset,"train_dataset.pt")
torch.save(test_dataset,"test_dataset.pt")
print("dataset made")
