import torch
import numpy as np
import torchvision.transforms as T
import torchvision
import torch.nn as nn
from sklearn.metrics import accuracy_score

df = torch.load("test_dataset.pt")
test_loader= torch.utils.data.DataLoader(df, batch_size=32, shuffle=True)

model_w = torch.hub.load("facebookresearch/swag", model="vit_h14_in1k")
model_w.head.layers.head = nn.Linear(in_features=1280, out_features=5, bias=True)
model_w.load_state_dict(torch.load("model_w1.pt"))

def test_w():
    model_w.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model_w(inputs)

            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        print(y_true)
        print(y_score)
        print("Accuracy: "+str(100*accuracy_score(y_true, np.argmax(y_score, axis=-1)))+"%")

test_w()
