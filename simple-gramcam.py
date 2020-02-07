# Author: Vinicius Arruda
# viniciusarruda.github.io
# Source code modified from: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
from MyDataset import MyDataset

# In the current directory create a folder called 'data' and inside it another folder called 'Elephant' 
# and inside, paste the image you want to check the GradCAM

torch.manual_seed(1)

# use the ImageNet transformation
dataset = MyDataset(
    txt_path=r"D:\\Ethan\\dlfnirs\\FFT_GRAY\\test.txt",
    transform=transforms.Compose([transforms.ToTensor()]), target_transform=None)

# define the dataloader to load that single image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        
        self.fc1 = nn.Linear(1*4480*3360, 2*16*16, bias=False)
        self.fc2 = nn.Linear(2*16*16, 2*2*2, bias=False)
        self.fc3 = nn.Linear(2*2*2, 2, bias=False)
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        
        x = x.view(1, 1*4480*3360)    
        x = self.fc1(x) 
        x = self.fc2(x)
        x = x.view(1, 2, 2, 2)    
        return x

    def get_activations_before(self, x):
        
        x = x.view(1, 1*4480*3360)    
        x = self.fc1(x)
        x = x.view(1, 2, 16, 16)     
        return x
        
    def forward(self, x):

        x = x.view(1, 1*4480*3360)    
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(1, 2, 2, 2)    

        x.register_hook(self.activations_hook)

        x = x.view(1, 2*2*2)    
        x = self.fc3(x)

        return x


sn = SimpleNet()

#####
# Training procedure to be placed here
#####

sn.eval()
img, _, _ = next(iter(dataloader))
pred = sn(img)
pred_idx = pred.argmax(dim=1)
pred[:, pred_idx].backward()
gradients = sn.get_activations_gradient()

# # pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# # get the activations of the last convolutional layer
activations = sn.get_activations(img).detach()

# weight the channels by corresponding gradients
for i in range(activations.size(1)):
    activations[:, i, :, :] *= pooled_gradients[i]

# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

# draw the heatmap
plt.matshow(heatmap.squeeze())

heatmap = heatmap.numpy()

plt.show()
img = cv2.imread('./data/Elephant/1.jpg')
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('./map.jpg', superimposed_img)