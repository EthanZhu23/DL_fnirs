import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2

# In the current directory create a folder called 'data' and inside it another folder called 'Elephant' 
# and inside, paste the image you want to check the GradCAM

torch.manual_seed(1)

# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((32, 32)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# define a 1 image dataset
dataset = datasets.ImageFolder(root='./data/', transform=transform)

# define the dataloader to load that single image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

class Combine(nn.Module):
    def __init__(self, input_nd, nf=64):
        super(Combine, self).__init__()
        self.output_num = [4, 2, 1]
        self.nd = input_nd
        self.conv1 = nn.Conv2d(input_nd, nf, 5, stride=2, bias=False)

        self.conv2 = nn.Conv2d(nf, nf * 2, 5, stride=2, bias=False)
        self.BN1 = nn.BatchNorm2d(nf * 2)

        self.conv3 = nn.Conv2d(nf * 2, nf * 4, 5, stride=2, bias=False)
        self.BN2 = nn.BatchNorm2d(nf * 4)

        self.conv4 = nn.Conv2d(nf * 4, nf * 8, 20, stride=10, bias=False)
        self.BN3 = nn.BatchNorm2d(nf * 8)

        self.conv5 = nn.Conv2d(nf * 8, 64, 20, stride=10, bias=False)
        self.fc1 = nn.Linear(5376, 2)

        #self.fc1 = nn.Linear(10752, 2)
        # self.fc2 = nn.Linear(4096, 1000)
        # self.softmax = nn.LogSoftmax(dim=1)
        # self.fc1 = nn.Linear(500‬, 50)
        # self.fc2 = nn.Linear(50, 10)

        # self.rnn = nn.LSTM(
        #    input_size=1000,
        #    hidden_size=64,
        #    num_layers=1,
        #   batch_first=True)
        # self.linear = nn.Linear(64, 2)
        # self.sigmoid = nn.Sigmoid()
    


    def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer

        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''
        # print(previous_conv.size())
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)

            h_wid = int(math.ceil(previous_conv_size[0] // out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] // out_pool_size[i]))
            h_pad = (h_wid * out_pool_size[i] - previous_conv_size[0] + 1) // 2
            w_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) // 2
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if (i == 0):
                spp = x.view(num_sample, -1)
                #print("spp size:",spp.size())
            else:
                #print("size:",spp.size())
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
        return spp

    def forward(self, x):

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(self.BN1(x))

        x = self.conv3(x)
        #x = F.leaky_relu(self.BN2(x))

        #x = self.conv4(x)
        # x = F.leaky_relu(self.BN3(x))
        # x = self.conv5(x)
        # spp = Modified_SPPLayer(4, x)
        spp = self.spatial_pyramid_pool(x, 1, [int(x.size(2)), int(x.size(3))], self.output_num)
        # print(spp.size())
        fc1 = self.fc1(spp)
        # fc2 = self.fc2(fc1)
        s = nn.Sigmoid()
        output = s(fc1)

        # LSTM
        # fc1 = fc1.view(-1,1000)
        # r_in = fc1.view(len(fc1),args.batch_size,-1)
        # r_out, _ = self.rnn(r_in)
        # r_out2 = self.linear(r_out[0])
        # r_out2 = self.sigmoid(r_out2)

        return output
    
#####
# Training procedure to be placed here
#####

sn.eval()
img, _ = next(iter(dataloader))
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