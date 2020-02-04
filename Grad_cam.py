from Combined_model import Combine
import torch
import os
import PIL
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from MyDataset import MyDataset
from torchvision import transforms

dataset = MyDataset(
    txt_path=r"C:\\Users\\EthanZhu\\Box Sync\\Project\\CNN_LSTM\\Github\\dlfnirs\\FFT_GRAY\\test.txt",
    transform=transforms.Compose([transforms.ToTensor()]), target_transform=None)

torch_img,_,_ = dataset.__getitem__(0)
normed_torch_img = torch_img


model = Combine(1)
model = torch.load("C:\\Users\\EthanZhu\\Box Sync\\Project\\CNN_LSTM\\Github\\DL_fnirs\\train6\\model_5\\model\\Kfold_5_epoch_21.pth")

configs = [
    dict(model_type='vgg', arch=model, layer_name='features')
]


for config in configs:
    config['arch'].to('cuda').eval()

cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
    for config in configs
]


images = []
for gradcam, gradcam_pp in cams:
    mask, _ = gradcam(normed_torch_img)
    heatmap, result = visualize_cam(mask, torch_img)

    mask_pp, _ = gradcam_pp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
    
    images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])
    
grid_image = make_grid(images, nrow=5)