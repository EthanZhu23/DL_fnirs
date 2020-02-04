import torch
from model_pytorch import Combine

model = Combine(8)
model.load_state_dict(torch.load('D:\\Ethan\\Final_data\\train3\\model\\Kfold_5_epoch_28.pth'))
model.eval()