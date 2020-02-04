import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        # Put image paths into a list from input text file
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            imgs.append(line)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        fn = fn.rstrip()
        paths = fn.split(',')
        image_dir_1 = paths[0]
        image_dir_2 = paths[1]
        image_dir_3 = paths[2]
        image_dir_4 = paths[3]
        image_dir_5 = paths[4]
        image_dir_6 = paths[5]
        image_dir_7 = paths[6]
        image_dir_8 = paths[7]
        label = int(paths[8])

        # f_nirs_1 = Image.open(image_dir_1).convert('RGB')
        f_nirs_1 = Image.open(image_dir_1)

        if self.transform is not None:
            f_nirs_1 = self.transform(f_nirs_1)

        # f_nirs_1 = pd.read_csv(image_dir_1)
        # f_nirs_1 = torch.tensor(f_nirs_1.values)
        # f_nirs_1.unsqueeze_(0)

        # f_nirs_2 = Image.open(image_dir_2).convert('RGB')
        f_nirs_2 = Image.open(image_dir_2)

        if self.transform is not None:
            f_nirs_2 = self.transform(f_nirs_2)

        # f_nirs_2 = pd.read_csv(image_dir_2)
        # f_nirs_2 = torch.tensor(f_nirs_2.values)
        # f_nirs_2.unsqueeze_(0)
        f_nirs_2 = torch.cat((f_nirs_1, f_nirs_2), 1)

        # f_nirs_3 = Image.open(image_dir_3).convert('RGB')
        f_nirs_3 = Image.open(image_dir_3)

        if self.transform is not None:
            f_nirs_3 = self.transform(f_nirs_3)
        # f_nirs_3 = pd.read_csv(image_dir_3)
        # f_nirs_3 = torch.tensor(f_nirs_3.values)
        # f_nirs_3.unsqueeze_(0)
        f_nirs_3 = torch.cat((f_nirs_2, f_nirs_3), 1)

        # f_nirs_4 = Image.open(image_dir_4).convert('RGB')
        f_nirs_4 = Image.open(image_dir_4)

        if self.transform is not None:
            f_nirs_4 = self.transform(f_nirs_4)

        # f_nirs_4 = pd.read_csv(image_dir_4)
        # f_nirs_4 = torch.tensor(f_nirs_4.values)
        # f_nirs_4.unsqueeze_(0)
        f_nirs_4 = torch.cat((f_nirs_3, f_nirs_4), 1)

        # f_nirs_5 = Image.open(image_dir_5).convert('RGB')
        f_nirs_5 = Image.open(image_dir_5)

        if self.transform is not None:
            f_nirs_5 = self.transform(f_nirs_5)

        # f_nirs_5 = pd.read_csv(image_dir_5)
        # f_nirs_5 = torch.tensor(f_nirs_5.values)
        # f_nirs_5.unsqueeze_(0)
        f_nirs_5 = torch.cat((f_nirs_4, f_nirs_5), 1)

        # f_nirs_6 = Image.open(image_dir_6).convert('RGB')
        f_nirs_6 = Image.open(image_dir_6)

        if self.transform is not None:
            f_nirs_6 = self.transform(f_nirs_6)

        # f_nirs_6 = pd.read_csv(image_dir_6)
        # f_nirs_6 = torch.tensor(f_nirs_6.values)
        # f_nirs_6.unsqueeze_(0)
        f_nirs_6 = torch.cat((f_nirs_5, f_nirs_6), 1)

        # f_nirs_7 = Image.open(image_dir_7).convert('RGB')
        f_nirs_7 = Image.open(image_dir_7)

        if self.transform is not None:
            f_nirs_7 = self.transform(f_nirs_7)

        # f_nirs_7 = pd.read_csv(image_dir_7)
        # f_nirs_7 = torch.tensor(f_nirs_7.values)
        # f_nirs_7.unsqueeze_(0)
        f_nirs_7 = torch.cat((f_nirs_6, f_nirs_7), 1)

        # f_nirs_8 = Image.open(image_dir_8).convert('RGB')
        f_nirs_8 = Image.open(image_dir_8)

        if self.transform is not None:
            f_nirs_8 = self.transform(f_nirs_8)

        # f_nirs_8 = pd.read_csv(image_dir_8)
        # f_nirs_8 = torch.tensor(f_nirs_8.values)
        # f_nirs_8.unsqueeze_(0)
        f_nirs_8 = torch.cat((f_nirs_7, f_nirs_8), 1)

        return f_nirs_8, label, index

    def __len__(self):
        return len(self.imgs)