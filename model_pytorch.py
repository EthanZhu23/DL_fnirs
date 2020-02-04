# pytorch mnist cnn + lstm

from __future__ import print_function
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image
import math
import numpy as np
from sklearn.metrics import confusion_matrix
from plotcm import plot_confusion_matrix
import pickle
import csv
from torch.utils.tensorboard import SummaryWriter


class Args:
    def __init__(self):
        self.cuda = True
        self.no_cuda = False
        self.seed = 1
        self.batch_size = 1
        self.test_batch_size = 1
        self.epochs = 35
        self.lr = 1e-05
        self.momentum = 0.5
        self.log_interval = 1


args = Args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


class MyDataset(Dataset):
    def __init__(self, txt_path,transform=None,target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        with open(txt_path,'r') as fh:
            next(fh)
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

        #f_nirs_1 = Image.open(image_dir_1).convert('RGB')
        f_nirs_1 = Image.open(image_dir_1)

        if self.transform is not None:
            f_nirs_1 = self.transform(f_nirs_1)

        #f_nirs_1 = pd.read_csv(image_dir_1)
        #f_nirs_1 = torch.tensor(f_nirs_1.values)
        #f_nirs_1.unsqueeze_(0)


        #f_nirs_2 = Image.open(image_dir_2).convert('RGB')
        f_nirs_2 = Image.open(image_dir_2)

        if self.transform is not None:
            f_nirs_2 = self.transform(f_nirs_2)

        #f_nirs_2 = pd.read_csv(image_dir_2)
        #f_nirs_2 = torch.tensor(f_nirs_2.values)
        #f_nirs_2.unsqueeze_(0)
        f_nirs_2 = torch.cat((f_nirs_1,f_nirs_2),0)

        #f_nirs_3 = Image.open(image_dir_3).convert('RGB')
        f_nirs_3 = Image.open(image_dir_3)

        if self.transform is not None:
            f_nirs_3 = self.transform(f_nirs_3)
        #f_nirs_3 = pd.read_csv(image_dir_3)
        #f_nirs_3 = torch.tensor(f_nirs_3.values)
        #f_nirs_3.unsqueeze_(0)
        f_nirs_3 = torch.cat((f_nirs_2,f_nirs_3),0)

        #f_nirs_4 = Image.open(image_dir_4).convert('RGB')
        f_nirs_4 = Image.open(image_dir_4)

        if self.transform is not None:
            f_nirs_4 = self.transform(f_nirs_4)

        #f_nirs_4 = pd.read_csv(image_dir_4)
        #f_nirs_4 = torch.tensor(f_nirs_4.values)
        #f_nirs_4.unsqueeze_(0)
        f_nirs_4 = torch.cat((f_nirs_3,f_nirs_4),0)

        #f_nirs_5 = Image.open(image_dir_5).convert('RGB')
        f_nirs_5 = Image.open(image_dir_5)

        if self.transform is not None:
            f_nirs_5 = self.transform(f_nirs_5)

        #f_nirs_5 = pd.read_csv(image_dir_5)
        #f_nirs_5 = torch.tensor(f_nirs_5.values)
        #f_nirs_5.unsqueeze_(0)
        f_nirs_5 = torch.cat((f_nirs_4,f_nirs_5),0)

        #f_nirs_6 = Image.open(image_dir_6).convert('RGB')
        f_nirs_6 = Image.open(image_dir_6)

        if self.transform is not None:
            f_nirs_6 = self.transform(f_nirs_6)

        #f_nirs_6 = pd.read_csv(image_dir_6)
        #f_nirs_6 = torch.tensor(f_nirs_6.values)
        #f_nirs_6.unsqueeze_(0)
        f_nirs_6 = torch.cat((f_nirs_5,f_nirs_6),0)

        #f_nirs_7 = Image.open(image_dir_7).convert('RGB')
        f_nirs_7 = Image.open(image_dir_7)

        if self.transform is not None:
            f_nirs_7 = self.transform(f_nirs_7)

        #f_nirs_7 = pd.read_csv(image_dir_7)
        #f_nirs_7 = torch.tensor(f_nirs_7.values)
        #f_nirs_7.unsqueeze_(0)
        f_nirs_7 = torch.cat((f_nirs_6,f_nirs_7),0)

        #f_nirs_8 = Image.open(image_dir_8).convert('RGB')
        f_nirs_8 = Image.open(image_dir_8)

        if self.transform is not None:
            f_nirs_8 = self.transform(f_nirs_8)

        #f_nirs_8 = pd.read_csv(image_dir_8)
        #f_nirs_8 = torch.tensor(f_nirs_8.values)
        #f_nirs_8.unsqueeze_(0)
        f_nirs_8 = torch.cat((f_nirs_7,f_nirs_8),0)

        return f_nirs_8, label, index

    def __len__(self):
        return len(self.imgs)


tb = SummaryWriter()


class Combine(nn.Module):
    def __init__(self, input_nd, nf=64):
        super(Combine, self).__init__()
        self.output_num = [4, 2, 1]
        
        self.conv1 = nn.Conv2d(input_nd, nf, 4, 2, 1, bias=False)
        
        self.conv2 = nn.Conv2d(nf, nf * 2, 4, 1, 1, bias=False)
        self.BN1 = nn.BatchNorm2d(nf * 2)

        self.conv3 = nn.Conv2d(nf * 2, nf * 4, 4, 1, 1, bias=False)
        self.BN2 = nn.BatchNorm2d(nf * 4)

        self.conv4 = nn.Conv2d(nf * 4, nf * 8, 4, 1, 1, bias=False)
        self.BN3 = nn.BatchNorm2d(nf * 8)

        self.conv5 = nn.Conv2d(nf * 8, 64, 4, 1, 0, bias=False)
        self.fc1 = nn.Linear(10752,2)
        #self.fc2 = nn.Linear(4096, 1000)
        #self.softmax = nn.LogSoftmax(dim=1)
        #self.fc1 = nn.Linear(500‬, 50)
        #self.fc2 = nn.Linear(50, 10)

        #self.rnn = nn.LSTM(
        #    input_size=1000,
        #    hidden_size=64,
        #    num_layers=1,
        #   batch_first=True)
        #self.linear = nn.Linear(64, 2)
        #self.sigmoid = nn.Sigmoid()

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
                # print("spp size:",spp.size())
            else:
                # print("size:",spp.size())
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
        return spp

    def forward(self, x):

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(self.BN1(x))

        x = self.conv3(x)
        x = F.leaky_relu(self.BN2(x))
        
        x = self.conv4(x)
        #x = F.leaky_relu(self.BN3(x))
        #x = self.conv5(x)
        #spp = Modified_SPPLayer(4, x)
        spp = self.spatial_pyramid_pool(x, 1, [int(x.size(2)), int(x.size(3))], self.output_num)
        # print(spp.size())
        fc1 = self.fc1(spp)
        #fc2 = self.fc2(fc1)
        s = nn.Sigmoid()
        output = s(fc1)
        
        
        
        
        #LSTM
        #fc1 = fc1.view(-1,1000)
        #r_in = fc1.view(len(fc1),args.batch_size,-1)
        #r_out, _ = self.rnn(r_in)
        #r_out2 = self.linear(r_out[0])
        #r_out2 = self.sigmoid(r_out2)

        return output


def train(epoch, model):
    model.train()
    epoch_loss = 0.0
    for batch_idx,(info, label, index) in enumerate(train_loader):

        #data = info[0]
        data = info
        data = data.float()
        target = label

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        #data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output= model(data)
        loss = criterion(output, target)
        #loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        #if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
            epoch, batch_idx * len(data)+1, len(train_loader.dataset),
            100. * (batch_idx+1) / len(train_loader), loss.item()), Variable(output.data[0]))
        torch.zero_(data)
        torch.cuda.empty_cache()


    print(epoch, epoch_loss / len(train_loader))
    train_loss = epoch_loss / len(train_loader)
    torch.cuda.empty_cache()

    return train_loss


def test(data_state, test_set, model):

    model.eval()
    t_loss = 0
    correct = 0
    all_labels, pred_labels, correct_index, wrong_index = [], [], [], []
    for event_index,(info,label,index) in enumerate(test_set):

        data = info
        data = data.float()
        target = label
        index = int(index)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        #data, target = Variable(data), Variable(target)
        with torch.no_grad():
            output = model(data)
            t_loss += criterion(output, target).data.item()  # sum up batch loss

        print(output.data)
        #test_loss += F.nll_loss(
        #    output, target, reduction='sum').data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        all_labels.append(int(target.data))
        pred_labels.append(int(pred.data))
        if pred.eq(target.data.view_as(pred)):
            correct_index.append(index)
        else:
            wrong_index.append(index)

        torch.cuda.empty_cache()

    t_loss /= len(test_loader.dataset)

    if data_state == 'test':
        t_acc = 100. * correct / len(test_set.dataset)
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\t'.format(
                t_loss, correct, len(test_set.dataset),
                100. * correct / len(test_set.dataset)), Variable(output.data[0]), '\n')
        print('Correct event index:', correct_index)
        print('Wrong event index', wrong_index)

        cm_test = confusion_matrix(all_labels, pred_labels)
        print(cm_test)

        with open("D:\\Ethan\\dlfnirs\\FFT_GRAY\\train5\\prediction_correct.txt", 'a') as correct_file:
            for item in correct_index:
                correct_file.write('%i\n' % item)
        with open("D:\\Ethan\\dlfnirs\\FFT_GRAY\\train5\\prediction_wrong.txt", 'a') as wrong_file:
            for item in wrong_index:
                wrong_file.write('%i\n' % item)
        return t_loss, t_acc, all_labels, pred_labels

    elif data_state == 'validation':
        v_acc = 100. * correct / len(test_set.dataset)
        cm_valid = confusion_matrix(all_labels, pred_labels)
        print(cm_valid)
        with open("D:\\Ethan\\dlfnirs\\FFT_GRAY\\train5\\prediction_correct.txt", 'a') as correct_file:
            for item in correct_index:
                correct_file.write('%i\n' % item)
        with open("D:\\Ethan\\dlfnirs\\FFT_GRAY\\train5\\prediction_wrong.txt", 'a') as wrong_file:
            for item in wrong_index:
                wrong_file.write('%i\n' % item)

        print(
            '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\t'.format(
                t_loss, correct, len(test_set.dataset),
                100. * correct / len(test_set.dataset)), Variable(output.data[0]), '\n')
        print('Correct event index:', correct_index)
        print('Wrong event index', wrong_index)

        return t_loss, v_acc


#def weight_reset(m):
#    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#        m.reset_parameters()

base_path = 'D:\\Ethan\\dlfnirs\\FFT_GRAY\\train5\\'
dataset = MyDataset(txt_path=r"D:\\Ethan\\dlfnirs\\FFT_GRAY\\data_path.csv", transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]), target_transform = None)
#dataset = MyDataset(txt_path=r"D:\Ethan\Final_data\test.txt", transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), target_transform = None)

fold1, fold2, fold3, fold4, fold5= torch.utils.data.random_split(dataset, (160,160,160,160,160))
#fold1, fold2, fold3, fold4, fold5= torch.utils.data.random_split(dataset, (11,11,11,11,11))

list_of_dataset = []


criterion = nn.CrossEntropyLoss()

all_label, pred_label = [], []
for k in range(1, 6):
    best_loss = 1000
    test_loss = 0
    test_acc = 0
    val_loss = 0
    val_acc = 0
    best_epoch = 0
    names = [0, 1]
    tem_alabel, tem_plabel= [], []

    if k == 1:

        list_of_dataset.append(fold2)
        list_of_dataset.append(fold3)
        list_of_dataset.append(fold4)
        list_of_dataset.append(fold5)
        ds = torch.utils.data.ConcatDataset(list_of_dataset)
        train_ds, valid_ds = torch.utils.data.random_split(ds, (576, 64))
        #train_ds, valid_ds = torch.utils.data.random_split(ds, (34, 10))
        test_ds = fold1
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

        model_1 = Combine(8)
        if args.cuda:
            model_1.cuda()
        optimizer = optim.Adam(model_1.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch, model_1)
            tb.add_scalar('Training_loss_fold_1', train_loss, epoch)
            torch.cuda.empty_cache()
            val_loss, val_acc= test('validation', valid_loader, model_1)
            tb.add_scalar('Validation_loss_fold_1', val_loss, epoch)
            tb.add_scalar('Validation_Accracy_fold_1', val_acc, epoch)

            if val_loss < best_loss:
                torch.save(model_1, base_path + 'model\\Kfold_'+str(k)+ '_epoch_' + str(epoch) + '.pth')
                best_loss = val_loss
                best_epoch = epoch

            torch.cuda.empty_cache()

        model_1 = torch.load(base_path + 'model\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.pth')
        test_loss, test_acc, tem_alabel, tem_plabel = test('test', test_loader, model_1)

        tb.add_scalar('Test_loss_fold_1', test_loss, k)
        tb.add_scalar('Test_Accracy_fold_1', test_acc, k)

        all_label = all_label + tem_alabel
        pred_label = pred_label + tem_plabel

        cm = confusion_matrix(tem_alabel, tem_plabel)
        print(cm)
        plot_confusion_matrix(cm, base_path + 'cm\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.png', target_names=names, cmap=None, normalize=False)

        #model.apply(weight_reset)
        del model_1
        list_of_dataset.clear()

    elif k == 2:
        list_of_dataset.append(fold1)
        list_of_dataset.append(fold3)
        list_of_dataset.append(fold4)
        list_of_dataset.append(fold5)
        ds = torch.utils.data.ConcatDataset(list_of_dataset)
        train_ds, valid_ds = torch.utils.data.random_split(ds, (576, 64))
        #train_ds, valid_ds = torch.utils.data.random_split(ds, (34, 10))
        test_ds = fold2
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

        model_1 = Combine(8)
        if args.cuda:
            model_1.cuda()
        optimizer = optim.Adam(model_1.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch, model_1)
            tb.add_scalar('Training_loss_fold_2', train_loss, epoch)
            torch.cuda.empty_cache()
            val_loss, val_acc= test('validation', valid_loader,model_1)
            tb.add_scalar('Validation_loss_fold_2', val_loss, epoch)
            tb.add_scalar('Validation_Accracy_fold_2', val_acc, epoch)

            if val_loss < best_loss:
                torch.save(model_1, base_path + 'model\\Kfold_' + str(k) + '_epoch_' + str(epoch) + '.pth')
                best_loss = val_loss
                best_epoch = epoch

            torch.cuda.empty_cache()

        model_1 = torch.load(base_path + 'model\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.pth')
        test_loss, test_acc, tem_alabel, tem_plabel = test('test', test_loader, model_1)

        tb.add_scalar('Test_loss_fold_2', test_loss, k)
        tb.add_scalar('Test_Accracy_fold_2', test_acc, k)

        all_label = all_label + tem_alabel
        pred_label = pred_label + tem_plabel

        cm = confusion_matrix(tem_alabel, tem_plabel)
        print(cm)
        plot_confusion_matrix(cm, base_path + 'cm\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.png', target_names=names, cmap=None, normalize=False)

        del model_1
        #model.apply(weight_reset)
        list_of_dataset.clear()

    elif k == 3:
        list_of_dataset.append(fold1)
        list_of_dataset.append(fold2)
        list_of_dataset.append(fold4)
        list_of_dataset.append(fold5)
        ds = torch.utils.data.ConcatDataset(list_of_dataset)
        train_ds, valid_ds = torch.utils.data.random_split(ds, (576, 64))
        #train_ds, valid_ds = torch.utils.data.random_split(ds, (34, 10))
        test_ds = fold3
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

        model_1 = Combine(8)
        if args.cuda:
            model_1.cuda()
        optimizer = optim.Adam(model_1.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch,model_1)
            tb.add_scalar('Training_loss_fold_3', train_loss, epoch)
            torch.cuda.empty_cache()
            val_loss, val_acc= test('validation', valid_loader,model_1)
            tb.add_scalar('Validation_loss_fold_3', val_loss, epoch)
            tb.add_scalar('Validation_Accracy_fold_3', val_acc, epoch)

            if val_loss < best_loss:
                torch.save(model_1, base_path + 'model\\Kfold_' + str(k) + '_epoch_' + str(epoch) + '.pth')
                best_loss = val_loss
                best_epoch = epoch

            torch.cuda.empty_cache()

        model_1 = torch.load(base_path + 'model\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.pth')
        test_loss, test_acc, tem_alabel, tem_plabel = test('test', test_loader, model_1)

        tb.add_scalar('Test_loss_fold_3', test_loss, k)
        tb.add_scalar('Test_Accracy_fold_3', test_acc, k)

        all_label = all_label + tem_alabel
        pred_label = pred_label + tem_plabel

        cm = confusion_matrix(tem_alabel, tem_plabel)
        print(cm)
        plot_confusion_matrix(cm, base_path + 'cm\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.png', target_names=names, cmap=None, normalize=False)

        del model_1
        #model.apply(weight_reset)
        list_of_dataset.clear()

    elif k == 4:
        list_of_dataset.append(fold1)
        list_of_dataset.append(fold2)
        list_of_dataset.append(fold3)
        list_of_dataset.append(fold5)
        ds = torch.utils.data.ConcatDataset(list_of_dataset)
        train_ds, valid_ds = torch.utils.data.random_split(ds, (576, 64))
        #train_ds, valid_ds = torch.utils.data.random_split(ds, (34, 10))
        test_ds = fold4
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

        model_1 = Combine(8)
        if args.cuda:
            model_1.cuda()
        optimizer = optim.Adam(model_1.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch,model_1)
            tb.add_scalar('Training_loss_fold_4', train_loss, epoch)
            torch.cuda.empty_cache()
            val_loss, val_acc = test('validation', valid_loader,model_1)
            tb.add_scalar('Validation_loss_fold_4', val_loss, epoch)
            tb.add_scalar('Validation_Accracy_fold_4', val_acc, epoch)

            if val_loss < best_loss:
                torch.save(model_1, base_path + 'model\\Kfold_' + str(k) + '_epoch_' + str(epoch) + '.pth')
                best_loss = val_loss
                best_epoch = epoch

            torch.cuda.empty_cache()

        model_1 = torch.load(base_path + 'model\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.pth')
        test_loss, test_acc, tem_alabel, tem_plabel = test('test', test_loader, model_1)

        tb.add_scalar('Test_loss_fold_4', test_loss, k)
        tb.add_scalar('Test_Accracy_fold_4', test_acc, k)

        all_label = all_label + tem_alabel
        pred_label = pred_label + tem_plabel

        cm = confusion_matrix(tem_alabel, tem_plabel)
        print(cm)
        plot_confusion_matrix(cm, base_path + 'cm\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.png', target_names=names, cmap=None, normalize=False)

        del model_1
        #model.apply(weight_reset)
        list_of_dataset.clear()

    elif k == 5:
        list_of_dataset.append(fold1)
        list_of_dataset.append(fold2)
        list_of_dataset.append(fold3)
        list_of_dataset.append(fold4)
        ds = torch.utils.data.ConcatDataset(list_of_dataset)
        train_ds, valid_ds = torch.utils.data.random_split(ds, (576, 64))
        #train_ds, valid_ds = torch.utils.data.random_split(ds, (34, 10))
        test_ds = fold4
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

        model_1 = Combine(8)
        if args.cuda:
            model_1.cuda()
        optimizer = optim.Adam(model_1.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch,model_1)
            tb.add_scalar('Training_loss_fold_5', train_loss, epoch)
            torch.cuda.empty_cache()
            val_loss, val_acc= test('validation', valid_loader, model_1)
            tb.add_scalar('Validation_loss_fold_5', val_loss, epoch)
            tb.add_scalar('Validation_Accracy_fold_5', val_acc, epoch)

            if val_loss < best_loss:
                torch.save(model_1, base_path + 'model\\Kfold_' + str(k) + '_epoch_' + str(epoch) + '.pth')
                best_loss = val_loss
                best_epoch = epoch

            torch.cuda.empty_cache()

        model_1 = torch.load(base_path + 'model\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.pth')
        test_loss, test_acc, tem_alabel, tem_plabel = test('test', test_loader, model_1)

        tb.add_scalar('Test_loss_fold_5', test_loss, k)
        tb.add_scalar('Test_Accracy_fold_5', test_acc, k)

        all_label = all_label + tem_alabel
        pred_label = pred_label + tem_plabel

        cm = confusion_matrix(tem_alabel, tem_plabel)
        print(cm)
        plot_confusion_matrix(cm, base_path + 'cm\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.png', target_names=names, cmap=None, normalize=False)

        cm_total = confusion_matrix(all_label,pred_label)
        print(cm_total)
        plot_confusion_matrix(cm_total, base_path + 'cm\\final.png', target_names=names, cmap=None, normalize=False)
        del model_1
        #model.apply(weight_reset)
        list_of_dataset.clear()




