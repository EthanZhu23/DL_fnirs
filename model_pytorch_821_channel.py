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
import gc
from Combined_model import Combine
from MyDataset import MyDataset


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


# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


tb = SummaryWriter()

def train(epoch, model):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (data, target, index) in enumerate(train_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
            epoch, batch_idx * len(data) + 1, len(train_loader.dataset),
                   100. * (batch_idx + 1) / len(train_loader), loss.item()), Variable(output.data[0]))

        del data, target, index, output, loss
        torch.cuda.empty_cache()

    print(epoch, epoch_loss / len(train_loader))
    train_loss = epoch_loss / len(train_loader)
    torch.cuda.empty_cache()

    return train_loss


def test(data_state, test_set, model, dir_path):
    model.eval()
    t_loss = 0
    correct = 0
    all_labels, pred_labels, correct_index, wrong_index = [], [], [], []
    for event_index, (info, label, index) in enumerate(test_set):

        data = info
        data = data.float()
        target = label
        index = int(index)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)
        with torch.no_grad():
            output = model(data)
            t_loss += criterion(output, target).data.item()  # sum up batch loss

        print(output.data)
        # test_loss += F.nll_loss(
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

        with open(dir_path + "\\test\\prediction_correct.txt", 'a') as correct_file:
            for item in correct_index:
                correct_file.write('%i\n' % item)
        with open(dir_path + "\\test\\prediction_wrong.txt", 'a') as wrong_file:
            for item in wrong_index:
                wrong_file.write('%i\n' % item)
        return t_loss, t_acc, all_labels, pred_labels

    elif data_state == 'validation':
        v_acc = 100. * correct / len(test_set.dataset)
        cm_valid = confusion_matrix(all_labels, pred_labels)
        print(cm_valid)
        with open(dir_path + "\\validation\\prediction_correct.txt", 'a') as correct_file:
            for item in correct_index:
                correct_file.write('%i\n' % item)
        with open(dir_path + "\\validation\\prediction_wrong.txt", 'a') as wrong_file:
            for item in wrong_index:
                wrong_file.write('%i\n' % item)

        print(
            '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\t'.format(
                t_loss, correct, len(test_set.dataset),
                100. * correct / len(test_set.dataset)), Variable(output.data[0]), '\n')
        print('Correct event index:', correct_index)
        print('Wrong event index', wrong_index)

        return t_loss, v_acc


# def weight_reset(m):
#    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#        m.reset_parameters()
base_path = 'D:\\Ethan\\dlfnirs\\FFT_GRAY\\train6\\'
dataset = MyDataset(
    txt_path=r"D:\\Ethan\\dlfnirs\\FFT_GRAY\\HHb_8_in_1_path.txt ",
    transform=transforms.Compose([transforms.ToTensor()]), target_transform=None)
# dataset = MyDataset(txt_path=r"D:\Ethan\Final_data\test.txt", transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), target_transform = None)

fold1, fold2, fold3, fold4, fold5 = torch.utils.data.random_split(dataset, (160, 160, 160, 160, 160))
# fold1, fold2, fold3, fold4, fold5= torch.utils.data.random_split(dataset, (11,11,11,11,11))

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
    tem_alabel, tem_plabel = [], []

    if k == 1:

        list_of_dataset.append(fold2)
        list_of_dataset.append(fold3)
        list_of_dataset.append(fold4)
        list_of_dataset.append(fold5)
        ds = torch.utils.data.ConcatDataset(list_of_dataset)
        train_ds, valid_ds = torch.utils.data.random_split(ds, (576, 64))
        # train_ds, valid_ds = torch.utils.data.random_split(ds, (34, 10))
        test_ds = fold1
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

        model_1 = Combine(1)
        if args.cuda:
            model_1.cuda()
        optimizer = optim.Adam(model_1.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch, model_1)
            tb.add_scalar('Training_loss_fold_1', train_loss, epoch)
            torch.cuda.empty_cache()
            val_loss, val_acc = test('validation', valid_loader, model_1, base_path + 'model_' + str(k))
            tb.add_scalar('Validation_loss_fold_1', val_loss, epoch)
            tb.add_scalar('Validation_Accracy_fold_1', val_acc, epoch)

            if val_loss < best_loss:
                torch.save(model_1,
                           base_path + 'model_' + str(k) + '\\model\\Kfold_' + str(k) + '_epoch_' + str(epoch) + '.pth')
                best_loss = val_loss
                best_epoch = epoch

            torch.cuda.empty_cache()

        model_1 = torch.load(
            base_path + 'model_' + str(k) + '\\model\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.pth')
        test_loss, test_acc, tem_alabel, tem_plabel = test('test', test_loader, model_1, base_path + 'model_' + str(k))

        tb.add_scalar('Test_loss_fold_1', test_loss, k)
        tb.add_scalar('Test_Accracy_fold_1', test_acc, k)

        all_label = all_label + tem_alabel
        pred_label = pred_label + tem_plabel

        cm = confusion_matrix(tem_alabel, tem_plabel)
        print(cm)
        plot_confusion_matrix(cm, base_path + 'model_' + str(k) + '\\cm\\Kfold_' + str(k) + '_epoch_' + str(
            best_epoch) + '.png', target_names=names, cmap=None, normalize=False)

        # model.apply(weight_reset)
        del model_1
        list_of_dataset.clear()

    elif k == 2:
        list_of_dataset.append(fold1)
        list_of_dataset.append(fold3)
        list_of_dataset.append(fold4)
        list_of_dataset.append(fold5)
        ds = torch.utils.data.ConcatDataset(list_of_dataset)
        train_ds, valid_ds = torch.utils.data.random_split(ds, (576, 64))
        # train_ds, valid_ds = torch.utils.data.random_split(ds, (34, 10))
        test_ds = fold2
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

        model_1 = Combine(1)
        if args.cuda:
            model_1.cuda()
        optimizer = optim.Adam(model_1.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch, model_1)
            tb.add_scalar('Training_loss_fold_2', train_loss, epoch)
            torch.cuda.empty_cache()
            val_loss, val_acc = test('validation', valid_loader, model_1, base_path + 'model_' + str(k))
            tb.add_scalar('Validation_loss_fold_2', val_loss, epoch)
            tb.add_scalar('Validation_Accracy_fold_2', val_acc, epoch)

            if val_loss < best_loss:
                torch.save(model_1,
                           base_path + 'model_' + str(k) + '\\model\\Kfold_' + str(k) + '_epoch_' + str(epoch) + '.pth')
                best_loss = val_loss
                best_epoch = epoch

            torch.cuda.empty_cache()

        model_1 = torch.load(
            base_path + 'model_' + str(k) + '\\model\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.pth')
        test_loss, test_acc, tem_alabel, tem_plabel = test('test', test_loader, model_1, base_path + 'model_' + str(k))

        tb.add_scalar('Test_loss_fold_2', test_loss, k)
        tb.add_scalar('Test_Accracy_fold_2', test_acc, k)

        all_label = all_label + tem_alabel
        pred_label = pred_label + tem_plabel

        cm = confusion_matrix(tem_alabel, tem_plabel)
        print(cm)
        plot_confusion_matrix(cm, base_path + 'model_' + str(k) + '\\cm\\Kfold_' + str(k) + '_epoch_' + str(
            best_epoch) + '.png', target_names=names, cmap=None, normalize=False)

        del model_1
        # model.apply(weight_reset)
        list_of_dataset.clear()

    elif k == 3:
        list_of_dataset.append(fold1)
        list_of_dataset.append(fold2)
        list_of_dataset.append(fold4)
        list_of_dataset.append(fold5)
        ds = torch.utils.data.ConcatDataset(list_of_dataset)
        train_ds, valid_ds = torch.utils.data.random_split(ds, (576, 64))
        # train_ds, valid_ds = torch.utils.data.random_split(ds, (34, 10))
        test_ds = fold3
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

        model_1 = Combine(1)
        if args.cuda:
            model_1.cuda()
        optimizer = optim.Adam(model_1.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch, model_1)
            tb.add_scalar('Training_loss_fold_3', train_loss, epoch)
            torch.cuda.empty_cache()
            val_loss, val_acc = test('validation', valid_loader, model_1, base_path + 'model_' + str(k))
            tb.add_scalar('Validation_loss_fold_3', val_loss, epoch)
            tb.add_scalar('Validation_Accracy_fold_3', val_acc, epoch)

            if val_loss < best_loss:
                torch.save(model_1,
                           base_path + 'model_' + str(k) + '\\model\\Kfold_' + str(k) + '_epoch_' + str(epoch) + '.pth')
                best_loss = val_loss
                best_epoch = epoch

            torch.cuda.empty_cache()

        model_1 = torch.load(
            base_path + 'model_' + str(k) + '\\model\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.pth')
        test_loss, test_acc, tem_alabel, tem_plabel = test('test', test_loader, model_1, base_path + 'model_' + str(k))

        tb.add_scalar('Test_loss_fold_3', test_loss, k)
        tb.add_scalar('Test_Accracy_fold_3', test_acc, k)

        all_label = all_label + tem_alabel
        pred_label = pred_label + tem_plabel

        cm = confusion_matrix(tem_alabel, tem_plabel)
        print(cm)
        plot_confusion_matrix(cm, base_path + 'model_' + str(k) + '\\cm\\Kfold_' + str(k) + '_epoch_' + str(
            best_epoch) + '.png', target_names=names, cmap=None, normalize=False)

        del model_1
        # model.apply(weight_reset)
        list_of_dataset.clear()

    elif k == 4:
        list_of_dataset.append(fold1)
        list_of_dataset.append(fold2)
        list_of_dataset.append(fold3)
        list_of_dataset.append(fold5)
        ds = torch.utils.data.ConcatDataset(list_of_dataset)
        train_ds, valid_ds = torch.utils.data.random_split(ds, (576, 64))
        # train_ds, valid_ds = torch.utils.data.random_split(ds, (34, 10))
        test_ds = fold4
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

        model_1 = Combine(1)
        if args.cuda:
            model_1.cuda()
        optimizer = optim.Adam(model_1.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch, model_1)
            tb.add_scalar('Training_loss_fold_4', train_loss, epoch)
            torch.cuda.empty_cache()
            val_loss, val_acc = test('validation', valid_loader, model_1, base_path + 'model_' + str(k))
            tb.add_scalar('Validation_loss_fold_4', val_loss, epoch)
            tb.add_scalar('Validation_Accracy_fold_4', val_acc, epoch)

            if val_loss < best_loss:
                torch.save(model_1,
                           base_path + 'model_' + str(k) + '\\model\\Kfold_' + str(k) + '_epoch_' + str(epoch) + '.pth')
                best_loss = val_loss
                best_epoch = epoch

            torch.cuda.empty_cache()

        model_1 = torch.load(
            base_path + 'model_' + str(k) + '\\model\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.pth')
        test_loss, test_acc, tem_alabel, tem_plabel = test('test', test_loader, model_1, base_path + 'model_' + str(k))

        tb.add_scalar('Test_loss_fold_4', test_loss, k)
        tb.add_scalar('Test_Accracy_fold_4', test_acc, k)

        all_label = all_label + tem_alabel
        pred_label = pred_label + tem_plabel

        cm = confusion_matrix(tem_alabel, tem_plabel)
        print(cm)
        plot_confusion_matrix(cm, base_path + 'model_' + str(k) + '\\cm\\Kfold_' + str(k) + '_epoch_' + str(
            best_epoch) + '.png', target_names=names, cmap=None, normalize=False)

        del model_1
        # model.apply(weight_reset)
        list_of_dataset.clear()

    elif k == 5:
        list_of_dataset.append(fold1)
        list_of_dataset.append(fold2)
        list_of_dataset.append(fold3)
        list_of_dataset.append(fold4)
        ds = torch.utils.data.ConcatDataset(list_of_dataset)
        train_ds, valid_ds = torch.utils.data.random_split(ds, (576, 64))
        # train_ds, valid_ds = torch.utils.data.random_split(ds, (34, 10))
        test_ds = fold4
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

        model_1 = Combine(1)
        if args.cuda:
            model_1.cuda()
        optimizer = optim.Adam(model_1.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch, model_1)
            tb.add_scalar('Training_loss_fold_5', train_loss, epoch)
            torch.cuda.empty_cache()
            val_loss, val_acc = test('validation', valid_loader, model_1, base_path + 'model_' + str(k))
            tb.add_scalar('Validation_loss_fold_5', val_loss, epoch)
            tb.add_scalar('Validation_Accracy_fold_5', val_acc, epoch)

            if val_loss < best_loss:
                torch.save(model_1,
                           base_path + 'model_' + str(k) + '\\model\\Kfold_' + str(k) + '_epoch_' + str(epoch) + '.pth')
                best_loss = val_loss
                best_epoch = epoch

            torch.cuda.empty_cache()

        model_1 = torch.load(
            base_path + 'model_' + str(k) + '\\model\\Kfold_' + str(k) + '_epoch_' + str(best_epoch) + '.pth')
        test_loss, test_acc, tem_alabel, tem_plabel = test('test', test_loader, model_1, base_path + 'model_' + str(k))

        tb.add_scalar('Test_loss_fold_5', test_loss, k)
        tb.add_scalar('Test_Accracy_fold_5', test_acc, k)

        all_label = all_label + tem_alabel
        pred_label = pred_label + tem_plabel

        cm = confusion_matrix(tem_alabel, tem_plabel)
        print(cm)
        plot_confusion_matrix(cm, base_path + 'model_' + str(k) + '\\cm\\Kfold_' + str(k) + '_epoch_' + str(
            best_epoch) + '.png', target_names=names, cmap=None, normalize=False)

        cm_total = confusion_matrix(all_label, pred_label)
        print(cm_total)
        plot_confusion_matrix(cm_total, base_path + 'model_' + str(k) + '\\cm\\final.png', target_names=names, cmap=None,
                              normalize=False)
        del model_1
        # model.apply(weight_reset)
        list_of_dataset.clear()




