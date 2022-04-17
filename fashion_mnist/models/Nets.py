#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
#--------------------------mlp---------------------------------------
class mlp(nn.Module):
    def __init__(self, args):
        super(mlp, self).__init__()
        self.layer_input = nn.Linear(args.num_channels, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(100, args.num_classes)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x
#----------------------vanilla CNN-----------------------------------------
class vanillacnn(nn.Module):
    def __init__(self, args):
        super(vanillacnn, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=5)  #in_channels, out_channels, kernel_size, stride, padding
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64*4*4, 512)
        self.fc2 = nn.Linear(512, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#------------------------lenet------------------------------------------------ 
class lenet_300_100(nn.Module):
    def __init__(self, args):
        super(lenet_300_100, self).__init__()
        self.fc1 = nn.Linear(args.num_channels, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, args.num_classes)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
        
class lenet5(nn.Module):
    def __init__(self, args):
        super(lenet5, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 6, kernel_size=5)  #in_channels, out_channels, kernel_size, stride, padding
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#-----------------------------------------------------------------------------------


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)  #in_channels, out_channels, kernel_size, stride, padding
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
#----------------------------------------------------------------------------
        
class cnn3(nn.Module):
    def __init__(self,args):
        super(cnn3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.Dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #32*32*48
        x = F.relu(self.conv2(x)) #32*32*96
        x = self.pool(x) #16*16*96
        x = self.Dropout(x)
        x = F.relu(self.conv3(x)) #16*16*192
        x = F.relu(self.conv4(x)) #16*16*256
        x = self.pool(x) # 8*8*256
        x = self.Dropout(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]) # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x
    

#----------------------------------------------------------------------------
class cfqk(nn.Module): 
    def __init__(self, args ):
        super(cfqk, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels= 6,
                               kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.pool(h)
        h = F.relu(self.conv2(h))
        h = self.pool(h)
        h = h.view(-1, 16 *5 *5)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return h
#----------------------------------------------------------------------------
class vgg16(nn.Module):
    def __init__(self, args):
        super(vgg16,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64,64,3,padding=1)    
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64,128, 3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)        
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(128,256, 3,padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(256, 256, 3,padding=1)        
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        
        self.conv7 = nn.Conv2d(256, 256, 3,padding=1)        
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()
        self.pool7 = nn.MaxPool2d(2, 2)
        
        self.conv8 = nn.Conv2d(256,512, 3,padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv2d(512, 512, 3,padding=1)        
        self.bn9 = nn.BatchNorm2d(512)
        self.relu9 = nn.ReLU()
        
        self.conv10 = nn.Conv2d(512, 512, 3,padding=1)        
        self.bn10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU()
        self.pool10 = nn.MaxPool2d(2, 2)
        
        self.conv11 = nn.Conv2d(512,512, 3,padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.relu11 = nn.ReLU()

        self.conv12 = nn.Conv2d(512, 512, 3,padding=1)        
        self.bn12 = nn.BatchNorm2d(512)
        self.relu12 = nn.ReLU()
        
        self.conv13 = nn.Conv2d(512, 512, 3,padding=1)        
        self.bn13 = nn.BatchNorm2d(512)
        self.relu13 = nn.ReLU()
        self.maxpool13 = nn.MaxPool2d(2, 2)
      
        self.fc14 = nn.Linear(512,4096)
        self.relu14 =nn.ReLU()
        self.drop14 = nn.Dropout()
        
        self.fc15 = nn.Linear(4096,4096)
        self.relu15 =nn.ReLU()
        self.drop15 = nn.Dropout()
        
        self.fc16 = nn.Linear(4096,args.num_classes)


    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.pool7(x)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)
        
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)
        x = self.pool10(x)
        
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)
        
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)
        
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu13(x)
        x = self.maxpool13(x)        
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc14(x))
        x = self.drop14(x)
        
        x = F.relu(self.fc15(x))
        x = self.drop15(x)
        
        x = self.fc16(x)

        return x
#----------------------------------------------------------------------------
