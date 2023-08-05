#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import torch
from torch import nn
import torch.nn.functional as F
import torchvision


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
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

class CNNCifar100(nn.Module):
    def __init__(self, args):
        super(CNNCifar100, self).__init__()
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

# class conv_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_c)
#         self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_c)
#         self.relu = nn.ReLU()
#     def forward(self, inputs):
#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         return x

# class encoder_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv = conv_block(in_c, out_c)
#         self.pool = nn.MaxPool2d((2, 2))
#     def forward(self, inputs):
#         x = self.conv(inputs)
#         p = self.pool(x)
#         return x, p

# class decoder_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
#         self.conv = conv_block(out_c+out_c, out_c)
#     def forward(self, inputs, skip):
#         x = self.up(inputs)
#         x = torch.cat([x, skip], axis=1)
#         x = self.conv(x)
#         return x
    
# class Salt_UNet(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         """ Encoder """
#         # self.e1 = encoder_block(3, 64)
#         self.e1 = encoder_block(10, 64)
#         self.e2 = encoder_block(64, 128)
#         self.e3 = encoder_block(128, 256)
#         self.e4 = encoder_block(256, 512)
#         """ Bottleneck """
#         self.b = conv_block(512, 1024)
#         """ Decoder """
#         self.d1 = decoder_block(1024, 512)
#         self.d2 = decoder_block(512, 256)
#         self.d3 = decoder_block(256, 128)
#         self.d4 = decoder_block(128, 64)
#         """ Classifier """
#         self.outputs = nn.Conv2d(64, 10, kernel_size=1, padding=0)
#     def forward(self, inputs):
#         """ Encoder """
#         s1, p1 = self.e1(inputs)
#         s2, p2 = self.e2(p1)
#         s3, p3 = self.e3(p2)
#         s4, p4 = self.e4(p3)
#         """ Bottleneck """
#         b = self.b(p4)
#         """ Decoder """
#         d1 = self.d1(b, s4)
#         d2 = self.d2(d1, s3)
#         d3 = self.d3(d2, s2)
#         d4 = self.d4(d3, s1)
#         """ Classifier """
#         outputs = self.outputs(d4)
#         return outputs
    
class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
    def forward(self, x1, x2):
        x1 = self.up(x1)
        print(x1.size())
        print(x1.shape)
        print(len(x1.shape))
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1

class Salt_UNet(nn.Module):
    def __init__(self,  args):
        super().__init__()
        
        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        # self.layer5 = self.base_layers[7]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        # self.conv_last = nn.Conv2d(64, args.num_classes, 1)
        self.conv_last = nn.Conv2d(64, 1, 1)
    def forward(self, input):
        e1 = self.layer1(input) # 64,128,128
        e2 = self.layer2(e1) # 64,64,64
        e3 = self.layer3(e2) # 128,32,32
        e4 = self.layer4(e3) # 256,16,16
        f = self.layer5(e4) # 512,8,8
        d4 = self.decode4(f, e4) # 256,16,16
        d3 = self.decode3(d4, e3) # 256,32,32
        d2 = self.decode2(d3, e2) # 128,64,64
        d1 = self.decode1(d2, e1) # 64,128,128
        d0 = self.decode0(d1) # 64,256,256
        out = self.conv_last(d0) # 1,256,256
        return out

# class DoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),  # in_ch、out_ch是通道数
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)
# class Salt_UNet(nn.Module):
#     def __init__(self, args, in_ch = 10, out_ch =10):
#         super(Salt_UNet, self).__init__()
#         self.conv1 = DoubleConv(in_ch, 64)
#       #  self.pool1 = nn.MaxPool2d(2)  # 每次把图像尺寸缩小一半
#         self.conv2 = DoubleConv(64, 128)
#        # self.pool2 = nn.MaxPool2d(2)
#         self.conv3 = DoubleConv(128, 256)
#         # self.pool3 = nn.MaxPool2d(2)
#         self.conv4 = DoubleConv(256, 512)
#         # self.pool4 = nn.MaxPool2d(2)
#         self.conv5 = DoubleConv(512, 1024)
#         # 逆卷积
#         self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
#         self.conv6 = DoubleConv(1024, 512)
#         self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.conv7 = DoubleConv(512, 256)
#         self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.conv8 = DoubleConv(256, 128)
#         self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.conv9 = DoubleConv(128, 64)

#         self.conv10 = nn.Conv2d(64, out_ch, 1)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         c1 = self.conv1(x)
#        # p1 = self.pool1(c1)
#         c2 = self.conv2(c1)
#       #  p2 = self.pool2(c2)
#         c3 = self.conv3(c2)
#        # p3 = self.pool3(c3)
#         c4 = self.conv4(c3)
#         # p4 = self.pool4(c4)
#         c5 = self.conv5(c4)
#         # c5 = self.conv5(p4)
#         up_6 = self.up6(c5)
#        # merge6 = torch.cat([up_6, c4], dim=1)  # 按维数1（列）拼接,列增加
#         c6 = self.conv6(up_6)
#         up_7 = self.up7(c6)
#        # merge7 = torch.cat([up_7, c3], dim=1)
#         c7 = self.conv7(up_7)
#         up_8 = self.up8(c7)
#        # merge8 = torch.cat([up_8, c2], dim=1)
#         c8 = self.conv8(up_8)
#         up_9 = self.up9(c8)
#        # merge9 = torch.cat([up_9, c1], dim=1)
#         c9 = self.conv9(up_9)
#         c10 = self.conv10(c9)

#         out = self.sigmoid(c10)
#         return out
    
class Mnist_2NN(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Classes 10
        self.fc1 = nn.Linear(1 * 28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, args.num_classes)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# emnist
class Emnist_NN(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Classes 10
        self.fc1 = nn.Linear(1 * 28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, args.num_classes)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

