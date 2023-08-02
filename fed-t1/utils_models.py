from utils_libs import *
import torchvision.models as models

class UNetDecoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(UNetDecoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1
#####
# class ConvBlock(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(ConvBlock, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.Dropout(0.3),
#             nn.LeakyReLU(),
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.Dropout(0.3),
#             nn.LeakyReLU()
#         )

#     def forward(self, x):
#         return self.layer(x)


# class DownSample(nn.Module):
#     def __init__(self, channel):
#         super(DownSample, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
#             nn.BatchNorm2d(channel),
#             nn.LeakyReLU()
#         )

#     def forward(self, x):
#         return self.layer(x)


# class UpSample(nn.Module):
#     def __init__(self, channel):
#         super(UpSample,  self).__init__()
#         self.layer = nn.Conv2d(channel, channel//2, 1, 1)

#     def forward(self, x, feature_map):
#         up = F.interpolate(x, scale_factor=2, mode='nearest')
#         out = self.layer(up)
#         return torch.cat((out, feature_map), dim=1)
#####


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=1, bias=False),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=1, bias=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#####

class client_model(nn.Module):
    def __init__(self, name, args=True):
        super(client_model, self).__init__()
        self.name = name
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)
          
        if self.name == 'mnist_2NN':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, self.n_cls)
            
        if self.name == 'emnist_NN':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, self.n_cls)
        
        if self.name == 'cifar10_LeNet':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)
            
        if self.name == 'cifar100_LeNet':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)
            
        if self.name == 'Resnet18':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 10)

            # Change BN to GN 
            resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
  
            self.model = resnet18

        if self.name == 'shakes_LSTM':
            embedding_dim = 8
            hidden_size = 100
            num_LSTM = 2
            input_length = 80
            self.n_cls = 80
            
            self.embedding = nn.Embedding(input_length, embedding_dim)
            self.stacked_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_LSTM)
            self.fc = nn.Linear(hidden_size, self.n_cls)

         ###################
         ### Testing !!! ###
         ###################

        if self.name == 'mnist_UNet':
            self.n_cls = 10
            # self.base_model = torchvision.models.resnet18(True)
            # self.base_model = models.resnet18(True)
            # self.base_layers = list(self.base_model.children())
            # self.layer1 = nn.Sequential(
            #     # nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            #     nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            #     self.base_layers[1],
            #     self.base_layers[2])
            # self.layer2 = nn.Sequential(*self.base_layers[3:5])
            # self.layer3 = self.base_layers[5]
            # self.layer4 = self.base_layers[6]
            # # self.layer5 = self.base_layers[7]
            # self.layer5 = self.base_layers[7]
            # self.decode4 = UNetDecoder(512, 256+256, 256)
            # self.decode3 = UNetDecoder(256, 256+128, 256)
            # self.decode2 = UNetDecoder(256, 128+64, 128)
            # self.decode1 = UNetDecoder(128, 64+64, 64)
            # self.decode0 = nn.Sequential(
            #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #     nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            #     nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            #     )
            # self.conv_last = nn.Conv2d(64, self.n_cls, 1)
#######
            # self.c1 = ConvBlock(3, 64)
            # self.d1 = DownSample(64)
            # self.c2 = ConvBlock(64, 128)
            # self.d2 = DownSample(128)
            # self.c3 = ConvBlock(128, 256)
            # self.d3 = DownSample(256)
            # self.c4 = ConvBlock(256, 512)
            # self.d4 = DownSample(512)
            # self.c5 = ConvBlock(512, 1024)

            # self.u1 = UpSample(1024)
            # self.c6 = ConvBlock(1024, 512)
            # self.u2 = UpSample(512)
            # self.c7 = ConvBlock(512, 256)
            # self.u3 = UpSample(256)
            # self.c8 = ConvBlock(256, 128)
            # self.u4 = UpSample(128)
            # self.c9 = ConvBlock(128, 64)

            # self.out = nn.Conv2d(64, 3, 3, 1, 1)
            # self.Th = nn.Sigmoid()


            # self.n_channels = 1
            self.n_channels = 3
            # self.n_channels = 4
            self.n_cls = 10
            self.bilinear = False

            self.inc = (DoubleConv(self.n_channels, 64))
            self.down1 = (Down(64, 128))
            # self.inc = (DoubleConv(self.n_channels, 50))
            # self.down1 = (Down(50, 128))
            self.down2 = (Down(128, 256))
            self.down3 = (Down(256, 512))
            factor = 2 if self.bilinear else 1
            self.down4 = (Down(512, 1024 // factor))
            self.up1 = (Up(1024, 512 // factor, self.bilinear))
            self.up2 = (Up(512, 256 // factor, self.bilinear))
            self.up3 = (Up(256, 128 // factor, self.bilinear))
            self.up4 = (Up(128, 64, self.bilinear))
            self.outc = (OutConv(64, self.n_cls))

            # Change here to adapt to your data
            # n_channels=3 for RGB images
            # n_classes is the number of probabilities you want to get per pixel

        if self.name == 'medical-mnist_MedicalMNISTCNN':
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.classifier = nn.Sequential(
                nn.Linear(in_features=256, out_features=128),
                nn.Dropout2d(p=0.4),
                nn.Linear(in_features=128, out_features=num_classes)
            )

    def forward(self, x):
        if self.name == 'Linear':
            x = self.fc(x)
            
        if self.name == 'mnist_2NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
  
        if self.name == 'emnist_NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        if self.name == 'cifar10_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
        if self.name == 'cifar100_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
        if self.name == 'Resnet18':
            x = self.model(x)

        if self.name == 'shakes_LSTM':
            x = self.embedding(x)
            x = x.permute(1, 0, 2) # lstm accepts in this style
            output, (h_, c_) = self.stacked_LSTM(x)
            # Choose last hidden layer
            last_hidden = output[-1,:,:]
            x = self.fc(last_hidden)

         ###################
         ### Testing !!! ###
         ###################

#####

        if self.name == 'medical-mnist_MedicalMNISTCNN':
            x = self.conv_block(x)
            bs, _, _, _ = x.shape
            x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
            x = self.classifier(x)
#####

        if self.name == 'mnist_UNet':
            # # e1 = self.layer1(input) # 64,128,128
            # e1 = self.layer1(x) # 64,128,128
            # e2 = self.layer2(e1) # 64,64,64
            # e3 = self.layer3(e2) # 128,32,32
            # e4 = self.layer4(e3) # 256,16,16
            # f = self.layer5(e4) # 512,8,8
            # d4 = self.decode4(f, e4) # 256,16,16
            # d3 = self.decode3(d4, e3) # 256,32,32
            # d2 = self.decode2(d3, e2) # 128,64,64
            # d1 = self.decode1(d2, e1) # 64,128,128
            # d0 = self.decode0(d1) # 64,256,256
            # x = self.conv_last(d0) # 1,256,256
            ###############
            # R1 = self.c1(x)
            # R2 = self.c2(self.d1(R1))
            # R3 = self.c3(self.d2(R2))
            # R4 = self.c4(self.d3(R3))
            # R5 = self.c5(self.d4(R4))

            # o1 = self.c6(self.u1(R5, R4))
            # o2 = self.c7(self.u2(o1, R3))
            # o3 = self.c8(self.u3(o2, R2))
            # o4 = self.c9(self.u4(o3, R1))
            # x = self.Th(self.out(o4))
            ###############
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outc(x)

        return x


""""

RuntimeError: Error(s) in loading state_dict for client_model:

model.load_state_dict(state_dict, strict=True)

Copies parameters and buffers from :attr:state_dict into this module and its descendants. If :attr:strict is True, then the keys of :attr:state_dict must exactly match the keys returned by this module’s :meth:~torch.nn.Module.state_dict function
从属性state_dict里面复制参数到这个模块和它的后代。如果strict为True, state_dict的keys必须完全与这个模块的方法返回的keys相匹配。如果为False,就不需要保证匹配。

Arguments:
state_dict (dict): a dict containing parameters and persistent buffers.
strict (bool, optional): whether to strictly enforce that the keys in :attr:state_dict match the keys returned by this module’s:meth:~torch.nn.Module.state_dict function. Default: True

Reference : https://blog.csdn.net/qq_29631521/article/details/92806793
"""