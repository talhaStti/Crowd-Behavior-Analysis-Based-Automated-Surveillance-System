import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
import threading


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1):

        super().__init__()

        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(conv1_t_size, 7, 7),
                                    stride=(conv1_t_stride, 2, 2),
                                    padding=(conv1_t_size // 2, 3, 3),
                                    bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))
        self.features = nn.Sequential(OrderedDict(self.features))

        # denseblock Computation
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch normalization
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out,
                                    output_size=(1, 1,
                                                 1)).view(features.size(0), -1)

        out = self.classifier(out)
        return out


model = DenseNet()
model_state = torch.load(
    r"C:\Users\ahmed\Desktop\DenseNet_state.pt", map_location=(torch.device('cpu')))
model.load_state_dict(model_state['model_state_dict'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


def Pred(path):
    batch_size = 16
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(171),
        torchvision.transforms.CenterCrop(112),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frames = []
    cap = cv2.VideoCapture(r"path")
    while (cap.isOpened()):
        # time.sleep(1)
        ret, frame = cap.read()
    # Transform the frame
        if ret == True:
            frame = Image.fromarray(frame)
            frame = transform(frame)
    # Add the frame to the list
    # frame=np.array(frame,dtype=(object))
    # frame=torch.as_tensor(frame)

            frames.append(frame)

    # If the list has enough frames, pass the entire batch through the model
            if len(frames) == batch_size:
                # print(frames)
                # frames=torch.unsqueeze(torch.as_tensor(frames),0)
                #frames=(np.array(frames, dtype=(object)))

                #frames=torch.unsqueeze(frames, 0)
                #batch = transform(frames)
                # print(batch.keys())
                #tensor_list = [transform(f) for f in frames]
                # print(tensor_list)
                batch = torch.stack(frames, dim=0)
        # Perform inference on the batch
                with torch.no_grad():
                    # print(batch.shape)
                    batch = torch.unsqueeze(batch, dim=0)
                    batch = torch.permute(batch, (0, 2, 1, 3, 4))
                    batch = batch.to(device)
                    # print(batch.shape())
                    output = model(batch)
                    output = output.cpu()
                    output = np.where(output > 0.5, 1, 0)
                    # print(output)
                    # pyscript.write('Val',output)
            # Clear the list to store new frames
                    frames = []
    cap.release()
# Closes all the frames
    return
