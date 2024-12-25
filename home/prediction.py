import numpy as np
import base64
import cv2
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
import os
import time
from .models import UploadedVideos
import allauth
from django.conf import settings
BASE_DIR = settings.BASE_DIR
import imageio
from io import BytesIO  
import base64
import requests


CONSECUTIVE_VIOLENT_FRAMES_THRESHOLD = 10
ALERT_COOLDOWN_DURATION = 60  # in seconds
ALERT_SOUND_COOLDOWN_DURATION = 10  # in seconds
consecutive_violent_frames = 0
last_alert_time = 0
last_alert_sound_time = 0



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
    rf"{BASE_DIR}\DenseNet_state.pt",
     map_location=(torch.device('cpu')))
model.load_state_dict(model_state['model_state_dict'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


def Pred(videoObj):
    dirName = f"violent_{videoObj.id}"                 
    fullPath = os.path.join(settings.MEDIA_ROOT, dirName)
    i = 0 #just for testing
    values = ['Normal', 'Violence detected']
    file_path = videoObj.video.path
    print(file_path)
    batch_size = 16
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(171),
        torchvision.transforms.CenterCrop(112),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frames = []
    pil_frames = []
    timestamps = []  # Added to store timestamps
    cap = cv2.VideoCapture(file_path)

    totalDuration = videoObj.duration
    while (cap.isOpened()):
        # time.sleep(1)
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame. Exiting loop.")
            break
    # Transform the frame
        if ret == True:
            pil_frame = Image.fromarray(frame)
            pil_frames.append(pil_frame)
            frame = transform(pil_frame)
        
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
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
                timestamps_np = np.array(timestamps)
        # Perform inference on the batch
                with torch.no_grad():
                    # print(batch.shape)
                    batch = torch.unsqueeze(batch, dim=0)
                    batch = torch.permute(batch, (0, 2, 1, 3, 4))
                    batch = batch.to(device)
                    # print(batch.shape())
                    output = model(batch)
                    output = output.cpu()
                    # print(output)
                    output = np.where(output > 0.5, 1, 0)
                    output = output[0]
                    output = output[0]
                    values[output]
                    # violence has been detected
                    videoObj.isVoilentThisIteration = False
                    if output == 1:
                        videoObj.isVoilentThisIteration = True
                        print("violence detected")
                        print("violent timeFrames " , timestamps)
                        dirName = f"violent_{videoObj.id}"
                        videoObj.violent = True
                        videoObj.classified = True
                        fullPath = os.path.join(settings.MEDIA_ROOT, dirName)
                        if not os.path.exists(fullPath):
                            os.makedirs(fullPath)
                            videoObj.violentFramesDir = dirName
                        videoObj.save()


                        # combined_image = torch.cat(frames, dim=-1)  
                        
                        # combined_np = combined_image.cpu().numpy().transpose((1, 2, 0))  # Convert tensor to numpy
                        # combined_np = (combined_np * 255).astype(np.uint8)
                        # combined_pil = Image.fromarray(combined_np)

                        # # Save the combined image
                        # combined_image_path = os.path.join(fullPath, f"combined_image_{i}.jpg")
                        # i +=1
                        # combined_pil.save(combined_image_path)
                        # print(f"Saved the combined image at {combined_image_path}.")

                        
                        
                        for i, timestamp in enumerate(timestamps_np):
                            image_path = os.path.join(fullPath, f"frame_{timestamp}.jpg")
                            pil_frames[i].save(image_path)
                            # violent_frames.append(image_path)
                            print(f"Saved frame {i} at timestamp {timestamp} seconds.")



                    print("timestamp" ,timestamps[15])
                    print("total duration" ,totalDuration)
                    progress = (timestamps[15] / totalDuration) * 100
                    timestamps = []
                    pil_frames = []
                    frames = []
                    print('progress = ',progress)
                    videoObj.progress  = progress
                    videoObj.save()
                    print("iteration done")
    cap.release()
    videoObj.classified = True
    videoObj.save()
    if videoObj.violent:
        create_video(fullPath,f'{fullPath}\\violent.mp4',30)
    print("finished",videoObj.classified)
    videoObj.progress  = 100.00
    videoObj.save()

# Closes all the frames

    return



def livePred(imageData):
    values = ['Normal', 'Violence detected']
    violent_frames = [] 
    if len(imageData) != 16:
        return

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(171),
        torchvision.transforms.CenterCrop(112),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frames = []
    for img in imageData:
        data_parts = img.split(',')
        base64_data = data_parts[1]
        binary_data = base64.b64decode(base64_data)
        img_io = BytesIO(binary_data)
        frame = Image.open(img_io)
        finalFrame = transform(frame)
        frames.append(finalFrame)
        # print('frame',img)
    if len(frames) == 16:
        batch = torch.stack(frames, dim=0)
        with torch.no_grad():
            batch = torch.unsqueeze(batch, dim=0)
            batch = torch.permute(batch, (0, 2, 1, 3, 4))
            batch = batch.to(device)
            output = model(batch)
            output = output.cpu()
            print(output)
            output = np.where(output > 0.5, 1, 0)
            output = output[0]
            output = output[0]
            values[output]
            if output == 1:
                return "Violent" 
                # print("violent timeFrames " , timestamps)
                # dirName = f"violent_{videoObj.id}"
                # videoObj.violent = True
                # videoObj.classified = True
                # fullPath = os.path.join(settings.MEDIA_ROOT, dirName)
                # if not os.path.exists(fullPath):
                #     os.makedirs(fullPath)
                #     videoObj.violentFramesDir = dirName
                # #     videoObj.save()
            return "normal"
                    
                        
                        # for i, timestamp in enumerate(timestamps_np):
                        #     image_path = os.path.join(fullPath, f"frame_{timestamp}.jpg")
                        #     pil_frames[i].save(image_path)
                        #     # violent_frames.append(image_path)
                        #     print(f"Saved frame {i} at timestamp {timestamp} seconds.")


# iterate over the lines and replace the value of myVariable
           

# write the modified lines back to the file
            
            # Clear the list to store new frames
                    # print("Timestamps:", timestamps_np)
                    # timestamps = []
                    # pil_frames = []
                    # frames = []
                    # print("iteration done")
    
#     while (cap.isOpened()):
#         # time.sleep(1)
#         ret, frame = cap.read()
#         if not ret:
#             print("Error reading frame. Exiting loop.")
#             break
#     # Transform the frame
#         if ret == True:
#             pil_frame = Image.fromarray(frame)
#             pil_frames.append(pil_frame)
#             frame = transform(pil_frame)
#             timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
#             frames.append(frame)

#             if len(frames) == 16:
#                 batch = torch.stack(frames, dim=0)
#                 timestamps_np = np.array(timestamps)
#                 with torch.no_grad():
#                     batch = torch.unsqueeze(batch, dim=0)
#                     batch = torch.permute(batch, (0, 2, 1, 3, 4))
#                     batch = batch.to(device)
#                     output = model(batch)
#                     output = output.cpu()
#                     output = np.where(output > 0.5, 1, 0)
#                     output = output[0]
#                     output = output[0]
#                     values[output]
#                     # if output == 1:
#                     #     print("violence detected")
#                     #     print("violent timeFrames " , timestamps)
#                     #     dirName = f"violent_{videoObj.id}"
#                     #     videoObj.violent = True
#                     #     videoObj.classified = True
#                     #     fullPath = os.path.join(settings.MEDIA_ROOT, dirName)
#                     #     if not os.path.exists(fullPath):
#                     #         os.makedirs(fullPath)
#                     #         videoObj.violentFramesDir = dirName
#                     #     videoObj.save()

                        
                        
#                         # for i, timestamp in enumerate(timestamps_np):
#                         #     image_path = os.path.join(fullPath, f"frame_{timestamp}.jpg")
#                         #     pil_frames[i].save(image_path)
#                         #     # violent_frames.append(image_path)
#                         #     print(f"Saved frame {i} at timestamp {timestamp} seconds.")


# # iterate over the lines and replace the value of myVariable
           

# # write the modified lines back to the file
            
#             # Clear the list to store new frames
#                     # print("Timestamps:", timestamps_np)
#                     timestamps = []
#                     pil_frames = []
#                     frames = []
#                     print("iteration done")
#     # cap.release()
    # videoObj.classified = True
    # videoObj.save()
    # if videoObj.violent:
    #     create_video(fullPath,f'{fullPath}\\violent.mp4',30)
    # print("finished",videoObj.classified)
# Closes all the frames

    return "this"


def create_video_from_frames(output_folder, frames):
    output_path = os.path.join(output_folder, "violent.mp4")
    framerate = 90  # Set the framerate for the video
    create_video(frames, output_path, framerate)

# def create_video_from_frames(imageData, frames):
#     dirName = f"violent_{imageData.id}"                 
#     fullPath = os.path.join(settings.MEDIA_ROOT, dirName)
#     output_path = "f'{fullPath}\\violent.mp4"
#     framerate = 90  # Set the framerate for the video
#     create_video(frames, output_path, framerate)

def create_video(image_folder, output_path, framerate):
    images = []

    # Sort the files based on their names
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])

    for filename in image_files:
        file_path = os.path.join(image_folder, filename)
        images.append(imageio.imread(file_path))

    imageio.mimsave(output_path, images, fps=framerate)


    # we can add functionality to delete all these frames 
def predict_single_frame(image_data_url):
    # Extract the base64-encoded image data from the data URL
    _, encoded_data = image_data_url.split(',', 1)
    image_data = base64.b64decode(encoded_data)

    # Convert the image data to a NumPy array
    image_array = np.frombuffer(image_data, np.uint8)
    try:
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error during image decoding: {e}")
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Perform the same transformations as in your original function
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(171),
        torchvision.transforms.CenterCrop(112),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pil_frame = Image.fromarray(img)
    frame = transform(pil_frame)

    # Expand dimensions to match the batch size expected by the model
    frame = frame.unsqueeze(0).unsqueeze(0)  # Add batch_size and channels dimensions

    # Perform inference on the single frame
    with torch.no_grad():
        frame = frame.permute(0, 1, 2, 3).to(device)
        output = model(frame)
        output = output.cpu()
        output = np.where(output > 0.5, 1, 0)
        output = output[0, 0]

    # Interpret the output and return the result
    result = 'Violence detected' if output == 1 else 'Normal'
    return result
