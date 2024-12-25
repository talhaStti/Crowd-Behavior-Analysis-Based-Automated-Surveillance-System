import time
import queue
from PIL import Image
import torch
import torchvision
import winsound
import cv2
import threading
import numpy as np
from .prediction import DenseNet
from.send_Alert import send_alert
video_buffer = queue.Queue()


def gen(user_email):
    model = DenseNet()
    model_state = torch.load(
        r"C:\Users\Sajeel Hashmi\Desktop\Crowd-Behavior-Analysis-Based-Automated-Surveillance-System-master\Crowd-Behavior-Analysis-Based-Automated-Surveillance-System-master\DenseNet_state.pt", map_location=(torch.device('cpu')))
    model.load_state_dict(model_state['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(171),
        torchvision.transforms.CenterCrop(112),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    values = ['Normal', 'Violence detected']
    img = []
    allImg = []
    cap = cv2.VideoCapture(0)
    # t_live = threading.Thread(target=collect, args=[])
    # t_live.start()
    cap.set(cv2.CAP_PROP_FPS, 30)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # video_buffer.put(frame)
        f = frame
        if not ret:
            video_buffer.queue.clear()
            break
        # _, jpeg = cv2.imencode('.jpg', frame)
        # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        f = cv2.blur(f, (7, 7))
        f = Image.fromarray(f)
        f = transform(f)
        img.append(f)
        allImg.append(f)
        if len(img) == 16:
            # p = Pred_live(img)
            batch = torch.stack(img, dim=0)
            with torch.no_grad():
                # print(batch.shape)
                batch = torch.unsqueeze(batch, dim=0)
                batch = torch.permute(batch, (0, 2, 1, 3, 4))
                batch = batch.cuda()
        # print(batch.shape())
                print("on pred function")
                output = model(batch)
                output = output.cpu()
                print("Prediction done")
                print(output)
                output = np.where(output > 0.5, 1, 0)
                output = output[0]
                output = output[0]
            if values[output] == "Violence detected":
                frame = cv2.rectangle(frame, (0, 0), (720-1,
                                                      480-1), (0, 0, 255), 20)
                send_alert(user_email, frame)
            img = []
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        # frame = video_buffer.get()
        if not cap.isOpened():
            break
