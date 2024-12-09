import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch import nn
from torchvision import models
import face_recognition
import numpy as np
import os
import cv2
from itertools import islice
from tqdm import tqdm
import dlib
from multiprocessing import Pool

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")

im_size = 112

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

sequence_length = 120

sm = nn.Softmax()
inv_normalize = transforms.Normalize(
    mean=-1*np.divide(mean, std), std=np.divide([1, 1, 1], std))

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])


def get_face_frame(frame):
    faces = face_recognition.face_locations(frame)
    top, right, bottom, left = faces[0]
    return frame[top:bottom, left:right, :]

# detector = dlib.get_frontal_face_detector()

# def get_face_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     faces = detector(gray)
    
#     if len(faces) > 0:
#         top, right, bottom, left = (faces[0].top(), faces[0].right(), faces[0].bottom(), faces[0].left())
#         return frame[top:bottom, left:right, :]
#     else:
#         return frame

# def process_frames_parallel(frames):
#     # Use multiple cores to process the frames in parallel
#     with Pool(processes=None) as pool:
#         return pool.map(get_face_frame, frames)

class create_dataset_and_predict(Dataset):
    def __init__(self, video_names, model, sequence_length=60, transform=None, use_faces=True):
        print(f"Using sequence length {sequence_length}")
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
        self.use_faces = use_faces
        self.model = model

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        for i, frame in tqdm(enumerate(self.frame_extract(video_path)), total=sequence_length, desc="Processing..."):
            if self.use_faces:
                try:
                    frame = get_face_frame(frame)
                except:
                    pass

            frame_tensor = self.transform(frame).unsqueeze(0)
            output, confidence = predict_single_frame(self.model, frame_tensor)
            tqdm.write(
                f"Frame {i + 1}: {output}, Confidence: {confidence:.1f}%")
            cv2.imwrite(
                f"./predictions/{i+1}_{output}_{confidence:.1f}%.jpg", frame)
            frames.append(self.transform(frame))
            if (len(frames) == self.count):
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        print(f"Extracted {len(frames)}")
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image


class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, gru_layers=2, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights='DEFAULT')
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.gru = nn.GRU(latent_dim, hidden_dim,
                          gru_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_gru, _ = self.gru(x)

        return fmap, self.dp(self.linear1(torch.mean(x_gru, dim=1)))


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image


def predict(model, img):
    _, logits = model(img)
    img = im_convert(img[:, -1, :, :, :])
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item()*100
    print('Confidence of prediction:', confidence)
    return [int(prediction.item()), confidence]


def predict_single_frame(model, frame):
    frame = frame.unsqueeze(0).to(device)
    _, logits = model(frame)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    output = "REAL" if prediction.item() == 1 else "FAKE"
    return output, confidence


def get_model(path_to_model):
    print(f"Using {device}")
    if (device == "cuda"):
        model = Model(2).cuda()
    else:
        model = Model(2).cpu()
    model.load_state_dict(torch.load(
        path_to_model, map_location=torch.device(device), weights_only=True))

    return model


def get_dataset(video_list, sequence_length, use_faces, model):
    video_dataset = create_dataset_and_predict(video_list, model=model,
                                               sequence_length=sequence_length, transform=train_transforms, use_faces=use_faces)

    return video_dataset


def predict_video(model, video_dataset):
    prediction = predict(model, video_dataset[0].to(device))
    confidence = round(prediction[1], 1)
    output = "REAL" if prediction[0] == 1 else "FAKE"
    return prediction[0], confidence, output


def predict_img_df(img_path):
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = probabilities.argmax().item()
    
    return 'Deepfake image' if predicted_class else 'Real image'