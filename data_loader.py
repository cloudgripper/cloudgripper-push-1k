import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

class TrainingDataset(Dataset):
    def __init__(self, video_data, position_data):
        self.video_data = video_data
        self.position_data = position_data

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, index):
        video = self.video_data[index]
        position = self.position_data[index]
        return video, position
    
def tensor_loader(training_path, MAP_ENABLE = False):
    train_video_dir = os.path.join(training_path, 'Video')
    train_trajectory_dir = os.path.join(training_path, 'Position')

    video_files = sorted(os.listdir(train_video_dir))
    position_files = sorted(os.listdir(train_trajectory_dir))

    assert len(video_files) == len(position_files), "Mismatch in the number of videos and position files."

    train_videos = [] 
    train_object_pos = []    

    for video_file, position_file in zip(video_files, position_files):
        assert video_file.split('.')[0] == position_file.split('.')[0], f"Mismatch in file names: {video_file}, {position_file}"

        video_path = os.path.join(train_video_dir, video_file)
        pos_path = os.path.join(train_trajectory_dir, position_file)

        cap = cv2.VideoCapture(video_path)

        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (224, 224))
            if frame_count % 2 == 0:  
                frames.append(frame_resized)
            frame_count += 1
        cap.release()
        video_tensor = torch.stack([torch.from_numpy(frame) for frame in frames])
        train_videos.append(video_tensor)

        stacked_poses = []
        stacked_cnt = 0
        with open(pos_path, 'r') as f:
            train_object_pos_tmp = []  
            for line in f:
                if stacked_cnt % 2 == 0:
                    values = line.strip()[1:-1].split()
                    values = [float(value) for value in values]
                    values[0] = values[0]/480
                    values[1] = values[1]/640
                    pos_tensor = torch.tensor(values)
                    train_object_pos_tmp.append(pos_tensor)
                stacked_cnt += 1 
            stacked_poses = map_generator(train_object_pos_tmp)
        train_object_pos.append(stacked_poses)
    return TrainingDataset(train_videos, train_object_pos)

def map_generator(train_object_pos_tmp):
    map_tensor = torch.zeros(16, 12)
    for i in range(len(train_object_pos_tmp)):
        x = train_object_pos_tmp[i][0] 
        y = train_object_pos_tmp[i][1]
        image_x = int(y * map_tensor.shape[0])
        image_y = int(x * map_tensor.shape[1])
        map_tensor[image_x, image_y] = 1
    return map_tensor

        