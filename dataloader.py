import numpy as np
import torch
import os
import pickle
from typing import Tuple, List
import cv2
import torchvision.transforms as transforms
import torchvision
import random
from torch.utils.data.dataloader import default_collate

def create_data_blobs(optical_flow_folder_path: str, transcriptions_folder_path: str, kinematics_folder_path: str, num_frames_per_blob: int, blobs_save_folder_path: str, spacing: int) -> None:
    if not os.path.exists(blobs_save_folder_path):
        os.makedirs(blobs_save_folder_path)
    
    blob_count = 0

    for file in os.listdir(transcriptions_folder_path):
        try:
            curr_file_path = os.path.join(transcriptions_folder_path, file)

            print('Processing file: {}'.format(curr_file_path.split('/')[-1]))

            curr_optical_flow_file = '_'.join([file.split('.')[0], 'capture1_resized.p'])
            curr_optical_flow_file = os.path.join(optical_flow_folder_path, curr_optical_flow_file)
            optical_flow_file = pickle.load(open(curr_optical_flow_file, 'rb'))

            curr_kinematics_file = '.'.join([file.split('.')[0], 'txt'])
            curr_kinematics_file = os.path.join(kinematics_folder_path, curr_kinematics_file)
            kinematics_list = []

            with open (curr_kinematics_file) as kf:
                for line in kf:
                    kinematics_list.append([float(v) for v in line.strip('\n').strip().split('     ')])
                kf.close()

            with open(curr_file_path, 'r') as f:
                for line in f:
                    line = line.strip('\n').strip()
                    line = line.split(' ')
                    start = int(line[0])
                    end = int(line[1])
                    gesture = line[2]
                    curr_blob = [torch.tensor(v) for v in optical_flow_file[start: start + spacing*num_frames_per_blob : spacing]]
                    curr_blob = torch.cat(curr_blob, dim = 2).permute(2, 0, 1)
                    curr_kinematics_blob = [torch.tensor(v).view(1, 76) for v in kinematics_list[start: start + spacing*num_frames_per_blob: spacing]]
                    curr_kinematics_blob = torch.stack(curr_kinematics_blob, dim = 0)
                    save_tuple = (curr_blob, curr_kinematics_blob)
                    curr_blob_save_path = 'blob_' + str(blob_count) + '_video_' + curr_file_path.split('/')[-1].split('.')[0].split('_')[-1] + '_gesture_' + gesture + '.p'
                    curr_blob_save_path = os.path.join(blobs_save_folder_path, curr_blob_save_path)
                    pickle.dump(save_tuple, open(curr_blob_save_path, 'wb'))

                    blob_count += 1
        except:
            pass

class gestureBlobDataset:
    def __init__(self, suturing_path: str) -> None:
        self.images_folder_path = suturing_path + 'video/'
        self.kinematics_folder_path = suturing_path + 'kinematics/AllGestures/'
        self.images_folder = os.listdir(self.images_folder_path)
        self.kinematics_folder = os.listdir(self.kinematics_folder_path)
        self.fps = 30
        self.min_frames = self.fps * 45
        self.max_frames = self.fps * 60
        self.kinematics_pred_frame_diff = self.fps * 1

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        idx = random.randint(0, len(self.kinematics_folder) - 1)
        curr_file_path = self.images_folder[idx * 2 + 1]
        curr_file_path = os.path.join(self.images_folder_path, curr_file_path)
        curr_kinematics_path = self.kinematics_folder[idx] 
        curr_kinematics_path = os.path.join(self.kinematics_folder_path, curr_kinematics_path) 
        print("Curr_File_Path: ",curr_file_path)
        video = torchvision.io.read_video(curr_file_path)[0]
        num_frames = 0
        if video.shape[0] < self.min_frames:
            num_frames = video.shape[0]
        elif video.shape[0] < self.max_frames:
            num_frames = random.randint(self.min_frames, video.shape[0])
        else:
            num_frames = random.randint(self.min_frames, self.max_frames)
        # num_frames = 30
        start_frame = random.randint(0, video.shape[0] - num_frames - self.kinematics_pred_frame_diff)
        video = video[start_frame : start_frame + num_frames]
        video = torch.permute(video, [0, 3, 1, 2])
        kinematics = torch.empty(size = [num_frames, 12*4])
        with open(curr_kinematics_path, "r") as f:
            for i, line in enumerate(f):
                if i >= start_frame + self.kinematics_pred_frame_diff and i < start_frame + self.kinematics_pred_frame_diff + num_frames:
                    line_nums = line.strip().split('     ')
                    if len(line_nums) < 76:
                        raise Exception("Not enough kinematic numbers")
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][0] = float(line_nums[0])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][1] = float(line_nums[1])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][2] = float(line_nums[2])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][3] = float(line_nums[3])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][4] = float(line_nums[4])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][5] = float(line_nums[5])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][6] = float(line_nums[6])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][7] = float(line_nums[7])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][8] = float(line_nums[8])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][9] = float(line_nums[9])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][10] = float(line_nums[10])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][11] = float(line_nums[11])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][12] = float(line_nums[0+19])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][13] = float(line_nums[1+19])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][14] = float(line_nums[2+19])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][15] = float(line_nums[3+19])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][16] = float(line_nums[4+19])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][17] = float(line_nums[5+19])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][18] = float(line_nums[6+19])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][19] = float(line_nums[7+19])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][20] = float(line_nums[8+19])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][21] = float(line_nums[9+19])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][22] = float(line_nums[10+19])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][23] = float(line_nums[11+19])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][24] = float(line_nums[0+38])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][25] = float(line_nums[1+38])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][26] = float(line_nums[2+38])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][27] = float(line_nums[3+38])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][28] = float(line_nums[4+38])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][29] = float(line_nums[5+38])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][30] = float(line_nums[6+38])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][31] = float(line_nums[7+38])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][32] = float(line_nums[8+38])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][33] = float(line_nums[9+38])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][34] = float(line_nums[10+38])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][35] = float(line_nums[11+38])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][36] = float(line_nums[0+57])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][37] = float(line_nums[1+57])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][38] = float(line_nums[2+57])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][39] = float(line_nums[3+57])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][40] = float(line_nums[4+57])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][41] = float(line_nums[5+57])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][42] = float(line_nums[6+57])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][43] = float(line_nums[7+57])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][44] = float(line_nums[8+57])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][45] = float(line_nums[9+57])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][46] = float(line_nums[10+57])
                    kinematics[i-self.kinematics_pred_frame_diff-start_frame][47] = float(line_nums[11+57])                
        return video, kinematics
    
class gestureBlobDataset2:
    def __init__(self, suturing_path: str) -> None:
        self.images_folder_path = suturing_path + 'video/'
        self.kinematics_folder_path = suturing_path + 'kinematics/AllGestures/'
        self.kinematics_folder = os.listdir(self.kinematics_folder_path)
        self.fps = 30
        self.num_frames = self.fps * 45
        self.kinematics_pred_frame_diff = self.fps * 1

    def __len__(self) -> int:
        return len(self.kinematics_folder)

    def __getitem__(self, idx: int) -> torch.Tensor:
        curr_kinematics_path = self.kinematics_folder[idx] 
        curr_kinematics_path = os.path.join(self.kinematics_folder_path, curr_kinematics_path) 
        
        count = 0
        with open(curr_kinematics_path, "r") as f:
            for count, line in enumerate(f):
                pass
                
        # num_frames = 30
        start_frame = random.randint(0, count - self.num_frames - self.kinematics_pred_frame_diff)
        kinematics_in = torch.empty(size = [self.num_frames, 12*4])
        kinematics_out = torch.empty(size = [self.num_frames, 12*4])
        with open(curr_kinematics_path, "r") as f:
            for i, line in enumerate(f):
                if i >= start_frame and i < start_frame + self.num_frames:
                    line_nums = line.strip().split('     ')
                    if len(line_nums) < 76:
                        raise Exception("Not enough kinematic numbers")
                    kinematics_in[i-start_frame][0] = float(line_nums[0])
                    kinematics_in[i-start_frame][1] = float(line_nums[1])
                    kinematics_in[i-start_frame][2] = float(line_nums[2])
                    kinematics_in[i-start_frame][3] = float(line_nums[3])
                    kinematics_in[i-start_frame][4] = float(line_nums[4])
                    kinematics_in[i-start_frame][5] = float(line_nums[5])
                    kinematics_in[i-start_frame][6] = float(line_nums[6])
                    kinematics_in[i-start_frame][7] = float(line_nums[7])
                    kinematics_in[i-start_frame][8] = float(line_nums[8])
                    kinematics_in[i-start_frame][9] = float(line_nums[9])
                    kinematics_in[i-start_frame][10] = float(line_nums[10])
                    kinematics_in[i-start_frame][11] = float(line_nums[11])
                    kinematics_in[i-start_frame][12] = float(line_nums[0+19])
                    kinematics_in[i-start_frame][13] = float(line_nums[1+19])
                    kinematics_in[i-start_frame][14] = float(line_nums[2+19])
                    kinematics_in[i-start_frame][15] = float(line_nums[3+19])
                    kinematics_in[i-start_frame][16] = float(line_nums[4+19])
                    kinematics_in[i-start_frame][17] = float(line_nums[5+19])
                    kinematics_in[i-start_frame][18] = float(line_nums[6+19])
                    kinematics_in[i-start_frame][19] = float(line_nums[7+19])
                    kinematics_in[i-start_frame][20] = float(line_nums[8+19])
                    kinematics_in[i-start_frame][21] = float(line_nums[9+19])
                    kinematics_in[i-start_frame][22] = float(line_nums[10+19])
                    kinematics_in[i-start_frame][23] = float(line_nums[11+19])
                    kinematics_in[i-start_frame][24] = float(line_nums[0+38])
                    kinematics_in[i-start_frame][25] = float(line_nums[1+38])
                    kinematics_in[i-start_frame][26] = float(line_nums[2+38])
                    kinematics_in[i-start_frame][27] = float(line_nums[3+38])
                    kinematics_in[i-start_frame][28] = float(line_nums[4+38])
                    kinematics_in[i-start_frame][29] = float(line_nums[5+38])
                    kinematics_in[i-start_frame][30] = float(line_nums[6+38])
                    kinematics_in[i-start_frame][31] = float(line_nums[7+38])
                    kinematics_in[i-start_frame][32] = float(line_nums[8+38])
                    kinematics_in[i-start_frame][33] = float(line_nums[9+38])
                    kinematics_in[i-start_frame][34] = float(line_nums[10+38])
                    kinematics_in[i-start_frame][35] = float(line_nums[11+38])
                    kinematics_in[i-start_frame][36] = float(line_nums[0+57])
                    kinematics_in[i-start_frame][37] = float(line_nums[1+57])
                    kinematics_in[i-start_frame][38] = float(line_nums[2+57])
                    kinematics_in[i-start_frame][39] = float(line_nums[3+57])
                    kinematics_in[i-start_frame][40] = float(line_nums[4+57])
                    kinematics_in[i-start_frame][41] = float(line_nums[5+57])
                    kinematics_in[i-start_frame][42] = float(line_nums[6+57])
                    kinematics_in[i-start_frame][43] = float(line_nums[7+57])
                    kinematics_in[i-start_frame][44] = float(line_nums[8+57])
                    kinematics_in[i-start_frame][45] = float(line_nums[9+57])
                    kinematics_in[i-start_frame][46] = float(line_nums[10+57])
                    kinematics_in[i-start_frame][47] = float(line_nums[11+57])
                if i >= start_frame + self.kinematics_pred_frame_diff and i < start_frame + self.kinematics_pred_frame_diff + self.num_frames:
                    line_nums = line.strip().split('     ')
                    if len(line_nums) < 76:
                        raise Exception("Not enough kinematic numbers")
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][0] = float(line_nums[0])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][1] = float(line_nums[1])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][2] = float(line_nums[2])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][3] = float(line_nums[3])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][4] = float(line_nums[4])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][5] = float(line_nums[5])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][6] = float(line_nums[6])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][7] = float(line_nums[7])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][8] = float(line_nums[8])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][9] = float(line_nums[9])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][10] = float(line_nums[10])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][11] = float(line_nums[11])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][12] = float(line_nums[0+19])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][13] = float(line_nums[1+19])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][14] = float(line_nums[2+19])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][15] = float(line_nums[3+19])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][16] = float(line_nums[4+19])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][17] = float(line_nums[5+19])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][18] = float(line_nums[6+19])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][19] = float(line_nums[7+19])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][20] = float(line_nums[8+19])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][21] = float(line_nums[9+19])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][22] = float(line_nums[10+19])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][23] = float(line_nums[11+19])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][24] = float(line_nums[0+38])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][25] = float(line_nums[1+38])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][26] = float(line_nums[2+38])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][27] = float(line_nums[3+38])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][28] = float(line_nums[4+38])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][29] = float(line_nums[5+38])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][30] = float(line_nums[6+38])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][31] = float(line_nums[7+38])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][32] = float(line_nums[8+38])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][33] = float(line_nums[9+38])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][34] = float(line_nums[10+38])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][35] = float(line_nums[11+38])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][36] = float(line_nums[0+57])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][37] = float(line_nums[1+57])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][38] = float(line_nums[2+57])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][39] = float(line_nums[3+57])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][40] = float(line_nums[4+57])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][41] = float(line_nums[5+57])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][42] = float(line_nums[6+57])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][43] = float(line_nums[7+57])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][44] = float(line_nums[8+57])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][45] = float(line_nums[9+57])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][46] = float(line_nums[10+57])
                    kinematics_out[i-self.kinematics_pred_frame_diff-start_frame][47] = float(line_nums[11+57])                
        return kinematics_in, kinematics_out

class gestureBlobBatchDataset:
    def __init__(self, gesture_dataset: gestureBlobDataset, random_tensor: str = 'random') -> None:
        self.gesture_dataset = gesture_dataset
        self.random_tensor = random_tensor
    
    def __len__(self) -> None:
        return(len(self.gesture_dataset))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        curr_tensor = self.gesture_dataset.__getitem__(idx)
        
        if self.random_tensor == 'random':
            rand_idx = np.random.randint(low = 0, high = len(self.gesture_dataset))
        
        elif self.random_tensor == 'next':
            if idx != len(self.gesture_dataset) - 1:
                rand_idx = idx + 1
            else:
                rand_idx = idx - 1
        else:
            raise ValueError('Value of random_tensor should be "random" or "next".')
        
        random_tensor = self.gesture_dataset.__getitem__(rand_idx)

        y_match = torch.tensor([1, 0], dtype = torch.float32).view(1, 2)
        if idx != rand_idx:
            y_rand = torch.tensor([0, 1], dtype = torch.float32).view(1, 2)
        else:
            y_rand = torch.tensor([1, 0], dtype = torch.float32).view(1, 2)

        return((curr_tensor, random_tensor, y_match, y_rand))

class gestureBlobMultiDataset:
    def __init__(self, blobs_folder_paths_list: List[str]) -> None:
        self.blobs_folder_paths_list = blobs_folder_paths_list
        self.blobs_folder_dict = {path: [] for path in self.blobs_folder_paths_list}
        for path in self.blobs_folder_paths_list:
            self.blobs_folder_dict[path] = os.listdir(path)
            self.blobs_folder_dict[path] = list(filter(lambda x: '.DS_Store' not in x, self.blobs_folder_dict[path]))
            self.blobs_folder_dict[path].sort(key = lambda x: int(x.split('_')[1]))
        
        self.dir_lengths = [len(os.listdir(path)) for path in self.blobs_folder_paths_list]
        for i in range(1, len(self.dir_lengths)):
            self.dir_lengths[i] += self.dir_lengths[i - 1]

    def __len__(self) -> int:
        return(self.dir_lengths[-1])

    def __getitem__(self, idx: int) -> torch.Tensor:
        dir_idx = 0
        while idx >= self.dir_lengths[dir_idx]:
            dir_idx += 1
        adjusted_idx = idx - self.dir_lengths[dir_idx]
        path = self.blobs_folder_paths_list[dir_idx]

        curr_file_path = self.blobs_folder_dict[path][adjusted_idx]
        curr_file_path = os.path.join(path, curr_file_path)
        curr_tensor_tuple = pickle.load(open(curr_file_path, 'rb'))
        # print(curr_tensor_tuple[0].size())
        if curr_tensor_tuple[0].size()[0] == 50:
            return(curr_tensor_tuple)
        else:
            return(None)

def size_collate_fn(batch: torch.Tensor) -> torch.Tensor:
    batch = list(filter(lambda x: x is not None, batch))
    return(default_collate(batch))

def main():
    optical_flow_folder_path = '../jigsaw_dataset/Knot_Tying/optical_flow/'
    transcriptions_folder_path = '../jigsaw_dataset/Knot_Tying/transcriptions'
    num_frames_per_blob = 25
    blobs_save_folder_path = '../jigsaw_dataset/Knot_Tying/blobs'
    spacing = 2
    kinematics_folder_path = '../jigsaw_dataset/Knot_Tying/kinematics/AllGestures/'

    # create_data_blobs(optical_flow_folder_path = optical_flow_folder_path, transcriptions_folder_path = transcriptions_folder_path, kinematics_folder_path = kinematics_folder_path, num_frames_per_blob = num_frames_per_blob, blobs_save_folder_path = blobs_save_folder_path, spacing = spacing)

    blobs_folder_paths_list = ['../jigsaw_dataset/Knot_Tying/blobs/', '../jigsaw_dataset/Needle_Passing/blobs/', '../jigsaw_dataset/Suturing/blobs/']
    # dataset = gestureBlobDataset(blobs_folder_path = '../jigsaw_dataset/Knot_Tying/blobs/')
    dataset = gestureBlobMultiDataset(blobs_folder_paths_list = blobs_folder_paths_list)
    out = dataset.__getitem__(3)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()



