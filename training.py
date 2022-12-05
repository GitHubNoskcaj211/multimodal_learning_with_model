import torch
import numpy as np
from neural_networks import multiModalRepresentation_diff, multiModalRepresentation, encoderDecoder, encoderDecoder2
from dataloader import gestureBlobDataset, gestureBlobBatchDataset, gestureBlobMultiDataset, gestureBlobDataset2, size_collate_fn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
from datetime import datetime
from typing import Tuple, List

# Use subset sampler for train test split

def calc_labels(y: torch.Tensor) -> torch.Tensor:
    # out = torch.ones(y.size()[0], 15, dtype = torch.float32)*(0.01/14)
    out = torch.zeros(y.size()[0], 15, dtype = torch.long)
    #print(out.size())
    for i in range(out.size()[0]):
        # out[i, int(y[i].item()) - 1] = 0.99
        out[i, int(y[i].item()) - 1] = 1
    return(out)

def train_encoder_decoder_embeddings(lr: float, num_epochs: int, suturing_path: str, weights_save_path: str, weight_decay: float, dataset_name: str) -> None:
    if torch.cuda.is_available():
        print('Using CUDA')

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    gesture_dataset = gestureBlobDataset2(suturing_path = suturing_path)
    dataloader = DataLoader(dataset = gesture_dataset, batch_size = 128, shuffle = False, collate_fn = size_collate_fn)

    loss_function = torch.nn.L1Loss()
    # loss_function = torch.nn.KLDivLoss()
    net = encoderDecoder2(num_lstm_layers = 2, embedding_dim = 512)
    net = net.train()
    if torch.cuda.is_available():
        net.cuda()

    # Remove comment to initialized at a pretrained point
    # net.load_state_dict(torch.load(os.path.join(weights_save_path,'two_stream_Knot_Tying_2020-02-01_20:16:16.pth' )))

    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr, weight_decay = weight_decay)
    
    for epoch in range(num_epochs):
        running_loss = 0
        count = 0
        print('Epoch {}'.format(epoch + 1))
        for data in dataloader:
            optimizer.zero_grad()
            input_sequence, target_sequence = data
            input_sequence, target_sequence = input_sequence.squeeze(), target_sequence.squeeze()
            for i in range(input_sequence.shape[0]):
                # input = input_sequence[i].unsqueeze(0).float()
                input = input_sequence[i].unsqueeze(0)
                target = target_sequence[i].unsqueeze(0)
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda() 
                out1 = net(input)
                loss = loss_function(out1, target)
                loss.backward()
                # print('Current loss = {}'.format(loss.item()))
                running_loss += loss.item()
                count += 1
            optimizer.step()
        with open("kin_to_kin_training.txt", "a") as f:
            f.write('\n Epoch: {}, Loss: {}'.format(epoch + 1, running_loss/count))
        print('\n Epoch: {}, Loss: {}'.format(epoch + 1, running_loss/count))

    print('Finished training.')
    print('Saving state dict.')

    file_name = 'multimodal_kin_to_kin.pth'
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    # print('State dict saved at timestamp {}'.format(now))

def main():
    blobs_folder_path = '../jigsaw_dataset/Knot_Tying/blobs'
    lr = 1e-3
    num_epochs = 1000
    weights_save_path = './weights_save'
    weight_decay = 1e-8
    dataset_name = 'Knot_Tying'

    blobs_folder_paths_list = ['../jigsaw_dataset/Knot_Tying/blobs', '../jigsaw_dataset/Needle_Passing/blobs', '../jigsaw_dataset/Suturing/blobs']

    # train_multimodal_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_path = blobs_folder_path, weights_save_path = weights_save_path, weight_decay = weight_decay)
    # train_encoder_decoder_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_path = blobs_folder_path, weights_save_path = weights_save_path, weight_decay = weight_decay, dataset_name = dataset_name)
    train_encoder_decoder_multidata_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_paths_list = blobs_folder_paths_list, weights_save_path = weights_save_path, weight_decay = weight_decay)

if __name__ == '__main__':
    main()
