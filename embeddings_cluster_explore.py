import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from dataloader import gestureBlobDataset, size_collate_fn
from neural_networks import encoderDecoder2, encoderDecoder2
import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import umap.umap_ as umap
import seaborn as sns
import matplotlib.pyplot as plt
from multipledispatch import dispatch
from typing import List, Tuple
from tqdm import tqdm
from barbar import Bar
from joblib import dump, load
import re
import json
import numpy as np
import torch
from typing import Tuple
import torch.nn as nn
import random

def store_embeddings_in_dict(blobs_folder_path: str, model: encoderDecoder2) -> dict:
    blobs_folder = os.listdir(blobs_folder_path)
    embeddings_list = []
    gestures_list = []
    user_list = []
    skill_dict = {'B': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 1, 'G': 0, 'H': 0, 'I': 0}
    skill_list = []
    file_list = []
    
    model.eval()

    for file in blobs_folder:
        print('Processing file {}'.format(file))

        curr_kinematics_path = os.path.join(blobs_folder_path, file)
        count = 0
        with open(curr_kinematics_path, "r") as f:
            for count, line in enumerate(f):
                pass
                
        num_frames = 30 * 45
        kinematics_pred_frame_diff = 30 * 1
        start_frame = random.randint(0, count - num_frames - kinematics_pred_frame_diff)
        kinematics_in = torch.empty(size = [num_frames, 12*4])
        with open(curr_kinematics_path, "r") as f:
            for i, line in enumerate(f):
                if i >= start_frame and i < start_frame + num_frames:
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
        try:
            kinematics_in = kinematics_in.unsqueeze(0)
            out = model.encoder(kinematics_in)
            out = out.cpu().detach().data.numpy()
            embeddings_list.append(out)

            file_list.append(file)
            file = file.split('_')
            gestures_list.append(file[-1].split('.')[0])
            user_list.append(file[-1].split('.')[0][0])
            skill_list.append(skill_dict[file[-1].split('.')[0][0]])
        except:
            print("PROBLEM")
            pass


    print("Gesture len: ",len(gestures_list))
    print("User len: ",len(user_list))
    print("Skill len: ",len(skill_list))
    print("Embedding len: ",len(embeddings_list))
    print("File len: ",len(file_list))
    final_dict = {'gesture': gestures_list, 'user': user_list, 'skill': skill_list, 'embeddings': embeddings_list, 'file_list': file_list}
    
    return(final_dict)

def cluster_statistics(blobs_folder_path: str, model: encoderDecoder2, num_clusters: int) -> pd.DataFrame:
    results_dict = store_embeddings_in_dict(blobs_folder_path = blobs_folder_path, model = model)
    k_means = KMeans(n_clusters = num_clusters)
    print("The shape: ",np.array(results_dict['embeddings']).shape)
    cluster_indices = k_means.fit_predict(np.array(results_dict['embeddings']).reshape(-1,512*1350))
    results_dict['cluster_indices'] = cluster_indices
    print("Cluster indices: ",cluster_indices.shape)
    df = pd.DataFrame(results_dict)
    return(df)

def cluster_statistics_multidata(blobs_folder_paths_list: List[str], model: encoderDecoder2, num_clusters: int) -> pd.DataFrame:
    results_dict = {'gesture': [], 'user': [], 'skill': [], 'embeddings': [], 'task': []}
    for idx, path in enumerate(blobs_folder_paths_list):
        temp_results_dict = store_embeddings_in_dict(blobs_folder_path = path, model = model)
        # import pdb; pdb.set_trace()
        temp_results_dict['task'] = [idx]*len(temp_results_dict['skill'])
        for key, value in temp_results_dict.items():
            results_dict[key].extend(value)
    k_means = KMeans(n_clusters = num_clusters)
    cluster_indices = k_means.fit_predict(np.array(results_dict['embeddings']).reshape(-1, 512))
    results_dict['cluster_indices'] = cluster_indices
    print("Results dict: ",results_dict)
    df = pd.DataFrame(results_dict)
    return(df)

def evaluate_model(blobs_folder_path: str, model: encoderDecoder2, num_clusters: int, save_embeddings: bool) -> None:
    df = cluster_statistics(blobs_folder_path = blobs_folder_path, model = model, num_clusters = num_clusters)
    if save_embeddings:
        print('Saving dataframe.')
        df.to_pickle('./df.p')
    y = df['gesture'].values.ravel()
    X = [np.array(v) for v in df['embeddings']]
    X = np.array(X).reshape(-1, 512)
    classifier = XGBClassifier(n_estimators = 1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8765)
    
    classifier.fit(X_train, y_train)
    y_hat = classifier.predict(X_train)
    y_hat_test = classifier.predict(X_test)
    
    print('Training set classification report.')
    print(classification_report(y_train, y_hat))
    
    print('Test set classification report.')
    print(classification_report(y_test, y_hat_test))

def evaluate_model_multidata(blobs_folder_paths_list: str, model: encoderDecoder2, num_clusters: int, save_embeddings: bool, classifier_save_path: str = './xgboost_save/multidata_xgboost.joblib') -> None:
    df = cluster_statistics_multidata(blobs_folder_paths_list = blobs_folder_paths_list, model = model, num_clusters = num_clusters)
    if save_embeddings:
        print('Saving dataframe.')
        df.to_pickle('./df.p')
    y = df['task'].values.ravel()
    X = [np.array(v) for v in df['embeddings']]
    X = np.array(X).reshape(-1, 512)
    classifier = XGBClassifier(n_estimators = 1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5113)
    
    print('Fitting classifier.')
    classifier.fit(X_train, y_train)
    y_hat = classifier.predict(X_train)
    y_hat_test = classifier.predict(X_test)
    
    print('Training set classification report.')
    print(classification_report(y_train, y_hat))
    
    print('Test set classification report.')
    print(classification_report(y_test, y_hat_test))

    print('Saving classifier.')
    dump(classifier, classifier_save_path)
    print('Classifier saved.')

def plot_umap_clusters(blobs_folder_path: str, model: encoderDecoder2, plot_store_path: str) -> None:
    results_dict = store_embeddings_in_dict(blobs_folder_path = blobs_folder_path, model = model)
    embeddings = np.array(results_dict['embeddings']).squeeze()
    print("Embeddings size: ",embeddings.shape)
    print('Training umap reducer.')    
    umap_reducer = umap.UMAP()
    embeddings = embeddings.reshape((34,1350*512))
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    print('Generating skill plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in results_dict['skill']])
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the Skill clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_skill.png')
    plt.savefig(save_path)
    plt.clf()

    #le_gest = LabelEncoder()
    #le_gest.fit(results_dict['gesture'])
    #print('Generating gesture plots.')
    #plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in le_gest.transform(results_dict['gesture'])])
    #plt.gca().set_aspect('equal', 'datalim')
    ## plt.title('UMAP projection of the Gesture clusters', fontsize=24);
    #save_path = os.path.join(plot_store_path, 'umap_gesture.png')
    #plt.savefig(save_path)
    #plt.clf()

    le_user = LabelEncoder()
    le_user.fit(results_dict['user'])
    print('Generating user plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in le_user.transform(results_dict['user'])])
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the User clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_user.png')
    plt.savefig(save_path)
    plt.clf()

def plot_umap_clusters_multidata(blobs_folder_paths_list: str, model: encoderDecoder2, plot_store_path: str) -> None:
    if not os.path.exists(plot_store_path):
        os.mkdir(plot_store_path)

    results_dict = {'gesture': [], 'user': [], 'skill': [], 'embeddings': [], 'task': []}
    for idx, path in enumerate(blobs_folder_paths_list):
        temp_results_dict = store_embeddings_in_dict(blobs_folder_path = path, model = model)
        temp_results_dict['task'] = [idx]*len(temp_results_dict['skill'])
        for key, value in temp_results_dict.items():
            results_dict[key].extend(value)

    # import pdb; pdb.set_trace()
    embeddings = np.array(results_dict['embeddings']).squeeze()
    
    print('Training umap reducer.')    
    umap_reducer = umap.UMAP()
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    print('Generating skill plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in results_dict['skill']])
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the Skill clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_skill.png')
    plt.savefig(save_path)
    plt.clf()

    le_gest = LabelEncoder()
    le_gest.fit(results_dict['gesture'])
    print('Generating gesture plots.')
    # import pdb; pdb.set_trace()
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])#, c=[sns.color_palette()[x] for x in le_gest.transform(results_dict['gesture'])])
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the Gesture clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_gesture.png')
    plt.savefig(save_path)
    plt.clf()

    le_user = LabelEncoder()
    le_user.fit(results_dict['user'])
    print('Generating user plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in le_user.transform(results_dict['user'])])
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the User clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_user.png')
    plt.savefig(save_path)
    plt.clf()

    le_task = LabelEncoder()
    le_task.fit(results_dict['task'])
    print('Generating task plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in results_dict['task']])
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the User clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_task.png')
    plt.savefig(save_path)
    plt.clf()

def label_surgical_study_video(optical_flow_path: str, model: encoderDecoder2, labels_store_path: str, num_frames_per_blob: int, spacing: int, classifier_load_path: str = './xgboost_save/multidata_xgboost.joblib') -> None:
    print('Loading dataset')
    dataset = surgeonStudyDataset(optical_flow_path = optical_flow_path, num_frames_per_blob = num_frames_per_blob, spacing = spacing)
    dataloader = DataLoader(dataset = dataset, batch_size = 128, shuffle = False, collate_fn = size_collate_fn)

    embeddings = []
    start_seconds_list = []
    end_seconds_list = []
    print('Generating embeddings.')
    for data, start_second, end_second in Bar(dataloader): 
        output = model.conv_net_stream(data)
        output = output.cpu().detach().data.numpy()
        embeddings.append(output)
        start_seconds_list.append(start_second)
        end_seconds_list.append(end_second)
    
    embeddings = np.concatenate(embeddings)
    start_seconds_list = np.concatenate(start_seconds_list)
    end_seconds_list = np.concatenate(end_seconds_list)

    print('Loading XGBoost')
    classifier = load(classifier_load_path)
    
    labels = classifier.predict(embeddings)

    labels = labels.tolist()
    start_seconds_list = start_seconds_list.tolist()
    end_seconds_list = end_seconds_list.tolist()

    labels.insert(0, 'labels')
    start_seconds_list.insert(0, 'start_second')
    end_seconds_list.insert(0, 'end_second')

    print('Saving labels.')

    # Labels are as follows: Needle passing: 0, Knot Tying: 1, Suturing: 2
    
    with open(labels_store_path, 'w') as f:
        for i in range(len(labels)):
            line = str(start_seconds_list[i]) + '\t' +  str(end_seconds_list[i]) + '\t' + str(labels[i])
            f.write(line)
            f.write('\n')

    f.close()
    print('Labels saved.')

def evaluate_model_superuser(blobs_folder_path: str, model: encoderDecoder2, transcriptions_path: str, experimental_setup_path: str) -> None:
    transcription_file_names = os.listdir(transcriptions_path)
    transcription_file_names = list(filter(lambda x: '.DS_Store' not in x, transcription_file_names))

    transcription_translation_dict = {}
    count = 0
    for file in transcription_file_names:
        curr_file_path = os.path.join(transcriptions_path, file)
        with open(curr_file_path, 'r') as f:
            for line in f:
                line = line.strip('\n').strip()
                line = line.split(' ')
                start = line[0]
                end = line[1]
                gesture = line[2]
                
                transcription_name = file.split('.')[0] + '_' + start.zfill(6) + '_' + end.zfill(6) + '.txt'
                new_name = 'blob_{}_video'.format(count) + '_'.join(file.split('.')[0].split('_')[0:3]) + '_gesture_' + gesture +'.p'
                new_name = re.sub('Knot_Tying', '', new_name)
                new_name = re.sub('Needle_Passing', '', new_name)
                new_name = re.sub('Suturing', '', new_name)
                transcription_translation_dict[transcription_name] = new_name
                count += 1

    df = cluster_statistics(blobs_folder_path = blobs_folder_path, model = model, num_clusters = 5)
    print("DONE")
    file_to_index_dict = {}
    file_count = 0
    for file in df['file_list']:
        file_to_index_dict[file] = file_count
        file_count += 1
    print("DONE 2")
    y = df['skill'].values.ravel()
    X = [np.array(v) for v in df['embeddings']]
    X = np.array(X).reshape(-1, 512*1350)
    print("DONE 3")
    sampler_list = []
    iterations = os.listdir(experimental_setup_path)
    iterations = list(filter(lambda x: '.DS_Store' not in x, iterations))
    print("DONE 4")
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    metrics_train = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    itr = 0
    for iter_num in tqdm(iterations):
        directory_path = os.path.join(experimental_setup_path, iter_num)
        train_indices = []
        test_indices = []
        print("Directory path: ",directory_path)
        with open(os.path.join(directory_path, 'Train.txt')) as f:
            for line in f:
                items = line.strip('\n').split('           ')
                try:
                    print("ITEM 0: ",items[0])
                    print("Transcription dict: ",transcription_translation_dict[items[0]])
                    print("File to index dict: ",file_to_index_dict)
                    print(file_to_index_dict[transcription_translation_dict[items[0]]])
                    exit()
                    train_indices.append(file_to_index_dict[transcription_translation_dict[items[0]]])
                except:
                    print("I stink")
                    exit()
                    pass
            f.close()
        
        with open(os.path.join(directory_path, 'Test.txt')) as f:
            for line in f:
                items = line.strip('\n').split('           ')
                try:
                    test_indices.append(file_to_index_dict[transcription_translation_dict[items[0]]])
                except:
                    pass
            f.close()
        print("Train indices: ",train_indices)
        print("Test indices: ",test_indices)
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        classifier = XGBClassifier(n_estimators = 1000,use_label_encoder=False)
        classifier.fit(X_train, y_train)

        y_hat = classifier.predict(X_train)
        y_hat_test = classifier.predict(X_test)
        report_train = classification_report(y_train,y_hat,output_dict = True)
        report_test = classification_report(y_test, y_hat_test, output_dict = True)

        # metrics['accuracy'] = (metrics['accuracy']*itr + report_test['accuracy'])/(itr + 1)
        # metrics['precision'] = (metrics['precision']*itr + report_test['weighted avg']['precision'])/(itr + 1)
        # metrics['recall'] = (metrics['recall']*itr + report_test['weighted avg']['recall'])/(itr + 1)
        # metrics['f1-score'] = (metrics['f1-score']*itr + report_test['weighted avg']['f1-score'])/(itr + 1)
        # metrics['support'] = (metrics['support']*itr + report_test['weighted avg']['support'])/(itr + 1)
        # itr += 1

        metrics['accuracy'].append(report_test['accuracy'])
        metrics['precision'].append(report_test['weighted avg']['precision'])
        metrics['recall'].append(report_test['weighted avg']['recall'])
        metrics['f1-score'].append(report_test['weighted avg']['f1-score'])
        metrics['support'].append(report_test['weighted avg']['support'])
        
        metrics_train['accuracy'].append(report_train['accuracy'])
        metrics_train['precision'].append(report_train['weighted avg']['precision'])
        metrics_train['recall'].append(report_train['weighted avg']['recall'])
        metrics_train['f1-score'].append(report_train['weighted avg']['f1-score'])
        metrics_train['support'].append(report_train['weighted avg']['support'])
    
    print("NUM: ",experimental_setup_path[-6])
    with open("Test"+experimental_setup_path[-6]+".txt",'w') as f:
        for key, val in metrics.items():
            f.write('Mean {} : {} \t \t Std {} : {}'.format(key, np.mean(val), key, np.std(val)))
            print('Mean {} : {} \t \t Std {} : {}'.format(key, np.mean(val), key, np.std(val)))
        f.close()
          
    with open("Train"+experimental_setup_path[-6]+".txt",'w') as f:
        for key, val in metrics_train.items():
            f.write('Mean {} : {} \t \t Std {} : {}'.format(key, np.mean(val), key, np.std(val)))
            print('Mean {} : {} \t \t Std {} : {}'.format(key, np.mean(val), key, np.std(val)))
        f.close()

def evaluate_model_superuser(blobs_folder_path: str, model: encoderDecoder2, user_index) -> None:
    i_to_user = {0: 'B', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I'}
    gesture_to_index = {'UNKNOWN': 0, 'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4, 'G5': 5, 'G6': 6, 'G7': 7, 'G8': 8, 'G8': 8, 'G9': 9, 'G10': 10, 'G11': 11, 'G12': 12}
    test_user = i_to_user[user_index]
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    metrics_train = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    model.eval()
    print("Real blobs folder path: ",blobs_folder_path)
    for trial in range(20):
        print('trial'+str(trial))
        blobs_folder = os.listdir(blobs_folder_path)
        train_embeddings_list = []
        test_embeddings_list = []
        train_out = []
        test_out = []
        
        print('loading')
        for file in blobs_folder:
            curr_kinematics_path = os.path.join(blobs_folder_path, file)
            count = 0
            with open(curr_kinematics_path, "r") as f:
                for count, line in enumerate(f):
                    pass
                    
            num_frames = 30 * 45
            kinematics_pred_frame_diff = 30 * 1
            start_frame = random.randint(0, count - num_frames - kinematics_pred_frame_diff)
            kinematics_in = torch.empty(size = [num_frames, 12*4])
            with open(curr_kinematics_path, "r") as f:
                for ii, line in enumerate(f):
                    if ii >= start_frame and ii < start_frame + num_frames:
                        line_nums = line.strip().split(' ')
                        if len(line_nums) < 77:
                            raise Exception("Not enough kinematic numbers")
                        kinematics_in[ii-start_frame][0] = float(line_nums[0])
                        kinematics_in[ii-start_frame][1] = float(line_nums[1])
                        kinematics_in[ii-start_frame][2] = float(line_nums[2])
                        kinematics_in[ii-start_frame][3] = float(line_nums[3])
                        kinematics_in[ii-start_frame][4] = float(line_nums[4])
                        kinematics_in[ii-start_frame][5] = float(line_nums[5])
                        kinematics_in[ii-start_frame][6] = float(line_nums[6])
                        kinematics_in[ii-start_frame][7] = float(line_nums[7])
                        kinematics_in[ii-start_frame][8] = float(line_nums[8])
                        kinematics_in[ii-start_frame][9] = float(line_nums[9])
                        kinematics_in[ii-start_frame][10] = float(line_nums[10])
                        kinematics_in[ii-start_frame][11] = float(line_nums[11])
                        kinematics_in[ii-start_frame][12] = float(line_nums[0+19])
                        kinematics_in[ii-start_frame][13] = float(line_nums[1+19])
                        kinematics_in[ii-start_frame][14] = float(line_nums[2+19])
                        kinematics_in[ii-start_frame][15] = float(line_nums[3+19])
                        kinematics_in[ii-start_frame][16] = float(line_nums[4+19])
                        kinematics_in[ii-start_frame][17] = float(line_nums[5+19])
                        kinematics_in[ii-start_frame][18] = float(line_nums[6+19])
                        kinematics_in[ii-start_frame][19] = float(line_nums[7+19])
                        kinematics_in[ii-start_frame][20] = float(line_nums[8+19])
                        kinematics_in[ii-start_frame][21] = float(line_nums[9+19])
                        kinematics_in[ii-start_frame][22] = float(line_nums[10+19])
                        kinematics_in[ii-start_frame][23] = float(line_nums[11+19])
                        kinematics_in[ii-start_frame][24] = float(line_nums[0+38])
                        kinematics_in[ii-start_frame][25] = float(line_nums[1+38])
                        kinematics_in[ii-start_frame][26] = float(line_nums[2+38])
                        kinematics_in[ii-start_frame][27] = float(line_nums[3+38])
                        kinematics_in[ii-start_frame][28] = float(line_nums[4+38])
                        kinematics_in[ii-start_frame][29] = float(line_nums[5+38])
                        kinematics_in[ii-start_frame][30] = float(line_nums[6+38])
                        kinematics_in[ii-start_frame][31] = float(line_nums[7+38])
                        kinematics_in[ii-start_frame][32] = float(line_nums[8+38])
                        kinematics_in[ii-start_frame][33] = float(line_nums[9+38])
                        kinematics_in[ii-start_frame][34] = float(line_nums[10+38])
                        kinematics_in[ii-start_frame][35] = float(line_nums[11+38])
                        kinematics_in[ii-start_frame][36] = float(line_nums[0+57])
                        kinematics_in[ii-start_frame][37] = float(line_nums[1+57])
                        kinematics_in[ii-start_frame][38] = float(line_nums[2+57])
                        kinematics_in[ii-start_frame][39] = float(line_nums[3+57])
                        kinematics_in[ii-start_frame][40] = float(line_nums[4+57])
                        kinematics_in[ii-start_frame][41] = float(line_nums[5+57])
                        kinematics_in[ii-start_frame][42] = float(line_nums[6+57])
                        kinematics_in[ii-start_frame][43] = float(line_nums[7+57])
                        kinematics_in[ii-start_frame][44] = float(line_nums[8+57])
                        kinematics_in[ii-start_frame][45] = float(line_nums[9+57])
                        kinematics_in[ii-start_frame][46] = float(line_nums[10+57])
                        kinematics_in[ii-start_frame][47] = float(line_nums[11+57])
                        gesture_id_num = int(gesture_to_index[line_nums[-1]])
                        if test_user in file:
                            test_out.append(gesture_id_num)
                        else:
                            train_out.append(gesture_id_num)
            try:
                kinematics_in = kinematics_in.unsqueeze(0)
                out = model.encoder(kinematics_in)
                out, h = model.LSTM(out)
                out = out.cpu().detach().data.numpy()
                if test_user in file:
                    for i in range(out.shape[1]):
                        test_embeddings_list.append(out[0][i].reshape(512))
                else:
                    for i in range(out.shape[1]):
                        train_embeddings_list.append(out[0][i].reshape(512))
            except:
                print("PROBLEM")
                pass
        train_embeddings_list = np.array(train_embeddings_list)
        test_embeddings_list = np.array(test_embeddings_list)
        train_out = np.array(train_out)
        test_out = np.array(test_out)
        classifier = XGBClassifier(n_estimators = 10)
        print('fitting')
        classifier.fit(train_embeddings_list, train_out)

        y_hat = classifier.predict(train_embeddings_list)
        y_hat_test = classifier.predict(test_embeddings_list)
        report_train = classification_report(train_out,y_hat,output_dict = True)
        report_test = classification_report(test_out, y_hat_test, output_dict = True)

        # metrics['accuracy'] = (metrics['accuracy']*itr + report_test['accuracy'])/(itr + 1)
        # metrics['precision'] = (metrics['precision']*itr + report_test['weighted avg']['precision'])/(itr + 1)
        # metrics['recall'] = (metrics['recall']*itr + report_test['weighted avg']['recall'])/(itr + 1)
        # metrics['f1-score'] = (metrics['f1-score']*itr + report_test['weighted avg']['f1-score'])/(itr + 1)
        # metrics['support'] = (metrics['support']*itr + report_test['weighted avg']['support'])/(itr + 1)
        # itr += 1

        metrics['accuracy'].append(report_test['accuracy'])
        metrics['precision'].append(report_test['weighted avg']['precision'])
        metrics['recall'].append(report_test['weighted avg']['recall'])
        metrics['f1-score'].append(report_test['weighted avg']['f1-score'])
        metrics['support'].append(report_test['weighted avg']['support'])
        
        metrics_train['accuracy'].append(report_train['accuracy'])
        metrics_train['precision'].append(report_train['weighted avg']['precision'])
        metrics_train['recall'].append(report_train['weighted avg']['recall'])
        metrics_train['f1-score'].append(report_train['weighted avg']['f1-score'])
        metrics_train['support'].append(report_train['weighted avg']['support'])
    
    with open("TestLeaveOut_"+str(user_index)+".txt",'w') as f:
        for key, val in metrics.items():
            f.write('Mean {} : {} \t \t Std {} : {}'.format(key, np.mean(val), key, np.std(val)))
            print('Mean {} : {} \t \t Std {} : {}'.format(key, np.mean(val), key, np.std(val)))
        f.close()
          
    with open("TrainLeaveOut_"+str(user_index)+".txt",'w') as f:
        for key, val in metrics_train.items():
            f.write('Mean {} : {} \t \t Std {} : {}'.format(key, np.mean(val), key, np.std(val)))
            print('Mean {} : {} \t \t Std {} : {}'.format(key, np.mean(val), key, np.std(val)))
        f.close()

def evaluate_model_umaps(blobs_folder_path: str, model: encoderDecoder2) -> None:
    user_letter_to_index = {'B': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7}
    gesture_to_index = {'UNKNOWN': 0, 'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4, 'G5': 5, 'G6': 6, 'G7': 7, 'G8': 8, 'G8': 8, 'G9': 9, 'G10': 10, 'G11': 11, 'G12': 12}
    user_to_skill = {'B': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 1, 'G': 0, 'H': 0, 'I': 0}
    model.eval()
    blobs_folder = os.listdir(blobs_folder_path)
    embeddings_list = []
    user_list = []
    skill_list = []
    gesture_list = []
    print('loading')
    for file in blobs_folder:
        user_letter = file[9]
        user_index = user_letter_to_index[user_letter]
        user_skill = user_to_skill[user_letter]
        curr_kinematics_path = os.path.join(blobs_folder_path, file)
        count = 0
        with open(curr_kinematics_path, "r") as f:
            for count, line in enumerate(f):
                pass
        count = count + 1
        kinematics_in = torch.empty(size = [count, 12*4])
        with open(curr_kinematics_path, "r") as f:
            for ii, line in enumerate(f):
                line_nums = line.strip().split(' ')
                if len(line_nums) < 77:
                    raise Exception("Not enough kinematic numbers")
                kinematics_in[ii][0] = float(line_nums[0])
                kinematics_in[ii][1] = float(line_nums[1])
                kinematics_in[ii][2] = float(line_nums[2])
                kinematics_in[ii][3] = float(line_nums[3])
                kinematics_in[ii][4] = float(line_nums[4])
                kinematics_in[ii][5] = float(line_nums[5])
                kinematics_in[ii][6] = float(line_nums[6])
                kinematics_in[ii][7] = float(line_nums[7])
                kinematics_in[ii][8] = float(line_nums[8])
                kinematics_in[ii][9] = float(line_nums[9])
                kinematics_in[ii][10] = float(line_nums[10])
                kinematics_in[ii][11] = float(line_nums[11])
                kinematics_in[ii][12] = float(line_nums[0+19])
                kinematics_in[ii][13] = float(line_nums[1+19])
                kinematics_in[ii][14] = float(line_nums[2+19])
                kinematics_in[ii][15] = float(line_nums[3+19])
                kinematics_in[ii][16] = float(line_nums[4+19])
                kinematics_in[ii][17] = float(line_nums[5+19])
                kinematics_in[ii][18] = float(line_nums[6+19])
                kinematics_in[ii][19] = float(line_nums[7+19])
                kinematics_in[ii][20] = float(line_nums[8+19])
                kinematics_in[ii][21] = float(line_nums[9+19])
                kinematics_in[ii][22] = float(line_nums[10+19])
                kinematics_in[ii][23] = float(line_nums[11+19])
                kinematics_in[ii][24] = float(line_nums[0+38])
                kinematics_in[ii][25] = float(line_nums[1+38])
                kinematics_in[ii][26] = float(line_nums[2+38])
                kinematics_in[ii][27] = float(line_nums[3+38])
                kinematics_in[ii][28] = float(line_nums[4+38])
                kinematics_in[ii][29] = float(line_nums[5+38])
                kinematics_in[ii][30] = float(line_nums[6+38])
                kinematics_in[ii][31] = float(line_nums[7+38])
                kinematics_in[ii][32] = float(line_nums[8+38])
                kinematics_in[ii][33] = float(line_nums[9+38])
                kinematics_in[ii][34] = float(line_nums[10+38])
                kinematics_in[ii][35] = float(line_nums[11+38])
                kinematics_in[ii][36] = float(line_nums[0+57])
                kinematics_in[ii][37] = float(line_nums[1+57])
                kinematics_in[ii][38] = float(line_nums[2+57])
                kinematics_in[ii][39] = float(line_nums[3+57])
                kinematics_in[ii][40] = float(line_nums[4+57])
                kinematics_in[ii][41] = float(line_nums[5+57])
                kinematics_in[ii][42] = float(line_nums[6+57])
                kinematics_in[ii][43] = float(line_nums[7+57])
                kinematics_in[ii][44] = float(line_nums[8+57])
                kinematics_in[ii][45] = float(line_nums[9+57])
                kinematics_in[ii][46] = float(line_nums[10+57])
                kinematics_in[ii][47] = float(line_nums[11+57])
                gesture_id_num = int(gesture_to_index[line_nums[-1]])
                gesture_list.append(gesture_id_num)
                user_list.append(user_index)
                skill_list.append(user_skill)
        try:
            kinematics_in = kinematics_in.unsqueeze(0)
            out = model.encoder(kinematics_in)
            out, h = model.LSTM(out)
            out = out.cpu().detach().data.numpy()
            for i in range(out.shape[1]):
                embeddings_list.append(out[0][i].reshape(512))
        except:
            print("PROBLEM")
            pass
    embeddings = np.array(embeddings_list)
    print("Embeddings size: ",embeddings.shape)
    print('Training umap reducer.')    
    umap_reducer = umap.UMAP()
    reduced_embeddings = umap_reducer.fit_transform(embeddings)
    
    print('Generating skill plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in skill_list])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Skill clusters', fontsize=24)
    plt.savefig('umap_skill.png')
    plt.clf()

    print('Generating gesture plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in gesture_list])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Gesture clusters', fontsize=24)
    plt.savefig('umap_gesture.png')
    plt.clf()

    print('Generating user plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in user_list])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the User clusters', fontsize=24)
    plt.savefig('umap_user.png')
    plt.clf()
    

def main():
    blobs_folder_path = '../jigsaw_dataset/Suturing/blobs'
    blobs_folder_paths_list = ['../jigsaw_dataset/Needle_Passing/blobs', '../jigsaw_dataset/Knot_Tying/blobs', '../jigsaw_dataset/Suturing/blobs']
    
    model = encoderDecoder2(embedding_dim = 2048)
    model.load_state_dict(torch.load('./weights_save/suturing_weights/suturing_2048.pth'))

    # store_embeddings_in_dict(blobs_folder_path = blobs_folder_path, model = model)
    # df = cluster_statistics(blobs_folder_path = blobs_folder_path, model = model, num_clusters = num_clusters)
    # return(df)

    # evaluate_model(blobs_folder_path = blobs_folder_path, model = model, num_clusters = 10, save_embeddings = False)
    # evaluate_model_multidata(blobs_folder_paths_list = blobs_folder_paths_list, model = model, num_clusters = 10, save_embeddings = False)

    # plot_umap_clusters(blobs_folder_path = blobs_folder_path, model = model, plot_store_path = './umap_plots/Needle_Passing')
    # plot_umap_clusters_multidata(blobs_folder_paths_list = blobs_folder_paths_list, model = model, plot_store_path = './umap_plots/multidata')

    optical_flow_path = '../jigsaw_dataset/Surgeon_study_videos/optical_flow/left_suturing.p'
    labels_store_path = '../jigsaw_dataset/Surgeon_study_videos/multimodal_labels/multidata_2048_labels.txt'
    num_frames_per_blob = 25
    spacing = 2
    label_surgical_study_video(optical_flow_path = optical_flow_path, model = model, labels_store_path = labels_store_path, num_frames_per_blob = num_frames_per_blob, spacing = spacing)

    transcriptions_path = '../jigsaw_dataset/Suturing/transcriptions/'
    experimental_setup_path = '../jigsaw_dataset/Experimental_setup/Suturing/Balanced/GestureClassification/UserOut/1_Out/'
    # evaluate_model_superuser(blobs_folder_path = blobs_folder_path, model = model, transcriptions_path = transcriptions_path, experimental_setup_path = experimental_setup_path)

if __name__ == '__main__':
    main()
