# Multimodal learning

How to use this repository:
1) Extract optical flows from the video
2) Create data blobs
2) Train a model 
3) Evaluate the trained model and get different results including U-map plots, gesture classification, skill classification, task classification

This project does take a fair bit of disk space. I generated something close to 200 GB. It generates the optical flow entirely before hand to reduce the time to train. A few thoughts on how to save space: 
1) Delete the input to each step as it finishes. For example, once you generate the resized images, you no longer need the original. Once you generate the optical flow, you can delete the resized images, and once you generate the blobs, you can delete the optical flow. If you have the space, it might be worth keeping the optical flow folder around for the project.
2) The optical flow representation is the largest. If you can't store thbe whole dataset as optical flow at once, you can generate a subset of it as optical flow, generate the data blobs from it, and then delete the flow representation to save space
3) You can rewrite the code to calculate the optical flow after the blob generation. The current implementation is to generate flow for the whole video, and then extract the first 50 frames of it for the training. This improves flexibility at the cost of disk space. First extracting the frames and then calculating flow would also speed up processing time
4) If all else fails, use a subset of the data (sampling across skill levels). The comparison to the original paper will be weaker but having a homework to hand in is better than not. 

You can use a local machine to do steps 1 and 2 so you can inspect the data preprocessing steps and they are both CPU only jobs. The main steps are each a mode in main.py and we will be using main.py to call them. It is recommended to read through main.py and see what options there are and what default parameters you may be using to call each step. We are only setting the path parameters below, not changing the default scaling and other frame counts so use the default parameters in your analysis. If for some reason a run fails in the middle, you may need to delete the files that have been generated. It is not guaranteed that the code will overwrite the previously generate files or just append to them. 

## Google Cloud VM Setup
* n1-standard-8
* NVIDIA Tesla V100
* 210 GB boot disk
* Ubuntu Pro 16.04
* Spot Instance

## Environment Setup
```
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda

# Verification: make sure a GPU shows up here
nvidia-smi

## Conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
source .bashrc

## Conda Environment
conda create --name torch python=3.6
conda activate torch
conda install pytorch cuda92 -c pytorch
pip install sklearn opencv-python tqdm pandas xgboost umap seaborn multipledispatch barbar
pip install -U scikit-learn scipy matplotlib
pip uninstall umap
pip install umap-learn
conda install torchvision -c pytorch
pip uninstall numpy
pip install numpy --upgrade --ignore-installed



# Get Dataset
sudo apt-get install zip unzip
pip3 install gdown
cd ~/
gdown <ID OF ZIP FILE IN GOOGLE DRIVE> -O <JIGSAWS.zip>
unzip <Path to zip file>
```

### To generate the optical flow, run the following command. You may need to adjust the path for where you downloaded your JIGSAWS directory. This call will perform two steps. The first step will resize the video and the second will generate the optical flow. 

<code>python3 main.py --mode 'optical_flow' --source_directory ../JIGSAWS/Suturing/video/ --resized_video_directory ../JIGSAWS/Suturing/resized_video --destination_directory ../JIGSAWS/Suturing/optical_flow  </code> 

Make sure the code started running OK (it should say 'Rescaling videos; Processing video Suturing_B001_capture1.avi; 100; 200;...') then go get some coffee, take a nap. The "generating optical flow" step can take a while. In the mean time, once the resizing step is done, you can check that you have the same number of videos in the resized_video folder as the original video directory and you can play the videos (they should be smaller than the original though). For reference, on my 6-years-old desktop that had a pretty good CPU for when it came out, the resizing took about 3 min and the optical flow generation took about 1 h 20 min. 

### To generate prepare the data for training, you need to generate 'data_blobs'. This step pre-extracts the first 50 frames (subsampled by 2) for each gesture based on the transcriptions. This preprocessing step should speed up the training so that you don't need to do the kinematics extraction each time. This step took me about half an hour. It would've been faster to do this before the optical flow and only calculate flow for the relevant frames, but this way offers more flexibility in case you want to use a different number of frames.  

<code>python3 main.py --mode 'data_blobs' --optical_flow_path ../JIGSAWS/Suturing/optical_flow --transcriptions_path ../JIGSAWS/Suturing/transcriptions --kinematics_path ../JIGSAWS/Suturing/kinematics/AllGestures --blobs_path ../JIGSAWS/Suturing/blobs</code> 


### To train the network using your data_blobs. Run the following code:  

<code>python3 main.py --num_epochs 1 --mode 'train' --suturing_folder_path ../JIGSAWS/Suturing/ --weights_save_path models/code> 

### To evaluate the network (remember to put your network into the models folder
<code>python3 main.py --mode 'eval' --blobs_folder_path '../JIGSAWS/Suturing/blobs' --weights_save_path models</code> 


The paper associated with this repository can be found at https://link.springer.com/article/10.1007/s11548-021-02343-y. The citation details are as follows.

@article{wu2021cross,
  title={Cross-modal self-supervised representation learning for gesture and skill recognition in robotic surgery},
  author={Wu, Jie Ying and Tamhane, Aniruddha and Kazanzides, Peter and Unberath, Mathias},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  volume={16},
  number={5},
  pages={779--787},
  year={2021},
  publisher={Springer}
}

