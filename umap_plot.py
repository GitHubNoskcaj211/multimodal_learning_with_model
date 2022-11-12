import argparse
from ast import parse
from training import train_encoder_decoder_embeddings, train_encoder_decoder_multidata_embeddings
from video_preprocessing import computeOpticalFlow, create_data_blobs
from embeddings_cluster_explore import evaluate_model, evaluate_model_multidata, plot_umap_clusters, plot_umap_clusters_multidata, evaluate_model_superuser
from neural_networks import encoderDecoder
import torch

def main() -> None:
    blobs_folder_path = '../JIGSAWS/Suturing/blobs'
    model_dim = 512
    weights_save_path = None
    model = encoderDecoder(embedding_dim = model_dim)
    model.load_state_dict(torch.load(weights_save_path))
    plot_path = '~/'
    plot_umap_clusters(blobs_folder_path=blobs_folder_path,model=model,plot_store_path=plot_path)


if __name__ == '__main__':
    main()
