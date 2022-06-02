# CF-Net
Code of "Cross Fusion Net: A Fast Semantic Segmentation Network for Small-scale Semantic Information Capturing in Aerial Scenes"
To train the network you can
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py
To test the network you can
CUDA_VISIBLE_DEVICES=0 python evaluate.py
To test the metrics and generate training and testing images, you can use the code provided in https://github.com/zhu-xlab/A-Relation-Augmented-Fully-Convolutional-Network-for-Semantic-Segmentation-in-Aerial-Scenes
