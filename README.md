# Clusterability as an Alternative to Anchor Points When Learning with Noisy Labels
This code is a PyTorch implementation of the paper "Clusterability as an Alternative to Anchor Points When Learning with Noisy Labels".
https://arxiv.org/abs/2102.05291

## Prerequisites
Python 3.6.6

PyTorch 1.3.0

Torchvision 0.4.1

Datasets will be downloaded to ./data/.

## Run HOC + forward loss correction
On CIFAR-10 with instance 0.6 noise. 

```
export CUDA_VISIBLE_DEVICES=0 && nohup python -u main.py --pre_type image --dataset cifar10 --loss fw --label_file_path ./data/IDN_0.6_C10_0.pt> ./out/test10.out &
```

On CIFAR-10 with real-world human-annotated labels

```
export CUDA_VISIBLE_DEVICES=0 && nohup python -u main.py --pre_type image --dataset cifar10 --loss fw --label_file_path ./data/noise_label_human.pt> ./out/test10.out &
```

On CIFAR-100 with instance 0.6 noise. 

```
export CUDA_VISIBLE_DEVICES=1 && nohup python -u main.py --pre_type image --dataset cifar100 --loss fw --label_file_path ./data/IDN_0.6_C100_0.pt> ./out/test100.out &
```

## Real-world human-annotated CIFAR-10
We collected them from Amazon Mechanical Turk (MTurk) and students at UC Santa Cruz in February 2020. We collect one annotation for each image with a cost of ¢10 per image.
