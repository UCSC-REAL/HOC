# Clusterability as an Alternative to Anchor PointsWhen Learning with Noisy Labels
This code is a PyTorch implementation of the paper "Clusterability as an Alternative to Anchor PointsWhen Learning with Noisy Labels" submitted anonymously to ICML2021.

The code is run on the NVIDIA TITAN V.
## Prerequisites
Python 3.6.6

PyTorch 1.3.0

Torchvision 0.4.1


## Runing HOC Estimator on CIFAR 10 and CIFAR 100
Run HOC Estimator on the CIFAR-10 with instance 0.6 noise. 

```
export CUDA_VISIBLE_DEVICES=0 && nohup python -u usePretrain.py --pre_type image --dataset cifar10 --loss fw --label_file_path ./data/IDN_0.6_C10_0.pt> ./out/test.out &
```

Datasets will be downloaded to ./data/.

