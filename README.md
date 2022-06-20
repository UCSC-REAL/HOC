# Clusterability as an Alternative to Anchor Points When Learning with Noisy Labels (ICML'21)
This code is a PyTorch implementation of the paper:

Zhaowei Zhu, Yiwen Song, and Yang Liu, "Clusterability as an Alternative to Anchor Points When Learning with Noisy Labels," https://proceedings.mlr.press/v139/zhu21e.html.

# Demo

http://peers.ai/

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
We collected them from Amazon Mechanical Turk (MTurk) and students at UC Santa Cruz in February 2020. We only collected one annotation for each image at the cost of ¢10 per image. The label file is available at ./data/noise_label_human.pt.


## Minimal implementation of HOC
G: the number of rounds needed to estimate the consensus probabilities (See details in Algorithm 1 [1])
max_iter: the maximum number of iterations to get an estimate of T 
```
CUDA_VISIBLE_DEVICES=0 python main_min.py --G 50 --max_iter 1500
```

## Run with three noisy labels
Save your noisy labels to ./data/test.csv.
Data format: N*3 matrix, where N is the number of instances. For example, a row [0,1,1] means three noisy labels for this instances are respectively 0, 1, and 1. 
Label classes MUST be consecutive integers.
```
python3 main_knwon2nn.py
```
The result of the default test case is 
```
[[87.7 12.3]
 [14.4 85.6]]
 ```

## Reference
```

@InProceedings{zhu2021clusterability,
  title = 	 {Clusterability as an Alternative to Anchor Points When Learning with Noisy Labels},
  author =       {Zhu, Zhaowei and Song, Yiwen and Liu, Yang},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {12912--12923},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/zhu21e/zhu21e.pdf},
  url = 	 {https://proceedings.mlr.press/v139/zhu21e.html}
}

```
