# Federated Learning [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4321561.svg)](https://doi.org/10.5281/zenodo.4321561)

This is partly the reproduction of the paper of [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)   
Only experiments on MNIST and CIFAR10 (both IID and non-IID) is produced by far.

Note: The scripts will be slow without the implementation of parallel computing. 

## Requirements

python>=3.6  
pytorch>=0.4

```
torch 2.0.1
Python 3.10.11
```
## Env.

**Build cython file**

build cython file for amplitude normalization
```bash
python utils/setup.py build_ext --inplace
```

## Run

The MLP and CNN models are produced by:
> python [main_nn.py](main_nn.py)

Federated learning with MLP and CNN is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](utils/options.py). 

For example:

```
python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  
```



`--all_clients` for averaging over all client models

NB: for CIFAR-10, `num_channels` must be 3.

## Results

### MNIST
Results are shown in Table 1 and Table 2, with the parameters C=0.1, B=10, E=5.

Table 1. results of 10 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP|  94.57%     | 70.44%         |
| FedAVG-CNN|  96.59%     | 77.72%         |

Table 2. results of 50 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP| 97.21%      | 93.03%         |
| FedAVG-CNN| 98.60%      | 93.81%         |


## Ackonwledgements
Acknowledgements give to [youkaichao](https://github.com/youkaichao).

## References
McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Artificial Intelligence and Statistics (AISTATS), 2017.

## Cite As
Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561

## CMD

```
python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

python main_nn.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

python main_fed.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 50 --gpu 0  

python main_nn.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 50 --gpu 0 

python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0 --all_clients

python main_fed.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 50 --gpu 0 --all_clients

python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

python main_nn.py --dataset cifar --num_channels 1 --model cnn --epochs 50 --gpu 0  

python main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 50 --gpu 0  

python main_nn.py --dataset cifar --num_channels 3 --model cnn --epochs 50 --gpu 0  

python main_fed.py --dataset cifar --iid --num_channels 3 --model mlp --epochs 50 --gpu 0  

python main_nn.py --dataset cifar --num_channels 3 --model mlp --epochs 50 --gpu 0 

python main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 50 --gpu 0 --all_clients

python main_fed.py --dataset cifar --iid --num_channels 3 --model mlp --epochs 50 --gpu 0 --all_clients

python main_fed.py --dataset cifar100 --iid --num_channels 3 --model cnn --epochs 50 --gpu 0 --num_classes 100

python main_fed.py --dataset cifar100 --iid --num_channels 3 --model mlp --epochs 50 --gpu 0 --num_classes 100

python main_nn.py --dataset cifar100 --num_channels 3 --model cnn --epochs 5 --gpu 0 --local_bs 64 --num_classes 100

python main_nn.py --dataset cifar100 --num_channels 3 --model mlp --epochs 5 --gpu 0 --local_bs 64 --num_classes 100

python main_fed.py --dataset mnist --iid --num_channels 1 --model 2nn --epochs 10 --gpu 0  

python main_fed.py --dataset mnist --iid --num_channels 1 --model 2nn --epochs 10 --gpu 0 --all_clients

python main_nn.py --dataset mnist --num_channels 1 --model 2nn --epochs 10 --gpu 0

python main_fed.py --dataset emnist --iid --num_channels 1 --model nn --epochs 10 --gpu 0  

python main_nn.py --dataset emnist --num_channels 1 --model nn --epochs 10 --gpu 0 

python main_fed.py --dataset salt --iid --num_channels 1 --model unet --epochs 5 --gpu 0

python main_fed.py --dataset salt --iid --num_channels 1 --model unet --epochs 15 --gpu 0  --lr 0.0001

python main_nn.py --dataset salt --num_channels 1 --model unet --epochs 15 --gpu 0 --lr 0.0001

python main_fed.py --dataset mnist --num_channels 1 --model cnn --epochs 5 --gpu 0

python main_fed.py --dataset cifar100 --num_channels 3 --model cnn --epochs 5 --gpu 0 --num_classes 100

python main_fed.py --dataset cifar100 --num_channels 3 --model cnn --epochs 5 --gpu 0 --num_classes 100 --iid

python main_fed.py --dataset salt --iid --num_channels 1 --model unet --epochs 5 --gpu 0 --method harmofl

python main_fed.py --dataset salt --iid --num_channels 1 --model unet --epochs 5 --gpu 0 --method harmofl --all_clients

python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 5 --gpu 0 --methods harmofl --all_clients

```

## Fed Test.

```
# MNIST CNN MLP FEDAVG HARMOFL
python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 1 --gpu 0 --methods harmofl

python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 1 --gpu 0 --methods harmofl --all_clients

python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 1 --gpu 0 

python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 1 --gpu 0 --all_clients

python main_fed.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 1 --gpu 0 --methods harmofl

python main_fed.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 1 --gpu 0 --methods harmofl --all_clients

python main_fed.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 1 --gpu 0 

python main_fed.py --dataset mnist --iid --num_channels 1 --model mlp --epochs 1 --gpu 0 --all_clients

# MNIST NN FEDAVG HARMOFL
python main_fed.py --dataset mnist --iid --num_channels 1 --model 2nn --epochs 1 --gpu 0  

python main_fed.py --dataset mnist --iid --num_channels 1 --model 2nn --epochs 1 --gpu 0 --all_clients

python main_fed.py --dataset mnist --iid --num_channels 1 --model 2nn --epochs 1 --gpu 0 --methods harmofl 

python main_fed.py --dataset mnist --iid --num_channels 1 --model 2nn --epochs 1 --gpu 0 --methods harmofl --all_clients

# CIFAR10 CNN MLP FEDAVG HARMOFL
python main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 1 --gpu 0

python main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 1 --gpu 0 --all_clients

python main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 1 --gpu 0 --methods harmofl 

python main_fed.py --dataset cifar --iid --num_channels 3 --model cnn --epochs 1 --gpu 0  --methods harmofl --all_clients

python main_fed.py --dataset cifar --iid --num_channels 3 --model mlp --epochs 1 --gpu 0

python main_fed.py --dataset cifar --iid --num_channels 3 --model mlp --epochs 1 --gpu 0 --all_clients

python main_fed.py --dataset cifar --iid --num_channels 3 --model mlp --epochs 1 --gpu 0 --methods harmofl 

python main_fed.py --dataset cifar --iid --num_channels 3 --model mlp --epochs 1 --gpu 0 --methods harmofl --all_clients

# CIFAR100 CNN MLP FEDAVG HARMOFL
python main_fed.py --dataset cifar100 --iid --num_channels 3 --model cnn --epochs 1 --gpu 0 --num_classes 100

python main_fed.py --dataset cifar100 --iid --num_channels 3 --model cnn --epochs 1 --gpu 0 --num_classes 100 --all_clients

python main_fed.py --dataset cifar100 --iid --num_channels 3 --model cnn --epochs 1 --gpu 0 --num_classes 100 --methods harmofl 

python main_fed.py --dataset cifar100 --iid --num_channels 3 --model cnn --epochs 1 --gpu 0 --num_classes 100 --methods harmofl --all_clients

python main_fed.py --dataset cifar100 --iid --num_channels 3 --model mlp --epochs 1 --gpu 0 --num_classes 100

python main_fed.py --dataset cifar100 --iid --num_channels 3 --model mlp --epochs 1 --gpu 0 --num_classes 100 --all_clients

python main_fed.py --dataset cifar100 --iid --num_channels 3 --model mlp --epochs 1 --gpu 0 --num_classes 100 --methods harmofl 

python main_fed.py --dataset cifar100 --iid --num_channels 3 --model mlp --epochs 1 --gpu 0 --num_classes 100 --methods harmofl --all_clients

# EMNIST NN FEDAVG HARMOFL
python main_fed.py --dataset emnist --iid --num_channels 1 --model nn --epochs 1 --gpu 0

python main_fed.py --dataset emnist --iid --num_channels 1 --model nn --epochs 1 --gpu 0 --all_clients

python main_fed.py --dataset emnist --iid --num_channels 1 --model nn --epochs 1 --gpu 0 --methods harmofl

python main_fed.py --dataset emnist --iid --num_channels 1 --model nn --epochs 1 --gpu 0 --methods harmofl --all_clients

# SALT UNET FEDAVG HARMOFL
python main_fed.py --dataset salt --iid --num_channels 1 --model unet --epochs 1 --gpu 0

python main_fed.py --dataset salt --iid --num_channels 1 --model unet --epochs 1 --gpu 0 --all_clients

python main_fed.py --dataset salt --iid --num_channels 1 --model unet --epochs 1 --gpu 0 --method harmofl

python main_fed.py --dataset salt --iid --num_channels 1 --model unet --epochs 1 --gpu 0 --method harmofl --all_clients

# CAMELYON17 DenseNet FEDAVG HARMOFL
python main_fed.py --dataset camelyon17 --gpu 0 --local_bs 1 --epochs 1

# python main_fed.py --dataset camelyon17 --gpu 0 --local_bs 128 --epochs 1

python main_fed.py --dataset camelyon17 --gpu 0 --local_bs 1 --epochs 1 --all_clients

python main_fed.py --dataset camelyon17 --gpu 0 --local_bs 1 --epochs 1 --method harmofl

python main_fed.py --dataset camelyon17 --gpu 0 --local_bs 1 --epochs 1 --method harmofl --all_clients

```