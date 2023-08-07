# Federated Learning Testing

1. Federated Learning on Non-IID Data with Local-drift Decoupling and Correction
Code for paper - **[Federated Learning on Non-IID Data with Local-drift Decoupling and Correction]**

We provide code to run FedDC, FedAvg, 
[FedDyn](https://openreview.net/pdf?id=B7v4QMR6Z9w), 
[Scaffold](https://openreview.net/pdf?id=B7v4QMR6Z9w), and [FedProx](https://arxiv.org/abs/1812.06127) methods.

2.  HarmoFL: Harmonizing Local and Global Drifts in Federated Learning on Heterogeneous Medical Images

This is the PyTorch implemention of our paper **HarmoFL: Harmonizing Local and Global Drifts in Federated Learning on Heterogeneous Medical Images** by [Meirui Jiang](https://meiruijiang.github.io/MeiruiJiang/), Zirui Wang and [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/).

3. FedUKD: Federated UNet Model with Knowledge Distillation for Land Use Classification from Satellite and Street Views

- https://arxiv.org/abs/2212.02196

4. FedTP: Federated Learning by Transformer Personalization



## Kan Note

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

pip install notebook

pip install medmnist
```

```
jupyter notebook
```


# HarmoFL: Harmonizing Local and Global Drifts in Federated Learning on Heterogeneous Medical Images
This is the PyTorch implemention of our paper **HarmoFL: Harmonizing Local and Global Drifts in Federated Learning on Heterogeneous Medical Images** by [Meirui Jiang](https://meiruijiang.github.io/MeiruiJiang/), Zirui Wang and [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/).

## Kan Note

- python == 3.10
- Cython==3.0.0
- numpy==1.25.0
- opencv_python==4.8.0.74
- Pillow==10.0.0
- setuptools==68.0.0
- SimpleITK==2.2.1
- torch==2.0.1
- torchvision==0.15.2

```
conda create --name [harmofl-env-name] python=3.10

conda activate [harmofl-env-name]

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

```
pip install -r requirements.txt
```



## Abstract
> Multiple medical institutions collaboratively training a model using federated learning (FL) has become a promising solution for maximizing the potential of data-driven models, yet the non-independent and identically distributed (non-iid) data in medical images is still an outstanding challenge in real-world practice. The feature heterogeneity caused by diverse scanners or sensors introduces a drift in the learning process, in both local (client) and global (server) optimizations, which harms the convergence as well as model performance. Many previous works have attempted to address the non-iid issue by tackling the drift locally or globally, but how to jointly solve the two essentially coupled drifts is still unclear. In this work, we concentrate on handling both local and global drifts and introduce a new harmonizing framework called HarmoFL. First, we propose to mitigate the local update drift by normalizing amplitudes of images transformed into the frequency domain to mimic a unified scanner/sensor, in order to generate a harmonized feature space across local clients. Second, based on harmonized features, we design a client weight perturbation guiding each local model to reach a flat optimum, where a neighborhood area of the local optimal solution has a uniformly low loss. Without any extra communication cost, the perturbation assists the global model to optimize towards a converged optimal solution by aggregating several local flat optima. We have theoretically analyzed the proposed method and empirically conducted extensive experiments on three medical image classification and segmentation tasks, showing that HarmoFL outperforms a set of recent state-of-the-art methods with promising convergence behavior.

<p align="center">
<img src="./assets/intro.png" alt="intro" width="70%"/>
</p>

## Usage
### Setup
**Conda**

We recommend using conda to setup the environment, See the `requirements.yaml` for environment configuration 

If there is no conda installed on your PC, please find the installers from https://www.anaconda.com/products/individual

If you have already installed conda, please use the following commands.

```bash
conda env create -f environment.yaml
conda activate harmofl
```

**Build cython file**

build cython file for amplitude normalization
```bash
python utils/setup.py build_ext --inplace
```

### Dataset & Trained Model
#### Classification
- Please download the histology breast cancer classification datasets [here](https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/), extract and put folder 'patches' under `data/camelyon17` directory:

#### Segmentation
- Please download the prostate MRI datasets [here](https://liuquande.github.io/SAML/), put the folder `data` under `data/prostate` directory.


### Train
`fed_train.py` is the main file to run the federated experiments
Please using following commands to train a model with federated learning strategy.
```bash
bash train.sh
```
Below please find some useful options:
- --alpha :specify the degree of weight perturbation, default is 0.05.
- --wk_iters :specify the local update epochs, default is 1.
### Test
suppose your test model's path is 'model/data/harmofl'
```bash
bash test.sh
```
