# Hierarchical Distribution-Aware Testing of Deep Learning
This repository contains the source code for testing Deep Neural Networks(DNNs). It is implemented in Python and includes the code and models required for replicating the studies in the paper.

## Introduction
The Hierarchical Distribution-Aware (HDA) testing implements the distribution-aware in the whole testing process, including test seeds selection and local test cases generation. The robustness and global distribution is combined to guide the test seeds selection, while a novel two-step Genetic Alogorithm (GA) based test case generation is developed to search Adeversarial Examples (AEs) in the balance of local distribution and prediction loss.

## Environment Setup
Requires Linux Platform with `Python 3.8.5`. We recommend to use anaconda for creating virtual environment. `requirements.txt` file contains the python packages required for running the code. Follow below steps for installing the packages:
- Create virtual environment and install necessary packages

	`conda create -n hda_test --file requirements.txt`

- Activate virtual environment

	`conda activate hda_test`


## Files
- `model` Directory contains scripts for training VAE and test models
- `checkpoints` Directory contains saved checkpoints for pre_trained VAE and test models

#### Note: 
- We only include a pre-trained VAE model and a test model for MNIST dataset due to the file size limit. For other dataset, please train the VAE models (fit global distribution) and test models first.

- You may get error 'zipfile.BadZipFile: File is not a zip file' when downloading CelebA dataset. Google Drive has a daily maximum quota for any file. Try to mannually download from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) and unzip the dataset. Move to the folder `Datasets/celeba`

## How To Use 
The tool can be run for HDA testing, and VAE and test model training with the following commands.

### Quick Start for HDA testing
You can quickly run the HDA testing on MNIST dataset by selecting 20 test seeds, using 'mse' as local perceptual quality metrics
``` 
python main.py --dataset mnist --no_seeds 20 --local_p mse
```
### Procedure from Scratch
For other datasets, like FashionMNIST, SVHN, CIFAR10, CELEBA, we need to first prepare the VAE and test model, and then run HDA testing. Here, we use SVHN as an example:

To train a test model
``` 
python main.py --dataset svhn --train True --vae_train False
```

To train a VAE model for seeds selection
``` 
python main.py --dataset svhn --train True --vae_train True
```
during the training of VAE model, the reconstruction images and new sampled images from VAE are created in `samples` folder.

When VAE model and test model are ready, run the HDA testing
``` 
python main.py --dataset svhn --no_seeds 20 --local_p mse
```

### Test Output
Test results are saved to the `dataset_output` folder, including the test seeds, generated AEs and a test report.

## About Adversarial Attack
We use torchattacks module to implement the adversarial attack. This package only support running on GPU(cuda service).



