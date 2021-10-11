# Equivariant Imaging: Learning Beyond the Range Space

This repository is the official implementation of [Equivariant Imaging: Learning Beyond the Range Space](https://arxiv.org/abs/2103.14756) (ICCV'2021 oral paper)

by [Dongdong Chen](https://dongdongchen.com), [Julián Tachella](https://tachella.github.io/), [Mike E. Davies](https://scholar.google.co.uk/citations?user=dwmfR3oAAAAJ&hl=en).


### [post](https://tachella.github.io/2021/04/16/equivariant-imaging-learning-beyond-the-range-space/) | [paper](https://arxiv.org/abs/2103.14756) | [toy example](https://github.com/edongdongchen/EI/blob/main/ei_demo_cs_usps.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edongdongchen/EI/blob/main/ei_demo_cs_usps.ipynb)



## Requirements

* [PyTorch](https://pytorch.org/) (1.6)

All used packages are listed in the Anaconda environment.yml file. You can create an environment and run
```
conda env create -f environment.yml
```

## Get quickly started

* Please find the [blog post](https://tachella.github.io/2021/04/16/equivariant-imaging-learning-beyond-the-range-space/) for a quick introduction of EI.
* Please find the core implementation of EI at './ei/closure/ei.py' ([ei.py](https://github.com/edongdongchen/EI/blob/master/ei/closure/ei.py)).
* Please find the 30 lines code [get_started.py](https://github.com/edongdongchen/EI/blob/master/get_started.py) and the [colab toy example](https://colab.research.google.com/github/edongdongchen/EI/blob/main/ei_demo_cs_usps.ipynb) to get started with EI.


## Test
We provide the trained models used in the paper which can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1Io0quD-RvoVNkCmE36aQYpoouEAEP5pF?usp=sharing).
Please put the downloaded folder 'ckp' in the root path. Then evaluate the trained models by running
```
python3 demo_test_inpainting.py
```
and
```
python3 demo_test_ct.py
```

## Train

To train EI for a given inverse problem (inpainting or CT), run
```
python3 demo_train.py --task 'inpainting'
```
or run a bash script to train all the models for both CT and inpainting tasks.
```
bash train_paper_bash.sh
```

### Train your models
To train your EI models on your dataset for a specific inverse problem (e.g. inpainting), run
```
python3 demo_train.py --h
```
* Note: you may have to implement the forward model (physics) if you manage to solve a new inverse problem.
* Note: you only need to specify some basic settings (e.g. the path of your training set).


### Citation

	@inproceedings{chen2021equivariant,
    title = {Equivariant Imaging: Learning Beyond the Range Space},
		author={Chen, Dongdong and Tachella, Juli{\'a}n and Davies, Mike E},
		booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
		year = {2021}
	}
