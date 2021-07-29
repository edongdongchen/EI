# Equivariant Imaging: Learning Beyond the Range Space

This repository is the official implementation of [Equivariant Imaging: Learning Beyond the Range Space](https://arxiv.org/abs/2103.14756) (ICCV'2021 oral paper)

by [Dongdong Chen](https://dongdongchen.com), [Julián Tachella](https://https://tachella.github.io/home/), [Mike E. Davies](https://scholar.google.co.uk/citations?user=dwmfR3oAAAAJ&hl=en).


## Requirements

* [PyTorch](https://pytorch.org/) (1.6)

All packages are listed in the below Anaconda environment.yml file, create an environment and run
```
conda env create -f environment.yml
```

## Train
```
python3 demo_train.py --gpu 0 --task 'inpainting' --mode 'ei'
```

## To do
*~~add core modules~~

*~~add demo of training~~

* add demo of testing

* add demo of single image reconstruction

* upload the trained models


### Citation

	@inproceedings{chen2021equivariant,
    title = {Equivariant Imaging: Learning Beyond the Range Space},
		author={Chen, Dongdong and Tachella, Juli{\'a}n and Davies, Mike E},
		booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
		year = {2021}
	}
