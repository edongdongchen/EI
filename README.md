# Equivariant Imaging: Learning Beyond the Range Space

This repository is the official implementation of [Equivariant Imaging: Learning Beyond the Range Space](https://arxiv.org/abs/2103.14756) (ICCV'2021 oral paper)

by [Dongdong Chen](https://dongdongchen.com), [Julián Tachella](https://https://tachella.github.io/home/), [Mike E. Davies](https://scholar.google.co.uk/citations?user=dwmfR3oAAAAJ&hl=en).


## Requirements

* [PyTorch](https://pytorch.org/) (1.6)

All used packages are listed in the below Anaconda environment.yml file, you can create an environment and run
```
conda env create -f environment.yml
```

## Test
We provide the trained models used in the paper and which can be downloaded at [Google Dirve](https://drive.google.com/drive/folders/1Io0quD-RvoVNkCmE36aQYpoouEAEP5pF?usp=sharing).
Please put the downloaded folder 'ckp' under the root path. Then evaluate the trained models by running
```
python3 demo_test_inpainting.py
```
and
```
python3 demo_test_ct.py
```

## Train

To train EI for a task (inpainting or CT), run
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
* Note: you may have to implement the forward model (physics) if you mannage to solve a new inverse problem.
* Note: what you only need to do then is specifying some hyperparameters (e.g. the path of your trainingset).


## To Do
* ~~add core modules~~ [DONE]

* add illustrations

* ~~add demo of training~~  [DONE]

* ~~add demo of testing~~  [DONE]

* add demo of single image reconstruction

* ~~upload the trained model~~  [DONE]


### Citation

	@inproceedings{chen2021equivariant,
    title = {Equivariant Imaging: Learning Beyond the Range Space},
		author={Chen, Dongdong and Tachella, Juli{\'a}n and Davies, Mike E},
		booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
		year = {2021}
	}
