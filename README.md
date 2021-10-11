# Equivariant Imaging: Learning Beyond the Range Space

### [post](https://tachella.github.io/2021/04/16/equivariant-imaging-learning-beyond-the-range-space/) | [paper](https://arxiv.org/abs/2103.14756) | [example](https://github.com/edongdongchen/EI/blob/main/ei_demo_cs_usps.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edongdongchen/EI/blob/main/ei_demo_cs_usps.ipynb)

[Equivariant Imaging: Learning Beyond the Range Space](https://arxiv.org/abs/2103.14756)

[Dongdong Chen](https://dongdongchen.com), [Julián Tachella](https://tachella.github.io/), [Mike E. Davies](https://www.research.ed.ac.uk/en/persons/michael-davies).

The University of Edinburgh

In ICCV 2021 (oral)


![flexible](https://github.com/edongdongchen/EI/blob/main/images/ct.png)
![flexible](https://github.com/edongdongchen/EI/blob/main/images/ipt.png)

Figure: **Learning to image from only measurements.** Training an imaging network through just measurement consistency (MC) does not significantly improve the reconstruction over the simple pseudo-inverse (<img src="https://render.githubusercontent.com/render/math?math=A^{\dagger}y">). However, by enforcing invariance in the reconstructed image set, equivariant imaging (EI) performs almost as well as a fully supervised network. Top: sparse view CT reconstruction, Bottom: pixel inpainting. PSNR is shown in top right corner of the images.

**EI** is a new `self-supervised`, `end-to-end` and `physics-based` learning framework for inverse problems with theoretical guarantees which leverages simple but fundamental priors about natural signals: `symmetry` and `low-dimensionality`.

## Get quickly started

* Please find the [blog post](https://tachella.github.io/2021/04/16/equivariant-imaging-learning-beyond-the-range-space/) for a quick introduction of EI.
* Please find the core implementation of EI at './ei/closure/ei.py' ([ei.py](https://github.com/edongdongchen/EI/blob/master/ei/closure/ei.py)).
* Please find the 30 lines code [get_started.py](https://github.com/edongdongchen/EI/blob/master/get_started.py) and the [toy cs example](https://github.com/edongdongchen/EI/blob/main/ei_demo_cs_usps.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edongdongchen/EI/blob/main/ei_demo_cs_usps.ipynb)
 to get started with EI.

## Overview
**The problem:** Imaging systems capture noisy 𝑚-dimensional measurements 𝑦 of a 𝑛-dimensional signal 𝑥 through a linear operator 𝐴: 𝑦=𝐴𝑥+𝜖. We aim to learn the reconstruction function 𝑓(𝑦)=𝑥 where
- `NO` groundtruth data {𝑥} for training as most inverse problems don’t have ground-truth;
- only a `single` forward operator 𝐴 is available;
- 𝐴 has a `non-trivial` nullspace (e.g. 𝑚<𝑛).

**The challenge:** 
- We have `NO` information about the signal set 𝒳 outside the range space of <img src="https://render.githubusercontent.com/render/math?math=A^{\top}"> or <img src="https://render.githubusercontent.com/render/math?math=A^{\dagger}">.
- It is `IMPOSSIBLE` to learn the signal set 𝒳 using {𝑦} alone.

**The motivation:** 
We assume the signal set <img src="https://render.githubusercontent.com/render/math?math=\mathcal{X}"> is `invariant` to a groups of transformations <img src="https://render.githubusercontent.com/render/math?math=T_g"> (orthgonal matrix, e.g. shift, rotation, scaling, reflection, etc.) related to a group <img src="https://render.githubusercontent.com/render/math?math=\mathcal{G}">, such that <img src="https://render.githubusercontent.com/render/math?math=T_gx\in \mathcal{X}"> and the sets <img src="https://render.githubusercontent.com/render/math?math=T_g\mathcal{X}">
and <img src="https://render.githubusercontent.com/render/math?math=\mathcal{X}"> are the same. For example,
1. natural images are shift invariant.
2. in CT/MRI data, organs can be imaged at different angles making the problem invariant to rotation.

Key observations: 
- Invariance provides access to implicit operators <img src="https://render.githubusercontent.com/render/math?math=A_g=AT_g"> with potentially different range spaces: <img src="https://render.githubusercontent.com/render/math?math=Ax=AT_gT_g^{\top}x=A_g\tilde{x}"> where <img src="https://render.githubusercontent.com/render/math?math=A_g=AT_g"> and <img src="https://render.githubusercontent.com/render/math?math=\tilde{x}=T_g^{\top}x">. Obviously, <img src="https://render.githubusercontent.com/render/math?math=\tilde{x}"> should also in the signal set.
- The composition 𝑓∘𝐴 is **equivariant** to the group of transformations <img src="https://render.githubusercontent.com/render/math?math={T_g}">: <img src="https://render.githubusercontent.com/render/math?math=f(AT_g x)=T_g f(Ax)">.

![overview](https://github.com/edongdongchen/EI/blob/main/images/invariance_iccv.png)

### Equivariant Imaging: to learn 𝑓 by using only measurements {𝑦}, all you need is to:
- Define:

1. define a transformation group  <img src="https://render.githubusercontent.com/render/math?math={T_g}"> based on the certain invariances to the signal set.
2. define a neural reconstruction function  <img src="https://render.githubusercontent.com/render/math?math=f_\theta: y\rightarrow x">, e.g. <img src="https://render.githubusercontent.com/render/math?math=f_\theta=G_\theta \circ A^{\dagger}"> where <img src="https://render.githubusercontent.com/render/math?math=A^{\dagger}"> is the pseudo-inverse of 𝐴 and <img src="https://render.githubusercontent.com/render/math?math=G_\theta: R^n\rightarrow R^n">  is a UNet-like neural net.

- Calculate:

1. calculate  <img src="https://render.githubusercontent.com/render/math?math=x^{(1)}=f_\theta(y)">  as the estimation of  𝑥.
2. calculate  <img src="https://render.githubusercontent.com/render/math?math=x^{(2)}=T_gx^{(1)}"> by transforming  <img src="https://render.githubusercontent.com/render/math?math=x^{(1)}">.
3. calculate  <img src="https://render.githubusercontent.com/render/math?math=x^{(3)}=f_\theta(Ax^{(2)})"> by reconstructing  <img src="https://render.githubusercontent.com/render/math?math=x^{(2)}">  from its measurement <img src="https://render.githubusercontent.com/render/math?math=Ax^{(2)}">.

![flowchart](https://github.com/edongdongchen/EI/blob/main/images/fig_flowchart.png)

- Train: finally learn the reconstruction function  <img src="https://render.githubusercontent.com/render/math?math=f_\theta">  by solving: <img src="https://render.githubusercontent.com/render/math?math=\arg\min_{\theta}\mathbb{E}_{y,g}"><img src="https://render.githubusercontent.com/render/math?math=\{L(Ax^{(1)}, y)"> + <img src="https://render.githubusercontent.com/render/math?math=\lambda L(x^{(2)}, x^{(3)})\}">


**Analysis:** check the paper for the detailed analysis. For more detailed theoretical guarantees see:

J. Tachella, D. Chen, M. E. Davies, Sampling Theorems for Unsupervised Learning in Inverse problems. Preprint 2021.


## Requirements

* [PyTorch](https://pytorch.org/) (1.6)

All used packages are listed in the Anaconda environment.yml file. You can create an environment and run
```
conda env create -f environment.yml
```


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
or run a bash script to train the models for both CT and inpainting tasks.
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
