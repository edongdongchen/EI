#!/usr/bin/env bash

python3 demo_train.py --gpu 0 --task 'inpainting' --mode 'ei' --epochs 2000 --lr 1e-3 --schedule 500 1000 1500

python3 demo_train.py --gpu 0 --task 'inpainting' --mode 'sup' --epochs 2000 --lr 1e-3 --schedule 500 1000 1500

python3 demo_train.py --gpu 0 --task 'inpainting' --mode 'mc' --epochs 2000 --lr 1e-3 --schedule 500 1000 1500

python3 demo_train.py --gpu 0 --task 'inpainting' --mode 'sup_ei' --epochs 2000 --lr 1e-3 --schedule 500 1000 1500

python3 demo_train.py --gpu 0 --task 'inpainting' --mode 'ei_adv' --epochs 2000 --lr 1e-3 --schedule 500 1000 1500

python3 demo_train.py --gpu 0 --task 'ct' --mode 'ei' --epochs 5000 --lr 5e-4 --batch-size 2 --ei-trans 5 --ei-alpha 100 --schedule 2000 3000 4000

python3 demo_train.py --gpu 0 --task 'ct' --mode 'sup' --epochs 5000 --lr 5e-4 --batch-size 2 --schedule 2000 3000 4000

python3 demo_train.py --gpu 0 --task 'ct' --mode 'mc' --epochs 5000 --lr 5e-4  --batch-size 2 --schedule 2000 3000 4000

python3 demo_train.py --gpu 0 --task 'ct' --mode 'sup_ei' --epochs 5000 --lr 5e-4 --batch-size 2 --ei-trans 5 --ei-alpha 100 --schedule 2000 3000 4000

python3 demo_train.py --gpu 0 --task 'ct' --mode 'ei_adv' --epochs 5000 --lr 5e-4 --batch-size 2 --ei-trans 5 --ei-alpha 100 --schedule 2000 3000 4000
