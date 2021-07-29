#!/usr/bin/env bash

python3 demo_train.py --gpu 0 --task 'inpainting' --mode 'ei'

python3 demo_train.py --gpu 0 --task 'inpainting' --mode 'sup'

python3 demo_train.py --gpu 0 --task 'inpainting' --mode 'mc'

python3 demo_train.py --gpu 0 --task 'inpainting' --mode 'sup_ei'

python3 demo_train.py --gpu 0 --task 'inpainting' --mode 'ei_adv'

python3 demo_train.py --gpu 0 --task 'ct' --mode 'ei' --epochs 5000 --lr 5e-4 --batch-size' 2 --ei-trans 5 --ei-alpha 100 --schedule [2000, 3000, 4000]

python3 demo_train.py --gpu 0 --task 'ct' --mode 'sup' --epochs 5000 --lr 5e-4 --batch-size' 2 --ei-trans 5 --ei-alpha 100 --schedule [2000, 3000, 4000]

python3 demo_train.py --gpu 0 --task 'ct' --mode 'mc' --epochs 5000 --lr 5e-4 --batch-size' 2 --ei-trans 5 --ei-alpha 100 --schedule [2000, 3000, 4000]

python3 demo_train.py --gpu 0 --task 'ct' --mode 'sup_ei' --epochs 5000 --lr 5e-4 --batch-size' 2 --ei-trans 5 --ei-alpha 100 --schedule [2000, 3000, 4000]

python3 demo_train.py --gpu 0 --task 'ct' --mode 'ei_adv' --epochs 5000 --lr 5e-4 --batch-size' 2 --ei-trans 5 --ei-alpha 100 --schedule [2000, 3000, 4000]