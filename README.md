# Introduction

- Misaligned image to image translation 문제를 해결하기위해 해당 코드에서 실험을 진행.

- 아래 논문에서 공유한 코드를 Baseline Model로 사용.

- 아래 논문에서는 대표적으로 pGAN, cGAN 모델을 소개함. (pGAN: supervised learning, cGAN: unsupervised learning)

- pGAN을 Base로 하는 'ourGAN'이라는 모델을 새롭게 구축하여 이곳에서 기능을 추가하며 비교 실험을 진행.

- [[paper]](https://ieeexplore.ieee.org/abstract/document/8653423) Dar SUH, Yurt M, Karacan L, Erdem A, Erdem E, Çukur T. Image Synthesis in Multi-Contrast MRI with Conditional Generative Adversarial Networks. IEEE Transaction on Medical Imaging. 2019. <br /><br /><br />

# Dataset

- IXI dataset을 통해 MRI T1, T2 이미지 translation을 수행.

- [Dataset Download](https://drive.google.com/drive/u/1/folders/1En_S9c081T2hV-joaFJv3xlMX2Eqzl5V)

- 위 데이터는 IXI dataset에서 전처리가 이루어진 데이터셋

- [전처리 내용]
  - train: 25명, val: 5명, test:10명
  - 각 환자에 대해서 유효한 slice만 활용 (환자당 91개 slice만 활용) 
  - ex. [3,910,256,256] -> [인접한 slice, 전체 slice(10명), x, y]
  - 인접한 slice: slice가 한 칸씩만 밀려진 데이터가 그대로 들어가 있음 (pGAN의 contribution 중 인접한 slice를 활용하여 loss를 구함)
  - 폴더 구성 방법: 'pGAN-cGAN/datasets/IXI' 이 경로에 train, val, test 폴더를 두고 각 폴더에 해당하는 mat 파일을 둔다. <br /><br /><br />

# Model 

>## ourGAN
- pGAN을 Base로 함 (Supervised learning)<br>
- 모듈을 추가해가며 비교 실험 진행 예정
### Training
python train.py --model ourGAN --dataroot datasets/IXI --name ourGAN_run --which_direction BtoA --lambda_A 100 --batchSize 16 --output_nc 1 --input_nc 3 --gpu_ids 0,1 --niter 50 --niter_decay 50 --save_epoch_freq 25 --lambda_vgg 100 --checkpoints_dir checkpoints/ --training --dataset_misalign
### Test
python test.py --model ourGAN --dataroot datasets/IXI --name ourGAN_run --which_direction BtoA --phase test --output_nc 1 --input_nc 3 --how_many 1200 --results_dir results/ --checkpoints_dir checkpoints/ --gpu_ids 0,2 <br /><br />

>## pGAN
### Training
python train.py --model pGAN --dataroot datasets/IXI --name pGAN_run --which_direction BtoA --lambda_A 100 --batchSize 16 --output_nc 1 --input_nc 3 --gpu_ids 0,1 --niter 50 --niter_decay 50 --save_epoch_freq 25 --lambda_vgg 100 --checkpoints_dir checkpoints/ --training --dataset_misalign
### Test
python test.py --model pGAN --dataroot datasets/IXI --name pGAN_run --which_direction BtoA --phase test --output_nc 1 --input_nc 3 --how_many 1200 --results_dir results/ --checkpoints_dir checkpoints/ --gpu_ids 1,2,4 <br /><br />

>## cGAN
### Training
python train.py --model cGAN --dataroot datasets/IXI --name cGAN_run --which_direction BtoA --batchSize 16 --output_nc 1 --input_nc 1 --gpu_ids 0,1 --niter 50 --niter_decay 50 --save_epoch_freq 25 --lambda_A 100 --lambda_B 100 --checkpoints_dir checkpoints/ --dataset_mode unaligned_mat --training --dataset_misalign
### Test
python test.py --model cGAN --dataroot datasets/IXI --name cGAN_run --phase test --output_nc 1 --input_nc 1 --how_many 1200 --results_dir results/ --checkpoints_dir checkpoints/ <br /><br />

>## 코드 추가 설명
1. Visdom 실행 (모든 코드 실행 전) <br>
python -m visdom.server -p 8097

2. Parser <br>
--model: 사용할 모델 (pGAN,cGAN,ourGAN)<br>
--dataroot: 데이터셋 경로<br>
--name: 결과 저장될 폴더명<br>
--dataset_misalign: 데이터셋에 misalign 적용 여부 (torchio.RandomAffine을 통해 의도적인 misalign을 가할 수 있어. /data/__init__.py 코드 참고)<br>
그 외 options 폴더 안 parser에 대한 코드 참고<br><br>

# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@article{dar2019image,
  title={Image Synthesis in Multi-Contrast MRI with Conditional Generative Adversarial Networks},
  author={Dar, Salman UH and Yurt, Mahmut and Karacan, Levent and Erdem, Aykut and Erdem, Erkut and {\c{C}}ukur, Tolga},
  journal={IEEE Transaction on Medical Imaging},
  year={2019},
  publisher={IEEE}
}
```
For any questions, comments and contributions, please contact Salman Dar (salman[at]ee.bilkent.edu.tr) <br />

(c) ICON Lab 2019<br><br>


# Acknowledgments
This code is based on implementations by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan) and [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

This code was written by [icon-lab/pGAN-cGAN](https://github.com/icon-lab/pGAN-cGAN) and we only modified it.