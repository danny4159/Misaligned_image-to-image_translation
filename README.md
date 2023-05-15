# Misalign learning code to MONAI /or Pytorch-lightning

## Installation guide
### Setup environment (conda environment 설치)

```python
conda create -n misalign python=3.9 # install python 3.9 version
conda activate misalign
```
### Pytorch installation (ltt package 사용하여 자동으로 CUDA 받아오기)
```python
pip install light-the-torch
ltt install torch torchvision torchaudio
```
`light-the-torch package` : 자동으로 CUDA 버젼에 맞는 pytorch, torchvision, torchaudio를 설치해줌


### 이후 dependency 설치
```
pip install -r requirements.txt # 다니엘 기존 코드를 테스트 해봄, 잘 돌아감 
```