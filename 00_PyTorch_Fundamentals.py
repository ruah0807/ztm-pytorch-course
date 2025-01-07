import pandas as pd
import numpy as np
import torch
import sklearn
import matplotlib
import torchinfo, torchmetrics

# Check PyTorch access (should print out a tensor)
# print(torch.randn(3, 3))

# Check for GPU (should return True)
# print(torch.cuda.is_available())

print(torch.__version__)
print(f"MPS 장치를 지원하도록 build가 되었는가? {torch.backends.mps.is_built()}")
print(f"MPS 장치가 사용 가능한가? {torch.backends.mps.is_available()}") 

# 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"사용 디바이스: {device}")


e = torch.rand(5, 3)
print(e)

## Introduction to Tensors

### 1. Creating Tensors
#### Pytorch tensors are created using`torch.tensor() = https://pytorch.org/docs/stable/tensors.html`
# scalar
print("\n\n########## SCALAR")
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim) # scalar의 차원

# Get tensor back as Python int
print(scalar.item()) # scalar의 값

# Vector
print("\n\n########## VECTOR")
vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim) # vector의 차원
print(vector.shape) # vector의 크기

# Matrix
print("\n\n########## MATRIX")
MATRIX = torch.tensor([[7,8],
                       [9,10]])
print(MATRIX)
print(MATRIX.ndim) # matrix의 차원
print(MATRIX[1])
print(MATRIX.shape) # matrix의 크기


# TENSOR
print("\n\n########## TENSOR")
TENSOR = torch.tensor([[[1,2,3,],
                        [3,6,9,],
                        [3,5,4]]])
print(TENSOR)
print(TENSOR.ndim) # tensor의 차원
print(TENSOR.shape) # tensor의 크기
print(TENSOR[0])


# Random tensors
print("\n\n########## RANDOM TENSORS")
### Why Random Tensors?
# Random tensors are important because the way many neural networks learn is that they start with tensors full of random numbers and then adjust those random numbers to better represent the data.

# - 데이터 분석 및 모델 테스트를 위해 무작위 데이터를 생성하는 것이 중요.
# - 무작위 데이터를 생성하면 모델이 데이터를 학습하고 예측하는 능력을 평가할 수 있음.

""" start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers """

#Create random tensor of size (3, 4)
random_tensor = torch.rand(1, 3, 4)
print(random_tensor)
print(random_tensor.ndim)# 텐서의 차원
print(random_tensor.shape)# 텐서의 크기

# Create random tensor with similar shape to an image
random_image_size_tensor = torch.rand(224, 224, 3) # height, width, color channels
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)
