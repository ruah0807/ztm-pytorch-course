{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 00_PyTorch_Fundamentals\n",
    "\n",
    "Resource notebook: https://www.learnpytorch.io/00_pytorch_fundamentals/\n",
    "\n",
    "If you have a question : https://github.com/mrdbourke/pytorch-deep-learning/discussions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import torch\n",
    "import sklearn\n",
    "import matplotlib\n",
    "import torchinfo, torchmetrics\n",
    "\n",
    "# Check PyTorch access (should print out a tensor)\n",
    "# print(torch.randn(3, 3))\n",
    "\n",
    "# Check for GPU (should return True)\n",
    "# print(torch.cuda.is_available())\n",
    "\n",
    "print(torch.__version__)\n",
    "print(f\"MPS 장치를 지원하도록 build가 되었는가? {torch.backends.mps.is_built()}\")\n",
    "print(f\"MPS 장치가 사용 가능한가? {torch.backends.mps.is_available()}\") \n",
    "\n",
    "# 디바이스 설정\n",
    "device = torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")\n",
    "print(f\"사용 디바이스: {device}\")\n",
    "\n",
    "\n",
    "e = torch.rand(5, 3)\n",
    "print(e)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "pytorch",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
