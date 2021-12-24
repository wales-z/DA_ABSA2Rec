import numpy as np
import torch
import torch.nn as nn
import torchvision
# print(torch.__version__) # 1.10.0+cu102
# print(torch.version.cuda) # 10.2
# print(torch.backends.cudnn.version()) # 7605
# print(torch.cuda.get_device_name(0)) # GeForce RTX 2080 Ti

# 固定随机种子以保证可复现性
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')