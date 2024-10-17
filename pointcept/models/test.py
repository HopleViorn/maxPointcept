
import numpy as np
import torch

a=np.array([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]]).astype(np.float64)

# a[:,:3]-=np.mean(a[:,:3],axis=0,keepdims=True)

print(a)