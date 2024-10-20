import torch
import torch.nn.functional as F

def cosine_similarity_torch(A, B):
    # Step 1: Compute cosine similarity using PyTorch built-in function
    cos_sim = F.cosine_similarity(A, B, dim=1)
    
    return cos_sim

# Example:
A = torch.tensor([[1.0, 2.0, 3.0], 
                  [-4.0, -5.0, -6.0],
                  [7.0, 8.0, 9.0]]
                  )

B = torch.tensor([[1.0, 2.0, 3.0], 
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]]
                  )

cos_sim = cosine_similarity_torch(A, B)
print(cos_sim)