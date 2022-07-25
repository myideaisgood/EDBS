import torch
import torch.nn as nn

class TripletLoss(nn.Module):
  def __init__(self, margin=1.0):
    super(TripletLoss, self).__init__()
    self.margin = margin
  
  def calc_euclidean(self, x1, x2):
    return (x1-x2).pow(2).sum(1)
  
  def forward(self, anchor, positive, negative):
    d_p = self.calc_euclidean(anchor, positive)
    d_n = self.calc_euclidean(anchor, negative)
    losses = torch.relu(d_p - d_n + self.margin)

    return losses.mean()