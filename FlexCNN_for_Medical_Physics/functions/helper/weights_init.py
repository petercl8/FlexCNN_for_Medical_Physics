import torch
from torch import nn

def weights_init(m): # 'm' represents layers in the generator or discriminator.

    # Function for initializing network weights to normal distribution, with mean 0 and s.d. 0.02
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def weights_init_he(m): # 'm' represents layers in the generator or discriminator.

    # Function for initializing network weights using He initialization (Kaiming initialization)
    # He initialization is designed for ReLU activations and helps with gradient flow
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)