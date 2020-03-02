# coding: utf-8
import torch

def save_model(model, path='model/unet.pt'):
    """
    Save the model `model` to `path`\\
    Args: 
        path: The path of the model to be saved
        model: The model to save
    """
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    """
    Load `model` from `path`, and push `model` to `device`\\
    Args: 
        model: The model to save
        path: The path of the model to be saved
        device: the torch device
    Return:
        the loaded model
    """
    saved_params = torch.load(path)
    model.load_state_dict(saved_params)
    model = model.to(device)
    return model
