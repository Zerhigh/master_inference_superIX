import torch
import yaml

from superIX.swin2_mose.model import Swin2MoSE


def to_shape(t1, t2):
    t1 = t1[None].repeat(t2.shape[0], 1)
    t1 = t1.view((t2.shape[:2] + (1, 1)))
    return t1


def norm(tensor, mean, std):
    # get stats
    mean = torch.tensor(mean).to(tensor.device)
    std = torch.tensor(std).to(tensor.device)
    # denorm
    return (tensor - to_shape(mean, tensor)) / to_shape(std, tensor)


def denorm(tensor, mean, std):
    # get stats
    mean = torch.tensor(mean).to(tensor.device)
    std = torch.tensor(std).to(tensor.device)
    # denorm
    return (tensor * to_shape(std, tensor)) + to_shape(mean, tensor)


def load_config(path):
    # load config
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_swin2_mose(model_weights, cfg):
    # load checkpoint
    checkpoint = torch.load(model_weights)

    # build model
    sr_model = Swin2MoSE(**cfg['super_res']['model'])
    sr_model.load_state_dict(
        checkpoint['model_state_dict'])

    sr_model.cfg = cfg

    return sr_model


def run_swin2_mose(model, lr, device='cuda:0'):
    # select 10m lr bands: B02, B03, B04, B08 and hr bands
    # automatically rearranged to r,g,b,nir
    lr = torch.from_numpy(lr)[None].float()[:, [3, 2, 1, 7]].to(device)

    # predict a image
    with torch.no_grad():
        sr = model(lr)
        if not torch.is_tensor(sr):
            sr, _ = sr

    sr = sr.cpu().numpy().squeeze()

    return sr