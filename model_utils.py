import torch
import torch.nn as nn
import os
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
#     checkpoint = torch.load(weights, map_location='cuda:1')
#     checkpoint = torch.load(weights, map_location='cpu')


    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_checkpoint_two(model, weights, target):
    checkpoint = torch.load(weights)
#     checkpoint = torch.load(weights, map_location='cuda:1')
#     checkpoint = torch.load(weights, map_location='cpu')


    try:
        model.load_state_dict(checkpoint[target])
    except:
        state_dict = checkpoint[target]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    from model import Uformer, Unet_SI, SimpleUnet, SimpleUnet_MoreLayers, SimpleUnet_MoreLayers2, UformerSAM, SimpleUnet_MoreLayers_SAM, UformerRecursiveSAM, UformerSAM_More
    from Restormer import Restormer, Restormer_1st,  Restormer_multiGPU, RestormerSimple, SwinRestormer, SwinRestormer_MR_PET
    arch = opt.arch

    print('You choose '+arch+'...')
    if arch == 'UNet':
        model_restoration = UNet(dim=opt.embed_dim)
    elif arch == 'Uformer':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_embed=opt.token_embed,token_mlp=opt.token_mlp)
    elif arch == 'UformerSAM':
        model_restoration = UformerSAM(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_embed=opt.token_embed,token_mlp=opt.token_mlp)
    elif arch == 'UformerSAM_More':
        model_restoration = UformerSAM_More(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_embed=opt.token_embed,token_mlp=opt.token_mlp)
    elif arch == 'UformerRecursiveSAM':
        model_restoration = UformerRecursiveSAM(embed_dim=opt.embed_dim,win_size=opt.win_size,token_embed=opt.token_embed,token_mlp=opt.token_mlp)
    elif arch == 'Restormer':
        model_restoration = Restormer()
    elif arch == 'Restormer_1st':
        model_restoration = Restormer_1st()
    elif arch == 'Restormer_multiGPU':
        model_restoration = Restormer_multiGPU()
    elif arch == 'RestormerSimple':
        model_restoration = RestormerSimple(locked_attn = opt.locked_attn, Conv_type=opt.conv_block, dim=opt.embed_dim, ffn_expansion_factor = opt.ffn_factor)
    elif arch == 'SwinRestormer':
        model_restoration = SwinRestormer(swin_stages = opt.swin_stages, locked_attn = opt.locked_attn, pre_post_norm=opt.pre_post_norm, Conv_type=opt.conv_block, dim=opt.embed_dim, ffn_expansion_factor = opt.ffn_factor, mr = opt.mr, cross = opt.cross )
    elif arch == 'SwinRestormer_MR_PET':
        model_restoration = SwinRestormer_MR_PET(swin_stages = opt.swin_stages, locked_attn = opt.locked_attn, pre_post_norm=opt.pre_post_norm, Conv_type=opt.conv_block, dim=opt.embed_dim, ffn_expansion_factor = opt.ffn_factor, mr = opt.mr, cross = opt.cross )
    elif arch == 'SwinRestormer_MR_PET_sum':
        model_restoration = SwinRestormer_MR_PET(swin_stages = opt.swin_stages, locked_attn = opt.locked_attn, pre_post_norm=opt.pre_post_norm, Conv_type=opt.conv_block, dim=opt.embed_dim, ffn_expansion_factor = opt.ffn_factor, mr = opt.mr )
        
        
    elif arch == 'Unet_SI':
        model_restoration = Unet_SI(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_embed=opt.token_embed,token_mlp=opt.token_mlp)
    elif arch == 'SimpleUnet':
        model_restoration = SimpleUnet(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_embed=opt.token_embed,token_mlp=opt.token_mlp)
    elif arch == 'SimpleUnet_MoreLayers':
        model_restoration = SimpleUnet_MoreLayers(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_embed=opt.token_embed,token_mlp=opt.token_mlp)
        
    elif arch == 'SimpleUnet_MoreLayers2':
        model_restoration = SimpleUnet_MoreLayers2(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_embed=opt.token_embed,token_mlp=opt.token_mlp)
    elif arch == 'SimpleUnet_MoreLayers_SAM':
        model_restoration = SimpleUnet_MoreLayers_SAM(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_embed=opt.token_embed,token_mlp=opt.token_mlp)
        
    elif arch == 'Uformer16':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=16,win_size=8,token_embed='linear',token_mlp='leff')
    elif arch == 'Uformer32':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_embed='linear',token_mlp='leff')
    else:
        raise Exception("Arch error!")

    return model_restoration