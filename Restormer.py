## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import os 

from einops import rearrange

from SwinTransformer import SwinTransformer  



##########################################################################
## Layer Norm

def to_3d(x):
#     print('x.size(): ', x.size())
#     return rearrange(x, 'b c h w -> b (h w) c')
    return rearrange(x, 'b c h w d -> b (h w d) c') # by SI

def to_4d(x,h,w,d):
#     print('x.size(): ', x.size())
#     return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
    return rearrange(x, 'b (h w d) c -> b c h w d',h=h,w=w,d=d)  # by SI

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
#         print('normalized_shape ', normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
#         print('x.size(): ', x.size())
#         print('mu.size(): ', mu.size())
#         print('sigma.size(): ', sigma.size())
#         print('self.weight.size(): ', self.weight.size())
#         print('self.bias.size(): ', self.bias.size())
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
#         h, w = x.shape[-2:]
        h, w, d = x.shape[-3:] # by SI
#         print('x.shape[-3:]: ', x.shape[-3:])
#         stx()
        return to_4d(self.body(to_3d(x)), h, w, d)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features  = int(dim*ffn_expansion_factor)

        self.md          = hidden_features*2
#         print('dim ', dim)
#         print('self.md ', self.md)
#         print('hidden_features ', hidden_features)
        self.project_in  = nn.Conv3d(dim,     self.md, kernel_size=1, bias=bias) #modification Apr21

        self.dwconv      = nn.Conv3d(self.md, self.md, kernel_size=3, stride=1, padding=1, groups=self.md, bias=bias)
#         self.dwconv      = nn.Conv3d(self.md, self.md, kernel_size=7, stride=1, padding=3, groups=self.md, bias=bias)

#         self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)  #modification Apr21

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SPFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(SPFeedForward, self).__init__()

        hidden_features  = int(dim*ffn_expansion_factor)
#         hidden_features  = int(dim)

        self.md          = hidden_features*2
        self.project_in  = nn.Conv3d(dim,     self.md, kernel_size=1, bias=bias)

        self.dwconv      = nn.Conv3d(self.md, self.md, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.dwconv      = nn.Conv3d(self.md, self.md, kernel_size=7, stride=1, padding=3, bias=bias)
    
#         self.dwconv5     = nn.Conv3d(self.md, self.md, kernel_size=7, stride=1, padding=3, bias=bias)

#         self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

#         x3, x4 = self.dwconv5(x).chunk(2, dim=1)
#         x = self.project_out(torch.cat([F.gelu(x1) * x2, F.gelu(x3) * x4], 1))
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, locked_attn):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        m        = 3
        self.qkv = nn.Conv3d(dim, dim*m, kernel_size=1,  bias=bias)  #modification Apr21
        self.qkv_dwconv = nn.Conv3d(dim*m, dim*m, kernel_size=3, stride=1, padding=1, groups=dim*m, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1,  bias=bias)  #modification Apr21
        self.locked_attn = locked_attn
#         self.w = nn.Parameter(torch.diag(torch.rand(int(dim/num_heads))))
        if (self.locked_attn == 1):
            self.w = nn.Parameter(torch.diag(torch.rand(int(dim/num_heads))))
        


    def forward(self, x):
        b,c,h,w,d = x.shape # by SI

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
#         print('self.q ', q.size())
        
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = rearrange(q, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads) # by SI
        k = rearrange(k, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads) # by SI
        v = rearrange(v, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads) # by SI

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        if (self.locked_attn == 1):
#             k = F.gelu(k)
#             print('self.w ', self.w.size())
#             print('k ', k.size())
#             print('k.transpose(-2, -1) ', k.transpose(-2, -1).size())
            k = self.w@k
#         print('self.temperature ', self.temperature)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = rearrange(out, 'b head c (h w d) -> b (head c) h w d', head=self.num_heads, h=h, w=w, d=d) # by SI

        out = self.project_out(out)
        return out
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, locked_attn):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        m        = 3
        self.qkv = nn.Conv3d(dim, dim*m, kernel_size=1,  bias=bias)  #modification Apr21
        self.qkv_dwconv = nn.Conv3d(dim*m, dim*m, kernel_size=3, stride=1, padding=1, groups=dim*m, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1,  bias=bias)  #modification Apr21
        self.locked_attn = locked_attn
#         self.w = nn.Parameter(torch.diag(torch.rand(int(dim/num_heads))))
        if (self.locked_attn == 1):
            self.w = nn.Parameter(torch.diag(torch.rand(int(dim/num_heads))))
        


    def forward(self, x):
        b,c,h,w,d = x.shape # by SI

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = rearrange(q, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads) # by SI
        k = rearrange(k, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads) # by SI
        v = rearrange(v, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads) # by SI

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        if (self.locked_attn == 1):
#             k = F.gelu(k)
#             print('self.w ', self.w.size())
#             print('k ', k.size())
#             print('k.transpose(-2, -1) ', k.transpose(-2, -1).size())
            k = self.w@k
#         print('self.temperature ', self.temperature)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = rearrange(out, 'b head c (h w d) -> b (head c) h w d', head=self.num_heads, h=h, w=w, d=d) # by SI

        out = self.project_out(out)
        return out

##########################################################################
## Spatial Attention 
class SPAttention(nn.Module):
    def __init__(self, dim, bias):
        super(SPAttention, self).__init__()
        self.conv0 = nn.Conv3d(dim,   dim*3, kernel_size=1,  bias=bias)
        self.conv1 = nn.Conv3d(dim*3, dim*3, kernel_size=3,  stride=1, padding=1, bias=bias)            
        self.conv2 = nn.Conv3d(dim*3, dim,   kernel_size=1,  bias=bias)
        


    def forward(self, x):
        b,c,h,w,d = x.shape # by SI

##
class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        gelu = nn.GELU()

        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, gelu)
        
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = Conv3dReLU(in_c, out_c, 3, 1, use_batchnorm=False)#Conv3dReLU(in_c, out_c)
        self.pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)

        return x
        

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm  (dim, LayerNorm_type)
        self.attn  = Attention  (dim, num_heads, bias)
        self.norm2 = LayerNorm  (dim, LayerNorm_type)
        self.ffn   = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class TransformerBlockSimple(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, Conv_type, pre_post_norm, locked_attn):
        super(TransformerBlockSimple, self).__init__()
#         if (Conv_type == 1):
#             self.norm0 = LayerNorm(dim, LayerNorm_type)
#             self.conv0 = nn.Conv3d(dim, dim*3, kernel_size=1,  bias=bias)
#             self.conv1 = nn.Conv3d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, bias=bias)
# #             self.conv1 = nn.Conv3d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
# #             self.conv11= nn.Conv3d(dim*3, dim*3, kernel_size=5, stride=1, padding=2, groups=dim*3, bias=bias)
            
#             self.conv2 = nn.Conv3d(dim*3, dim, kernel_size=1,  bias=bias)
            
        self.conv_type = Conv_type
        self.pre_post_norm = pre_post_norm
#         self.norm11 = LayerNorm(dim, LayerNorm_type)
#         self.norm22 = LayerNorm(dim, LayerNorm_type)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, locked_attn)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        if (self.conv_type == 1):
#             self.cv1  = encoder_block(dim, dim)
#             self.norm3 = LayerNorm(dim, LayerNorm_type)
            self.norm3 = LayerNorm(dim, LayerNorm_type)
            self.ffn2 = SPFeedForward(dim, ffn_expansion_factor, bias)
        

    def forward(self, x):
#         x = self.norm0(x)
#         x0 = self.conv0(x)
#         x1 = self.conv1(x)
#         x = torch.cat([self.conv0(x), self.conv1(x)], 1)
#         x = x + self.conv2(torch.cat([self.conv0(self.norm0(x)), self.conv1(self.norm0(x))], 1))
#         print('self.conv_type: ', self.conv_type)
#         if (self.conv_type == 1):
#             x =  F.gelu(self.conv2(self.conv1(self.conv0(self.norm0(x)))))
#             self.conv1(self.conv0(self.norm0(x)))
#             x =  self.conv2(torch.cat([self.conv1(self.conv0(self.norm0(x))), self.conv11(self.conv0(self.norm0(x)))], 1))
#             t = self.conv0(self.norm0(x))
#             x =  self.conv2(torch.cat([self.conv1(t), self.conv11(t)], 1))
#         if (self.conv_type == 1):
# #             x = self.cv1(self.norm3(x))
#             x = x + self.ffn2(self.norm3(x))
        
        if (self.pre_post_norm == 0): # pre norm
            x =     x + self.attn(self.norm1(x))
            x =     x + self.ffn (self.norm2(x))
            if (self.conv_type == 1):
                x = x + self.ffn2(self.norm3(x))
        else:
            x =     x + self.norm1(self.attn((x)))
            x =     x + self.norm2(self.ffn ((x)))
            if (self.conv_type == 1):
                x = x + self.norm3(self.ffn2(x))

        return x
    
    



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
#         print('x ', x.size())
        x = self.proj(x)

        return x

    


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
class DownsampleSimple(nn.Module):
    def __init__(self, n_feat):
        super(DownsampleSimple, self).__init__()
        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=False))
#         self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
#                                   nn.AvgPool3d(3, stride=2, padding=1))

    def forward(self, x):
        return self.body(x)

class UpsampleSimple(nn.Module):
    def __init__(self, n_feat):
        super(UpsampleSimple, self).__init__()
        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))

    def forward(self, x):
        return self.body(x)
    
############################
class SwinRestormer(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
#         dim = 96,
#         dim = 48, # dim = 96
        dim = 16, # by SI
#         dim = embed_dim,
#         num_blocks = [1,4,4,6], 
         num_blocks = [1,2,4,6], 
#         num_blocks = [4,6,6,8], 
        num_refinement_blocks = 1,
#         heads = [1,1,1,1],
        heads = [1,2,4,8],
#         ffn_expansion_factor = 2.66,
        ffn_expansion_factor = 1,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        Conv_type = 0,   ## Other option 'BiasFree'
        pre_post_norm = 0,
        locked_attn  = 0,
        swin_stages  = 2,
        mr = 0,
        cross = 0,
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(SwinRestormer, self).__init__()
        print('dim ', dim)
        print('ffn_factor ', ffn_expansion_factor)
        print('pre_post_norm ', pre_post_norm)
        print('locked_attn ', locked_attn)
        print('swin_stages ', swin_stages)
        print('mr ', mr)
        print('cross ', cross)
        if (mr == 1):
            inp_channels = 2
        if (cross == 1):
            inp_channels = 1
            
        self.mr = mr
        self.cross = cross
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
#         self.stride      = 0
#         self.kernel_s    = 3
        self.encoder_level1 = nn.Sequential( 
            *[TransformerBlockSimple(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[0])]
        )
        
        self.down1_2 = DownsampleSimple(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential( 
                *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[1])])
        
        self.down2_3 = DownsampleSimple(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[2])])

        self.down3_4 = DownsampleSimple(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[3])])
        
        self.reduce_chan_level4 = nn.Conv3d(int(dim*2**4), int(dim*2**3), kernel_size=1, bias=bias)
#         print('int(dim*2**4) :', int(dim*2**4))
#         print('int(dim*2**3) :', int(dim*2**3))
        #### swin transformer modifictaion !!!!!!!!!!!!!!!!!!
#         self.decoder_level4 = nn.Sequential( 
#             *[TransformerBlockSimple(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[3])])

    
        self.up4_3 = UpsampleSimple(int(dim*2**3)) ## From Level 4 to Level 3
#         self.reduce_chan_level3 = nn.Conv3d(int(dim*2**3)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.reduce_chan_level3 = nn.Conv3d(int(dim*2**3)+int(dim*4), int(dim*2**2), kernel_size=1, bias=bias)
#         self.reduce_chan_level3 = nn.Conv3d(int(dim*2**3)+int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        
        self.decoder_level3 = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[2])])


        self.up3_2 = UpsampleSimple(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[1])])
        
        self.up2_1 = UpsampleSimple(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv3d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output  = nn.Conv3d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.swin_stages = swin_stages
        if (self.swin_stages == 2):
            swin_depths      = (4, 8)
            swin_num_heads   = (4, 8)
            swin_out_indices = (0, 1)
            window_size      = (6, 6, 6)
        elif (self.swin_stages == 4):
            swin_depths      = (2, 2, 18, 2)
#             swin_num_heads   = (4, 4, 8, 8)
            swin_num_heads   = (6, 12, 24, 48)
            swin_out_indices = (0, 1, 2, 3)
            window_size      = (3, 3, 3)
            
            self.swin_up4_3 = UpsampleSimple(int(dim*2**5))  
            self.swin_reduce_chan_level3 = nn.Conv3d(int(dim*2**5), int(dim*2**4), kernel_size=1, bias=bias)
            self.swin_decoder_level3 = nn.Sequential(
                *[TransformerBlockSimple(dim=int(dim*2**4), num_heads=swin_num_heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(swin_depths[3])])
            
            self.swin_up3_2 = UpsampleSimple(int(dim*2**4))  
            self.swin_reduce_chan_level2 = nn.Conv3d(int(dim*2**4), int(dim*2**3), kernel_size=1, bias=bias)
            self.swin_decoder_level2 = nn.Sequential(
                *[TransformerBlockSimple(dim=int(dim*2**3), num_heads=swin_num_heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(swin_depths[2])])

            
        self.swin    = SwinTransformer(patch_size= 4,
                                           in_chans= inp_channels,
                                           embed_dim= dim*2**2,
                                           depths= swin_depths,   #(1, 5), #( 2, 2, 10, 2) ,
                                           num_heads= swin_num_heads, #(4, 4), #(4, 4, 8, 8),
                                           window_size= window_size,#(3, 3, 3),
                                           mlp_ratio= 4,
                                           qkv_bias= False,
                                           drop_rate= 0,
                                           drop_path_rate= 0.3,
                                           ape= True,
                                           spe= False,
                                           patch_norm= True,
                                           use_checkpoint= False,
                                           out_indices= swin_out_indices,
                                           pat_merg_rf= 4,
                                           pos_embed_method= 'relative',
                                           locked_attn=locked_attn)
        

        
        
    def forward(self, inp_img):
#         print('inp_img ', inp_img.size()) # torch.Size([1, 1, 96, 96, 96])
        pet_img = inp_img[:, 0, :, :, :].unsqueeze(1)
        mr_img = inp_img[:, 1, :, :, :].unsqueeze(1)
#         print('pet_img ', pet_img.size()) # torch.Size([1, 1, 96, 96, 96])
#         print('mr_img ', mr_img.size()) # torch.Size([1, 1, 96, 96, 96])
        if (self.cross == 0):
            inp_enc_level1 = self.patch_embed(inp_img)
        else:
            inp_enc_level1 = self.patch_embed(pet_img)
        
        
#         print('inp_enc_level1 ', inp_enc_level1.size())
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
#         print('out_enc_level1 ', out_enc_level1.size()) # torch.Size([1, 48, 96, 96, 96])
        inp_enc_level2 = self.down1_2(out_enc_level1)
#         print('inp_enc_level2 ', inp_enc_level2.size())
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
#         print('out_enc_level2 ', out_enc_level2.size()) # torch.Size([1, 96, 48, 48, 48])
        inp_enc_level3 = self.down2_3(out_enc_level2)
#         print('inp_enc_level3 ', inp_enc_level3.size())
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
#         print('out_enc_level3 ', out_enc_level3.size()) # torch.Size([1, 192, 24, 24, 24])
        inp_enc_level4 = self.down3_4(out_enc_level3)        
#         print('inp_enc_level4 ', inp_enc_level4.size()) #torch.Size([1, 384, 12, 12, 12])
        latent = self.latent(inp_enc_level4) 
#         print('latent ', latent.size()) # torch.Size([1, 384, 12, 12, 12])
        if (self.cross == 0):
            out = self.swin(inp_img)
        else:
            out = self.swin(pet_img)
        
#         print('inp_img ', inp_img.size()) # torch.Size([1, 1, 96, 96, 96])
        if (self.swin_stages == 2):
            swin_latent         = out[-1]
            swin_out_enc_level1 = out[-2]
#             print('latent :', latent.size())
#             print('swin_latent :', swin_latent.size())
            latent = torch.cat([latent, swin_latent], 1)
#             print('latent2 :', latent.size())
            latent = self.reduce_chan_level4(latent)
#             latent = self.decoder_level4(latent)
#             print('latent3 :', latent.size())
            inp_dec_level3 = self.up4_3(latent)
#             print('inp_dec_level3 :', inp_dec_level3.size())
#             print('out_enc_level3 :', out_enc_level3.size())
#             print('swin_out_enc_level1 :', swin_out_enc_level1.size())
            
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3, swin_out_enc_level1], 1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        elif (self.swin_stages == 4):
            swin_latent         = out[-1]
            swin_out_enc_level3 = out[-2]
            swin_out_enc_level2 = out[-3]
            swin_out_enc_level1 = out[-4]
            
#             print('swin_latent ', swin_latent.size())
#             print('self.swin_up4_3(swin_latent) ', self.swin_up4_3(swin_latent).size())
            swin_latent         = torch.cat([self.swin_up4_3(swin_latent), swin_out_enc_level3], 1)
            swin_latent         = self.swin_reduce_chan_level3(swin_latent)
            swin_out_dec_level3 = self.swin_decoder_level3(swin_latent)
            
            swin_out_enc_level2 = torch.cat([self.swin_up3_2(swin_out_dec_level3), swin_out_enc_level2], 1)
            swin_out_enc_level2 = self.swin_reduce_chan_level2(swin_out_enc_level2)
            swin_out_dec_level2 = self.swin_decoder_level2(swin_out_enc_level2)
            
            
            
            latent = torch.cat([latent, swin_out_dec_level2], 1)
            latent = self.reduce_chan_level4(latent)

            inp_dec_level3 = self.up4_3(latent)
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3, swin_out_enc_level1], 1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            out_dec_level3 = self.decoder_level3(inp_dec_level3) 
            
#         print('f0 ', swin_latent.size())
#         print('f1 ', swin_out_enc_level3.size())
#         print('f2 ', swin_out_enc_level2.size())
#         print('f3 ', swin_out_enc_level1.size())
#         f0  torch.Size([1, 1536, 3, 3, 3])
#         f1  torch.Size([1, 768, 6, 6, 6])
#         f2  torch.Size([1, 384, 12, 12, 12])
#         f3  torch.Size([1, 192, 24, 24, 24])

#         f0  torch.Size([1, 96*2, 12, 12, 12])
#         f1  torch.Size([1, 48*2, 24, 24, 24])
# f0  torch.Size([1, 384, 12, 12, 12])
# f1  torch.Size([1, 192, 24, 24, 24])
#         print('f2 ', f2.size())
#         print('f3 ', f3.size())
#         print('latent ', latent.size())

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
#         print('out_enc_level1: ', out_enc_level1.size())
        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
#             out_dec_level1 = self.UpSam(out_dec_level1)
            if self.mr == 1:
                inp_img = torch.unsqueeze(inp_img[:, 0, :, :, :], 1)
#             print('inp_img: ', inp_img.size())
            out_dec_level1 = self.output(out_dec_level1) + inp_img
            
#         print('out_enc_level1: ', out_enc_level1.size())
#         out_dec_level1 = self.UpSam(out_dec_level1 )
#         print('out_enc_level1: ', out_enc_level1.size())
        return out_dec_level1
    def ExtractFeatures(self, inp_img):
        x = self.patch_embed(inp_img)

        x = self.encoder_level1(x)

        
        x = self.down1_2(x)
        x = self.encoder_level2(x)

        x = self.down2_3(x)
        x = self.encoder_level3(x) 

        x = self.down3_4(x)        
        x = self.latent(x) 
        
        x1 = self.swin(inp_img)
        x1 = x1[-1]
        x = torch.cat([x, x1], 1)
        
        return x

class SwinRestormer_MR_PET(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
#         dim = 96,
#         dim = 48, # dim = 96
        dim = 16, # by SI
#         dim = embed_dim,
#         num_blocks = [1,4,4,6], 
         num_blocks = [1,2,4,6], 
#         num_blocks = [4,6,6,8], 
        num_refinement_blocks = 1,
#         heads = [1,1,1,1],
        heads = [1,2,4,8],
#         ffn_expansion_factor = 2.66,
        ffn_expansion_factor = 1,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        Conv_type = 0,   ## Other option 'BiasFree'
        pre_post_norm = 0,
        locked_attn  = 0,
        swin_stages  = 2,
        mr = 0,
        cross = 0,
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(SwinRestormer_MR_PET, self).__init__()
        print('dim ', dim)
        print('ffn_factor ', ffn_expansion_factor)
        print('pre_post_norm ', pre_post_norm)
        print('locked_attn ', locked_attn)
        print('swin_stages ', swin_stages)
        print('mr ', mr)
        print('cross ', cross)
                    
        self.mr = mr
        self.cross = cross

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential( 
            *[TransformerBlockSimple(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[0])]
        )
        
        self.down1_2 = DownsampleSimple(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential( 
                *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[1])])
        
        self.down2_3 = DownsampleSimple(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[2])])

        self.down3_4 = DownsampleSimple(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[3])])
        
        if (self.mr == 1):
            self.reduce_chan_level4 = nn.Conv3d(int(dim*2**4)*2, int(dim*2**3), kernel_size=1, bias=bias)
        else:
            self.reduce_chan_level4 = nn.Conv3d(int(dim*2**4), int(dim*2**3), kernel_size=1, bias=bias)

        
    
        self.up4_3 = UpsampleSimple(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv3d(int(dim*2**3)+int(dim*4), int(dim*2**2), kernel_size=1, bias=bias)
        
        self.decoder_level3 = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[2])])


        self.up3_2 = UpsampleSimple(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[1])])
        
        self.up2_1 = UpsampleSimple(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv3d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output  = nn.Conv3d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.swin_stages = swin_stages
        if (self.swin_stages == 2):
            swin_depths      = (4, 8)
            swin_num_heads   = (4, 8)
            swin_out_indices = (0, 1)
            window_size      = (6, 6, 6)
        elif (self.swin_stages == 4):
            swin_depths      = (2, 2, 18, 2)
#             swin_num_heads   = (4, 4, 8, 8)
            swin_num_heads   = (6, 12, 24, 48)
            swin_out_indices = (0, 1, 2, 3)
            window_size      = (3, 3, 3)
            
            self.swin_up4_3 = UpsampleSimple(int(dim*2**5))  
            self.swin_reduce_chan_level3 = nn.Conv3d(int(dim*2**5), int(dim*2**4), kernel_size=1, bias=bias)
            self.swin_decoder_level3 = nn.Sequential(
                *[TransformerBlockSimple(dim=int(dim*2**4), num_heads=swin_num_heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(swin_depths[3])])
            
            self.swin_up3_2 = UpsampleSimple(int(dim*2**4))  
            self.swin_reduce_chan_level2 = nn.Conv3d(int(dim*2**4), int(dim*2**3), kernel_size=1, bias=bias)
            self.swin_decoder_level2 = nn.Sequential(
                *[TransformerBlockSimple(dim=int(dim*2**3), num_heads=swin_num_heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(swin_depths[2])])

            
        self.swin    = SwinTransformer(patch_size= 4,
                                           in_chans= inp_channels,
                                           embed_dim= dim*2**2,
                                           depths= swin_depths,   #(1, 5), #( 2, 2, 10, 2) ,
                                           num_heads= swin_num_heads, #(4, 4), #(4, 4, 8, 8),
                                           window_size= window_size,#(3, 3, 3),
                                           mlp_ratio= 4,
                                           qkv_bias= False,
                                           drop_rate= 0,
                                           drop_path_rate= 0.3,
                                           ape= True,
                                           spe= False,
                                           patch_norm= True,
                                           use_checkpoint= False,
                                           out_indices= swin_out_indices,
                                           pat_merg_rf= 4,
                                           pos_embed_method= 'relative',
                                           locked_attn=locked_attn)

        
        
    def forward(self, inp_img, latent_tmp=torch.tensor([]), mr_img = torch.tensor([])):
        if mr_img.numel() != 0:
#             print('mr_img ', mr_img.size())
            inp_img = inp_img + mr_img


        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 

        out = self.swin(inp_img)
        
        if (self.swin_stages == 2):
            swin_latent         = out[-1]
            swin_out_enc_level1 = out[-2]
            latent = torch.cat([latent, swin_latent], 1)
#             print('latent ', latent.size())
#             print('latent_tmp.numel() ', latent_tmp.numel())
            
            if latent_tmp.numel() != 0:
                latent = torch.cat([latent, latent_tmp], 1)
#                 print('latent ', latent.size())
#                 print('latent_tmp ', latent_tmp.size())
#                 print('mr_img ', mr_img.size())
                
            latent = self.reduce_chan_level4(latent)
            inp_dec_level3 = self.up4_3(latent)
            
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3, swin_out_enc_level1], 1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        elif (self.swin_stages == 4):
            swin_latent         = out[-1]
            swin_out_enc_level3 = out[-2]
            swin_out_enc_level2 = out[-3]
            swin_out_enc_level1 = out[-4]
            

            swin_latent         = torch.cat([self.swin_up4_3(swin_latent), swin_out_enc_level3], 1)
            swin_latent         = self.swin_reduce_chan_level3(swin_latent)
            swin_out_dec_level3 = self.swin_decoder_level3(swin_latent)
            
            swin_out_enc_level2 = torch.cat([self.swin_up3_2(swin_out_dec_level3), swin_out_enc_level2], 1)
            swin_out_enc_level2 = self.swin_reduce_chan_level2(swin_out_enc_level2)
            swin_out_dec_level2 = self.swin_decoder_level2(swin_out_enc_level2)
            
            
            
            latent = torch.cat([latent, swin_out_dec_level2], 1)
            latent = self.reduce_chan_level4(latent)

            inp_dec_level3 = self.up4_3(latent)
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3, swin_out_enc_level1], 1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            out_dec_level3 = self.decoder_level3(inp_dec_level3) 
            

 
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:

#             if self.mr == 1:
#                 inp_img = torch.unsqueeze(inp_img[:, 0, :, :, :], 1)

            out_dec_level1 = self.output(out_dec_level1) + inp_img
            
        return out_dec_level1
    
    
    def ExtractFeatures(self, inp_img):
        x = self.patch_embed(inp_img)

        x = self.encoder_level1(x)

        
        x = self.down1_2(x)
        x = self.encoder_level2(x)

        x = self.down2_3(x)
        x = self.encoder_level3(x) 

        x = self.down3_4(x)        
        x = self.latent(x) 
        
        x1 = self.swin(inp_img)
        x1 = x1[-1]
        x = torch.cat([x, x1], 1)
        
        return x

##########################################################################
class RestormerSimple(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
#         dim = 96,
#         dim = 48, # dim = 96
        dim = 16, # by SI
#         dim = embed_dim,
        num_blocks = [1,4,4,6],  # for restormer_96_2
#          num_blocks = [1,2,4,6], # for the others
#         num_blocks = [4,6,6,8], 
        num_refinement_blocks = 1,
#         heads = [1,1,1,1],
        heads = [1,2,4,8],
#         ffn_expansion_factor = 2.66,
        ffn_expansion_factor = 1,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        Conv_type = 0,   ## Other option 'BiasFree'
        pre_post_norm = 0,
        locked_attn  = 0,
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(RestormerSimple, self).__init__()
        print('dim ', dim)
        print('ffn_factor ', ffn_expansion_factor)
        print('pre_post_norm ', pre_post_norm)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
#         self.stride      = 0
#         self.kernel_s    = 3
        self.encoder_level1 = nn.Sequential( 
            *[TransformerBlockSimple(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[0])]
        )
        
        self.down1_2 = DownsampleSimple(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential( 
                *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[1])])
        
        self.down2_3 = DownsampleSimple(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[2])])

        self.down3_4 = DownsampleSimple(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[3])])
        
        self.up4_3 = UpsampleSimple(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv3d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[2])])


        self.up3_2 = UpsampleSimple(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[1])])
        
        self.up2_1 = UpsampleSimple(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type, pre_post_norm = pre_post_norm, locked_attn = locked_attn) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv3d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output  = nn.Conv3d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.UpSam   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.DownSam = nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=False)
    def forward(self, inp_img):
#         print('inp_img ', inp_img.size())
        inp_enc_level1 = self.patch_embed(inp_img)
#         print('inp_enc_level1 ', inp_enc_level1.size())
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
#         print('out_enc_level1 ', out_enc_level1.size())
        inp_enc_level2 = self.down1_2(out_enc_level1)
#         print('inp_enc_level2 ', inp_enc_level2.size())
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
#         print('out_enc_level2 ', out_enc_level2.size())
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
#         print('out_enc_level3 ', out_enc_level3.size())
        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
#         print('latent ', latent.size())
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
#             out_dec_level1 = self.UpSam(out_dec_level1)
            out_dec_level1 = self.output(out_dec_level1) + inp_img
            
#         print('out_enc_level1: ', out_enc_level1.size())
#         out_dec_level1 = self.UpSam(out_dec_level1 )
#         print('out_enc_level1: ', out_enc_level1.size())
        return out_dec_level1
    def ExtractFeatures(self, inp_img):
        x = self.patch_embed(inp_img)

        x = self.encoder_level1(x)

        
        x = self.down1_2(x)
        x = self.encoder_level2(x)

        x = self.down2_3(x)
        x = self.encoder_level3(x) 

        x = self.down3_4(x)        
        x = self.latent(x) 
        
        return x
    
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
#         dim = 48,
        dim = 32, # by SI
#         num_blocks = [4,6,6,8], 
        num_blocks = [2,2,2,6], 
        num_refinement_blocks = 1,
#         heads = [1,2,4,8],
        heads = [1,1,1,1],
#         heads = [1,2,4,2],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()
#         self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
#         self.downres1 = DownsampleRes(1)
#         self.upres1  = UpsampleRes(1)
#         self.patch_embed = OverlapPatchEmbed(inp_channels, inp_channels)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim//2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim//2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim//2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim//2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim//2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim//2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv3d(int(dim//2**1), int(dim//2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim//2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim//2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim//2**0), int(dim//2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim//2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim//2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv3d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output  = nn.Conv3d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.UpSam   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.DownSam = nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=False)
        
#         self.DownSamSf = DownsampleRes(16)
#         self.UpSamSf   = UpsampleRes(32)
    def forward(self, inp_img):
#         inp_img =     self.downres1(inp_img)
#         inp_img =     self.upres1(inp_img)
        
#         inp_avg_img    = self.avg_pool(inp_img)  # by SI

        inp_enc_level1 = self.patch_embed(inp_img)
#         inp_enc_level1 = self.DownSamSf(inp_enc_level1) # by SI
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)

        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
#             out_dec_level1 = self.UpSamSf(out_dec_level1) 
            out_dec_level1 = self.output(out_dec_level1) + inp_img
            
#             out_dec_level1 = self.UpSam(out_dec_level1) + inp_img# by SI
            

            
        return out_dec_level1

    def ExtractFeatures(self, inp_img):
#         inp_avg_img    = self.DownSam(inp_img)
#         inp_avg_img    = self.avg_pool(inp_img)  # by SI
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
        
        return latent
# The first triral
class Restormer_1st(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
#         dim = 48,
        dim = 16, # by SI
#         num_blocks = [2,4,4,6], 
        num_blocks = [1,1,1,1], 
        num_refinement_blocks = 1,
#         heads = [1,2,4,8],
        heads = [1,1,1,1],
#         heads = [1,2,4,2],
        ffn_expansion_factor = 2.66/2,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer_1st, self).__init__()
#         self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim//2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim//2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim//2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim//2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim//2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim//2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv3d(int(dim//2**1), int(dim//2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim//2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim//2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim//2**0), int(dim//2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim//2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim//2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv3d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output  = nn.Conv3d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.output  = nn.Conv3d(int(dim*3), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.UpSam   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.DownSam = nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=False)
#         self.DownSamSf = DownsampleRes(32)
#         self.UpSamSf   = UpsampleRes(64)
    def forward(self, inp_img):

#         inp_img_tmp    = self.DownSam(inp_img)
#         inp_img        = self.avg_pool(inp_img)
        
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
                
        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            
            out_dec_level1 = self.output(out_dec_level1) + inp_img
            

            
        return out_dec_level1

    def ExtractFeatures(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
        
        return latent

import torch
import torch.nn as nn
import torch.optim as optim

# The first triral
class Restormer_multiGPU(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
#         dim = 96,
        dim = 48, # dim = 96
#         dim = 16, # by SI
#         num_blocks = [1,4,4,6], 
         num_blocks = [1,2,4,6], 
#         num_blocks = [4,6,6,8], 
        num_refinement_blocks = 1,
#         heads = [1,1,1,1],
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
#         ffn_expansion_factor = 1,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        Conv_type = 0,   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer_multiGPU, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim).to('cuda:0')

        self.encoder_level1 = nn.Sequential( 
            *[TransformerBlockSimple(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[0])]
        ).to('cuda:0')
        
        self.down1_2 = Downsample(dim).to('cuda:0') ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential( 
                *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[1])]).to('cuda:0')
        
        self.down2_3 = DownsampleSimple(int(dim*2**1)).to('cuda:0') ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[2])]).to('cuda:0')

        self.down3_4 = DownsampleSimple(int(dim*2**2)).to('cuda:0') ## From Level 3 to Level 4
        self.latent = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[3])]).to('cuda:0')
        
        self.up4_3 = UpsampleSimple(int(dim*2**3)).to('cuda:1') ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv3d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias).to('cuda:1')
        self.decoder_level3 = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[2])]).to('cuda:1')


        self.up3_2 = UpsampleSimple(int(dim*2**2)).to('cuda:1') ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias).to('cuda:1')
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[1])]).to('cuda:1')
        
        self.up2_1 = UpsampleSimple(int(dim*2**1)).to('cuda:1')  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[0])]).to('cuda:1')
        
        self.refinement = nn.Sequential( 
            *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_refinement_blocks)]).to('cuda:1')
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task 
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv3d(dim, int(dim*2**1), kernel_size=1, bias=bias).to('cuda:1')
        ###########################
            
        self.output  = nn.Conv3d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias).to('cuda:1')                                
                                  
                                  

    def forward(self, inp_img):
#         latent = self.seq1(inp_img.to('cuda:0'))
#         print(x.size())
#         print(x.to('cuda:1').size())
        
        inp_enc_level1 = self.patch_embed(inp_img.to('cuda:0'))
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4).to('cuda:1')
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3.to('cuda:1')], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2.to('cuda:1')], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1.to('cuda:1')], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
                
        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            
            out_dec_level1 = self.output(out_dec_level1) + inp_img.to('cuda:1')
            

            
        return out_dec_level1            

            
        return x

    def ExtractFeatures(self, inp_img):
        x = self.patch_embed(inp_img.to('cuda:0'))

        x = self.encoder_level1(x)

        
        x = self.down1_2(x)
        x = self.encoder_level2(x)

        x = self.down2_3(x)
        x = self.encoder_level3(x) 

        x = self.down3_4(x)        
        x = self.latent(x) 
        
        return x


# class RestormerSimple(nn.Module):
#     def __init__(self, 
#         inp_channels=1, 
#         out_channels=1, 
#         dim = 96,
# #         dim = 48,
# #         dim = 16, # by SI
# #         num_blocks = [1,1,1,1], 
#         num_blocks = [4,6,6,8], 
#         num_refinement_blocks = 4,
#         heads = [1,1,1,1],
# #         heads = [1,2,4,8],
#         ffn_expansion_factor = 2.66,
# #         ffn_expansion_factor = 1,
#         bias = False,
#         LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
#         Conv_type = 0,   ## Other option 'BiasFree'
#         dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
#     ):

#         super(RestormerSimple, self).__init__()
#         self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
# #         self.stride      = 0
# #         self.kernel_s    = 3
#         self.encoder_level1 = nn.Sequential( 
#             *[TransformerBlockSimple(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[0])]
#         )
        
#         self.down1_2 = DownsampleSimple(dim) ## From Level 1 to Level 2
#         self.encoder_level2 = nn.Sequential( 
#                 *[TransformerBlockSimple(dim=int(dim//2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[1])])
        
#         self.down2_3 = DownsampleSimple(int(dim//2**1)) ## From Level 2 to Level 3
#         self.encoder_level3 = nn.Sequential( 
#             *[TransformerBlockSimple(dim=int(dim//2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[2])])

#         self.down3_4 = DownsampleSimple(int(dim//2**2)) ## From Level 3 to Level 4
#         self.latent = nn.Sequential( 
#             *[TransformerBlockSimple(dim=int(dim//2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[3])])
        
#         self.up4_3 = UpsampleSimple(int(dim//2**3)) ## From Level 4 to Level 3
#         self.reduce_chan_level3 = nn.Conv3d(int(dim//2**1), int(dim//2**2), kernel_size=1, bias=bias)
#         self.decoder_level3 = nn.Sequential( 
#             *[TransformerBlockSimple(dim=int(dim//2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[2])])


#         self.up3_2 = UpsampleSimple(int(dim//2**2)) ## From Level 3 to Level 2
#         self.reduce_chan_level2 = nn.Conv3d(int(dim//2**0), int(dim//2**1), kernel_size=1, bias=bias)
#         self.decoder_level2 = nn.Sequential(
#             *[TransformerBlockSimple(dim=int(dim//2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[1])])
        
#         self.up2_1 = UpsampleSimple(int(dim//2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

#         self.decoder_level1 = nn.Sequential( 
#             *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_blocks[0])])
        
#         self.refinement = nn.Sequential( 
#             *[TransformerBlockSimple(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, Conv_type = Conv_type) for i in range(num_refinement_blocks)])
        
#         #### For Dual-Pixel Defocus Deblurring Task ####
#         self.dual_pixel_task = dual_pixel_task
#         if self.dual_pixel_task:
#             self.skip_conv = nn.Conv3d(dim, int(dim*2**1), kernel_size=1, bias=bias)
#         ###########################
            
#         self.output  = nn.Conv3d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
# #         self.UpSam   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
# #         self.DownSam = nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=False)
#     def forward(self, inp_img):

#         inp_enc_level1 = self.patch_embed(inp_img)
#         out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
#         inp_enc_level2 = self.down1_2(out_enc_level1)
#         out_enc_level2 = self.encoder_level2(inp_enc_level2)

#         inp_enc_level3 = self.down2_3(out_enc_level2)
#         out_enc_level3 = self.encoder_level3(inp_enc_level3) 

#         inp_enc_level4 = self.down3_4(out_enc_level3)        
#         latent = self.latent(inp_enc_level4) 
                        
#         inp_dec_level3 = self.up4_3(latent)
#         inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
#         inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
#         out_dec_level3 = self.decoder_level3(inp_dec_level3) 

#         inp_dec_level2 = self.up3_2(out_dec_level3)
#         inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
#         inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
#         out_dec_level2 = self.decoder_level2(inp_dec_level2) 

#         inp_dec_level1 = self.up2_1(out_dec_level2)
#         inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
#         out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
#         out_dec_level1 = self.refinement(out_dec_level1)

#         #### For Dual-Pixel Defocus Deblurring Task ####
#         if self.dual_pixel_task:
#             out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
#             out_dec_level1 = self.output(out_dec_level1)
#         ###########################
#         else:
# #             out_dec_level1 = self.UpSam(out_dec_level1)
#             out_dec_level1 = self.output(out_dec_level1) + inp_img
            
# #         print('out_enc_level1: ', out_enc_level1.size())
# #         out_dec_level1 = self.UpSam(out_dec_level1 )
# #         print('out_enc_level1: ', out_enc_level1.size())
#         return out_dec_level1
#     def ExtractFeatures(self, inp_img):
#         inp_enc_level1 = self.patch_embed(inp_img)

#         out_enc_level1 = self.encoder_level1(inp_enc_level1)

        
#         inp_enc_level2 = self.down1_2(out_enc_level1)
#         out_enc_level2 = self.encoder_level2(inp_enc_level2)

#         inp_enc_level3 = self.down2_3(out_enc_level2)
#         out_enc_level3 = self.encoder_level3(inp_enc_level3) 

#         inp_enc_level4 = self.down3_4(out_enc_level3)        
#         latent = self.latent(inp_enc_level4) 
        
#         return latent
