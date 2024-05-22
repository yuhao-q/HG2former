import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from pdb import set_trace as stx
import matplotlib.colors as mcolors


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class HSV_Gamma(nn.Module):
    def __init__(self, n_fea_middle=40, n_fea_in=3, n_fea_out=3, gamma=2.2):
        super(HSV_Gamma, self).__init__()

        self.gamma = gamma
        self.gamma_conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.gamma_depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_middle)
        self.gamma_conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

        self.hsl_conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.hsl_depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_middle)
        self.hsl_conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

        self.combine_conv = nn.Conv2d(n_fea_out * 2, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # Gamma branch
        gamma_img = img.permute(0, 2, 3, 1)  
        gamma_img = gamma_img.to(torch.float32) / 255.0  
        gamma_img = torch.pow(gamma_img, self.gamma) 
        gamma_img = gamma_img.permute(0, 3, 1, 2)  

        gamma_features = self.gamma_conv1(gamma_img)
        gamma_features = self.gamma_depth_conv(gamma_features)
        gamma_map = self.gamma_conv2(gamma_features)

        # HSL branch
        hsl_img = img.permute(0, 2, 3, 1)  
        hsl_img = hsl_img.to(torch.float32) / 255.0 
        hsl_img = mcolors.rgb_to_hsv(hsl_img.cpu().numpy()) 
        hsl_img = torch.from_numpy(hsl_img).permute(0, 3, 1, 2).to(img.device)

        hsl_features = self.hsl_conv1(hsl_img)
        hsl_features = self.hsl_depth_conv(hsl_features)
        hsl_map = self.hsl_conv2(hsl_features)

        combined_map = torch.cat([gamma_map, hsl_map], dim=1)
        combined_map = self.combine_conv(combined_map)

        return combined_map


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Channelatt(nn.Module):
    def __init__(self, dim, hidden_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()  
        hidden_dim = hidden_dim or dim 
        self.act = act_layer() 
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1) 
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim) 
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1) 
        self.drop = nn.Dropout(drop) 

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channelConv1 = nn.Conv1d(1, 1, 3, padding=1) 
        self.channelConv2 = nn.Conv1d(1, 1, kernel_size=5, padding=2) 

        self.apply(self._init_weights) 

    def _init_weights(self, m): 
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0) 
        elif isinstance(m, (nn.LayerNorm,nn.GroupNorm, nn.LayerNorm)): 
            nn.init.constant_(m.bias, 0)  
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d): 
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  
            fan_out //= m.groups 
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out)) 
            if m.bias is not None:
                m.bias.data.zero_() 

    def forward(self, x): 
        x = self.fc1(x)  
        x = self.dwconv(x)  
        x = self.act(x)  
        x = self.drop(x) 
        x = self.fc2(x)
        x = self.drop(x)

        res = x.clone()
        x = self.avg_pool(self.act(x)) 
        x = self.channelConv1(x.squeeze(-1).transpose(-1, -2)) 
        x = self.act(x)  
        x = self.channelConv2(x)
        x = x.transpose(-1, -2).unsqueeze(-1)  

        return res + x 



class FFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = torch.nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.depthwise = torch.nn.Conv2d(hidden_features,hidden_features, kernel_size=3,stride=1,padding=1,dilation=1,groups=hidden_features)
        self.pointwise2 = torch.nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Positional embedding
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)


        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_p = self.pos_emb(out)
        out = out + out_p

        out = self.project_out(out)
        return out



class IMSA(nn.Module):

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(IMSA, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FFN(dim, dim*ffn_expansion_factor, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class HSV_Gam_transformer(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            num_blocks=2,
            ffn_expansion_factor=4,
            bias=True  
    ):
        super().__init__()
        self.blocks = nn.ModuleList([IMSA(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for _ in range(num_blocks)])

    def forward(self, x):

        for imsa in self.blocks:
            x = imsa(x)
        out = x
        return out


class Enc_Dec(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[3, 6, 6]):
        super(Enc_Dec, self).__init__()
        self.dim = dim
        self.level = level

        # HSV_Gamma
        self.HSV_Gamma = HSV_Gamma(dim)

        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        self.encoder1 = HSV_Gam_transformer(dim=self.dim, num_blocks=num_blocks[0])
        self.downsample1 = nn.Conv2d(self.dim, self.dim * 2, 4, 2, 1, bias=False)
        self.encoder2 = HSV_Gam_transformer(dim=self.dim * 2, num_blocks=num_blocks[1])
        self.downsample2 = nn.Conv2d(self.dim * 2, self.dim * 4, 4, 2, 1, bias=False)
        
        self.bottleneck = HSV_Gam_transformer(dim=self.dim * 4, num_blocks=num_blocks[-1])

        # Decoder
        self.upsample1 = nn.ConvTranspose2d(self.dim * 4, self.dim * 2, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.fusion1 = nn.Conv2d(self.dim * 4, self.dim * 2, 1, 1, bias=False)
        self.decoder1 = HSV_Gam_transformer(dim=self.dim * 2, num_blocks=num_blocks[1])
        self.upsample2 = nn.ConvTranspose2d(self.dim * 2, self.dim, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.fusion2 = nn.Conv2d(self.dim * 2, self.dim, 1, 1, bias=False)
        self.decoder2 = HSV_Gam_transformer(dim=self.dim, num_blocks=num_blocks[0])

        self.out = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img):
        input_img = img * self.HSV_Gamma(img) + img

        embedded_x = self.embedding(input_img)
        encoder1_out = self.encoder1(embedded_x)
        downsample1_out = self.downsample1(encoder1_out)
        encoder2_out = self.encoder2(downsample1_out)
        downsample2_out = self.downsample2(encoder2_out)

        bottleneck_out = self.bottleneck(downsample2_out)

        upsample1_out = self.upsample1(bottleneck_out)
        fusion1_out = self.fusion1(torch.cat([upsample1_out, encoder2_out], dim=1))
        decoder1_out = self.decoder1(fusion1_out)
        upsample2_out = self.upsample2(decoder1_out)
        fusion2_out = self.fusion2(torch.cat([upsample2_out, encoder1_out], dim=1))

        decoder2_out = self.decoder2(fusion2_out)
        out = self.out(decoder2_out) + img

        return out

class HG2former(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[3,6,6]):
        super(HG2former, self).__init__()
        self.stage = stage

        self.modules_body = nn.ModuleList([Enc_Dec(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=2, num_blocks=num_blocks)
                        for _ in range(stage)])
    
    def forward(self, x):
        for module in self.modules_body:
            x = module(x)
        return x
