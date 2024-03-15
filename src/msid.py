import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_



class MSIDEncoder(VisionTransformer):
    def __init__(self, ext_depthes,*args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.head = None

        trunc_normal_(self.pos_embed, std=.02)

        self.ext_depthes=ext_depthes
        self.norms=nn.ModuleDict({f'layer_{i}':nn.Sequential(nn.BatchNorm1d(self.embed_dim),nn.LayerNorm(self.embed_dim)) for i in self.ext_depthes})
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        feats=[]
        for i,blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.ext_depthes:
                feats.append(self.norms[f'layer_{i}'](x[:,0]))
        return feats
        
    def forward(self, x):
        feats = self.forward_features(x)
        feats = torch.cat(feats,1) #[feat_1,feat_2,...] -> (B,D*T)
        return feats
        
    def extract_mlfeat(self,x,ext_depthes):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        feats=[]
        for i,blk in enumerate(self.blocks):
            x = blk(x)
            if i in ext_depthes:
                if f'layer_{i}' in self.norms:
                    y=self.norms[f'layer_{i}'](x[:,0])
                else:
                    y=x[:,0]
                feats.append(y)
        feats = torch.cat(feats,1)
        return feats

@register_model
def msid_base_patch8_112(pretrained=False, ext_depthes=[11],**kwargs):
    model = MSIDEncoder(ext_depthes=ext_depthes,
        img_size=112, patch_size=8, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model