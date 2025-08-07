import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class MaskExtractor(nn.Module):
    def __init__(self, config, mm_hidden_size, depth=2):
        super(MaskExtractor, self).__init__()
        self.mask_pooling = MaskPooling()
        modules = [nn.Linear(mm_hidden_size, config.hidden_size)]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.feat_linear =  nn.Sequential(*modules)

    def forward(self, feats, masks):
        query_feats = []
        
        if masks is None: #infer
            return None

        num_imgs = len(masks)
        region_token_nums = []
        image_idx = 0
        for idx in range(num_imgs):
            if masks[idx]==None:
                continue
            for mask_idx in range(len(masks[idx])):
                mask = masks[idx][mask_idx].unsqueeze(0).unsqueeze(0).float()
                if len(mask[0])==0:
                    mask = torch.zeros((1, 1, 336, 336)).to(feats.device).float()

                feat = feats[image_idx].unsqueeze(0)
                image_idx+=1
                
                # h, w = feat.shape[1:3]
                feat = feat.permute(0,3,1,2)

                raw_dtype = feat.dtype
                feat = feat.to(mask.dtype)
                
                mask_feat_raw = self.mask_pooling(feat, mask) # [n, 1024]

                query_feats.append(mask_feat_raw)
        if len(query_feats)==0:
            return None
        mask_feats = torch.cat(query_feats, dim=0)
        mask_feats = mask_feats.to(feats[0].dtype)
        mask_feats_linear = self.feat_linear(mask_feats)
        return mask_feats_linear

    
class MaskPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):

        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # b, c, h ,w = x.shape
        # b, q, h, w = mask.shape
        mask = (mask > 0).to(mask.dtype)
        mask = mask.permute(1,0,2,3)
        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = (x * mask/denorm).sum(-1).sum(-1)

        return mask_pooled_x


def build_region_encoder(config, mm_hidden_size):

    return MaskExtractor(config, mm_hidden_size)
   