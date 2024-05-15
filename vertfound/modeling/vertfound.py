import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import Any, Dict, List, Tuple
from .segment_anything.utils.transforms import ResizeLongestSide
from .segment_anything.build_sam import sam_model_registry
from .decoder import build_decoder
from .pan_decoder import PANDecoder
from vertfound.util.preprocessing import prepare_logits
from . import constants
from einops import rearrange

from .segment_anything.modeling.prompt_engineering import prompt_engineering, get_prompt_templates
from .clip import load as load_clip
import clip

class MLP(nn.Module):
    def __init__(self, feature_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VertFound(nn.Module):
    TEXT_FEATS_MAP = {
        'coco': 'text_feats_coco',
        'objects365': 'text_feats_objects365',
        'v3det': 'text_feats_v3det',
        'lvis': 'text_feats_lvis',
        'openimages': 'text_feats_openimages',
        # 'vertdet':'text_feats_vertdet'
    }

    def __init__(self, sam_checkpoint='./sam_checkpoints/sam_vit_b_01ec64.pth',
                 clip_type='CLIP_400M_Large', is_training=True, custom_vocabulary=None, image_size=224):
        super().__init__()

        self.use_layer = [23, 21, 23]

        self.sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
        self._freeze_module(self.sam)

        self.clip_model, self.text_dim, self.clip_dim = self._load_clip_model(clip_type, image_size)
        self.clip_model.eval()
        self._freeze_module(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale.exp()

        self.to_clip = nn.Linear(256, self.clip_dim)
        self.ln_clip = nn.LayerNorm(self.clip_dim, elementwise_affine=False)
        self.projector = nn.Linear(257, 768)
        # self.decoder = build_decoder(d_model=self.clip_dim)

        self.pan_decoder = PANDecoder([257] * 3, self.clip_dim, 8)

        self.sk_proj = MLP(self.text_dim, 1)
       
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        
        # Dynamically set attributes based on the datasets in the map
        # if is_training:
        #     datasets_to_load = ['objects365', 'v3det', 'openimages']
        #     for dataset in datasets_to_load:
        #         setattr(self, self.TEXT_FEATS_MAP[dataset], self.get_text_feat(dataset))
        # else:
        #     # dataset_name = 'custom' if custom_vocabulary else 'lvis'
        #     dataset_name = 'custom' if custom_vocabulary else 'coco'
        #     self.text_feats = self.get_text_feat(dataset_name, custom_class=custom_vocabulary)

    @staticmethod
    def _freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    def _load_clip_model(self, clip_type, image_size):
        clip_model_map = {
            'CLIP_400M': ("ViT-B/16", 512, 768),
            'CLIP_400M_Large': ("ViT-L/14", 768, 1024),
            'CLIP_400M_Large_336': ("ViT-L/14@336px", 768, 1024)
        }
        model_type, text_dim, clip_dim = clip_model_map[clip_type]
        clip_model, _ = load_clip(model_type, image_size=image_size)
        # clip_model.visual.register_hooks([7, 9])
        return clip_model, text_dim, clip_dim

    @torch.no_grad()
    def get_text_feat(self, dataset_name: str, custom_class=None) -> torch.Tensor:
        dataset_map = {
            'coco': constants.COCO_INSTANCE_CLASSES,
            'objects365': constants.OBJECTS365V1,
            'v3det': constants.V3DET,
            'lvis': constants.LVIS_CATEGORIES,
            'openimages': constants.OPENIMAGE,
            'custom': custom_class,
            'vertdet': constants.VERT_CLASSES_1
        }
        # Error handling for custom dataset without custom classes provided
        if dataset_name == 'custom' and custom_class is None:
            raise ValueError("For custom datasets, you must provide the 'custom_class' parameter.")
        
        class_names = dataset_map.get(dataset_name, [])

        def clean_class_name(clss: str) -> str:
            """Clean class names for prompt templates."""
            return clss.replace('-other', '').replace('-merged', '').replace('-stuff', '')

        def extract_mean_emb(text: str) -> torch.Tensor:
            """Extract mean embeddings from text using the clip model."""
            tokens = clip.tokenize(text).cuda()
            
            if len(tokens) > 10000:
                split_idx = len(tokens) // 2
                text_features = torch.cat([
                    self.clip_model.encode_text(tokens[:split_idx]),
                    self.clip_model.encode_text(tokens[split_idx:])],
                    dim=0)
            else:
                text_features = self.clip_model.encode_text(tokens)

            return torch.mean(text_features, 0, keepdims=True)[0]

        templates = get_prompt_templates()
        clss_embeddings = []
        # for clss in range(len(class_names)):
        for clss in range(len(class_names)):
            # txts = [template.format(clss.replace('-other','').replace('-merged','').replace('-stuff','')) for template in templates]
            # txts = templates[0].format(clss.replace('-other','').replace('-merged','').replace('-stuff',''))
            txts = str(clss) #'a photo of a ' + clss
            clss_embeddings.append(extract_mean_emb(txts))

        text_emb = torch.stack(clss_embeddings, dim=0)
        text_emb /= text_emb.norm(dim=-1, keepdim=True) 
        
        return text_emb

    def sigmoid_focal_loss(self, inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, reduction=True):
        """Compute the sigmoid focal loss."""
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss

        return loss.mean(1).sum() / num_boxes

    def get_logits(self, region_features, text_features, logit_scale):
        """Compute logits for region and text features."""
        region_features = region_features / (region_features.norm(dim=-1, keepdim=True) + 1e-7)
        logits_per_image = logit_scale * region_features @ text_features.unsqueeze(0).transpose(1, 2) # [?, gt_boxes, 25]
        logits_per_text = logit_scale * text_features.unsqueeze(0) @ region_features.transpose(1, 2)
        return logits_per_image, logits_per_text
        
    def sinkhorn(self, C, mu, nu, eps=0.1, max_iter=100, reduction='mean'):

        bz, n, m = C.shape
        d = C.device
        # a = torch.ones(bz, n, device=d) / n
        # b = torch.ones(bz, m, device=d) / m

        # Using log domain for numerical stability
        K = torch.exp(-C / eps)
        u = torch.ones_like(mu, device=d)

        # Step 2: Sinkhorn Iterations
        for _ in range(max_iter):
            v = nu / torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
            u = mu / torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)
        
        T = u.unsqueeze(-1) * K * v.unsqueeze(-2)
        distance = torch.sum(u.unsqueeze(-1) * K * C * v.unsqueeze(-2), dim=(1, 2))

        # Reduction
        if reduction == 'mean':
            distance = torch.mean(distance)
        elif reduction == 'sum':
            distance = torch.sum(distance)

        return T, distance

    def sk_loss(self, region_features, logit_scale):
        text_feats_ = self.get_text_feat('vertdet')
        region_feat = self.sk_proj(region_features).squeeze(-1)
        text_feat = self.sk_proj(text_feats_).squeeze(-1)

        region_feat = F.softmax(region_feat, dim=-1)
        text_feat = F.softmax(text_feat, dim=-1)

        logits_per_image, _ = self.get_logits(region_features, text_feats_, logit_scale)
        # logits_per_image, _ = self.get_logits(region_features, text_feats_, logit_scale)

        _, dis = self.sinkhorn(F.softmax(-logits_per_image, dim=-1), region_feat, text_feat)

        loss = torch.exp(dis * logit_scale) / logits_per_image.size(1)

        return loss

    def cn_loss(self, region_features, logit_scale, label=None, thres=0.5):
        text_feats_ = self.get_text_feat('vertdet')
        logits_per_image, _ = self.get_logits(region_features, text_feats_, logit_scale)
        
        bz, n_box, d = region_features.shape
        logits = prepare_logits(logits_per_image)
        
        total_loss = 0
        logits_per_image_prob = F.softmax(logits, dim=2)

        for b in range(bz):
            exists = logits_per_image_prob[b].max(dim=1).values > thres

            loss = torch.tensor(0.0, device=logits_per_image.device)
        
            for i in range(1, n_box):
                if exists[i] and exists[i-1]:
                    diff = torch.abs(torch.argmax(logits_per_image_prob[b][i]) - torch.argmax(logits_per_image_prob[b][i-1]))
                    loss += torch.clamp(diff - 1, min=0)
            total_loss += loss
        return total_loss / bz

    def ce_loss(self, region_features, label, logit_scale, dataset_name, focal_alpha=0.25):
        """Compute the cross-entropy loss."""
        b, n_box, d = region_features.shape
        # text_feats = getattr(self, self.TEXT_FEATS_MAP[dataset_name])
        # text_feats = getattr(self, self.TEXT_FEATS_MAP['vertdet'])
        text_feats_ = self.get_text_feat('vertdet')
        logits_per_image, _ = self.get_logits(region_features, text_feats_, logit_scale) # [bs, ?, 25]

        ## NEW
        # logits_per_image = self.cn_conf(logits_per_image)
        # logits_per_image = torch.softmax(logits_per_image, dim=-1)

        target_classes_onehot = torch.zeros(logits_per_image.shape, dtype=logits_per_image.dtype, device=logits_per_image.device)
        label = label.long()
        target_classes_onehot.scatter_(2, label.unsqueeze(-1), 1) ## one_hot_code

        loss_ce = self.sigmoid_focal_loss(logits_per_image, target_classes_onehot, n_box, alpha=focal_alpha, gamma=2) * logits_per_image.shape[1]

        return loss_ce
    
    def compute_loss(self, region_features, label, logit_scale, dataset_name, focal_alpha=0.25):
        loss_ce = self.ce_loss(region_features, label, logit_scale, dataset_name, focal_alpha)
        loss_cn = self.cn_loss(region_features, logit_scale, label)
        loss_sk = self.sk_loss(region_features, logit_scale)
        
        return loss_ce, loss_cn, loss_sk

    def forward_train(self, batched_input: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        """Training forward pass."""
        resized_image = torch.stack([x["resized_image"] for x in batched_input], dim=0) # [16, 3, 224, 224]

        with torch.no_grad():
            # clip_feat = self.clip_model.encode_image_featuremap(resized_image).detach() # [16, 257, 1024]
            clip_feats = self.clip_model.encode_all(resized_image, self.use_layer) # [16, 257, 1024]
           
        masks_token = torch.stack([x["mask_tokens"] for x in batched_input], dim=0).squeeze(2) # [bs, max_boxes, 256]
        dataset_name = batched_input[0]["dataset_name"] # 'vert'
        masks_token = self.to_clip(masks_token) #nn.Linear(256, self.clip_dim) -> [bs, max_boxes, 64, 64, 1024]

        pan_stack = []
        for feat in clip_feats:
            # x = F.interpolate(feat, feat.shape[-1] // (4 ** i))
            pan_stack.append(feat.transpose(0, 1).detach())

        out = self.pan_decoder(pan_stack, masks_token)

        # semantic_token = self.projector(self.decoder(masks_token, clip_feat))
        # final_feat = out.view(-1, out.size(1), out.size(2) * out.size(3))

        semantic_token = self.projector(torch.bmm(masks_token, out.transpose(-1, -2)))
        
        label = torch.stack([x["label"] for x in batched_input], dim=0)

        return self.compute_loss(semantic_token, label, self.logit_scale, dataset_name)

    def forward_eval(self, batched_input: List[Dict[str, Any]], multimask_output=False) -> List[Dict[str, torch.Tensor]]:
        """Inference forward pass."""


        sam_output = self.sam(batched_input, multimask_output=multimask_output)
        masks_token = torch.stack([x["masks_token"] for x in sam_output], dim=0).squeeze(2)

        # masks_token = torch.stack([torch.tensor(x['extra_info']["mask_tokens"]) for x in batched_input], dim=0).squeeze(2).cuda()
        
        
        #masks_token = torch.tensor(batched_input[0]['extra_info']["mask_tokens"])
        # pred_mask = torch.stack([x["masks"] for x in sam_output], dim=0)
        resized_image = torch.stack([x["resized_image"] for x in batched_input], dim=0)

        with torch.no_grad():
            self.pan_decoder.eval()

            # clip_feat = self.clip_model.encode_image_featuremap(resized_image).detach()
            clip_feats = self.clip_model.encode_all(resized_image, self.use_layer)
        
            masks_token = self.to_clip(masks_token)

            pan_stack = []
            for feat in clip_feats:
                pan_stack.append(feat.transpose(0, 1).detach())

            out = self.pan_decoder(pan_stack, masks_token)

            # semantic_token = self.projector(self.decoder(masks_token, clip_feat))
            # semantic_token = self.projector(self.decoder(masks_token, out))
            semantic_token = self.projector(torch.bmm(masks_token, out.transpose(-1, -2)))

        logits_per_image, _ = self.get_logits(semantic_token, self.get_text_feat('vertdet'), self.logit_scale)
        logits = prepare_logits(logits_per_image)
        logits_per_image_prob = F.softmax(logits, dim=2)

        return logits_per_image_prob # pred_mask

    def forward_inference(self, clip_feat, masks_token, resized_image,) -> List[Dict[str, torch.Tensor]]:
        """Inference forward pass."""
    #    if masks_token.shape
        masks_token = masks_token[None,:]
        if masks_token.shape[2] == 1:
            masks_token = masks_token.squeeze(2)
        else:
            masks_token = masks_token.permute(2, 1, 0, 3).squeeze(2)
            clip_feat = clip_feat.repeat(3, 1, 1)
        with torch.no_grad():
            self.decoder.eval()
            self.pan_decoder.eval()
            masks_token = self.to_clip(masks_token)

            clip_feats = self.clip_model.encode_all(resized_image, self.use_layer)
        
            pan_stack = []
            for feat in clip_feats:
                pan_stack.append(feat.transpose(0, 1).detach())

            out = self.pan_decoder(pan_stack, masks_token)

            semantic_token = self.projector(self.decoder(masks_token, out))

            # semantic_token = self.projector(self.decoder(masks_token, clip_feat))
            
            logits_per_image, _ = self.get_logits(semantic_token, self.text_feats, self.logit_scale)
            if logits_per_image.shape[0] == 3:
                logits_per_image = logits_per_image.permute(1, 0, 2)

        logits = prepare_logits(logits_per_image)
        logits_per_image_prob = F.softmax(logits, dim=2)

        return logits_per_image_prob
            
        
def build_vertfound_model(clip_type='CLIP_400M_Large', is_training=True, pretrain_ckpt=None, image_size=224, custom_vocabulary=None):
    model = VertFound(clip_type=clip_type, is_training=is_training, image_size=image_size, custom_vocabulary=custom_vocabulary)
    if pretrain_ckpt:
        checkpoint = torch.load(pretrain_ckpt, map_location='cpu')['model']
        
        # Remove the 'model.' prefix
        new_checkpoint = {}
        for key in checkpoint.keys():
            if key.startswith('model.'):
                new_key = key[len('model.'):]
                new_checkpoint[new_key] = checkpoint[key]
            else:
                new_checkpoint[key] = checkpoint[key]
        
        # Load the modified state dict
        msg = model.load_state_dict(new_checkpoint, strict=False)
    else:
        msg= 'training stage'
    return model, msg