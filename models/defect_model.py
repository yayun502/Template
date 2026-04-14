import os
import torch
import torch.nn as nn
import timm
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone_type="resnet50_local",
        model_name="convnext_tiny",
        pretrained=True,
        pretrained_path=None,
        out_dim=256
    ):
        super().__init__()

        self.backbone_type = backbone_type

        if backbone_type == "resnet50_local":
            backbone = models.resnet50(weights=None)

            if pretrained_path is not None and os.path.exists(pretrained_path):
                state_dict = torch.load(pretrained_path, map_location="cpu")

                # 有些 .pth 可能包成 {"state_dict": ...}
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                # 若 key 前面有 "module."，移除
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        k = k[len("module."):]
                    new_state_dict[k] = v

                missing, unexpected = backbone.load_state_dict(new_state_dict, strict=False)
                print(f"[ImageEncoder] Loaded local pretrained weights from: {pretrained_path}")
                print(f"[ImageEncoder] Missing keys: {missing}")
                print(f"[ImageEncoder] Unexpected keys: {unexpected}")
            else:
                print("[ImageEncoder] No local pretrained weight loaded. Using random init.")

            in_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()

            self.backbone = backbone
            self.proj = nn.Linear(in_dim, out_dim)

        elif backbone_type == "timm":
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg"
            )
            in_dim = self.backbone.num_features
            self.proj = nn.Linear(in_dim, out_dim)

        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)
        return feat


class AttentionPooling(nn.Module):
    def __init__(self, feat_dim=256, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats, mask=None):
        """
        feats: [B, N, D]
        mask:  [B, N], 1 valid / 0 padding
        """
        scores = self.attn(feats).squeeze(-1)  # [B, N]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(feats * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class DefectClassifier(nn.Module):
    def __init__(
        self,
        backbone_type="resnet50_local",
        backbone_name="convnext_tiny",
        pretrained=True,
        pretrained_path=None,
        feat_dim=256,
        num_classes=4,
        num_attrs=4
    ):
        super().__init__()

        self.encoder = ImageEncoder(
            backbone_type=backbone_type,
            model_name=backbone_name,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            out_dim=feat_dim
        )

        self.local_pool = AttentionPooling(feat_dim=feat_dim)
        self.global_pool = AttentionPooling(feat_dim=feat_dim)

        fusion_dim = feat_dim * 2

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.main_head = nn.Linear(fusion_dim, num_classes)
        self.attr_head = nn.Linear(fusion_dim, num_attrs)

    def encode_views(self, x):
        """
        x: [B, N, C, H, W]
        return: [B, N, D]
        """
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        feats = self.encoder(x)
        feats = feats.view(B, N, -1)
        return feats

    def forward(self, local_imgs, global_imgs, local_mask=None, global_mask=None):
        local_feats = self.encode_views(local_imgs)
        global_feats = self.encode_views(global_imgs)

        local_pooled, local_attn = self.local_pool(local_feats, local_mask)
        global_pooled, global_attn = self.global_pool(global_feats, global_mask)

        fused = torch.cat([local_pooled, global_pooled], dim=-1)
        fused = self.fusion(fused)

        logits = self.main_head(fused)
        attr_logits = self.attr_head(fused)

        return {
            "logits": logits,
            "attr_logits": attr_logits,
            "local_attn": local_attn,
            "global_attn": global_attn,
            "fused_feat": fused
        }
