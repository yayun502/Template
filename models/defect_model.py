import os
import sys
import torch
import torch.nn as nn
import timm
from torchvision import models


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for k in ["state_dict", "model", "teacher", "student", "backbone"]:
        if k in checkpoint:
            return checkpoint[k]

    return checkpoint


def clean_state_dict_keys(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        if k.startswith("encoder."):
            k = k[len("encoder."):]
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v
    return cleaned


class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone_type="resnet50_local",
        model_name="convnext_tiny",
        pretrained=True,
        pretrained_path=None,
        out_dim=256,
        dino_repo_dir=None
    ):
        super().__init__()

        self.backbone_type = backbone_type

        if backbone_type == "resnet50_local":
            backbone = models.resnet50(weights=None)

            if pretrained_path is not None and os.path.exists(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location="cpu")
                state_dict = extract_state_dict(checkpoint)
                state_dict = clean_state_dict_keys(state_dict)

                missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
                print(f"[ImageEncoder] Loaded local pretrained weights from: {pretrained_path}")
                print(f"[ImageEncoder] Missing keys: {missing}")
                print(f"[ImageEncoder] Unexpected keys: {unexpected}")
            else:
                print("[ImageEncoder] No local pretrained weight loaded. Using random init.")

            in_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()

            self.backbone = backbone
            self.proj = nn.Linear(in_dim, out_dim)

        elif backbone_type == "dinov2_local":
            if dino_repo_dir is None:
                raise ValueError("dino_repo_dir must be provided when backbone_type='dinov2_local'")
            if not os.path.isdir(dino_repo_dir):
                raise ValueError(f"DINO repo dir not found: {dino_repo_dir}")

            if dino_repo_dir not in sys.path:
                sys.path.insert(0, dino_repo_dir)

            # model_name: dinov2_vits14 / dinov2_vitb14 / dinov2_vitl14
            try:
                backbone = torch.hub.load(
                    dino_repo_dir,
                    model_name,
                    source="local",
                    pretrained=False
                )
            except TypeError:
                # 如果本地 hub entrypoint 不接受 pretrained=False
                backbone = torch.hub.load(
                    dino_repo_dir,
                    model_name,
                    source="local"
                )

            if pretrained_path is not None and os.path.exists(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location="cpu")
                state_dict = extract_state_dict(checkpoint)
                state_dict = clean_state_dict_keys(state_dict)

                missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
                print(f"[ImageEncoder] Loaded DINOv2 local weights from: {pretrained_path}")
                print(f"[ImageEncoder] Missing keys: {missing}")
                print(f"[ImageEncoder] Unexpected keys: {unexpected}")
            else:
                print("[ImageEncoder] No DINOv2 local pretrained weight loaded. Using hub/default init.")

            if hasattr(backbone, "embed_dim"):
                in_dim = backbone.embed_dim
            elif hasattr(backbone, "num_features"):
                in_dim = backbone.num_features
            else:
                raise ValueError("Cannot infer output feature dim from DINOv2 backbone.")

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
        self._check_feat_shape(feat)
        feat = self.proj(feat)
        return feat

    def _check_feat_shape(self, feat):
        if feat.ndim != 2:
            raise RuntimeError(
                f"Expected backbone output shape [B, D], but got {tuple(feat.shape)}. "
                "Please check whether your local DINOv2 hub model returns pooled features "
                "or a token sequence in this repo version."
            )


class AttentionPooling(nn.Module):
    def __init__(self, feat_dim=256, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats, mask=None):
        scores = self.attn(feats).squeeze(-1)

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
        dino_repo_dir=None
    ):
        super().__init__()

        self.encoder = ImageEncoder(
            backbone_type=backbone_type,
            model_name=backbone_name,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            out_dim=feat_dim,
            dino_repo_dir=dino_repo_dir
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
        self.hier_head = nn.Linear(fusion_dim, 3)

    def encode_views(self, x):
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
        hier_logits = self.hier_head(fused)

        return {
            "logits": logits,
            "hier_logits": hier_logits,
            "local_attn": local_attn,
            "global_attn": global_attn,
            "fused_feat": fused
        }
