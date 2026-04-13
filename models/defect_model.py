import torch
import torch.nn as nn
import timm


class ImageEncoder(nn.Module):
    def __init__(self, model_name="convnext_tiny", pretrained=True, out_dim=256):
        super().__init__()
        # 先用 timm backbone，例如 convnext_tiny、resnet50、efficientnet_b0 都行
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        in_dim = self.backbone.num_features
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # backbone 抽 feature
        feat = self.backbone(x)   # [B, in_dim]
        # 線性層投影到固定維度
        feat = self.proj(feat)    # [B, out_dim]
        return feat


class AttentionPooling(nn.Module):
    def __init__(self, feat_dim=256, hidden_dim=128):
        super().__init__()
        # 把多張圖融合成一個 feature
        # 假設同一個 sample 有 6 張 low-FOV 圖，模型不一定要平均看待。
        # Attention pooling 會自動學：哪些圖比較重要、哪些圖可以忽略（比投票或平均分數更合理）
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

        weights = torch.softmax(scores, dim=1)  # [B, N]
        pooled = torch.sum(feats * weights.unsqueeze(-1), dim=1)  # [B, D]
        return pooled, weights


class DefectClassifier(nn.Module):
    def __init__(
        self,
        backbone_name="convnext_tiny",
        feat_dim=256,
        num_classes=4,
        num_attrs=4,
        pretrained=True
    ):
        super().__init__()

        self.encoder = ImageEncoder(
            model_name=backbone_name,
            pretrained=pretrained,
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
        feats = self.encoder(x)          # [B*N, D]
        feats = feats.view(B, N, -1)     # [B, N, D]
        return feats

    def forward(self, local_imgs, global_imgs, local_mask=None, global_mask=None):
        # local_imgs: 同一個 sample 裡所有 FOV <= 29 的圖
        # global_imgs: 同一個 sample 裡所有 FOV > 29 的圖
        # ==> 每個 sample 張數不同，就用 padding + mask 解
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
