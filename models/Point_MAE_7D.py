import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)
from utils.logger import *
import random

# from knn_cuda import KNN
from pytorch3d.ops import knn_points

# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from pytorch3d.loss import chamfer_distance
from functools import partial

from utils import misc
from models.Point_MAE import Group, TransformerEncoder, TransformerDecoder


class ChamferDistanceL1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return chamfer_distance(x, y, norm=1)[0]


class ChamferDistanceL2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return chamfer_distance(x, y, norm=2)[0]




class Encoder_7D(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        point_groups : B G N 3
        -----------------
        feature_global : B G C
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

# Pretrain model
class MaskTransformer_7D(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f"[args] {config.transformer_config}", logger="Transformer")


        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_vis_token, pos_vis_embed):
        x_vis_token = self.blocks(x_vis_token, pos_vis_embed)
        x_vis_token = self.norm(x_vis_token)

        return x_vis_token


@MODELS.register_module()
class Point_MAE_7D(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f"[Point_MAE] ", logger="Point_MAE")
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer_7D(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)
        ]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(
            f"[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...",
            logger="Point_MAE",
        )
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=0.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)
        
        self.mask_type = config.transformer_config.mask_type
        self.mask_ratio = config.transformer_config.mask_ratio
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder_embed = Encoder_7D(encoder_channel=self.encoder_dims)
        
        self.encoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == "cdl2":
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()
            
    def _mask_center_block(self, center, noaug=False):
        """
        center : B G 3
        --------------
        mask : B G (bool)
        """
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(
                points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1
            )  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        """
        center : B G 3
        --------------
        mask : B G (bool)
        """
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack(
                [
                    np.zeros(G - self.num_mask),
                    np.ones(self.num_mask),
                ]
            )
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G


    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)
        
        # generate mask
        if self.mask_type == "rand":
            bool_masked_pos = self._mask_center_rand(center, noaug=False)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=False)
        
        # embed the neighbors
        all_input_tokens = self.encoder_embed(neighborhood)  #  B G C

        batch_size, seq_len, C = all_input_tokens.size()

        # mask the input tokens
        x_vis_token = all_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        
        # mask the centers
        center_vis = center[~bool_masked_pos]
        center_mask = center[bool_masked_pos]
        
        neighborhood_vis = neighborhood[~bool_masked_pos]
        neighborhood_mask = neighborhood[bool_masked_pos]
        
        # embed the centers to positional embeddings
        pos_vis_embed_encoder = self.encoder_pos_embed(center_vis.reshape(batch_size, -1, 3))
        
        x_vis_token = self.MAE_encoder(x_vis_token, pos_vis_embed_encoder)
        B, _, C = x_vis_token.shape  # B VIS C

        pos_vis_embed_decoder = self.decoder_pos_embed(center_vis).reshape(B, -1, C)
        pos_mask_embed_decoder = self.decoder_pos_embed(center_mask).reshape(B, -1, C)

        _, N_mask, _ = pos_mask_embed_decoder.shape
        mask_token = self.mask_token.expand(B, N_mask, -1)
        x_full = torch.cat([x_vis_token, mask_token], dim=1)
        pos_full = torch.cat([pos_vis_embed_decoder, pos_mask_embed_decoder], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N_mask)

        B, M, C = x_rec.shape
        rebuild_points = (
            self.increase_dim(x_rec.transpose(1, 2))
            .transpose(1, 2)
            .reshape(B * M, -1, 3)
        )  # B M 1024

        gt_points = neighborhood_mask.reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        if vis:  # visualization
            vis_points = neighborhood_vis.reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center_vis.unsqueeze(1)
            full_rebuild = rebuild_points + center_mask.unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center_mask, center_vis], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1


# finetune model
@MODELS.register_module()
class PointTransformer_7D(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder_7D(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim),
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {
                k.replace("module.", ""): v for k, v in ckpt["base_model"].items()
            }

            for k in list(base_ckpt.keys()):
                if k.startswith("MAE_encoder"):
                    base_ckpt[k[len("MAE_encoder.") :]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith("base_model"):
                    base_ckpt[k[len("base_model.") :]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log("missing_keys", logger="Transformer")
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger="Transformer",
                )
            if incompatible.unexpected_keys:
                print_log("unexpected_keys", logger="Transformer")
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger="Transformer",
                )

            print_log(
                f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
                logger="Transformer",
            )
        else:
            print_log("Training from scratch!!!", logger="Transformer")
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret
