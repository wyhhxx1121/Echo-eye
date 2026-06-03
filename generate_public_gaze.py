#!/usr/bin/env python3
"""
bus_gaze_generate_sam.py

- 引入层次化 Prompt 调制（Hierarchical Prompt Conditioning）：在 ViT 每一层用 FiLM（由 Prompt 生成的 gamma/beta）调制 token。
- 在 CNN 分支也用 Prompt FiLM 调制特征。
- 在联合训练阶段加入 Supervised Contrastive 年资对比损失（高/低年资两个“类”）。
"""

import os, sys, random, math, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def env_path(name, default):
    return os.environ.get(name, default)

# -------------------------
# CONFIG (edit here)
# -------------------------
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # private paired gaze
    junior_img_root = env_path("ECHO_EYE_JUNIOR_IMAGE_ROOT", "data/private/images")
    junior_gaze_root = env_path("ECHO_EYE_JUNIOR_GAZE_ROOT", "data/private/gaze_junior")
    senior_img_root = env_path("ECHO_EYE_SENIOR_IMAGE_ROOT", "data/private/images_senior")
    senior_gaze_root = env_path("ECHO_EYE_SENIOR_GAZE_ROOT", "data/private/gaze_senior")

    # public base
    public_base = env_path("ECHO_EYE_PUBLIC_BASE", "data/public")

    # datasets
    ds_mask_only = {
        "BUS_UCLM_im": {
            "image": os.path.join(public_base, "BUS_UCLM_im", "image"),
            "mask":  os.path.join(public_base, "BUS_UCLM_im", "mask")
        },
        "BUS_WHU_im": {
            "image": os.path.join(public_base, "BUS_WHU_im", "image"),
            "mask":  os.path.join(public_base, "BUS_WHU_im", "mask")
        }
    }

    ds_mask_label = {
        "BrEaST": {
            "image_ben": os.path.join(public_base, "BrEaST", "benign"),
            "image_mal": os.path.join(public_base, "BrEaST", "malignant"),
            "mask_ben":  os.path.join(public_base, "BrEaST_mask", "benign"),
            "mask_mal":  os.path.join(public_base, "BrEaST_mask", "malignant"),
        },
        "BUS_UC": {
            "image_ben": os.path.join(public_base, "BUS_UC", "benign"),
            "image_mal": os.path.join(public_base, "BUS_UC", "malignant"),
            "mask_ben":  os.path.join(public_base, "BUS_UC_mask", "benign"),
            "mask_mal":  os.path.join(public_base, "BUS_UC_mask", "malignant")
        },
        "BUSBRA": {
            "image_ben": os.path.join(public_base, "BUSBRA", "benign"),
            "image_mal": os.path.join(public_base, "BUSBRA", "malignant"),
            "mask_ben":  os.path.join(public_base, "BUSBRA_mask", "benign"),
            "mask_mal":  os.path.join(public_base, "BUSBRA_mask", "malignant")
        },
        "QAMEBI": {
            "image_ben": os.path.join(public_base, "QAMEBI", "benign"),
            "image_mal": os.path.join(public_base, "QAMEBI", "malignant"),
            "mask_ben":  os.path.join(public_base, "QAMEBI_mask", "benign"),
            "mask_mal":  os.path.join(public_base, "QAMEBI_mask", "malignant")
        },
        "UDIAT": {
            "image_ben": os.path.join(public_base, "UDIAT", "benign"),
            "image_mal": os.path.join(public_base, "UDIAT", "malignant"),
            "mask_ben":  os.path.join(public_base, "UDIAT_mask", "benign"),
            "mask_mal":  os.path.join(public_base, "UDIAT_mask", "malignant")
        },
        "BUSI": {
            "image_ben": os.path.join(public_base, "BUSI", "benign"),
            "image_mal": os.path.join(public_base, "BUSI", "malignant"),
            "mask_ben":  os.path.join(public_base, "BUSI_mask", "benign"),
            "mask_mal":  os.path.join(public_base, "BUSI_mask", "malignant")
        },
    }

    ds_label_only = {
        "GDPH_SYSUCC_i": {
            "image_ben": os.path.join(public_base, "GDPH&SYSUCC_i", "benign"),
            "image_mal": os.path.join(public_base, "GDPH&SYSUCC_i", "malignant")
        },
        "us-dataset_i": {
            "image_ben": os.path.join(public_base, "us-dataset_i", "benign"),
            "image_mal": os.path.join(public_base, "us-dataset_i", "malignant")
        },
        "US3M_i": {
            "image_ben": os.path.join(public_base, "US3M_i", "benign"),
            "image_mal": os.path.join(public_base, "US3M_i", "malignant")
        },
        "BUS_COT": {
            "image_ben": os.path.join(public_base, "BUS_COT", "benign"),
            "image_mal": os.path.join(public_base, "BUS_COT", "malignant")
        }
    }

    predicted_mask_root = env_path("ECHO_EYE_PREDICTED_MASK_ROOT", "runs/predicted_masks")
    out_public_gaze = env_path("ECHO_EYE_PUBLIC_GAZE_OUT", "runs/public_gaze_sam")

    # model/encoder
    img_size = 256
    patch_size = 16
    base_dim = 64
    transformer_layers = 4
    transformer_heads = 4

    # train
    batch_size = 8
    seg_pretrain_epochs = 30
    joint_epochs = 30
    lr = 1e-4
    runs_dir = "./runs"

    # losses & sampling
    gaze_weight = 1.0
    seg_weight = 1.0
    sparsity_w = 0.01

    max_points_jun = 1200
    topk_ratio_jun = 0.06
    bg_ratio_jun = 0.01

    max_points_sen = 800
    topk_ratio_sen = 0.03
    bg_ratio_sen = 0.005

    seg_threshold = 80
    seg_dilate_px = 3
    dot_radius_base = 1

    lambda_entropy_s = 1.0
    lambda_entropy_j = 0.8
    lambda_containment = 2.0
    topk_ratio_for_containment = 0.12

    # seniority contrastive
    lambda_seniority = 0.2
    con_temperature = 0.07
    con_embed_dim = 128

    seed = 42

    # pretrained samus weights
    pretrained_samus_ckpt = env_path("ECHO_EYE_SAMUS_CKPT", "weights/SAMUS.pth")

cfg = Config()

# -------------------------
# Utilities
# -------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def list_files(folder, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    p = Path(folder)
    if not p.exists(): return []
    files=[]
    for e in exts:
        files += sorted([str(x) for x in p.rglob(f"*{e}")])
    return files

def read_gray(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)/255.0

def read_color(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: return None
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def norm01_np(x):
    x = x.astype(np.float32)
    mn = x.min(); mx = x.max()
    if mx - mn < 1e-8: return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def gaze_image_to_density_from_img(gaze_img, sigma=3, size_out=None):
    if gaze_img is None: return None
    if len(gaze_img.shape) == 3:
        g = cv2.cvtColor(gaze_img, cv2.COLOR_BGR2GRAY)
    else: g = gaze_img
    mask = (g > 10).astype(np.uint8) * 255
    k = max(3, int(2 * round(sigma) + 1))
    den = cv2.GaussianBlur(mask.astype(np.float32), (k,k), sigmaX=sigma, sigmaY=sigma)
    den = norm01_np(den)
    if size_out is not None:
        den = cv2.resize((den*255).astype(np.uint8), (size_out, size_out), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    return den

def normalize_name(s):
    if s is None: return ""
    s2 = str(s).lower()
    s2 = s2.replace(" ", "").replace("-", "").replace("_", "").replace("@", "").replace("(", "").replace(")", "")
    s2 = s2.replace("image", "").replace("mask", "").replace("anno", "").replace("tumor", "").replace("cropped", "")
    return s2

def find_dir_by_fragment(base, frag):
    if not base or not os.path.exists(base): return None
    frag_low = frag.lower()
    for child in sorted(Path(base).iterdir()):
        if child.is_dir() and frag_low in child.name.lower():
            return str(child)
    for child in sorted(Path(base).rglob("*")):
        if child.is_dir() and frag_low in child.name.lower():
            return str(child)
    for child in sorted(Path(base).rglob("*")):
        if child.is_dir() and normalize_name(frag_low) in normalize_name(child.name):
            return str(child)
    return None

def resolve_paths(cfg):
    print("[Resolver] Searching under:", cfg.public_base)
    for ds, info in cfg.ds_mask_only.items():
        for key in ["image","mask"]:
            p = info.get(key)
            if p and os.path.exists(p): continue
            found = find_dir_by_fragment(cfg.public_base, ds)
            if found:
                cand = os.path.join(found, key)
                info[key] = cand if os.path.exists(cand) else found
                print(f"  resolved {ds}.{key} => {info[key]}")
    for ds, info in cfg.ds_mask_label.items():
        for field in ["image_ben","image_mal","mask_ben","mask_mal"]:
            p = info.get(field)
            if p and os.path.exists(p): continue
            found = find_dir_by_fragment(cfg.public_base, ds)
            if found:
                cand = os.path.join(found, "benign") if "ben" in field else os.path.join(found, "malignant")
                if os.path.exists(cand): info[field]=cand; print(f"  resolved {ds}.{field} => {cand}")
    for ds, info in cfg.ds_label_only.items():
        for field in ["image_ben","image_mal"]:
            p = info.get(field)
            if p and os.path.exists(p): continue
            found = find_dir_by_fragment(cfg.public_base, ds)
            if found:
                cand = os.path.join(found, "benign") if "ben" in field else os.path.join(found, "malignant")
                info[field] = cand if os.path.exists(cand) else found
                print(f"  fallback {ds}.{field} => {info[field]}")

def try_mask_candidates(mask_dir, basename_noext):
    if not mask_dir or not os.path.exists(mask_dir): return None
    candidates=[]
    for ext in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]:
        candidates.append(os.path.join(mask_dir, basename_noext+ext))
    for ext in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]:
        candidates.append(os.path.join(mask_dir, basename_noext+"_mask"+ext))
        candidates.append(os.path.join(mask_dir, basename_noext+"_anno"+ext))
        candidates.append(os.path.join(mask_dir, basename_noext+"_annotation"+ext))
    for c in candidates:
        if os.path.exists(c): return c
    nums = re.findall(r"\d+", basename_noext)
    if nums:
        num = nums[0]
        for p in Path(mask_dir).iterdir():
            if not p.is_file(): continue
            if num in p.name: return str(p)
    norm_base = normalize_name(basename_noext)
    for p in Path(mask_dir).iterdir():
        if not p.is_file(): continue
        if norm_base in normalize_name(p.name): return str(p)
    return None

def pair_images_masks_for_dataset(dataset_name):
    pairs=[]
    if dataset_name in cfg.ds_mask_only:
        info = cfg.ds_mask_only[dataset_name]
        img_dir = info.get("image")
        mask_dir = info.get("mask")
        if not img_dir or not os.path.exists(img_dir): return pairs
        imgs = list_files(img_dir)
        for img in imgs:
            name_noext = os.path.splitext(os.path.basename(img))[0]
            m = try_mask_candidates(mask_dir, name_noext)
            if m: pairs.append((img, m, ""))
        return pairs
    if dataset_name in cfg.ds_mask_label:
        info = cfg.ds_mask_label[dataset_name]
        for cls_key, cls in [("image_ben","benign"),("image_mal","malignant")]:
            img_dir = info.get(cls_key)
            mask_dir = info.get("mask_ben") if cls=="benign" else info.get("mask_mal")
            if (mask_dir is None or not os.path.exists(mask_dir)):
                mask_dir_guess = find_dir_by_fragment(cfg.public_base, dataset_name + "_mask") or find_dir_by_fragment(cfg.public_base, dataset_name + "mask")
                if mask_dir_guess: mask_dir = mask_dir_guess
            if not img_dir or not os.path.exists(img_dir): continue
            imgs = list_files(img_dir)
            for img in imgs:
                name_noext = os.path.splitext(os.path.basename(img))[0]
                m = try_mask_candidates(mask_dir, name_noext)
                if m: pairs.append((img, m, cls))
        return pairs
    return pairs

def gather_all_seg_pairs():
    all_pairs=[]
    for ds in cfg.ds_mask_only.keys():
        prs = pair_images_masks_for_dataset(ds)
        for img,mask,cls in prs: all_pairs.append((img,mask,ds,cls))
    for ds in cfg.ds_mask_label.keys():
        prs = pair_images_masks_for_dataset(ds)
        for img,mask,cls in prs: all_pairs.append((img,mask,ds,cls))
    return all_pairs

def print_dataset_stats_and_preview():
    print("== Dataset discovery report ==")
    for ds, info in cfg.ds_mask_only.items():
        imgs = list_files(info.get("image","")); masks = list_files(info.get("mask",""))
        print(f"{ds} (mask-only): images={len(imgs)}, masks={len(masks)}")
    for ds, info in cfg.ds_mask_label.items():
        imgs_b = list_files(info.get("image_ben","")); imgs_m = list_files(info.get("image_mal",""))
        masks_b = list_files(info.get("mask_ben","")); masks_m = list_files(info.get("mask_mal",""))
        print(f"{ds} (mask+label): images_ben={len(imgs_b)}, images_mal={len(imgs_m)}, masks_ben={len(masks_b)}, masks_mal={len(masks_m)}")
    for ds, info in cfg.ds_label_only.items():
        imgs_b = list_files(info.get("image_ben","")); imgs_m = list_files(info.get("image_mal",""))
        print(f"{ds} (label-only): images_ben={len(imgs_b)}, images_mal={len(imgs_m)}")
    print("== End report ==")

# -------------------------
# Datasets
# -------------------------
class CombinedSegDataset(Dataset):
    def __init__(self, pairs, img_size=256):
        self.pairs = pairs; self.img_size = img_size
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        img_p, mask_p, ds, cls = self.pairs[idx]
        img = read_gray(img_p, size=self.img_size)
        mask = read_gray(mask_p, size=self.img_size)
        if img is None or mask is None:
            raise RuntimeError(f"Failed read pair: {img_p} / {mask_p}")
        img_t = torch.from_numpy(img).unsqueeze(0).float()
        mask_t = torch.from_numpy((mask > 0.5).astype(np.float32)).unsqueeze(0).float()
        return img_t, mask_t, img_p, ds, cls

class PairedGazeDataset(Dataset):
    def __init__(self, cfg):
        self.img_size = cfg.img_size; self.samples=[]
        classes = ["Benign","Malignant","benign","malignant"]
        for cls in classes:
            jun_img_dir = Path(cfg.junior_img_root) / cls
            jun_gaze_dir = Path(cfg.junior_gaze_root) / cls
            sen_img_dir = Path(cfg.senior_img_root) / cls
            sen_gaze_dir = Path(cfg.senior_gaze_root) / cls
            if not jun_img_dir.exists(): continue
            for p in sorted(jun_img_dir.iterdir()):
                if not p.is_file(): continue
                fname = p.name
                jun_gaze_p = jun_gaze_dir / fname
                sen_img_p = sen_img_dir / fname
                sen_gaze_p = sen_gaze_dir / fname
                if jun_gaze_p.exists() and sen_img_p.exists() and sen_gaze_p.exists():
                    self.samples.append((str(p), str(jun_gaze_p), str(sen_img_p), str(sen_gaze_p)))
        if len(self.samples) == 0:
            raise RuntimeError("No matched paired private gaze samples found. Check dataset paths.")
        random.shuffle(self.samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_p, jun_gaze_p, sen_img_p, sen_gaze_p = self.samples[idx]
        img = read_gray(img_p, size=self.img_size).astype(np.float32)
        jun_gaze_img = cv2.imread(jun_gaze_p); sen_gaze_img = cv2.imread(sen_gaze_p)
        den_j = gaze_image_to_density_from_img(jun_gaze_img, sigma=3.0, size_out=self.img_size)
        den_s = gaze_image_to_density_from_img(sen_gaze_img, sigma=1.5, size_out=self.img_size)
        img_t = torch.from_numpy(img).unsqueeze(0).float()
        den_j_t = torch.from_numpy(den_j).unsqueeze(0).float()
        den_s_t = torch.from_numpy(den_s).unsqueeze(0).float()
        return img_t, den_j_t, den_s_t, img_p

# -------------------------
# Model
# -------------------------
class PatchEmbedOverlap(nn.Module):
    def __init__(self, in_ch=1, embed_dim=64, patch_size=16):
        super().__init__()
        stride = patch_size // 2
        padding = patch_size // 4
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size
    def forward(self, x):
        y = self.proj(x)
        B,D,Hp,Wp = y.shape
        tokens = y.flatten(2).transpose(1,2)
        tokens = self.norm(tokens)
        return tokens, (Hp, Wp)

class TransformerBlock(nn.Module):
    def __init__(self, dim, nhead, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop)
        )
    def forward(self, x):
        h = x
        x = self.norm1(x)
        x,_ = self.attn(x, x, x, need_weights=False)
        x = x + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        return x

class HierPromptViT(nn.Module):
    """ViT encoder with Hierarchical Prompt FiLM at each layer."""
    def __init__(self, dim, depth=4, nhead=4, prompt_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, nhead) for _ in range(depth)])
        self.depth = depth
        self.prompt_dim = prompt_dim if prompt_dim is not None else dim
        # per-layer prompt -> (gamma,beta) for FiLM
        self.prompt_mlps = nn.ModuleList([nn.Linear(self.prompt_dim, 2*dim) for _ in range(depth)])
    def forward(self, tokens, prompt_emb=None):
        # tokens: [B, N, D]
        for l in range(self.depth):
            if prompt_emb is not None:
                gb = self.prompt_mlps[l](prompt_emb)  # [B, 2D]
                g, b = gb.chunk(2, dim=1)            # [B, D], [B, D]
                g = g.unsqueeze(1)                   # [B,1,D]
                b = b.unsqueeze(1)
                tokens = tokens * (1 + g) + b        # FiLM before block
            tokens = self.layers[l](tokens)
        return tokens

class CNNBranch(nn.Module):
    def __init__(self, in_ch=1, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//2, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_ch//2), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//2, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class CrossBranchAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
    def forward(self, vit_tokens, cnn_tokens):
        q = self.q_proj(vit_tokens)
        k = self.k_proj(cnn_tokens)
        v = self.v_proj(cnn_tokens)
        att = torch.bmm(q, k.transpose(1,2)) / math.sqrt(q.shape[-1])
        att = F.softmax(att, dim=-1)
        agg = torch.bmm(att, v)
        out = self.out(agg) + vit_tokens
        return out

class PromptEncoder(nn.Module):
    def __init__(self, img_size=256, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, embed_dim//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim//2), nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(embed_dim, embed_dim)
    def forward(self, gaze):
        if gaze is None: return None
        x = self.conv(gaze)
        x = x.view(x.shape[0], -1)
        emb = self.fc(x)
        return emb  # [B, embed_dim]

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.conv(x)

class SAMUSModel(nn.Module):
    def __init__(self, cfg, preload_pos_embed=None):
        super().__init__()
        self.cfg = cfg
        self.patch = PatchEmbedOverlap(in_ch=1, embed_dim=cfg.base_dim, patch_size=cfg.patch_size)
        self.vit = HierPromptViT(dim=cfg.base_dim, depth=cfg.transformer_layers, nhead=cfg.transformer_heads, prompt_dim=cfg.base_dim)
        self.cnn = CNNBranch(in_ch=1, out_ch=cfg.base_dim)
        self.cba = CrossBranchAttention(cfg.base_dim)
        # CNN FiLM from prompt
        self.cnn_film = nn.Linear(cfg.base_dim, 2*cfg.base_dim)

        stride = cfg.patch_size // 2
        Hp = cfg.img_size // stride
        Wp = Hp
        self.Hp = Hp; self.Wp = Wp

        self.token2map = nn.Linear(cfg.base_dim, cfg.base_dim)

        self.dec1 = DecoderBlock(cfg.base_dim, cfg.base_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DecoderBlock(cfg.base_dim, cfg.base_dim//2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DecoderBlock(cfg.base_dim//2, cfg.base_dim//4)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = DecoderBlock(cfg.base_dim//4, cfg.base_dim//8)

        self.prompt_enc = PromptEncoder(img_size=cfg.img_size, embed_dim=cfg.base_dim)
        self.film_map4 = nn.Linear(cfg.base_dim, (cfg.base_dim//8)*2)

        self.head_mask = nn.Conv2d(cfg.base_dim//8, 1, 1)
        self.head_sen = nn.Conv2d(cfg.base_dim//8, 1, 1)
        self.head_jun = nn.Conv2d(cfg.base_dim//8, 1, 1)

        # projection heads for contrastive (z embeddings)
        C = cfg.base_dim // 8
        self.proj_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(C, cfg.con_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.con_embed_dim, cfg.con_embed_dim)
        )

        # pos_embed register (if provided)
        if preload_pos_embed is not None:
            self.register_parameter('pos_embed', nn.Parameter(preload_pos_embed))
        else:
            N = Hp * Wp
            self.register_parameter('pos_embed', nn.Parameter(torch.zeros(1, N, cfg.base_dim)))

    def apply_film_2d(self, feat, gamma, beta):
        if gamma is None or beta is None: return feat
        B,C,H,W = feat.shape
        g = gamma.view(B,C,1,1)
        b = beta.view(B,C,1,1)
        return feat * (1 + g) + b

    def prompt_to_film_2d(self, prompt_emb, out_ch):
        if prompt_emb is None: return None, None
        # map to 2*out_ch
        mapper = nn.Linear(self.cfg.base_dim, 2*out_ch).to(prompt_emb.device)
        gb = mapper(prompt_emb)
        g,b = gb.chunk(2, dim=1)
        if g.shape[1] != out_ch:
            proj = nn.Linear(g.shape[1], out_ch).to(prompt_emb.device)
            g = proj(g); b = proj(b)
        return g, b

    def forward(self, x, prompt_s=None, prompt_j=None):
        B = x.shape[0]
        tokens, (Hp, Wp) = self.patch(x)                   # [B, N, D]
        pe = self.pos_embed
        if pe.shape[1] != tokens.shape[1]:
            N = tokens.shape[1]
            if pe.shape[1] > N:
                pe = pe[:, :N, :].to(tokens.device)
            else:
                pad = torch.zeros(1, N - pe.shape[1], pe.shape[2], device=tokens.device)
                pe = torch.cat([pe.to(tokens.device), pad], dim=1)
        tokens = tokens + pe.to(tokens.device)

        # prompt embeddings
        prompt_s_emb = self.prompt_enc(prompt_s) if prompt_s is not None else None
        prompt_j_emb = self.prompt_enc(prompt_j) if prompt_j is not None else None
        # encoder侧使用“mask风格”的综合 prompt（层次化调制）
        if prompt_s_emb is not None and prompt_j_emb is not None:
            prompt_mask_emb = 0.5 * (prompt_s_emb + prompt_j_emb)
        elif prompt_s_emb is not None:
            prompt_mask_emb = prompt_s_emb
        elif prompt_j_emb is not None:
            prompt_mask_emb = prompt_j_emb
        else:
            prompt_mask_emb = None

        vit_out = self.vit(tokens, prompt_emb=prompt_mask_emb)  # Hierarchical FiLM inside

        cnn_feat = self.cnn(x)                                  # [B, C, Hc, Wc]
        # CNN FiLM (single-shot)
        if prompt_mask_emb is not None:
            gb = self.cnn_film(prompt_mask_emb)                 # [B, 2C]
            g, b = gb.chunk(2, dim=1)                           # [B,C]
            cnn_feat = self.apply_film_2d(cnn_feat, g, b)

        cnn_resized = F.interpolate(cnn_feat, size=(self.Hp, self.Wp), mode='bilinear', align_corners=False)
        cnn_tokens = cnn_resized.flatten(2).transpose(1,2)      # [B, N, D]

        fused = self.cba(vit_out, cnn_tokens)                   # [B, N, D]
        map_feat = self.token2map(fused).transpose(1,2).view(B, self.cfg.base_dim, self.Hp, self.Wp)

        # upsample decoder (shared backbone)
        d = self.dec1(map_feat)
        d = self.up1(d)
        d = self.dec2(d)
        d = self.up2(d)
        d = self.dec3(d)
        d = self.up3(d)
        d = self.dec4(d)
        d = F.interpolate(d, size=(self.cfg.img_size, self.cfg.img_size), mode='bilinear', align_corners=False)

        # head-level FiLM：mask 使用平均 prompt；senior/junior 分别使用对应 prompt
        def film_map_4(pe):
            if pe is None: return None, None
            gb = self.film_map4(pe)  # [B, 2*(C/8)]
            g,b = gb.chunk(2, dim=1)
            C = self.cfg.base_dim // 8
            if g.shape[1] != C:
                proj = nn.Linear(g.shape[1], C).to(g.device)
                g = proj(g); b = proj(b)
            return g, b

        gb_mask = film_map_4(prompt_mask_emb)
        gb_sen  = film_map_4(prompt_s_emb)
        gb_jun  = film_map_4(prompt_j_emb)

        feat_mask = d.clone(); feat_sen = d.clone(); feat_jun = d.clone()
        if gb_mask[0] is not None: feat_mask = self.apply_film_2d(feat_mask, gb_mask[0], gb_mask[1])
        if gb_sen[0]  is not None: feat_sen  = self.apply_film_2d(feat_sen,  gb_sen[0],  gb_sen[1])
        if gb_jun[0]  is not None: feat_jun  = self.apply_film_2d(feat_jun,  gb_jun[0],  gb_jun[1])

        out_mask = torch.sigmoid(self.head_mask(feat_mask))
        out_sen  = torch.sigmoid(self.head_sen(feat_sen))
        out_jun  = torch.sigmoid(self.head_jun(feat_jun))

        # contrastive embeddings
        zh = F.normalize(self.proj_head(feat_sen), dim=1)  # [B, D]
        zl = F.normalize(self.proj_head(feat_jun), dim=1)  # [B, D]

        return out_mask, out_sen, out_jun, zh, zl

# -------------------------
# ckpt utils
# -------------------------
def clean_state_dict(raw_state):
    new = {}
    for k,v in raw_state.items():
        if k.startswith('module.'):
            new[k[len('module.'):]] = v
        else:
            new[k] = v
    return new

def load_ckpt_flexible(ckpt_path, model, device):
    if not os.path.exists(ckpt_path):
        print("[load_ckpt_flexible] ckpt not found:", ckpt_path); return {}
    ck = torch.load(ckpt_path, map_location='cpu')
    state = ck.get('model_state', ck) if isinstance(ck, dict) else ck
    state = clean_state_dict(state)
    if 'pos_embed' in state and 'pos_embed' not in model.state_dict():
        pe = state['pos_embed']
        pe_t = torch.tensor(pe).to(device) if not isinstance(pe, torch.Tensor) else pe.to(device)
        model.register_parameter('pos_embed', nn.Parameter(pe_t))
        print("[load_ckpt_flexible] registered pos_embed from checkpoint with shape", tuple(pe_t.shape))
    state_on_device = {k: (torch.tensor(v).to(device) if not isinstance(v, torch.Tensor) else v.to(device)) for k,v in state.items()}
    missing, unexpected = model.load_state_dict(state_on_device, strict=False)
    print("[load_ckpt_flexible] load_state_dict done. missing keys:", missing, "unexpected keys:", unexpected)
    return {"missing": missing, "unexpected": unexpected}

# -------------------------
# Losses & sampling
# -------------------------
def dice_loss(pred, target, eps=1e-6):
    pred_f = pred.view(pred.size(0), -1)
    target_f = target.view(target.size(0), -1)
    inter = (pred_f * target_f).sum(dim=1)
    union = pred_f.sum(dim=1) + target_f.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()

@torch.no_grad()
def _masked_fill_diagonal(mat, fill=0.0):
    i = torch.arange(mat.size(0), device=mat.device)
    mat[i, i] = fill
    return mat

def supervised_contrastive_loss(features, labels, temperature=0.07, eps=1e-8):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS'20)
    features: [N, D] (L2 normalized)
    labels:   [N] (int), same-class positives, others negatives
    """
    N, D = features.size()
    if N < 2: 
        return torch.tensor(0.0, device=features.device)
    sim = torch.div(features @ features.t(), temperature)  # [N,N]
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()  # stability
    # mask: positives (same label, excl. self)
    labels = labels.contiguous().view(-1,1)
    mask = torch.eq(labels, labels.T).float().to(features.device)
    mask = mask - torch.eye(N, device=features.device)  # remove diagonal
    # denominator
    exp_sim = torch.exp(sim) * (1 - torch.eye(N, device=features.device))
    denom = exp_sim.sum(dim=1, keepdim=True) + eps
    # for each anchor, positives indices
    log_prob = sim - torch.log(denom)
    # only count positives
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + eps)
    loss = - mean_log_prob_pos.mean()
    return loss

def density_to_points_topk(density, max_points=500, topk_ratio=0.1, bg_ratio=0.05):
    H, W = density.shape
    flat = density.flatten()
    N = flat.size
    k = int(max(1, round(N * topk_ratio)))
    thresh = np.partition(flat, -k)[-k]
    mask_fg = (density >= thresh)
    fg_idxs = np.argwhere(mask_fg)
    bg_idxs = np.argwhere(~mask_fg)
    n_fg = int(round(max_points * (1.0 - bg_ratio)))
    n_bg = max_points - n_fg
    coords = []
    if fg_idxs.shape[0] > 0 and n_fg > 0:
        M = fg_idxs.shape[0]; size_fg = min(n_fg, M)
        fg_vals = density[mask_fg].flatten()
        if fg_vals.sum() <= 1e-12 or np.count_nonzero(fg_vals) < size_fg:
            idxs = np.random.choice(M, size=size_fg, replace=True)
        else:
            probs = fg_vals / fg_vals.sum()
            if np.count_nonzero(probs) < size_fg:
                idxs = np.random.choice(M, size=size_fg, replace=True)
            else:
                idxs = np.random.choice(M, size=size_fg, replace=False, p=probs)
        for ii in idxs:
            yx = fg_idxs[int(ii)]; coords.append((int(yx[1]), int(yx[0])))
    if bg_idxs.shape[0] > 0 and n_bg > 0:
        Mbg = bg_idxs.shape[0]; size_bg = min(n_bg, Mbg)
        bg_vals = density[~mask_fg].flatten()
        if bg_vals.sum() <= 1e-12 or np.count_nonzero(bg_vals) < size_bg:
            idxs = np.random.choice(Mbg, size=size_bg, replace=True)
        else:
            probs_bg = bg_vals / bg_vals.sum()
            if np.count_nonzero(probs_bg) < size_bg:
                idxs = np.random.choice(Mbg, size=size_bg, replace=True)
            else:
                idxs = np.random.choice(Mbg, size=size_bg, replace=False, p=probs_bg)
        for ii in idxs:
            yx = bg_idxs[int(ii)]; coords.append((int(yx[1]), int(yx[0])))
    if len(coords) < max_points and fg_idxs.shape[0] > 0:
        need = max_points - len(coords)
        remaining = [tuple(x) for x in fg_idxs.tolist()]
        remaining = [yx for yx in remaining if (int(yx[1]), int(yx[0])) not in coords]
        if len(remaining) > 0:
            take = min(need, len(remaining)); sel = random.sample(remaining, take)
            for yx in sel: coords.append((int(yx[1]), int(yx[0])))
    seen = set(); out=[]
    for (x,y) in coords:
        if (x,y) not in seen:
            seen.add((x,y)); out.append((x,y))
    return out

def density_to_points_smart(density, patch_size=8, max_pts_per_patch=5, min_patch_mean=0.005):
    H, W = density.shape; coords=[]
    for y0 in range(0, H, patch_size):
        for x0 in range(0, W, patch_size):
            patch = density[y0:min(y0+patch_size,H), x0:min(x0+patch_size,W)]
            p_mean = float(patch.mean())
            if p_mean < min_patch_mean: continue
            n_points = max(1, min(max_pts_per_patch, int(round(max_pts_per_patch * (p_mean / (p_mean + 1e-8))))))
            flat = patch.flatten()
            if flat.sum() <= 1e-8 or np.count_nonzero(flat) < n_points:
                for _ in range(n_points):
                    yy = y0 + random.randint(0, patch.shape[0]-1)
                    xx = x0 + random.randint(0, patch.shape[1]-1)
                    coords.append((xx, yy))
            else:
                cand = np.where(flat > 0)[0]
                probs = flat[cand] / flat[cand].sum()
                chosen = np.random.choice(cand, size=min(n_points, cand.size), replace=False, p=probs)
                for idx in chosen:
                    yy = y0 + (idx // patch.shape[1]); xx = x0 + (idx % patch.shape[1]); coords.append((int(xx), int(yy)))
    if len(coords)>0: coords = list(dict.fromkeys(coords))
    return coords

def overlay_red_dots(orig_bgr, coords, radius=1):
    img = orig_bgr.copy()
    for (x,y) in coords:
        xi = max(0, min(img.shape[1]-1, int(round(x))))
        yi = max(0, min(img.shape[0]-1, int(round(y))))
        cv2.circle(img, (xi, yi), int(radius), (0,0,255), -1)
    return img

def find_mask_for_image(img_path, mask_root):
    if not mask_root: return None
    base = os.path.basename(img_path); name, ext = os.path.splitext(base)
    m = try_mask_candidates(mask_root, name)
    return m

def spatial_entropy_map(output_tensor, eps=1e-8):
    B = output_tensor.shape[0]; flat = output_tensor.view(B, -1); s = flat.sum(dim=1, keepdim=True)
    p = flat / (s + eps); ent = -(p * torch.log(p + eps)).sum(dim=1); return ent.mean()

def topk_mask_from_density(batch_tensor, topk_ratio=0.1):
    B,C,H,W = batch_tensor.shape; flat = batch_tensor.view(B, -1); N = flat.shape[1]
    k = max(1, int(round(N * topk_ratio))); vals,_ = torch.topk(flat, k, dim=1); kth = vals[:, -1].view(B,1)
    thresh = kth.repeat(1, N); mask_flat = (flat >= thresh)
    return mask_flat.view(B,1,H,W)

# -------------------------
# Training / inference flows
# -------------------------
def train_segmentation(cfg, device, pairs):
    ds = CombinedSegDataset(pairs, img_size=cfg.img_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    preload_pos = None
    if cfg.pretrained_samus_ckpt and os.path.exists(cfg.pretrained_samus_ckpt):
        ck = torch.load(cfg.pretrained_samus_ckpt, map_location='cpu')
        state = ck.get('model_state', ck) if isinstance(ck, dict) else ck
        state = clean_state_dict(state)
        if 'pos_embed' in state:
            preload_pos = state['pos_embed']
            if not isinstance(preload_pos, torch.Tensor):
                preload_pos = torch.tensor(preload_pos)
            print("[train_segmentation] Found pos_embed in pretrained ckpt shape:", tuple(preload_pos.shape))
    model = SAMUSModel(cfg, preload_pos_embed=preload_pos.to(cfg.device) if preload_pos is not None else None).to(device)
    if cfg.pretrained_samus_ckpt and os.path.exists(cfg.pretrained_samus_ckpt):
        print("[train_segmentation] Loading pretrained SAMUS weights (flexible)...")
        load_ckpt_flexible(cfg.pretrained_samus_ckpt, model, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    bce = nn.BCELoss()
    best=1e9; ensure_dir(cfg.runs_dir)
    ckpt_path = os.path.join(cfg.runs_dir, "seg_pretrained_all.pth")
    for epoch in range(1, cfg.seg_pretrain_epochs+1):
        model.train(); running=0.0
        for xb, mask, *_ in tqdm(loader, desc=f"SegPre E{epoch}", leave=False):
            xb = xb.to(device); mask = mask.to(device)
            out_seg,_,_,_,_ = model(xb, prompt_s=None, prompt_j=None)
            loss = bce(out_seg, mask) + dice_loss(out_seg, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.item()) * xb.size(0)
        avg = running / len(ds)
        print(f"[SegTrain] Epoch {epoch} loss {avg:.4f}")
        if avg < best:
            best = avg; torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_path); print("Saved seg ckpt:", ckpt_path)
    return ckpt_path

def generate_pseudo_masks_for_label_only(cfg, device, seg_ckpt):
    ensure_dir(cfg.predicted_mask_root)
    model = SAMUSModel(cfg).to(device)
    if seg_ckpt and os.path.exists(seg_ckpt):
        print("[generate_pseudo_masks_for_label_only] loading seg ckpt into model (flexible)...")
        load_ckpt_flexible(seg_ckpt, model, device)
    model.eval()
    generated_pairs=[]
    with torch.no_grad():
        for ds, info in cfg.ds_label_only.items():
            for cls_key, cls in [("image_ben","benign"),("image_mal","malignant")]:
                img_dir = info.get(cls_key)
                if img_dir is None or not os.path.exists(img_dir): continue
                out_mask_dir = os.path.join(cfg.predicted_mask_root, ds, cls); ensure_dir(out_mask_dir)
                imgs = list_files(img_dir)
                for img_p in tqdm(imgs, desc=f"Pred masks {ds}/{cls}", leave=False):
                    img_rs = read_gray(img_p, size=cfg.img_size)
                    if img_rs is None: continue
                    x = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to(device)
                    out_seg,_,_,_,_ = model(x, prompt_s=None, prompt_j=None)
                    seg_np = (out_seg[0,0].cpu().numpy() * 255).astype(np.uint8)
                    mask_bin = (seg_np > cfg.seg_threshold).astype(np.uint8) * 255
                    base = os.path.basename(img_p); mask_path = os.path.join(out_mask_dir, base)
                    cv2.imwrite(mask_path, mask_bin); generated_pairs.append((img_p, mask_path, ds, cls))
    return generated_pairs

def joint_train_with_style(cfg, device, seg_ckpt=None):
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    paired_ds = PairedGazeDataset(cfg)
    paired_loader = DataLoader(paired_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    seg_pairs = gather_all_seg_pairs()
    for ds in cfg.ds_label_only.keys():
        for cls in ["benign","malignant"]:
            pred_dir = os.path.join(cfg.predicted_mask_root, ds, cls)
            if os.path.exists(pred_dir):
                for mask_p in list_files(pred_dir):
                    base = os.path.basename(mask_p)
                    cand1 = os.path.join(cfg.ds_label_only[ds].get("image_ben",""), base)
                    cand2 = os.path.join(cfg.ds_label_only[ds].get("image_mal",""), base)
                    img_p = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)
                    if img_p:
                        seg_pairs.append((img_p, mask_p, ds, cls))
    if len(seg_pairs) == 0:
        raise RuntimeError("No segmentation pairs available for joint training.")
    seg_dataset = CombinedSegDataset(seg_pairs, img_size=cfg.img_size)
    seg_loader = DataLoader(seg_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = SAMUSModel(cfg).to(device)
    if seg_ckpt is not None and os.path.exists(seg_ckpt):
        print("[joint_train_with_style] loading seg ckpt into joint model (flexible)...")
        load_ckpt_flexible(seg_ckpt, model, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    bce = nn.BCELoss(); l1 = nn.L1Loss()
    ckpt_out = os.path.join(cfg.runs_dir, "joint_style_full.pth")
    steps = max(len(paired_loader), len(seg_loader)); model.train()

    for epoch in range(1, cfg.joint_epochs+1):
        running = 0.0; it_p = iter(paired_loader); it_s = iter(seg_loader)
        for _ in tqdm(range(steps), desc=f"JointStyle E{epoch}", leave=False):
            loss_total = 0.0
            # ----- gaze + contrastive -----
            try:
                xb, den_j, den_s, _ = next(it_p)
                xb = xb.to(device); den_j = den_j.to(device); den_s = den_s.to(device)
                out_seg_p, out_s_p, out_j_p, zh, zl = model(xb, prompt_s=den_s, prompt_j=den_j)
                # gaze supervision
                loss_s = bce(out_s_p, den_s) + l1(out_s_p, den_s)
                loss_j = bce(out_j_p, den_j) + l1(out_j_p, den_j)
                loss_total += cfg.gaze_weight * (loss_s + loss_j)
                # entropy/style regularization
                H_s = spatial_entropy_map(out_s_p); H_j = spatial_entropy_map(out_j_p)
                loss_total += cfg.lambda_entropy_s * H_s
                loss_total += -cfg.lambda_entropy_j * H_j
                mask_j_top = topk_mask_from_density(out_j_p, topk_ratio=cfg.topk_ratio_for_containment).to(device)
                outside = out_s_p * (~mask_j_top).float(); loss_contain = outside.mean()
                loss_total += cfg.lambda_containment * loss_contain
                loss_total += cfg.sparsity_w * out_s_p.mean()
                # seniority supervised contrastive
                # 构造 2B 样本：B个 senior + B个 junior
                feats = torch.cat([zh, zl], dim=0)  # [2B, D]
                labels = torch.cat([torch.ones(zh.size(0), device=zh.device, dtype=torch.long),
                                    torch.zeros(zl.size(0), device=zl.device, dtype=torch.long)], dim=0)
                loss_con = supervised_contrastive_loss(feats, labels, temperature=cfg.con_temperature)
                loss_total += cfg.lambda_seniority * loss_con
            except StopIteration:
                pass
            # ----- segmentation -----
            try:
                xb2, mask2, *_ = next(it_s)
                xb2 = xb2.to(device); mask2 = mask2.to(device)
                out_seg_b, _, _, _, _ = model(xb2, prompt_s=None, prompt_j=None)
                loss_seg = bce(out_seg_b, mask2) + dice_loss(out_seg_b, mask2)
                loss_total += cfg.seg_weight * loss_seg
            except StopIteration:
                pass

            if loss_total == 0: continue
            opt.zero_grad(); loss_total.backward(); opt.step()
            running += float(loss_total.item())
        avg = running / steps
        print(f"[JointStyle] Epoch {epoch}/{cfg.joint_epochs} avg loss {avg:.4f}")
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "opt_state": opt.state_dict()}, ckpt_out)
    return ckpt_out

def infer_and_save_gaze(cfg, device, joint_ckpt):
    model = SAMUSModel(cfg).to(device)
    if joint_ckpt and os.path.exists(joint_ckpt):
        print("[infer_and_save_gaze] loading joint ckpt (flexible)...")
        load_ckpt_flexible(joint_ckpt, model, device)
    model.eval()
    ensure_dir(cfg.out_public_gaze)
    skip_ds = set(cfg.ds_mask_only.keys())
    produce = list(cfg.ds_mask_label.keys()) + list(cfg.ds_label_only.keys())
    produce = [d for d in produce if d not in skip_ds]
    with torch.no_grad():
        for ds in produce:
            print(f"[Infer] Dataset: {ds}")
            if ds in cfg.ds_mask_label:
                info = cfg.ds_mask_label[ds]
                for cls_key, cls in [("image_ben","benign"),("image_mal","malignant")]:
                    img_dir = info.get(cls_key)
                    if not img_dir or not os.path.exists(img_dir): continue
                    out_j_dir = os.path.join(cfg.out_public_gaze, ds, "junior", cls); ensure_dir(out_j_dir)
                    out_s_dir = os.path.join(cfg.out_public_gaze, ds, "senior", cls); ensure_dir(out_s_dir)
                    imgs = list_files(img_dir)
                    print(f"  class {cls}: images={len(imgs)}")
                    for img_p in tqdm(imgs, desc=f"{ds}/{cls}", leave=False):
                        orig_color = read_color(img_p, size=None); 
                        if orig_color is None: continue
                        Horig,Worig = orig_color.shape[:2]
                        img_rs = read_gray(img_p, size=cfg.img_size)
                        if img_rs is None: continue
                        x = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to(device)
                        out_seg, out_s, out_j, _, _ = model(x, prompt_s=None, prompt_j=None)
                        pred_s = norm01_np(out_s[0,0].cpu().numpy()); pred_j = norm01_np(out_j[0,0].cpu().numpy())
                        pred_s_full = cv2.resize((pred_s*255).astype(np.uint8),(Worig,Horig))
                        pred_j_full = cv2.resize((pred_j*255).astype(np.uint8),(Worig,Horig))
                        mask_root_b = info.get("mask_ben"); mask_root_m = info.get("mask_mal")
                        mask_try=None
                        for mask_root in [mask_root_b, mask_root_m]:
                            if not mask_root: continue
                            m = find_mask_for_image(img_p, mask_root)
                            if m:
                                mask_try = m; break
                        if mask_try is None:
                            pred_m = os.path.join(cfg.predicted_mask_root, ds, cls, os.path.basename(img_p))
                            if os.path.exists(pred_m): mask_try = pred_m
                        if mask_try and os.path.exists(mask_try):
                            mask_img = cv2.imread(mask_try, cv2.IMREAD_GRAYSCALE)
                            seg_bin = (cv2.resize(mask_img, (Worig,Horig), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)*255
                        else:
                            seg_bin = (pred_s_full > cfg.seg_threshold).astype(np.uint8)*255
                        k = max(1, int(cfg.seg_dilate_px * min(Worig,Horig) / 256)); kernel = np.ones((k,k), np.uint8)
                        seg_bin = cv2.dilate(seg_bin, kernel, iterations=1); seg_mask_float = (seg_bin>0).astype(np.float32)
                        masked_j = (pred_j_full.astype(np.float32)/255.0)*seg_mask_float
                        masked_s = (pred_s_full.astype(np.float32)/255.0)*seg_mask_float
                        if masked_j.sum() <= 1e-8: masked_j = pred_j_full.astype(np.float32)/255.0
                        if masked_s.sum() <= 1e-8: masked_s = pred_s_full.astype(np.float32)/255.0
                        coords_j = density_to_points_topk(masked_j, max_points=cfg.max_points_jun, topk_ratio=cfg.topk_ratio_jun, bg_ratio=cfg.bg_ratio_jun)
                        coords_s = density_to_points_topk(masked_s, max_points=cfg.max_points_sen, topk_ratio=cfg.topk_ratio_sen, bg_ratio=cfg.bg_ratio_sen)
                        if len(coords_j)==0: coords_j = density_to_points_smart(masked_j)
                        if len(coords_s)==0: coords_s = density_to_points_smart(masked_s)
                        radius = max(1, int(min(Worig,Horig)/256 * cfg.dot_radius_base))
                        over_j = overlay_red_dots(orig_color, coords_j, radius=radius)
                        over_s = overlay_red_dots(orig_color, coords_s, radius=radius)
                        base_name = os.path.basename(img_p)
                        cv2.imwrite(os.path.join(out_j_dir, base_name), over_j)
                        cv2.imwrite(os.path.join(out_s_dir, base_name), over_s)
            elif ds in cfg.ds_label_only:
                info = cfg.ds_label_only[ds]
                for cls_key, cls in [("image_ben","benign"),("image_mal","malignant")]:
                    img_dir = info.get(cls_key)
                    if not img_dir or not os.path.exists(img_dir): continue
                    out_j_dir = os.path.join(cfg.out_public_gaze, ds, "junior", cls); ensure_dir(out_j_dir)
                    out_s_dir = os.path.join(cfg.out_public_gaze, ds, "senior", cls); ensure_dir(out_s_dir)
                    imgs = list_files(img_dir)
                    print(f"  class {cls}: images={len(imgs)}")
                    for img_p in tqdm(imgs, desc=f"{ds}/{cls}", leave=False):
                        orig_color = read_color(img_p, size=None)
                        if orig_color is None: continue
                        Horig,Worig = orig_color.shape[:2]
                        img_rs = read_gray(img_p, size=cfg.img_size)
                        if img_rs is None: continue
                        x = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to(device)
                        out_seg, out_s, out_j, _, _ = model(x, prompt_s=None, prompt_j=None)
                        pred_s = norm01_np(out_s[0,0].cpu().numpy()); pred_j = norm01_np(out_j[0,0].cpu().numpy())
                        pred_s_full = cv2.resize((pred_s*255).astype(np.uint8),(Worig,Horig))
                        pred_j_full = cv2.resize((pred_j*255).astype(np.uint8),(Worig,Horig))
                        base = os.path.basename(img_p)
                        pred_mask = os.path.join(cfg.predicted_mask_root, ds, cls, base)
                        if os.path.exists(pred_mask):
                            mask_img = cv2.imread(pred_mask, cv2.IMREAD_GRAYSCALE)
                            seg_bin = (cv2.resize(mask_img, (Worig,Horig), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)*255
                        else:
                            seg_bin = (pred_s_full > cfg.seg_threshold).astype(np.uint8)*255
                        k = max(1, int(cfg.seg_dilate_px * min(Worig,Horig) / 256)); kernel = np.ones((k,k), np.uint8)
                        seg_bin = cv2.dilate(seg_bin, kernel, iterations=1); seg_mask_float = (seg_bin>0).astype(np.float32)
                        masked_j = (pred_j_full.astype(np.float32)/255.0)*seg_mask_float
                        masked_s = (pred_s_full.astype(np.float32)/255.0)*seg_mask_float
                        if masked_j.sum() <= 1e-8: masked_j = pred_j_full.astype(np.float32)/255.0
                        if masked_s.sum() <= 1e-8: masked_s = pred_s_full.astype(np.float32)/255.0
                        coords_j = density_to_points_topk(masked_j, max_points=cfg.max_points_jun, topk_ratio=cfg.topk_ratio_jun, bg_ratio=cfg.bg_ratio_jun)
                        coords_s = density_to_points_topk(masked_s, max_points=cfg.max_points_sen, topk_ratio=cfg.topk_ratio_sen, bg_ratio=cfg.bg_ratio_sen)
                        if len(coords_j)==0: coords_j = density_to_points_smart(masked_j)
                        if len(coords_s)==0: coords_s = density_to_points_smart(masked_s)
                        radius = max(1, int(min(Worig,Horig)/256 * cfg.dot_radius_base))
                        over_j = overlay_red_dots(orig_color, coords_j, radius=radius)
                        over_s = overlay_red_dots(orig_color, coords_s, radius=radius)
                        base_name = os.path.basename(img_p)
                        cv2.imwrite(os.path.join(out_j_dir, base_name), over_j)
                        cv2.imwrite(os.path.join(out_s_dir, base_name), over_s)
            else:
                continue
    print("[Infer] gaze generation finished.")

# -------------------------
# Main
# -------------------------
def main():
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    ensure_dir(cfg.runs_dir); ensure_dir(cfg.predicted_mask_root); ensure_dir(cfg.out_public_gaze)

    resolve_paths(cfg)
    print_dataset_stats_and_preview()

    seg_pairs = gather_all_seg_pairs()
    print(f"Total segmentation pairs discovered: {len(seg_pairs)}")
    if len(seg_pairs) == 0:
        print("No segmentation pairs found. Please check dataset paths and naming. Exiting.")
        return

    device = cfg.device
    seg_ckpt = train_segmentation(cfg, device, seg_pairs)

    pseudo_pairs = generate_pseudo_masks_for_label_only(cfg, device, seg_ckpt)
    print(f"Generated pseudo masks: {len(pseudo_pairs)}")

    joint_ckpt = joint_train_with_style(cfg, device, seg_ckpt=seg_ckpt)

    infer_and_save_gaze(cfg, device, joint_ckpt)
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
