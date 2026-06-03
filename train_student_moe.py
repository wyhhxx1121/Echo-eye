# moe_kd_three_backbones.py
# Student-MoE pipeline (teachers: multi single-model experts with gaze; students: experts -> Student-MoE)
# - Default student experts: resnet18, convnext_tiny, swin_tiny
# - Optional extra backbones provided for later comparisons
# - Teacher: trained per backbone x gaze_group (junior/senior). No teacher MoE; fusion = weighted average by val AUC.
# - Student: KD from teacher ensemble (differential KD); each student trained independently (skip if ckpt exists).
# - Student gating (MoE) trained on frozen student experts; final Student-MoE saved as single file for deployment.
# Requirements: torch, torchvision, pillow, numpy, sklearn, tqdm

import os, glob, random, math, copy
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def env_path(name, default):
    return os.environ.get(name, default)

# ---------------------------
# Config - set paths & hyperparams
# ---------------------------
IMAGE_ROOT = env_path("ECHO_EYE_PRIVATE_IMAGE_ROOT", "data/private/images")
PRIVATE_GAZE_JUNIOR = env_path("ECHO_EYE_PRIVATE_GAZE_JUNIOR", "data/private/gaze_junior")
PRIVATE_GAZE_SENIOR = env_path("ECHO_EYE_PRIVATE_GAZE_SENIOR", "data/private/gaze_senior")
PUBLIC_BASE = env_path("ECHO_EYE_PUBLIC_BASE", "data/public")

PUBLIC_IMAGE_ROOTS = {
    "BrEaST": os.path.join(PUBLIC_BASE, "BrEaST"),
    "BUS_UC": os.path.join(PUBLIC_BASE, "BUS_UC"),
    "BUSBRA": os.path.join(PUBLIC_BASE, "BUSBRA"),
    "QAMEBI": os.path.join(PUBLIC_BASE, "QAMEBI"),
    "UDIAT": os.path.join(PUBLIC_BASE, "UDIAT"),
    "BUSI": os.path.join(PUBLIC_BASE, "BUSI"),
    "GDPH_SYSUCC_i": os.path.join(PUBLIC_BASE, "GDPH&SYSUCC_i"),
    "us-dataset_i": os.path.join(PUBLIC_BASE, "us-dataset_i"),
    "US3M_i": os.path.join(PUBLIC_BASE, "US3M_i"),
    "BUS_COT": os.path.join(PUBLIC_BASE, "BUS_COT")
}
PUBLIC_GAZE_BASE = env_path("ECHO_EYE_PUBLIC_GAZE_BASE", "runs/public_gaze_sam")

# Default student expert backbones 
STUDENT_BACKBONES = ["resnet18", "convnext_tiny", "swin_tiny"]
# Optional set 
OPTIONAL_BACKBONES = ["densenet121", "resnet50", "efficientnet_b0"]

# teacher/backbone list for teachers
TEACHER_BACKBONES = STUDENT_BACKBONES  # modify if teachers should include optional backbones

BATCH_SIZE = 32
IMG_SIZE = 224
NUM_EPOCHS_TEACHER = 40    # smoke-test; increase for real runs
NUM_EPOCHS_STUDENT = 40
NUM_EPOCHS_GATE = 40

LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# KD / differential config
KD_T = 4.0
ALPHA_KD_MAX = 0.7
KD_RAMPUP_EPOCHS = 0.3
LAMBDA_KD_BASE = 1.0
FEAT_ATT_SCALE = 1.0
DIFF_KD_CONFIG = {
    0: {"LAMBDA_FEAT": 100.0 * FEAT_ATT_SCALE, "LAMBDA_ATT": 50.0 * FEAT_ATT_SCALE},  # junior emphasis on feat
    1: {"LAMBDA_FEAT": 50.0 * FEAT_ATT_SCALE, "LAMBDA_ATT": 100.0 * FEAT_ATT_SCALE}   # senior emphasis on att
}

# domain weighting
PRIVATE_SAMPLE_WEIGHT = 2.0
PUBLIC_SAMPLE_WEIGHT = 1.0

GAZE_ALPHA = 1.0
USE_PRETRAINED = True   # use pretrained weights
USE_CLASS_WEIGHT = True

CLASSES = ["benign", "malignant"]
NUM_CLASSES = len(CLASSES)

CHECKPOINT_DIR = env_path("ECHO_EYE_CHECKPOINT_DIR", "runs/checkpoints_moe_kd")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# num_workers for DataLoader (adjust per environment)
NUM_WORKERS = 4

# ---------------------------
# Utilities for collecting images & gaze
# ---------------------------
def collect_image_list(root, dataset_tag="private"):
    items = []
    for label, cls in enumerate(CLASSES):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif"):
            for fp in glob.glob(os.path.join(cls_dir, ext)):
                fname = os.path.basename(fp)
                items.append((fp, label, dataset_tag, cls, fname))
    return items

def find_gaze_for_image(dataset_tag, class_name, filename, teacher_group):
    """teacher_group: 0 junior, 1 senior. Search private first (if dataset_tag private), else public gaze."""
    if dataset_tag == "private":
        gaze_root = PRIVATE_GAZE_JUNIOR if teacher_group==0 else PRIVATE_GAZE_SENIOR
        if os.path.isdir(gaze_root):
            candidate = os.path.join(gaze_root, class_name, filename)
            if os.path.exists(candidate): return candidate
            matches = glob.glob(os.path.join(gaze_root, class_name, os.path.splitext(filename)[0] + "*"))
            if matches: return matches[0]
    else:
        grp = "junior" if teacher_group==0 else "senior"
        base = os.path.join(PUBLIC_GAZE_BASE, dataset_tag, grp, class_name)
        if os.path.isdir(base):
            candidate = os.path.join(base, filename)
            if os.path.exists(candidate): return candidate
            matches = glob.glob(os.path.join(base, os.path.splitext(filename)[0] + "*"))
            if matches: return matches[0]
    # fallback scan all public gaze datasets
    grp = "junior" if teacher_group==0 else "senior"
    for ds_name in PUBLIC_IMAGE_ROOTS.keys():
        base = os.path.join(PUBLIC_GAZE_BASE, ds_name, grp, class_name)
        if os.path.isdir(base):
            matches = glob.glob(os.path.join(base, filename))
            if matches: return matches[0]
            matches = glob.glob(os.path.join(base, os.path.splitext(filename)[0] + "*"))
            if matches: return matches[0]
    return None

def is_private_path(p):
    try:
        return os.path.abspath(p).startswith(os.path.abspath(IMAGE_ROOT))
    except Exception:
        return False

# ---------------------------
# Dataset classes
# ---------------------------
class MasterDataset(Dataset):
    def __init__(self, items, gaze_groups=[0,1], transform_img=None, transform_gaze=None):
        self.items = items
        self.gaze_groups = gaze_groups
        self.transform_img = transform_img
        self.transform_gaze = transform_gaze
        self.gaze_map = []
        for (img_path, label, dataset_tag, class_name, fname) in items:
            per = []
            for g in gaze_groups:
                gp = find_gaze_for_image(dataset_tag, class_name, fname, g)
                per.append(gp)
            self.gaze_map.append(per)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        img_path, label, dataset_tag, class_name, fname = self.items[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("Open image fail:", img_path, e)
            return None
        img_t = self.transform_img(img) if self.transform_img else transforms.ToTensor()(img)
        per_gaze_paths = self.gaze_map[idx]
        return img_t, per_gaze_paths, label, img_path

class TeacherDatasetFromSamples(Dataset):
    def __init__(self, items, gaze_group, transform_img=None, transform_gaze=None):
        self.samples = []
        self.transform_img = transform_img
        self.transform_gaze = transform_gaze
        for (img_path, label, dataset_tag, class_name, fname) in items:
            gp = find_gaze_for_image(dataset_tag, class_name, fname, gaze_group)
            if gp is not None and os.path.exists(gp) and os.path.exists(img_path):
                self.samples.append((img_path, gp, label))
        print(f"[TeacherDataset] group {gaze_group} samples: {len(self.samples)}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, gaze_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            gaze_img = Image.open(gaze_path).convert("RGB")
        except Exception as e:
            print("Open teacher sample fail:", img_path, gaze_path, e)
            return None
        gaze_np = np.array(gaze_img)[:,:,0].astype(np.float32)
        if gaze_np.max() == gaze_np.min():
            gaze_np = np.zeros_like(gaze_np)
        else:
            gaze_np = (gaze_np - gaze_np.min()) / (gaze_np.max() - gaze_np.min() + 1e-8)
        gaze_pil = Image.fromarray((gaze_np * 255).astype(np.uint8))
        img_t = self.transform_img(img) if self.transform_img else transforms.ToTensor()(img)
        gaze_t = self.transform_gaze(gaze_pil) if self.transform_gaze else transforms.ToTensor()(gaze_pil)
        if gaze_t.dim() == 3:
            gaze_t = gaze_t[0:1,:,:]
        return img_t, gaze_t, label, img_path

# ---------------------------
# Backbone helper & model defs
# ---------------------------
def get_backbone_truncated(backbone_name, pretrained=True):
    """Return (features_module, avgpool_module, out_dim). Tries multiple torchvision names for swin/efficientnet."""
    backbone_name = backbone_name.lower()
    if backbone_name.startswith("resnet"):
        if backbone_name == "resnet18":
            base = models.resnet18(pretrained=pretrained)
        elif backbone_name == "resnet50":
            base = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(backbone_name)
        features = nn.Sequential(*list(base.children())[:-2])
        avgpool = base.avgpool
        out_dim = base.fc.in_features
        return features, avgpool, out_dim

    if backbone_name == "densenet121":
        base = models.densenet121(pretrained=pretrained)
        features = base.features
        avgpool = nn.AdaptiveAvgPool2d((1,1))
        try:
            out_dim = base.classifier.in_features
        except:
            out_dim = 1024
        return features, avgpool, out_dim

    if backbone_name == "convnext_tiny":
        base = models.convnext_tiny(pretrained=pretrained)
        features = base.features
        avgpool = nn.AdaptiveAvgPool2d((1,1))
        try:
            out_dim = base.classifier[2].in_features
        except Exception:
            out_dim = base.classifier[-1].in_features
        return features, avgpool, out_dim

    if backbone_name.startswith("swin"):
        # try several torchvision variants
        try:
            base = models.swin_t(pretrained=pretrained)
            features = nn.Sequential(*list(base.children())[:-2])
            avgpool = nn.AdaptiveAvgPool2d((1,1))
            out_dim = base.head.in_features if hasattr(base, 'head') else 768
            return features, avgpool, out_dim
        except Exception:
            try:
                base = models.swin_tiny_patch4_window7_224(pretrained=pretrained)
                features = nn.Sequential(*list(base.children())[:-2])
                avgpool = nn.AdaptiveAvgPool2d((1,1))
                out_dim = base.head.in_features if hasattr(base, 'head') else 768
                return features, avgpool, out_dim
            except Exception as e:
                raise ValueError("Swin not available in this torchvision build: " + str(e))

    if backbone_name.startswith("efficientnet"):
        # try efficientnet_b0
        try:
            base = models.efficientnet_b0(pretrained=pretrained)
            features = nn.Sequential(*list(base.features))
            avgpool = nn.AdaptiveAvgPool2d((1,1))
            try:
                out_dim = base.classifier[1].in_features
            except:
                out_dim = 1280
            return features, avgpool, out_dim
        except Exception as e:
            raise ValueError("EfficientNet not available: " + str(e))

    raise ValueError("Unsupported backbone: " + backbone_name)

class TeacherNetBackbone(nn.Module):
    def __init__(self, backbone_name="resnet18", gaze_alpha=1.0, num_classes=2, pretrained=True):
        super().__init__()
        self.features, self.avgpool, self.outdim = get_backbone_truncated(backbone_name, pretrained=pretrained)
        self.fc = nn.Linear(self.outdim, num_classes)
        self.gaze_alpha = gaze_alpha
    def forward(self, x, gaze=None):
        feat = self.features(x)
        att_map = None
        if gaze is not None:
            att_map = F.interpolate(gaze, size=feat.shape[2:], mode='bilinear', align_corners=False)
            amin = att_map.amin(dim=(-2,-1), keepdim=True)
            amax = att_map.amax(dim=(-2,-1), keepdim=True)
            att_map = (att_map - amin) / (amax - amin + 1e-8)
            feat = feat * (1.0 + self.gaze_alpha * att_map)
        pooled = torch.flatten(self.avgpool(feat), 1)
        logits = self.fc(pooled)
        return logits, feat, pooled, att_map

class StudentNetBackbone(nn.Module):
    def __init__(self, backbone_name="resnet18", num_classes=2, pretrained=True):
        super().__init__()
        self.features, self.avgpool, self.outdim = get_backbone_truncated(backbone_name, pretrained=pretrained)
        self.fc = nn.Linear(self.outdim, num_classes)
        C = self.outdim
        h = max(1, C//2)
        self.att_pred = nn.Sequential(
            nn.Conv2d(C, h, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        feat = self.features(x)
        att = self.att_pred(feat)
        pooled = torch.flatten(self.avgpool(feat), 1)
        logits = self.fc(pooled)
        return logits, feat, pooled, att

# ---------------------------
# Losses, collate, sched
# ---------------------------
def distillation_loss(student_logits, teacher_logits, T):
    s_logprob = F.log_softmax(student_logits / T, dim=1)
    t_prob = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s_logprob, t_prob, reduction='batchmean') * (T*T)

def kd_alpha_schedule(epoch, total_epochs):
    ramp_epochs = max(1, int(total_epochs * KD_RAMPUP_EPOCHS))
    if epoch <= ramp_epochs:
        return ALPHA_KD_MAX * (epoch / ramp_epochs)
    else:
        return ALPHA_KD_MAX

from torch.utils.data.dataloader import default_collate
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch)==0: return None
    first = batch[0]
    if not isinstance(first, (tuple,list)) or len(first)<4:
        return default_collate(batch)
    second = first[1]
    if isinstance(second, list):
        imgs = torch.stack([b[0] for b in batch], dim=0)
        per_gaze_paths = [b[1] for b in batch]
        labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
        img_paths = [b[3] for b in batch]
        return imgs, per_gaze_paths, labels, img_paths
    else:
        return default_collate(batch)

# ---------------------------
# Prepare data (private + public)
# ---------------------------
print("Collecting data...")
private_items = collect_image_list(IMAGE_ROOT, dataset_tag="private")
if len(private_items)==0:
    raise RuntimeError("No private images found under IMAGE_ROOT")
# stratified split private -> train_private / test_private
idxs = list(range(len(private_items)))
labs = [lab for (_,lab,_,_,_) in private_items]
train_idx_priv, test_idx_priv = train_test_split(idxs, test_size=0.2, stratify=labs, random_state=SEED)
train_private = [private_items[i] for i in train_idx_priv]
test_private = [private_items[i] for i in test_idx_priv]
print(f"Private train {len(train_private)}  Private test {len(test_private)}")

# collect public image lists
public_items = []
for ds_name, ds_root in PUBLIC_IMAGE_ROOTS.items():
    if not os.path.isdir(ds_root):
        print("[WARN] skip missing public root:", ds_root)
        continue
    its = collect_image_list(ds_root, dataset_tag=ds_name)
    print(f"Public {ds_name}: {len(its)}")
    public_items.extend(its)
print("Total public images:", len(public_items))

# training items: train_private + public
train_items = list(train_private) + list(public_items)
print("Total train items:", len(train_items))

# transforms
transform_img_train = transforms.Compose([
    transforms.RandomResizedCrop((IMG_SIZE,IMG_SIZE), scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
transform_img_val = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
transform_gaze = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
])

# teacher loaders (train uses val-style transforms)
teacher_train_loaders = []
teacher_val_loaders = []
for grp in [0,1]:
    tr_ds = TeacherDatasetFromSamples(train_items, gaze_group=grp, transform_img=transform_img_val, transform_gaze=transform_gaze)
    val_ds = TeacherDatasetFromSamples(test_private, gaze_group=grp, transform_img=transform_img_val, transform_gaze=transform_gaze)
    teacher_train_loaders.append(DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn))
    teacher_val_loaders.append(DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn))

# master datasets for student (train on train_items including public)
master_train_ds = MasterDataset(train_items, gaze_groups=[0,1], transform_img=transform_img_train, transform_gaze=transform_gaze)
master_test_ds  = MasterDataset(test_private, gaze_groups=[0,1], transform_img=transform_img_val, transform_gaze=transform_gaze)
train_loader_student = DataLoader(master_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
test_loader_student = DataLoader(master_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

# ---------------------------
# Training functions: teacher & student
# ---------------------------
def train_teacher_one_epoch(model, loader, optimizer, device):
    model.train()
    losses=[]; y_true=[]; y_pred=[]
    for batch in tqdm(loader, desc="Teacher train", leave=False):
        if batch is None: continue
        imgs, gazes, labels, _ = batch
        imgs=imgs.to(device); gazes=gazes.to(device); labels=labels.to(device)
        logits, feat, pooled, att = model(imgs, gaze=gazes)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_true.extend(labels.cpu().numpy().tolist()); y_pred.extend(preds)
    acc = accuracy_score(y_true,y_pred) if len(y_true)>0 else 0.0
    return np.mean(losses) if losses else 0.0, acc

def eval_teacher(model, loader, device):
    model.eval()
    y_true=[]; y_pred=[]; y_score=[]
    with torch.no_grad():
        for batch in tqdm(loader, desc="Teacher val", leave=False):
            if batch is None: continue
            imgs, gazes, labels, _ = batch
            imgs=imgs.to(device); gazes=gazes.to(device); labels=labels.to(device)
            logits, feat, pooled, att = model(imgs, gaze=gazes)
            probs = F.softmax(logits, dim=1)[:,1].cpu().numpy().tolist()
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_true.extend(labels.cpu().numpy().tolist()); y_pred.extend(preds); y_score.extend(probs)
    acc = accuracy_score(y_true,y_pred) if len(y_true)>0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_score) if len(set(y_true))>1 else float('nan')
    except:
        auc = float('nan')
    return {"acc":acc, "auc":auc, "f1": f1_score(y_true,y_pred, zero_division=0) if len(y_true)>0 else 0.0}

def train_student_kd_one_epoch(student, teachers, teacher_perf_weights, master_loader, opt_student, device, scheduler=None,
                               temperature=KD_T, lambda_kd_base=LAMBDA_KD_BASE, epoch_idx=1, total_epochs=1):
    student.train()
    for t in teachers: t.eval()
    losses=[]; y_true=[]; y_pred=[]
    alpha = kd_alpha_schedule(epoch_idx, total_epochs)
    loop = tqdm(master_loader, desc=f"Student KD epoch {epoch_idx}", leave=False)
    for batch in loop:
        if batch is None: continue
        imgs, per_gaze_paths_list, labels, img_paths = batch
        imgs=imgs.to(device); labels=labels.to(device)
        B = imgs.size(0)

        # collect teacher outputs (ensemble by teacher_perf_weights)
        teacher_logits_all = []
        teacher_pooled_all = []
        teacher_att_all = []
        for ti, teacher in enumerate(teachers):
            # for teacher i (gaze-group aware), build gaze batch where available
            gaze_paths = [pg[ti%2] if len(pg)>ti%2 else None for pg in per_gaze_paths_list]  # teacher index ordering matches gaze group (we assume teachers list built as [g0_of_backbone1,g1_of_backbone1,...])
            idxs = [i for i,p in enumerate(gaze_paths) if p is not None]
            if len(idxs)==0:
                # forward all imgs w/o gaze
                with torch.no_grad():
                    l_all, f_all, p_all, a_all = teacher(imgs, gaze=None)
                teacher_logits_all.append(l_all)
                teacher_pooled_all.append(p_all)
                teacher_att_all.append(a_all)
                continue
            # build gaze tensors for present indices
            gaze_tensors = []
            for i in idxs:
                gp = gaze_paths[i]
                try:
                    g_img = Image.open(gp).convert("RGB")
                    g_np = np.array(g_img)[:,:,0].astype(np.float32)
                    if g_np.max()==g_np.min():
                        g_np = np.zeros_like(g_np)
                    else:
                        g_np = (g_np - g_np.min())/(g_np.max()-g_np.min()+1e-8)
                    g_pil = Image.fromarray((g_np*255).astype(np.uint8))
                    g_t = transform_gaze(g_pil)
                except Exception:
                    g_t = torch.zeros(1, IMG_SIZE, IMG_SIZE)
                gaze_tensors.append(g_t)
            gaze_batch = torch.stack(gaze_tensors, dim=0).to(device)
            sub_imgs = imgs[idxs]
            with torch.no_grad():
                t_logits_sub, t_feat_sub, t_pooled_sub, t_att_sub = teacher(sub_imgs, gaze=gaze_batch)
            logits_full = torch.zeros((B, NUM_CLASSES), device=device)
            pooled_full = torch.zeros((B, t_pooled_sub.size(1)), device=device)
            att_full = None
            if t_att_sub is not None:
                att_full = torch.zeros((B, t_att_sub.size(1), t_att_sub.size(2), t_att_sub.size(3)), device=device)
            for k, orig in enumerate(idxs):
                logits_full[orig] = t_logits_sub[k]
                pooled_full[orig] = t_pooled_sub[k]
                if att_full is not None:
                    att_full[orig] = t_att_sub[k]
            teacher_logits_all.append(logits_full)
            teacher_pooled_all.append(pooled_full)
            teacher_att_all.append(att_full)

        # ensemble teacher logits by teacher_perf_weights
        ensemble_logits = torch.zeros((B, NUM_CLASSES), device=device)
        denom = 0.0
        for ti, tlog in enumerate(teacher_logits_all):
            if tlog is None: continue
            w = teacher_perf_weights[ti] if teacher_perf_weights is not None else 1.0
            ensemble_logits += w * tlog
            denom += w
        if denom > 0:
            ensemble_logits = ensemble_logits / denom

        # student forward
        s_logits, s_feat, s_pooled, s_att = student(imgs)

        # kd loss (logit-level)
        kd_loss = distillation_loss(s_logits, ensemble_logits.detach(), temperature)

        # feat/att losses (differential)
        feat_loss_total = torch.tensor(0.0, device=device)
        att_loss_total = torch.tensor(0.0, device=device)
        for ti in range(len(teacher_pooled_all)):
            pooled_full = teacher_pooled_all[ti]
            att_full = teacher_att_all[ti]
            if pooled_full is None: continue
            minc = min(pooled_full.size(1), s_pooled.size(1))
            feat_loss_i = F.mse_loss(s_pooled[:, :minc], pooled_full[:, :minc])
            if att_full is not None and s_att is not None:
                try:
                    att_resized = F.interpolate(att_full, size=(s_att.size(2), s_att.size(3)), mode='bilinear', align_corners=False)
                    att_loss_i = F.mse_loss(s_att, att_resized)
                except Exception:
                    att_loss_i = torch.tensor(0.0, device=device)
            else:
                att_loss_i = torch.tensor(0.0, device=device)
            cfg = DIFF_KD_CONFIG.get(ti%2, {"LAMBDA_FEAT":1.0, "LAMBDA_ATT":1.0})
            # weight by teacher performance weight magnitude (so stronger teachers contribute more)
            gate_w = teacher_perf_weights[ti] if teacher_perf_weights is not None else 1.0
            feat_loss_total += gate_w * cfg["LAMBDA_FEAT"] * feat_loss_i
            att_loss_total += gate_w * cfg["LAMBDA_ATT"] * att_loss_i

        # classification loss with domain sample weighting
        sample_weights = [PRIVATE_SAMPLE_WEIGHT if is_private_path(p) else PUBLIC_SAMPLE_WEIGHT for p in img_paths]
        sw = torch.tensor(sample_weights, dtype=torch.float32, device=device)
        per_example_loss = F.cross_entropy(s_logits, labels, reduction='none')
        cls_loss = (per_example_loss * sw).mean()

        loss = (1.0 - kd_alpha_schedule(epoch_idx, total_epochs)) * cls_loss + kd_alpha_schedule(epoch_idx, total_epochs) * (LAMBDA_KD_BASE * kd_loss) + feat_loss_total + att_loss_total

        opt_student.zero_grad(); loss.backward(); opt_student.step()
        if scheduler is not None:
            try: scheduler.step()
            except: pass

        losses.append(loss.item())
        preds = torch.argmax(s_logits, dim=1).cpu().numpy().tolist()
        y_true.extend(labels.cpu().numpy().tolist()); y_pred.extend(preds)
        loop.set_postfix({'loss': np.mean(losses) if losses else 0.0})

    acc = accuracy_score(y_true,y_pred) if len(y_true)>0 else 0.0
    return np.mean(losses) if losses else 0.0, acc

def eval_student_on_loader(model, loader, device):
    model.eval()
    y_true=[]; y_pred=[]; y_score=[]
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            if batch is None: continue
            imgs, per_gaze_paths, labels, _ = batch
            imgs=imgs.to(device); labels=labels.to(device)
            logits, feat, pooled, att = model(imgs)
            probs = F.softmax(logits, dim=1)[:,1].cpu().numpy().tolist()
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_true.extend(labels.cpu().numpy().tolist()); y_pred.extend(preds); y_score.extend(probs)
    y_true=np.array(y_true); y_pred=np.array(y_pred); y_score=np.array(y_score)
    acc = accuracy_score(y_true,y_pred) if len(y_true)>0 else 0.0
    prec = precision_score(y_true,y_pred, zero_division=0) if len(y_true)>0 else 0.0
    rec = recall_score(y_true,y_pred, zero_division=0) if len(y_true)>0 else 0.0
    f1 = f1_score(y_true,y_pred, zero_division=0) if len(y_true)>0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = float('nan')
    cm = confusion_matrix(y_true,y_pred) if len(y_true)>0 else None
    return {"acc":acc,"prec":prec,"rec":rec,"f1":f1,"auc":auc,"cm":cm,"y_true":y_true,"y_pred":y_pred,"y_score":y_score}

# ---------------------------
# Main: train/load teachers & students per backbone
# ---------------------------
summary_results = []

# optional class weight (not used directly because we use per-example weighting)
if USE_CLASS_WEIGHT:
    counts = defaultdict(int)
    for _p, lab, _, _, _ in train_items:
        counts[lab] += 1
    totalc = sum(counts.values()) if sum(counts.values())>0 else 1
    class_weights = [0.0]*NUM_CLASSES
    for k in counts:
        class_weights[k] = totalc / (counts[k]+1e-8)
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print("class weights:", class_weights)
else:
    class_weight_tensor = None

# Build teachers list (teacher experts per backbone x gaze_group)
teacher_experts = []
teacher_meta = []  # (backbone, gaze_group)
teacher_ckpt_paths = []
print("\n=== Prepare teacher experts ===")
for backbone in TEACHER_BACKBONES:
    for g in [0,1]:
        ckpt_name = f"teacher_{backbone}_g{g}.pth"
        ckpt = os.path.join(CHECKPOINT_DIR, ckpt_name)
        model = TeacherNetBackbone(backbone_name=backbone, gaze_alpha=GAZE_ALPHA, num_classes=NUM_CLASSES, pretrained=USE_PRETRAINED).to(DEVICE)
        # if checkpoint exists, load; otherwise train (if training data exists)
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print(f"[LOAD] Teacher loaded from {ckpt}")
        else:
            # train if paired samples exist
            loader_train = teacher_train_loaders[g]
            loader_val = teacher_val_loaders[g]
            if len(loader_train.dataset) == 0:
                print(f"[WARN] No paired train samples for teacher {backbone} g{g}. Model remains randomly initialized.")
            else:
                print(f"[TRAIN] Teacher {backbone} g{g} will train; ckpt -> {ckpt}")
                opt = torch.optim.Adam(model.parameters(), lr=LR)
                best_auc = -1.0; best_epoch = 0
                for e in range(1, NUM_EPOCHS_TEACHER+1):
                    loss_tr, acc_tr = train_teacher_one_epoch(model, teacher_train_loaders[g], opt, DEVICE)
                    metrics = eval_teacher(model, teacher_val_loaders[g], DEVICE) if len(teacher_val_loaders[g].dataset)>0 else {"auc":float('nan')}
                    auc_val = metrics.get("auc", float('nan'))
                    print(f"[Teacher {backbone} g{g}] epoch {e}/{NUM_EPOCHS_TEACHER} loss {loss_tr:.4f} acc {acc_tr:.4f} val_auc {auc_val:.4f}")
                    if not math.isnan(auc_val) and auc_val > best_auc:
                        best_auc = auc_val; best_epoch = e
                        torch.save(model.state_dict(), ckpt)
                print(f"Teacher {backbone} g{g} done. best val AUC {best_auc:.4f} (epoch {best_epoch})")
        # freeze and append
        model.eval()
        for p in model.parameters(): p.requires_grad = False
        teacher_experts.append(model)
        teacher_meta.append((backbone, g))
        teacher_ckpt_paths.append(ckpt)

# compute teacher performance weights by evaluating on private val (teacher_val_loaders)
print("\nEvaluating teachers on private val subset to compute fusion weights...")
teacher_val_aucs = []
for idx, model in enumerate(teacher_experts):
    g = teacher_meta[idx][1]
    metrics = eval_teacher(model, teacher_val_loaders[g], DEVICE) if len(teacher_val_loaders[g].dataset)>0 else {"auc":float('nan')}
    a = metrics.get("auc", float('nan'))
    if math.isnan(a): a = 0.0
    teacher_val_aucs.append(max(0.0, a))
    print(f"Teacher {teacher_meta[idx]} val auc: {a:.4f}")
s = sum(teacher_val_aucs)
if s == 0:
    teacher_perf_weights = [1.0/len(teacher_val_aucs)] * len(teacher_val_aucs)
else:
    teacher_perf_weights = [a/s for a in teacher_val_aucs]
print("Teacher fusion weights (by val AUC):", teacher_perf_weights)

# ---------------------------
# Train / load student experts
# ---------------------------
student_models = []
student_ckpts = {}
print("\n=== Train or load student experts ===")
for backbone in STUDENT_BACKBONES:
    print("\n" + "="*60)
    print("Student expert backbone:", backbone)
    print("="*60)
    student = StudentNetBackbone(backbone_name=backbone, num_classes=NUM_CLASSES, pretrained=USE_PRETRAINED).to(DEVICE)
    student_ckpt = os.path.join(CHECKPOINT_DIR, f"{backbone}_student_kd_best.pth")
    student_ckpts[backbone] = student_ckpt

    if os.path.exists(student_ckpt):
        student.load_state_dict(torch.load(student_ckpt, map_location=DEVICE))
        print(f"[LOAD] Student expert loaded: {student_ckpt}")
    else:
        # train student via KD from teacher ensemble
        print(f"[TRAIN] Student {backbone} will be trained via KD -> ckpt: {student_ckpt}")
        opt_student = torch.optim.Adam(student.parameters(), lr=LR)
        steps = max(1, len(train_loader_student))
        try:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(opt_student, max_lr=LR*6, total_steps=NUM_EPOCHS_STUDENT * steps)
        except Exception:
            scheduler = None
        best_auc = -1.0; best_epoch = 0
        for e in range(1, NUM_EPOCHS_STUDENT+1):
            loss_tr, acc_tr = train_student_kd_one_epoch(student, teacher_experts, teacher_perf_weights, train_loader_student, opt_student, DEVICE, scheduler=scheduler, temperature=KD_T, lambda_kd_base=LAMBDA_KD_BASE, epoch_idx=e, total_epochs=NUM_EPOCHS_STUDENT)
            metrics = eval_student_on_loader(student, test_loader_student, DEVICE)
            print(f"[Student {backbone}] epoch {e}/{NUM_EPOCHS_STUDENT} train_loss {loss_tr:.4f} train_acc {acc_tr:.4f} val_auc {metrics['auc']:.4f} val_f1 {metrics['f1']:.4f}")
            if not math.isnan(metrics['auc']) and metrics['auc'] > best_auc:
                best_auc = metrics['auc']; best_epoch = e
                torch.save(student.state_dict(), student_ckpt)
        print(f"Student {backbone} training done. best val AUC {best_auc:.4f} epoch {best_epoch}")

    # final eval student
    student.load_state_dict(torch.load(student_ckpt, map_location=DEVICE))
    metrics = eval_student_on_loader(student, test_loader_student, DEVICE)
    print(f"Final Student ({backbone}) on private test: Acc {metrics['acc']:.4f} Prec {metrics['prec']:.4f} Rec {metrics['rec']:.4f} F1 {metrics['f1']:.4f} AUC {metrics['auc']:.4f}")
    print("Confusion matrix:\n", metrics['cm'])
    print("Classification report:\n", classification_report(metrics['y_true'], metrics['y_pred'], target_names=CLASSES, zero_division=0))

    student.eval()
    for p in student.parameters(): p.requires_grad = False
    student_models.append((backbone, student))

# ---------------------------
# Train student gating (Student-MoE)
# ---------------------------
print("\n=== Train or load Student-Gating (Student-MoE) ===")
pooled_dims_students = [m.outdim for _, m in student_models]
class GatingNet(nn.Module):
    def __init__(self, pooled_dims, hidden=256):
        super().__init__()
        self.in_dim = sum(pooled_dims)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, len(pooled_dims))
        )
    def forward(self, pooled_list):
        x = torch.cat(pooled_list, dim=1)
        logits = self.mlp(x)
        weights = F.softmax(logits, dim=1)
        return weights

student_gating = GatingNet(pooled_dims=pooled_dims_students, hidden=256).to(DEVICE)
student_gating_ckpt = os.path.join(CHECKPOINT_DIR, "student_gating_best.pth")

# prepare expert_models list for gating (backbone name, model)
expert_models = student_models  # same structure

def train_student_gating_epoch(gating, experts, train_loader, opt, device, entropy_reg=0.01):
    gating.train()
    losses=[]; y_true=[]; y_pred=[]
    loop = tqdm(train_loader, desc="Student gating train", leave=False)
    for batch in loop:
        if batch is None: continue
        imgs, per_gaze_paths, labels, img_paths = batch
        imgs=imgs.to(device); labels=labels.to(device)
        pooled_list=[]; logits_list=[]
        for _, m in experts:
            with torch.no_grad():
                l, f, p, a = m(imgs)
            pooled_list.append(p)
            logits_list.append(l)
        weights = gating(pooled_list)
        stacked_logits = torch.stack(logits_list, dim=1)
        ensemble_logits = (weights.unsqueeze(2) * stacked_logits).sum(dim=1)
        sample_weights = torch.tensor([PRIVATE_SAMPLE_WEIGHT if is_private_path(p) else PUBLIC_SAMPLE_WEIGHT for p in img_paths], dtype=torch.float32, device=device)
        per_example_loss = F.cross_entropy(ensemble_logits, labels, reduction='none')
        cls_loss = (per_example_loss * sample_weights).mean()
        gate_entropy = -(weights * (torch.log(weights + 1e-12))).sum(dim=1).mean()
        loss = cls_loss + entropy_reg * gate_entropy
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        preds = torch.argmax(ensemble_logits, dim=1).cpu().numpy().tolist()
        y_true.extend(labels.cpu().numpy().tolist()); y_pred.extend(preds)
        loop.set_postfix({'loss': np.mean(losses) if losses else 0.0})
    acc = accuracy_score(y_true,y_pred) if len(y_true)>0 else 0.0
    return np.mean(losses) if losses else 0.0, acc

def eval_student_moe(gating, experts, loader, device):
    gating.eval()
    y_true=[]; y_pred=[]; y_score=[]
    with torch.no_grad():
        for batch in tqdm(loader, desc="Student-MoE Eval", leave=False):
            if batch is None: continue
            imgs, per_gaze_paths, labels, _ = batch
            imgs=imgs.to(device); labels=labels.to(device)
            pooled_list=[]; logits_list=[]
            for _, m in experts:
                l, f, p, a = m(imgs)
                pooled_list.append(p); logits_list.append(l)
            weights = gating(pooled_list)
            stacked_logits = torch.stack(logits_list, dim=1)
            ensemble_logits = (weights.unsqueeze(2) * stacked_logits).sum(dim=1)
            probs = F.softmax(ensemble_logits, dim=1)[:,1].cpu().numpy().tolist()
            preds = torch.argmax(ensemble_logits, dim=1).cpu().numpy().tolist()
            y_true.extend(labels.cpu().numpy().tolist()); y_pred.extend(preds); y_score.extend(probs)
    try:
        auc = roc_auc_score(y_true, y_score) if len(set(y_true))>1 else float('nan')
    except:
        auc = float('nan')
    return {"acc": accuracy_score(y_true,y_pred) if len(y_true)>0 else 0.0,
            "auc": auc,
            "f1": f1_score(y_true,y_pred, zero_division=0) if len(y_true)>0 else 0.0,
            "y_true": np.array(y_true), "y_pred": np.array(y_pred), "y_score": np.array(y_score),
            "cm": confusion_matrix(y_true,y_pred) if len(y_true)>0 else None}

# If gating ckpt exists -> load; else train
if os.path.exists(student_gating_ckpt):
    student_gating.load_state_dict(torch.load(student_gating_ckpt, map_location=DEVICE))
    print(f"[LOAD] Student gating loaded: {student_gating_ckpt}")
else:
    print("[TRAIN] Student gating will be trained (Student-MoE).")
    opt_gate = torch.optim.Adam(student_gating.parameters(), lr=LR)
    best_sg_auc = -1.0; best_sg_epoch = 0
    for e in range(1, NUM_EPOCHS_GATE+1):
        loss_g, acc_g = train_student_gating_epoch(student_gating, expert_models, train_loader_student, opt_gate, DEVICE)
        metrics = eval_student_moe(student_gating, expert_models, test_loader_student, DEVICE)
        print(f"[Student-Gating] epoch {e}/{NUM_EPOCHS_GATE} loss {loss_g:.4f} train_acc {acc_g:.4f} val_auc {metrics['auc']:.4f}")
        if not math.isnan(metrics['auc']) and metrics['auc'] > best_sg_auc:
            best_sg_auc = metrics['auc']; best_sg_epoch = e
            torch.save(student_gating.state_dict(), student_gating_ckpt)
    print(f"Student-Gating best AUC: {best_sg_auc:.4f} (epoch {best_sg_epoch})")
    if os.path.exists(student_gating_ckpt):
        student_gating.load_state_dict(torch.load(student_gating_ckpt, map_location=DEVICE))

# Evaluate Student-MoE
moe_metrics = eval_student_moe(student_gating, expert_models, test_loader_student, DEVICE)
print("\nStudent-MoE on private test:", moe_metrics)
print("Classification report:\n", classification_report(moe_metrics['y_true'], moe_metrics['y_pred'], target_names=CLASSES, zero_division=0))

# ---------------------------
# Export Student-MoE as single deployable file
# ---------------------------
print("\nSaving Student-MoE package...")
student_moe_pkg = {
    "expert_backbones": [b for b,_ in expert_models],
    "experts_state": {},
    "gating_state": student_gating.state_dict(),
    "num_classes": NUM_CLASSES,
    "img_size": IMG_SIZE,
}
for b, m in expert_models:
    # load checkpoint (we have them saved earlier)
    ckpt = student_ckpts.get(b, None)
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location='cpu')
    else:
        sd = m.state_dict()
    student_moe_pkg["experts_state"][b] = sd

pkg_path = os.path.join(CHECKPOINT_DIR, "student_moe.pth")
torch.save(student_moe_pkg, pkg_path)
print("Saved Student-MoE package to:", pkg_path)
print("You can load it and reconstruct Student-MoE by instantiating each backbone and loading state_dicts, then loading gating.")

# ---------------------------
# Final per-expert evaluations (re-print)
# ---------------------------
print("\n=== Final per-expert student evaluations ===")
for b, m in expert_models:
    metrics = eval_student_on_loader(m, test_loader_student, DEVICE)
    print(f"\nStudent ({b}) on private test: Acc {metrics['acc']:.4f} Prec {metrics['prec']:.4f} Rec {metrics['rec']:.4f} F1 {metrics['f1']:.4f} AUC {metrics['auc']:.4f}")
    print("Confusion matrix:\n", metrics['cm'])
    print("Classification report:\n", classification_report(metrics['y_true'], metrics['y_pred'], target_names=CLASSES, zero_division=0))

print("\nAll checkpoints and Student-MoE saved under:", CHECKPOINT_DIR)
