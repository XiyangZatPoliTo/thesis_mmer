"""
ExpD_Plus · TimeSformer-8f + AudioEmbedding + AttentionFusion
增强: SpecAug++(Audio)  TubeletMask(Video)
技巧: LLRD  CosineRestart  LoRA(rank16)
验证目标 ≥ 0.90
0510实验记录：
1, 局部解冻高层

2025年5月18日11:31:10
目前的结果表示，增设LoRA之后训练时长相较LLRD缩短了一半，但是正确率有略微下降。
"""

# ───────── imports ─────────
import os, time, math, random
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
import torchaudio.transforms as T
from transformers import TimesformerModel, TimesformerConfig, AutoImageProcessor
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from dataset.dataset import RAVDESSMultimodalDataset
from models.audio_modelx import AudioEmbedding
from models.attention_fusion import AttentionFusion
from utils.metrics import calculate_metrics
from utils.train_utils import EarlyStopping, save_training_logs, plot_training_curves

import re
layer_pat = re.compile(r"encoder\.layer\.(\d+)\.")


# ───────── config ──────────
cfg = {
    "exp":        "ExpD_Plus_TimeSformer_LoRA_8f_0509",
    "batch":      6,        # RTX3060, grad_ckpt=True
    "grad_accum": 6,        # eff batch = 36
    "frames":     8,
    "epochs":     25,
    "lr":         2e-4,
    "min_lr":     1e-6,
    "wd":         1e-4,
    "dropout":    0.3,
    "a_dim":      256,
    "n_mels":     128,
    "patience":   8,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


# ───────── dataset ─────────
dset = RAVDESSMultimodalDataset(
    dataset_dir="D:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24",
    mel_specs_kwargs={"n_mels": cfg["n_mels"]},
    frames=cfg["frames"],
    verbose=False
)

tr_len = int(0.8 * len(dset)); va_len = len(dset) - tr_len
tr_set, va_set = random_split(dset, [tr_len, va_len])
tr_loader = DataLoader(tr_set, batch_size=cfg["batch"], shuffle=True)
va_loader = DataLoader(va_set, batch_size=cfg["batch"])

# ───────── augmentation ❶ ─────────
freq_mask = T.FrequencyMasking(16)
time_mask = T.TimeMasking(24)

def tubelet_mask(video, prob=0.15):       # video: [T,3,224,224]
    if random.random() > prob: return video
    t, c, h, w = video.shape
    ph, pw = 16, 16                       # patch size
    th = random.randrange(0, h // ph)
    tw = random.randrange(0, w // pw)
    video[:, :, th*ph:(th+1)*ph, tw*pw:(tw+1)*pw] = 0
    return video

# ───────── models ──────────
aud_model = AudioEmbedding(embedding_dim=cfg["a_dim"], input_shape=(128, 128)).to(cfg["device"])

# --- TimeSformer base, 8 frames
ts_cfg = TimesformerConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
ts_cfg.num_frames = cfg["frames"]
vid_model = TimesformerModel.from_pretrained(
    "facebook/timesformer-base-finetuned-k400", config=ts_cfg).to(cfg["device"])

# # 补丁代码
# from types import MethodType
# emb = vid_model.embeddings
# orig_forward = emb.forward
# def safe_patch(self, pixel_values):
#
#     patches, *_ = self.patch_embeddings(pixel_values)
#     return patches
#
# emb.forward = MethodType(safe_patch, emb)
#
# vid_model.gradient_checkpointing_enable()                       # 显存减半
processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
vid_dim = ts_cfg.hidden_size                                    # 768

# --- LoRA ❷
lora_cfg = LoraConfig(
    target_modules=["qkv"],
    r=16, lora_alpha=32, lora_dropout=0.05)
vid_model = get_peft_model(vid_model, lora_cfg)

# Fusion
fusion = AttentionFusion(
    audio_dim=cfg["a_dim"],
    video_dim=vid_dim,
    num_classes=8,
    dropout=cfg["dropout"]).to(cfg["device"])

# ───────── optimizer & LLRD ❸ ─────────
decay = 0.95
param_groups = []
for n, p in vid_model.named_parameters():
    if not p.requires_grad:
        continue

    m = layer_pat.match(n)
    if m:
        layer_idx = int(m.group(1))
        lr = cfg["lr"] * (decay ** (11-layer_idx))     # 12 层
    else:
        lr = cfg["lr"]
    param_groups.append({"params": p, "lr": lr})
param_groups += [{"params": aud_model.parameters()},
                 {"params": fusion.parameters()}]

optim  = AdamW(param_groups, weight_decay=cfg["wd"])
steps  = math.ceil(len(tr_loader) / cfg["grad_accum"])
sched  = CosineAnnealingWarmRestarts(optim, T_0=steps*3, T_mult=2, eta_min=cfg["min_lr"])
scaler = GradScaler(device="cuda")
stopper= EarlyStopping(patience=cfg["patience"], verbose=True)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)           # + LS

# ───────── helper ──────────
def vid_forward(frames):                                       # [B,T,3,H,W]
    B,T,C,H,W = frames.shape
    videos = []
    for b in range(B):
        clip = frames[b].cpu()
        clip = tubelet_mask(clip)
        # processor要list[帧]，因此拆分
        videos.append([clip[t] for t in range(T)])
    inputs = processor(videos, return_tensors="pt",
                       do_rescale=False, input_data_format="channels_first").to(cfg["device"])
    feat = vid_model(**inputs).last_hidden_state.mean(1)       # [B,768]
    return feat

# ───────── training ────────
print(f"🚀 Start {cfg['exp']} on {cfg['device']}")
tr_accs, va_accs, losses = [], [], []; t0_all = time.time()

for epoch in range(cfg["epochs"]):
    aud_model.train(); vid_model.train(); fusion.train()
    total, correct, ep_loss = 0, 0, 0; t0 = time.time()

    for step,(aud, vid, lab) in enumerate(tqdm(tr_loader, desc=f"Epoch{epoch+1}")):
        aud, vid, lab = aud.to(cfg["device"]), vid.to(cfg["device"]), lab.to(cfg["device"])
        aud = time_mask(freq_mask(aud))                        # SpecAug++
        if random.random()<0.5: vid = vid.flip(-1)            # H-flip

        with autocast(device_type="cuda"):
            out = fusion(aud_model(aud), vid_forward(vid))
            loss = criterion(out, lab) / cfg["grad_accum"]

        scaler.scale(loss).backward()
        ep_loss += loss.item()*cfg["grad_accum"]
        correct += (out.argmax(1)==lab).sum().item(); total+=lab.size(0)

        if (step+1)%cfg["grad_accum"]==0 or step+1==len(tr_loader):
            scaler.step(optim); scaler.update()
            optim.zero_grad(set_to_none=True); sched.step()

    tr_acc = correct/total
    va_acc,_ = calculate_metrics(
        lambda a,v: fusion(aud_model(a), vid_forward(v)),
        va_loader, cfg["device"], mode="acc")

    tr_accs.append(tr_acc); va_accs.append(va_acc); losses.append(ep_loss/len(tr_loader))
    print(f"📘 Epoch {epoch+1}/{cfg['epochs']} | TrainAcc {tr_acc:.4f} | ValAcc {va_acc:.4f}"
          f" | Time {(time.time()-t0)/60:.2f} min")

    stopper(va_acc)
    if stopper.early_stop:
        print("⛔ Early stop")
        break

# ───────── save ───────────
os.makedirs("saved_models",exist_ok=True);os.makedirs("logs",exist_ok=True);os.makedirs("plots",exist_ok=True)
ts = time.strftime("%m%d_%H%M")
torch.save({
    "audio": aud_model.state_dict(),
    "video": vid_model.state_dict(),
    "fusion":fusion.state_dict()
}, f"saved_models/{cfg['exp']}_{ts}.pt")
save_training_logs(tr_accs,va_accs,losses,f"logs/{cfg['exp']}_{ts}.csv")
plot_training_curves(tr_accs,va_accs,losses,f"plots/{cfg['exp']}_{ts}.png")
print(f"✅ Done · total {(time.time()-t0_all)/60:.1f} min")