"""
Experiment D  ·  TimeSformer‑Base + Audio‑CNN  + Attention Fusion
Target: ≥ 60 % ValAcc on RAVDESS         (GPU: RTX‑3060, 12 GB)
在本文件expD的基础上加了LLRD，效果非常好的一次，最佳valacc为89.24，总训练时长296.3
"""
import gc
# ─────────────────── imports ───────────────────
import os, math, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import torchaudio.transforms as T
# from pytorchvideo.models.hub import timesformer_base
import torch.nn as nn
from transformers import TimesformerModel, AutoImageProcessor

from dataset.dataset import RAVDESSMultimodalDataset
from models.audio_modelx import AudioEmbedding
from models.attention_fusion import AttentionFusion
from utils.train_utils import EarlyStopping, save_training_logs, plot_training_curves
from utils.metrics import calculate_metrics

# ──────────────── configuration ────────────────
config = {
    "exp":         "ExpD_TimeSformer",
    "batch_size":  5,           # fits 12 GB with AMP
    "frames":      8,
    "epochs":      25,
    "grad_accum":  6,           # logical batch = 16
    "lr_init":     2e-4,
    "min_lr":      1e-6,
    "weight_decay":1e-4,
    "dropout":     0.3,
    "emb_dim":     256,         # audio embedding
    "n_mels":      128,
    "patience":    8,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ───────────── dataset & augmentation ───────────
dataset = RAVDESSMultimodalDataset(
    dataset_dir="D:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24",
    mel_specs_kwargs={"n_mels": config["n_mels"]},
    frames=config["frames"],
    verbose=False
)
tr_size = int(0.8 * len(dataset))
va_size = len(dataset) - tr_size
tr_set, va_set = random_split(dataset, [tr_size, va_size])
tr_loader = DataLoader(tr_set, batch_size=config["batch_size"], shuffle=True)
va_loader = DataLoader(va_set, batch_size=config["batch_size"])

freq_mask = T.FrequencyMasking(16)
time_mask = T.TimeMasking(24)

# ──────────────── models ────────────────────────
# 1️⃣  audio branch
aud_model = AudioEmbedding(embedding_dim=config["emb_dim"], input_shape=(128, 128)).to(config["device"])

# 2️⃣  TimeSformer video branch
tsf = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
tsf.num_frames = config["frames"]  # 8帧
vid_model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400",
                                             config=tsf.config).to(config["device"])
vid_dim = tsf.config.hidden_size # 768
processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

# 3️⃣  fusion
fusion = AttentionFusion(
    audio_dim=config["emb_dim"],
    video_dim=vid_dim,
    num_classes=8,
    dropout=config["dropout"]
).to(config["device"])

# ───────────── optimisation objects ─────────────
criterion = nn.CrossEntropyLoss()
## 第一次实验D
# params    = list(aud_model.parameters()) + list(tsf.parameters()) + list(fusion.parameters())
# optim     = AdamW(params, lr=config["lr_init"], weight_decay=config["weight_decay"])

## 第二次尝试：Layer-wise Learning rate decay(LLRD)
params = []
decay = 0.95
for i, (n, p) in enumerate(vid_model.named_parameters()):
    if "encoder" in n:
        layer_idx = int(n.split('.')[2])
        lr = config["lr_init"] * (decay ** (12 - layer_idx))
    else:
        lr = config["lr_init"]
    params.append({"params": p, "lr": lr})
optim = AdamW(params, weight_decay=config["weight_decay"])

steps_ep  = math.ceil(len(tr_loader)/config["grad_accum"])
sched     = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=config["epochs"]*steps_ep, eta_min=config["min_lr"])
scaler    = GradScaler(device="cuda")
stopper   = EarlyStopping(patience=config["patience"], verbose=True)

# ─────────────── training loop ────────────────
def vid_forward(frames):  # frames: [B, T, 3, H, W]
    # reshape to list[tensor] -> processor -> model
    B, T, C, H, W = frames.shape
    # 将B*T张量拆成list(B) of Tensor[T,3,H,W]
    videos = [
        [frames[b, t] for t in range(T)]
        for b in range(B)
    ]
    inputs = processor(
        videos,
        return_tensors="pt",
        do_rescale=False
    ).to(config["device"])
    feat = vid_model(**inputs).last_hidden_state.mean(1)  # [B,768]
    return feat


print(f"🚀 Start {config['exp']} on {config['device']}")
tr_accs, va_accs, losses = [], [], []
start_all = time.time()

gc.collect()
torch.cuda.empty_cache()

for epoch in range(config["epochs"]):
    aud_model.train(); vid_model.train(); fusion.train()
    ep_loss, correct, total = 0.0, 0, 0
    ep_start = time.time()

    for step, (aud, vid, lab) in enumerate(tr_loader):
        aud, vid, lab = aud.to(config["device"]), vid.to(config["device"]), lab.to(config["device"])

        # augment
        aud = freq_mask(time_mask(aud))
        if torch.rand(1) < 0.5: vid = vid.flip(-1)

        with autocast(device_type="cuda"):
            aud_feat = aud_model(aud)  # [B,256]
            vid_feat = vid_forward(vid)  # [B,768]
            out  = fusion(aud_feat, vid_feat)
            loss = criterion(out, lab) / config["grad_accum"]

        scaler.scale(loss).backward()
        ep_loss += loss.item() * config["grad_accum"]
        correct += (out.argmax(1) == lab).sum().item()
        total   += lab.size(0)

        if (step+1) % config["grad_accum"] == 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            sched.step()

    tr_acc = correct / total
    va_acc, _ = calculate_metrics(
        lambda a, v: fusion(aud_model(a), vid_forward(v)),
        va_loader, config["device"],
        # aud_model, vid_model,
        mode="acc")

    tr_accs.append(tr_acc)
    va_accs.append(va_acc)
    losses.append(ep_loss/len(tr_loader))

    print(f"📘 Epoch {epoch+1}/{config['epochs']} | "
          f"TrainAcc {tr_acc:.4f} | ValAcc {va_acc:.4f} | "
          f"Time {(time.time()-ep_start)/60:.2f} min")

    stopper(va_acc)          # we only save fusion weights for simplicity
    if stopper.early_stop:
        print("⛔ Early stop"); break

    torch.cuda.empty_cache()

# ───────────────── save artefacts ───────────────
os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)
stamp = time.strftime("%m%d_%H%M")
torch.save({'audio': aud_model.state_dict(),
            'video': tsf.state_dict(),
            'fusion': fusion.state_dict()},
           f"saved_models/{config['exp']}_{stamp}.pt")

save_training_logs(tr_accs, va_accs, losses,
                   f"logs/{config['exp']}_{stamp}.csv")
plot_training_curves(tr_accs, va_accs, losses,
                     f"plots/{config['exp']}_{stamp}.png")
print(f"✅ Finished · total {(time.time()-start_all)/60:.1f} min")

