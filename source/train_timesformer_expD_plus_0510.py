"""
ExpD+  : TimeSformer + LoRA 8-frame
改动  : ① 关闭 gradient-checkpointing
        ② 仅解冻 encoder.layer.9-11
        ③ CosineWarmRestarts 学习率
2025-05-10

2025年5月18日10:55:57 目前此代码因为相关库文件无法适配，未通过测试
"""

import os, time, math, random
import torch
import torch.nn as nn
from torch import GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset.dataset import RAVDESSMultimodalDataset
from models.audio_modelx import AudioEmbedding
from models.attention_fusion import AttentionFusion
from transformers import TimesformerModel, TimesformerConfig, AutoImageProcessor
from peft import LoraConfig, get_peft_model, TaskType

# ------------------------
def main():
    # ==================== 配置 ====================
    cfg = dict(
        exp_name="ExpD_Plus_TimeSformer_UnfreezeTop3_0510",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch=6,  # 显存≈10 GB (RTX3060)
        grad_accum=6,  # 6×6=36 等效批
        epoch=25,
        lr=1e-4,
        weight_decay=1e-5,
        patience=4,
        n_mels=128,
        frames=8,  # clip 长度
        audio_emb_dim=256,
        num_classes=8,
        log_dir="logs",
        model_dir="saved_models",
        plot_dir="plots",
    )

    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["model_dir"], exist_ok=True)
    os.makedirs(cfg["plot_dir"], exist_ok=True)

    # ==================== 数据集 ====================
    dataset = RAVDESSMultimodalDataset(
        dataset_dir="D:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24",
        mel_specs_kwargs={"n_mels": cfg["n_mels"]},
        frames=cfg["frames"],
        verbose=False
    )
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=cfg["batch"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch"], shuffle=False, num_workers=4, pin_memory=True)

    # ==================== 模型 ====================
    ## 音频
    aud_model = AudioEmbedding(embedding_dim=cfg["audio_emb_dim"]).to(cfg["device"])

    ## 视频 – TimeSformer + LoRA
    ts_cfg = TimesformerConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
    vid_model = TimesformerModel.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",
        config=ts_cfg
    ).to(cfg["device"])

    # 关闭 gradient-checkpointing !
    # vid_model.gradient_checkpointing_enable()  # ← 不再调用

    # ---- 解冻最后 3 层 Transformer ----
    for p in vid_model.parameters():
        p.requires_grad = False
    for i, (n, p) in enumerate(vid_model.named_parameters()):
        if "encoder" in n:
            layer_idx = int(n.split('.')[2])
            if layer_idx >= ts_cfg.num_hidden_layers - 3:  # 9,10,11
                p.requires_grad = True
    # head / LayerNorm 保持训练
    for n, p in vid_model.named_parameters():
        if any(k in n for k in ["temporal_embedding", "layernorm", "pooler"]):
            p.requires_grad = True

    # LoRA 注入到 qkv
    lora_cfg = LoraConfig(
        lora_alpha=32,
        target_modules=["qkv"],
        bias="none",
        lora_dropout=0.1,
        task_type=TaskType.FEATURE_EXTRACTION
    )
    vid_model = get_peft_model(vid_model, lora_cfg)
    print(f"Trainable video params: {sum(p.numel() for p in vid_model.parameters() if p.requires_grad):,}")

    ## 融合
    fusion = AttentionFusion(
        audio_dim=cfg["audio_emb_dim"],
        video_dim=vid_model.config.hidden_size,  # 768
        num_classes=cfg["num_classes"],
        dropout=0.3
    ).to(cfg["device"])

    # ==================== 优化器 + LLRD ====================
    decay = 0.95
    param_groups = []

    # 音频
    param_groups += [{"params": aud_model.parameters(), "lr": cfg["lr"]}]

    # 视频 – LLRD 仅对可训练参数生效
    import re
    layer_pat = re.compile(r"encoder\.layer\.(\d+)\.")

    for n, p in vid_model.named_parameters():
        if not p.requires_grad:
            continue

        m = layer_pat.match(n)
        if m:
            layer_idx = int(m.group(1))
            lr = cfg["lr"] * (decay ** (11 - layer_idx))  # 12 层
        else:
            lr = cfg["lr"]
        param_groups.append({"params": p, "lr": lr})

    # 融合
    param_groups += [{"params": fusion.parameters(), "lr": cfg["lr"]}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])
    scaler = GradScaler(device="cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ==================== 辅助工具 ====================
    def accuracy(outputs, labels):
        return (outputs.argmax(1) == labels).float().mean().item()

    class EarlyStopping:
        def __init__(self, patience=4):
            self.best = 0.0
            self.counter = 0
            self.patience = patience

        def step(self, metric):
            if metric > self.best:
                self.best = metric
                self.counter = 0
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience

    stopper = EarlyStopping(cfg["patience"])
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

    # TubeletMask 与 SpecAugment 按你之前实现保持不变
    def tubelet_mask(frames, prob=0.15):
        if random.random() > prob:
            return frames
        T, C, H, W = frames.shape
        ph, pw = 16, 16
        th = random.randrange(0, H // ph)
        tw = random.randrange(0, W // pw)
        frames[..., th * ph:(th + 1) * ph, tw * pw:(tw + 1) * pw] = 0
        return frames

    def vid_forward(frames):  # [B,T,3,H,W]
        B, T, C, H, W = frames.shape
        videos = []
        for b in range(B):
            clip = frames[b].cpu()
            clip = tubelet_mask(clip)
            # processor要list[帧]，因此拆分
            videos.append([clip[t] for t in range(T)])
        proc_out = processor(
            videos,
            return_tensors="pt",
            do_rescale=False,
            input_data_format="channels_first")

        # print("process_keys:", proc_out.keys())

        pix = proc_out["pixel_values"].to(cfg["device"])
        with torch.no_grad():
            out = vid_model(pixel_values=pix)
        feat = out.last_hidden_state.mean(1)  # [B,768]
        return feat

    # ==================== 训练循环 ====================
    def run_epoch(loader, train=True):
        if train:
            aud_model.train()
            vid_model.train()
            fusion.train()
        else:
            aud_model.eval()
            vid_model.eval()
            fusion.eval()

        total_loss, total_acc, steps = 0, 0, 0
        optimizer.zero_grad()
        for i, (aud, vid, y) in enumerate(tqdm(loader, desc="Epoch" + ("Train" if train else "Val"))):
            aud, vid, y = aud.to(cfg["device"]), vid.to(cfg["device"]), y.to(cfg["device"])

            # try:
            #     vid_feat = vid_forward(vid)
            #     out = fusion(aud_model(aud), vid_feat)
            # except Exception as e:
            #     print(f"\n捕捉到异常：{type(e).__name__} - {e}")
            #     import traceback, sys
            #     traceback.print_exc()
            #     sys.exit(1)
            # Forward
            out = fusion(aud_model(aud), vid_forward(vid))
            loss = criterion(out, y)
            acc = accuracy(out, y)

            if train:
                loss.backward()
                if (i + 1) % cfg["grad_accum"] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step(epoch + i / len(loader))

            total_loss += loss.item()
            total_acc += acc
            steps += 1
        return total_loss / steps, total_acc / steps

    print(f"🚀 Start {cfg['exp_name']} on {cfg['device']}")
    best_acc, start = 0.0, time.time()
    for epoch in range(cfg["epoch"]):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)
        mins = (time.time() - t0) / 60
        print(f"📘 Epoch {epoch + 1}/{cfg['epoch']} | "
              f"TrainAcc {tr_acc:.4f} | ValAcc {val_acc:.4f} | Time {mins:.2f} min")

        if stopper.step(val_acc):
            print("EarlyStopping - patience reached")
            break

    # ==================== 保存 ====================
    ts = time.strftime("%m%d_%H%M")
    torch.save({
        "audio": aud_model.state_dict(),
        "video": vid_model.state_dict(),
        "fusion": fusion.state_dict()
    }, f"{cfg['model_dir']}/{cfg['exp_name']}_{ts}.pt")
    print("✅ Done · total %.1f min" % ((time.time() - start) / 60))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
