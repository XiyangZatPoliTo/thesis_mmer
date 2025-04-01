#########################################################################
## Train AudioEmoRec for Speech Emotion Recognition on RAVDESS dataset  ##
#########################################################################
# 若使用conda命令安装依赖包，请用管理员方式打开cmd进行安装，可以避免channel错误
# 请安装低于numpy==2.0的版本，否则会出现依赖项无法加载的情况
import torch
from torch.utils.data import DataLoader
from trainer import train, evaluate
from adabelief_pytorch import AdaBelief
from utils.env import create_config, save_model, create_folder
from utils.criterion import LabelSmoothingLoss
from audio_model import AudioEmoRec
from dataset import RAVDESSMelS
from audio_dataloader import get_audio_meta, split_meta, mels2batch
import pickle
import matplotlib.pyplot as plt
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

config = create_config()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # I tested on the CPU
print("here is the checkpoint 1")
print("train model on {}".format(device))

# Extract meta information from .wav file name
meta = get_audio_meta(config.dataset_dir)
print(meta.head(3))
print("here is the checkpoint 2")
# Mel spectrogram parameters
mel_kwargs = {"n_mels": 64, "fmax": 6000, "win_length": 2048, "hop_length": 2048 // 3 * 2, "n_fft": 2048 * 2}

if config.kfcv:
    # Stratified 5-Fold CV
    splitted_meta = split_meta(meta=meta, kfcv=True, n_splits=5, target='emotion', random_state=config.random_state)
else:
    # Hold-out 8:2
    splitted_meta = split_meta(meta=meta, kfcv=False, test_ratio=0.2, stratify=True, target='emotion',
                               random_state=config.random_state)

print("here is the checkpoint 3")
print('OUTPUT_PATH: {}'.format(config.out_dir))

for k, (tr, ts) in enumerate(splitted_meta):
    if config.kfcv:
        # K-Fold CV
        print(f'----------- Fold {k + 1} -----------')
        train_meta = meta.iloc[tr, :]
        test_meta = meta.iloc[ts, :]
    else:
        # Hold-out
        train_meta, test_meta = splitted_meta
    train_dataset = RAVDESSMelS(dataset_dir=config.dataset_dir, mel_specs_kwargs=mel_kwargs,
                                speech_list=train_meta.speech_list.tolist())
    test_dataset = RAVDESSMelS(dataset_dir=config.dataset_dir, mel_specs_kwargs=mel_kwargs,
                               speech_list=test_meta.speech_list.tolist())

    # Train/test dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=mels2batch, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=mels2batch, shuffle=False)

    # Define model
    model = AudioEmoRec(input_dim=mel_kwargs['n_mels'], hidden_dim=128, kernel_size=3, num_classes=8).to(device)
    loss = LabelSmoothingLoss(n_classes=8, smoothing=.1, dim=-1)
    optimizer = AdaBelief(model.parameters(), lr=1e-3, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-7,
                          weight_decouple=False, rectify=False, print_change_log=False)
    # 学习率调度器，通过监控模型的性能指标来动态调整学习率，帮助模型更好地收敛。
    # UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.7, patience=2, cooldown=3, verbose=True,
                                                           min_lr=1e-6, mode='min', threshold=0.001)

    MODEL_SAVE_DIR = os.path.join(config.out_dir, f'model/fold{k + 1}/' if config.kfcv else 'model/')
    LOG_OUT_DIR = os.path.join(config.out_dir, 'logs/')
    LOG_FILENAME = os.path.join(LOG_OUT_DIR, f'fold{k + 1}_{config.log}' if config.kfcv else config.log)
    # noinspection PyPackageRequirements
    N_EPOCHS = config.n_epochs
    create_folder(config.out_dir)
    create_folder(MODEL_SAVE_DIR)
    create_folder(LOG_OUT_DIR)
    print('LOG_OUT_PATH: {}'.format(LOG_FILENAME))
    print('MODEL_OUT_PATH: {}'.format(os.path.join(MODEL_SAVE_DIR, config.model_name)))

    # Training
    print('Training {} Epochs'.format(N_EPOCHS))
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    best_test_loss = torch.inf  # initialized the best loss to infinity
    best_test_acc = 0  # initialized the best accuracy to 0

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(train_dataloader, model, optimizer, loss, epoch + 1, device=device)
        test_loss, test_acc = evaluate(test_dataloader, model, loss, epoch + 1, device=device)

        f = open(f"{LOG_FILENAME}", "a")
        epoch_result = ("[Epoch {e}] train Loss: {trainL:.5f} Accuracy: {trainAcc:.4f} | test Loss: {testL:.5f} "
                        "Accuracy: {testAcc:.4f} ").format(
            e=epoch + 1, trainL=train_loss, trainAcc=train_acc, testL=test_loss, testAcc=test_acc)
        print(epoch_result, file=f)
        print(epoch_result)

        scheduler.step(train_loss)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        f.close()

        if test_acc > best_test_acc:
            save_model(MODEL_SAVE_DIR, test_loss=test_loss, test_acc=test_acc, model=model,
                       model_name=config.model_name)
            best_test_acc = test_acc
            best_test_loss = test_loss

        elif ((best_test_acc - test_acc) < 1e-3) & (test_loss < best_test_loss):
            save_model(MODEL_SAVE_DIR, test_loss=test_loss, test_acc=test_acc, model=model,
                       model_name=config.model_name)
            best_test_acc = test_acc
            best_test_loss = test_loss
        else:
            pass

    # Save training log
    metric_log_dict = {
        'train': {
            'acc': train_acc_list,
            'loss': train_loss_list
        },
        'test': {
            'acc': test_acc_list,
            'loss': test_loss_list
        }
    }

    with open(os.path.join(LOG_OUT_DIR, (f'fold{k + 1}_' if config.kfcv else '') + 'metric_log_dict.pkl'), 'wb') as f:
        pickle.dump(metric_log_dict, f)

    # Last epoch, save the model
    save_model(MODEL_SAVE_DIR, test_loss=test_loss, test_acc=test_acc, model=model, model_name="last_epoch", init=False)
    if not config.kfcv:
        break
