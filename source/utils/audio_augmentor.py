import torch
import torchaudio
import random

class AudioAugmentor:
    def __init__(self, time_mask_param=30, freq_mask_param=15, noise_level=0.005,
                 p_time_mask=0.5, p_freq_mask=0.5, p_noise=0.3):
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.noise_level = noise_level
        self.p_time_mask = p_time_mask
        self.p_freq_mask = p_freq_mask
        self.p_noise = p_noise

    def __call__(self, mel_spec: torch.Tensor):
        """
        mel_spec: [n_mels, time] or [1, n_mels, time]
        """
        original_shape = mel_spec.shape
        if mel_spec.ndim == 2:
            mel_spec = mel_spec.unsqueeze(0)

        if random.random() < self.p_time_mask:
            mel_spec = self.time_mask(mel_spec)
        if random.random() < self.p_freq_mask:
            mel_spec = self.freq_mask(mel_spec)
        if random.random() < self.p_noise:
            noise = torch.randn_like(mel_spec) * self.noise_level
            mel_spec = mel_spec + noise

        return mel_spec.view(original_shape)
