
import torch
import torchvision

print("✅ PyTorch Version:", torch.__version__)
print("✅ Torchvision Version:", torchvision.__version__)
print("✅ CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ CUDA Device:", torch.cuda.get_device_name(0))

# Try importing ViT model
try:
    from torchvision.models.vision_transformer import vit_b_16
    model = vit_b_16(pretrained=False)
    print("✅ ViT model loaded successfully!")
except Exception as e:
    print("❌ Failed to load ViT model:", str(e))
