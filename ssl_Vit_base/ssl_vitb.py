import os
import random
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import warnings
from medmnist.dataset import OCTMNIST
from transformers import ViTConfig, ViTForImageClassification
from fvcore.nn import FlopCountAnalysis

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
SEED = 42
BATCH_SIZE = 64  
EPOCHS = 1  
LR = 1e-4
MOMENTUM = 0.999
ACCUM_STEPS = 8  
OUTPUT_DIR = "./ssp_retinaloct_iccv2025/ssl_vit_base/"
PRETRAINED_MODEL_PATH = os.path.join(OUTPUT_DIR, "octmnist_ssl_vit_base_model.pth")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Data Augmentation Pipeline
class DualViewTransform:
    def __init__(self, augment_transform):
        self.augment_transform = augment_transform

    def __call__(self, x):
        return self.augment_transform(x), self.augment_transform(x)

augment_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# DataLoader function
def load_octmnist_data(batch_size):
    dataset = OCTMNIST(split="train", transform=DualViewTransform(augment_transform), download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

unlabeled_dataloader = load_octmnist_data(BATCH_SIZE)

# Vision Transformer Backbone with Correct Feature Extraction
class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=768, 
            num_hidden_layers=12,
            num_attention_heads=12,  
            intermediate_size=3072,
            num_labels=0,
            output_hidden_states=True  
        )
        self.vit = ViTForImageClassification(config)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        output = self.vit(x)
        if output.hidden_states is None:
            raise ValueError("Hidden states are not available. Ensure `output_hidden_states=True` in ViTConfig.")
        return self.pool(output.hidden_states[-1].permute(0, 2, 1)).squeeze(-1)

# Dual Stream Network
class DualStreamNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.online_network = nn.ModuleList([ViTBackbone() for _ in range(2)])
        self.target_network = nn.ModuleList([ViTBackbone() for _ in range(2)])

        # Freeze target networks
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.projection_head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x1, x2):
        online_feats = [net(x) for net, x in zip(self.online_network, [x1, x2])]
        target_feats = [net(x).detach() for net, x in zip(self.target_network, [x1, x2])]
        
        online_proj_feat = self.projection_head(torch.mean(torch.stack(online_feats), dim=0))
        online_pred_feat = self.prediction_head(online_proj_feat)
        target_proj_feat = self.projection_head(torch.mean(torch.stack(target_feats), dim=0))
        
        return online_pred_feat, target_proj_feat

# Initialize Model, Optimizer, and Criterion
model = DualStreamNetwork().to(DEVICE)
model = nn.DataParallel(model)  
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CosineSimilarity(dim=1)
scaler = GradScaler()

# Log GPU memory usage
def log_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9  
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU Memory - Allocated: {allocated:.4f} GB, Reserved: {reserved:.4f} GB")

# Compute FLOPs (GFLOPs) 
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
flops = FlopCountAnalysis(model.module.online_network[0], dummy_input)
print(f"Model FLOPs: {flops.total() / 1e9:.4f} GFLOPs")

# Training function with performance tracking
def train_self_supervised(model, dataloader, epochs, optimizer, criterion):
    total_time, total_samples, total_throughput = 0.0, 0.0, 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0  
        optimizer.zero_grad()  
        epoch_start = time.time()  

        for i, (views, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            view1, view2 = views[0].to(DEVICE), views[1].to(DEVICE)
            batch_size = view1.size(0)
            total_samples += batch_size

            with autocast():
                online_pred_feat, target_proj_feat = model(view1, view2)
                loss = -torch.mean(criterion(online_pred_feat, target_proj_feat)) / ACCUM_STEPS

            scaler.scale(loss).backward()
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUM_STEPS

        epoch_time = time.time() - epoch_start
        throughput = total_samples / epoch_time
        total_time += epoch_time
        total_throughput += throughput

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}, Time: {epoch_time:.4f}s, Throughput: {throughput:.4f} samples/sec")
        log_gpu_memory()

    #print(f"\nTotal Training Time: {total_time:.4f}s")

# Measure Inference Latency after training
def measure_inference_latency(model, sample_input):
    model.eval()
    with torch.no_grad():
        start = time.time()
        model(sample_input, sample_input)
        end = time.time()
    print(f"Inference Latency: {(end - start) * 1000:.4f} ms")

# Create a sample input to measure latency
sample_input = torch.randn(1, 3, 224, 224).to(DEVICE)

# Start training
train_self_supervised(model, unlabeled_dataloader, EPOCHS, optimizer, criterion)

# Measure inference latency
measure_inference_latency(model, sample_input)

# Save the trained model
torch.save(model.module.online_network[0].state_dict(), PRETRAINED_MODEL_PATH)