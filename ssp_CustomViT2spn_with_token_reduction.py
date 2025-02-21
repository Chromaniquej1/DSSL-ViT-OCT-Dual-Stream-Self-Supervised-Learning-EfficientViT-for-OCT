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
from fvcore.nn import FlopCountAnalysis
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
SEED = 42
BATCH_SIZE = 128  # Increased batch size to potentially improve training
EPOCHS = 5
LR = 1e-4
MOMENTUM = 0.999
ACCUM_STEPS = 8
OUTPUT_DIR = "./ssp_retinaloct_iccv2025/custom_transformer_reduced/"
PRETRAINED_MODEL_PATH = os.path.join(OUTPUT_DIR, "octmnist_custom_transformer_reduced_model.pth")
TOK_RATIO = 0.5 # Token keep ratio


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
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# DataLoader function
def load_octmnist_data(batch_size, split="train"):
    dataset = OCTMNIST(split=split, transform=DualViewTransform(augment_transform), download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

unlabeled_dataloader = load_octmnist_data(BATCH_SIZE, split="train")
labeled_dataloader = load_octmnist_data(BATCH_SIZE, split="val")  # For linear evaluation

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, attention_type='full'):
        super().__init__()
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.attention_type == 'full':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        elif self.attention_type == 'linear':
            q = q.softmax(dim=-2)
            k = k.softmax(dim=-1)
            x = (q @ k.transpose(-2, -1) @ v).transpose(1, 2).reshape(B, N, C)
        elif self.attention_type == 'local':
            # Simplified local attention
            window_size = min(7, N)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = attn * torch.tril(torch.ones_like(attn), diagonal=window_size-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")

        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., attention_type='full'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, attention_type)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class CustomTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=32, in_chans=3, num_classes=0, embed_dim=128, depth=4,  # Reduced embed_dim and depth
                 num_heads=4, mlp_ratio=4., attention_types=['full', 'linear', 'local'], token_keep_ratio=TOK_RATIO):  # Reduced num_heads and increased patch_size
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, attention_types[i % len(attention_types)])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Token Reduction Components
        self.token_selector = nn.Linear(embed_dim, 1)  # Predicts a score for each token
        self.token_keep_ratio = token_keep_ratio

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Token Reduction
        token_scores = self.token_selector(x).squeeze(-1)  # [B, num_patches]
        num_tokens_to_keep = max(1, int(self.token_keep_ratio * x.shape[1]))
        selected_tokens = torch.topk(token_scores, k=num_tokens_to_keep, dim=1)[1]
        x = x.gather(1, selected_tokens.unsqueeze(-1).expand(-1, -1, x.shape[2]))  # [B, num_tokens_to_keep, embed_dim]


        x = x.mean(dim=1)  # Global average pooling on the reduced set of tokens
        x = self.head(x)
        return x

class CustomTransformerBackbone(nn.Module):
    def __init__(self, token_keep_ratio=TOK_RATIO):
        super().__init__()
        self.transformer = CustomTransformer(img_size=256, patch_size=32, in_chans=3, embed_dim=128, depth=4,
                                             num_heads=4, mlp_ratio=4., attention_types=['full', 'linear', 'local'],
                                             token_keep_ratio=token_keep_ratio)  # Pass token_keep_ratio
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.transformer(x)
        x = self.pool(x.unsqueeze(-1)).squeeze(-1)
        return x

# Dual Stream Network
class DualStreamNetwork(nn.Module):
    def __init__(self, token_keep_ratio=TOK_RATIO):
        super().__init__()
        self.online_network = nn.ModuleList([CustomTransformerBackbone(token_keep_ratio=token_keep_ratio) for _ in range(2)])
        self.target_network = nn.ModuleList([CustomTransformerBackbone(token_keep_ratio=token_keep_ratio) for _ in range(2)])

        # Freeze target networks
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.projection_head = nn.Sequential(
            nn.Linear(128, 512),   # Adjusted Linear layer
            nn.ReLU(),
            nn.Linear(512, 64)     # Adjusted Linear layer
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(64, 64),      # Adjusted Linear layer
            nn.ReLU(),
            nn.Linear(64, 64)       # Adjusted Linear layer
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

# Compute FLOPs (GFLOPs) before training
dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)
flops = FlopCountAnalysis(model.module.online_network[0], dummy_input)
print(f"Model FLOPs: {flops.total() / 1e9:.4f} GFLOPs")

def train_self_supervised(model, dataloader, epochs, optimizer, criterion):
    scaler = GradScaler()
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
                loss = 1 - torch.mean(criterion(online_pred_feat, target_proj_feat)) / ACCUM_STEPS
            
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
        
        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}, "
              f"Time: {epoch_time:.4f}s, Throughput: {throughput:.4f} samples/sec")
        
        log_gpu_memory()

    print(f"\nTotal Training Time: {total_time:.4f}s")


# Measure Inference Latency
def measure_inference_latency(model, sample_input):
    model.eval()
    torch.cuda.synchronize()
    with torch.no_grad():
        start = time.time()
        model(sample_input, sample_input)
        torch.cuda.synchronize()
        end = time.time()
    print(f"Inference Latency: {(end - start) * 1000:.4f} ms")

# Create a sample input
sample_input = torch.randn(1, 3, 256, 256).to(DEVICE)

# Measure Inference Latency Before Training
measure_inference_latency(model, sample_input)

# Start training
train_self_supervised(model, unlabeled_dataloader, EPOCHS, optimizer, criterion)

# Measure inference latency after training
measure_inference_latency(model, sample_input)

# Compute FLOPs after training
flops_after_training = FlopCountAnalysis(model.module.online_network[0], dummy_input)
print(f"Model FLOPs after training: {flops_after_training.total() / 1e9:.4f} GFLOPs")

# Perform Linear Evaluation
#linear_evaluation(model, unlabeled_dataloader, labeled_dataloader)

# Save the trained model
torch.save(model.module.online_network[0].state_dict(), PRETRAINED_MODEL_PATH)
