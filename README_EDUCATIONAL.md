# DiT Educational Setup Guide

This guide helps you set up DiT for educational/learning purposes with single-GPU training, mixed precision, and gradient accumulation.

## 🚀 Quick Start (From Scratch)

### 1. Git Configuration (One-Time Setup)

```bash
# Set your identity
git config --global user.name "YourGitHubUsername"
git config --global user.email "your.email@example.com"

# Verify
git config --global --list | grep user
```

### 2. Clone and Setup Repository

```bash
# Clone the official repo (or your fork)
git clone https://github.com/facebookresearch/DiT.git
cd DiT

# Add your fork as origin (if you forked it)
git remote rename origin upstream
git remote add origin https://github.com/YourUsername/YourFork.git

# Verify remotes
git remote -v
# upstream → official repo (pull updates from here)
# origin   → your fork (push your changes here)

# Create your working branch
git checkout -b educational
```

### 3. Install PyTorch with RTX 5080 Support (sm_120)

**Important:** Standard PyTorch doesn't support RTX 5080 (Blackwell architecture). You need PyTorch nightly with CUDA 12.8:

```bash
# Check your CUDA driver
nvidia-smi  # Should show CUDA 13.1 or compatible

# Install uv (if not already installed)
pip install uv

# Uninstall old PyTorch (if any)
uv pip uninstall torch torchvision

# Install PyTorch nightly with CUDA 12.8 (includes sm_120 support)
UV_HTTP_TIMEOUT=300 uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install DiT dependencies
uv pip install timm diffusers accelerate

# Verify PyTorch recognizes your GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Architectures: {torch.cuda.get_arch_list()}')"
# Should show sm_120 in the architecture list
```

### 4. Prepare Dataset (Tiny ImageNet)

```bash
# Install unzip if needed
sudo apt install unzip

# Download Tiny ImageNet
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

# Flatten directory structure (ImageFolder format)
cd tiny-imagenet-200/train
for class_dir in */; do
    mv "$class_dir"images/* "$class_dir"
    rmdir "$class_dir"images
done
cd ../..
```

## 🎓 Educational Training Features

This modified version includes:

### New Command-Line Flags

- `--single-gpu`: Train without DDP (easier debugging, no torchrun needed)
- `--mixed-precision`: Enable FP16 training (50% less memory)
- `--gradient-accumulation-steps N`: Accumulate gradients over N steps (larger effective batch size)

### Updated Defaults

```python
--model DiT-S/2                      # Smaller model (33M params vs 675M)
--num-classes 200                    # For Tiny ImageNet
--global-batch-size 8                # Memory-friendly
--gradient-accumulation-steps 2      # Effective batch = 8 (uses 4 per step)
--mixed-precision (enabled)          # FP16 by default
--log-every 10                       # Frequent logging
--single-gpu (enabled)               # Single GPU mode by default
```

## 🏃 Running Training

### Basic Training (Minimal Memory)

```bash
python3 train.py
```

That's it! The defaults are optimized for learning on a single GPU.

### Custom Training Examples

```bash
# Larger batch size (if you have enough memory)
python3 train.py \
  --global-batch-size 16 \
  --gradient-accumulation-steps 4

# Different model sizes
python3 train.py --model DiT-S/2   # Small (33M params) - Recommended
python3 train.py --model DiT-B/4   # Base (130M params)
python3 train.py --model DiT-L/8   # Large (458M params)

# Disable mixed precision (if debugging)
python3 train.py --no-mixed-precision

# Use distributed training (original behavior)
torchrun --nnodes=1 --nproc_per_node=1 train.py --no-single-gpu
```

### Memory Usage Guide

| Config | Micro-Batch | Memory Usage | Speed |
|--------|-------------|--------------|-------|
| Default | 4 | ~4-6 GB | Fast |
| `--global-batch-size 16 --gradient-accumulation-steps 4` | 4 | ~4-6 GB | Medium |
| `--global-batch-size 32 --gradient-accumulation-steps 1` | 32 | ~12-14 GB | Fastest |
| `--global-batch-size 2` | 2 | ~2-3 GB | Slower |

## 🔧 Testing Your Setup

### Quick Test (Sample from Pre-trained Model)

```bash
python sample.py --image-size 256 --seed 1
```

This downloads a pre-trained model and generates samples. If this works, your PyTorch is correctly installed.

### Training Test (1 Epoch)

```bash
python3 train.py --epochs 1 --log-every 5
```

Should complete without errors and show training progress.

## 📚 Git Workflow (Daily Use)

### Make Changes

```bash
# Work on your educational branch
git checkout educational

# Edit files...

# Commit changes
git add .
git commit -m "describe your changes"

# Push to your fork
git push origin educational
```

### Get Upstream Updates

```bash
# Update main branch from official repo
git checkout main
git pull upstream main

# Merge into your educational branch
git checkout educational
git merge main

# Resolve conflicts if any, then push
git push origin educational
```

## 🐛 Troubleshooting

### PyTorch CUDA Error: "no kernel image is available"

**Problem:** Your PyTorch doesn't support RTX 5080 (sm_120).

**Solution:** Install PyTorch nightly with CUDA 12.8 (see step 3 above).

### Out of Memory Error

**Solutions:**
1. Reduce batch size: `--global-batch-size 4`
2. Increase gradient accumulation: `--gradient-accumulation-steps 4`
3. Use smaller model: `--model DiT-S/2`
4. Ensure mixed precision is enabled (default)

### Git Authentication Failed

**Solution:** Set up GitHub authentication:

```bash
# Option 1: GitHub CLI (easiest)
sudo apt install gh
gh auth login

# Option 2: SSH keys
ssh-keygen -t ed25519 -C "your.email@example.com"
cat ~/.ssh/id_ed25519.pub
# Add to https://github.com/settings/keys
git remote set-url origin git@github.com:YourUsername/YourFork.git
```

### "ValueError: environment variable RANK expected"

**Problem:** Trying to run distributed training without torchrun.

**Solution:** Use `--single-gpu` flag (now default) or use torchrun:

```bash
# Single GPU (simple)
python3 train.py --single-gpu

# Distributed
torchrun --nnodes=1 --nproc_per_node=1 train.py --no-single-gpu
```

## 📊 Valid Model Names (Case-Sensitive!)

**Small (Recommended):**
- `DiT-S/2`, `DiT-S/4`, `DiT-S/8`

**Base:**
- `DiT-B/2`, `DiT-B/4`, `DiT-B/8`

**Large:**
- `DiT-L/2`, `DiT-L/4`, `DiT-L/8`

**Extra Large:**
- `DiT-XL/2`, `DiT-XL/4`, `DiT-XL/8`

## 🎯 Summary Checklist

- [ ] Git configured (name, email)
- [ ] Repository cloned and remotes set up
- [ ] PyTorch nightly with CUDA 12.8 installed
- [ ] GPU recognized by PyTorch (sm_120 support)
- [ ] Tiny ImageNet downloaded and formatted
- [ ] Test sampling works
- [ ] Test training works
- [ ] Changes committed and pushed

## 📖 Additional Resources

- Original DiT Paper: http://arxiv.org/abs/2212.09748
- Project Page: https://www.wpeebles.com/DiT
- PyTorch Memory Management: https://pytorch.org/docs/stable/notes/cuda.html

---

**Modified by:** KennethWang1221  
**Purpose:** Educational single-GPU training with memory optimizations  
**Based on:** [facebookresearch/DiT](https://github.com/facebookresearch/DiT)
