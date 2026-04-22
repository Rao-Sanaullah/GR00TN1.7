# GR00T N1.7 on WSL2 Ubuntu 22.04

A complete, battle-tested installation guide for [NVIDIA Isaac GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) on Windows Subsystem for Linux — including real fixes for issues not covered in the official docs.

📖 **Full interactive guide → [index.html](https://rao-sanaullah.github.io/GR00TN1.7/)** (open via GitHub Pages)

---

## Tested On

| | |
|---|---|
| **OS** | WSL2 — Ubuntu 22.04 |
| **GPU** | NVIDIA RTX 5090 (Blackwell / sm_120) |
| **CUDA** | 12.8 |
| **Python** | 3.10 |

---

## What's Fixed Here

Issues not covered in the official docs — full details with terminal output in the [HTML guide](./index.html):

- 🔴 **Permission denied on `.venv`** — repo cloned as root, `uv sync` fails
- 🔴 **Triton crash on RTX 5090** — sm_120 (Blackwell) not recognised by Triton 3.3.1 pinned in PyTorch 2.7
- 🔴 **HuggingFace gated model** — `Cosmos-Reason2-2B` backbone requires access request + `huggingface-cli login`

---

## Quick Start

```bash
# 1. Install git-lfs BEFORE cloning
sudo apt-get install -y git-lfs ffmpeg curl build-essential
git lfs install

# 2. Clone with submodules
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

# 3. Fix permissions if cloned as root
sudo chown -R $USER:$USER .

# 4. Install uv and sync
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --python 3.10

# 5. Set CUDA_HOME
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
source ~/.bashrc

# 6. Patch Triton for RTX 5090 / Blackwell
uv run bash scripts/patch_triton_cuda13.sh

# 7. Authenticate with HuggingFace (gated model)
uv run huggingface-cli login

# 8. Verify
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

> For step-by-step details, real terminal output, and fine-tuning commands → **[see the full guide](./index.html)**

---


## Related

- [NVIDIA Isaac GR00T (official repo)](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T N1 Paper](https://arxiv.org/abs/2503.14734)
- [HuggingFace: nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B)

---

## License

Community documentation. GR00T N1.7 is licensed under [Apache 2.0](https://github.com/NVIDIA/Isaac-GR00T/blob/main/LICENSE) (code) and the [NVIDIA Open Model License](https://developer.nvidia.com/isaac/gr00t) (weights).
