# VideoRAG 在 RunPod 上的部署安装指南

## 系统环境信息
- **镜像**: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
- **Python**: 3.10.12
- **PyTorch**: 2.2.0+cu121 (已预装在系统Python环境中)
- **torchvision**: 0.17.0+cu121 (已预装)
- **torchaudio**: 2.2.0 (已预装)
- **CUDA**: 12.1
- **GPU**: NVIDIA RTX 2000 Ada (16GB)

**重要**: 系统已预装PyTorch，可以考虑使用 `--system-site-packages` 选项创建虚拟环境以复用系统PyTorch，节省空间并避免版本冲突。

## 安装步骤

### 步骤 1: 更新系统和基础工具

```bash
# 更新apt包管理器
apt-get update

# 安装必要的系统依赖
apt-get install -y wget curl git git-lfs build-essential

# 更新pip到最新版本
pip3 install --upgrade pip
```

### 步骤 2: 安装 Git LFS（用于下载大模型文件）

```bash
# 安装git-lfs
apt-get install -y git-lfs

# 初始化git-lfs
git lfs install
```

### 步骤 3: 创建Python虚拟环境

```bash
# 检查网络卷挂载（RunPod的网络卷通常挂载在/workspace）
df -h /workspace

# 创建虚拟环境（系统已预装PyTorch 2.2.0）
# 选项1：使用系统site-packages（推荐，复用系统PyTorch，节省空间）
python3 -m venv --system-site-packages /workspace/videorag_env

# 选项2：完全隔离的虚拟环境（需要重新安装PyTorch）
# python3 -m venv /workspace/videorag_env

# 选项3：放在系统卷上（如果网络卷访问速度较慢）
# python3 -m venv --system-site-packages /root/videorag_env

# 激活虚拟环境
source /workspace/videorag_env/bin/activate
# 如果使用选项2，则使用：source /root/videorag_env/bin/activate

# 验证Python版本
python --version  # 应该显示 Python 3.10.12
```

### 步骤 4: 克隆VideoRAG项目

```bash
# 使用网络卷创建工作目录（重要：避免占用系统卷空间）
# RunPod的网络卷通常挂载在 /workspace，有大量可用空间
cd /workspace

# 克隆VideoRAG仓库
git clone https://github.com/HKUDS/VideoRAG.git
cd VideoRAG/VideoRAG-algorithm
```

### 步骤 5: 安装核心依赖包

```bash
# 确保虚拟环境已激活（根据步骤3的选择使用对应路径）
source /workspace/videorag_env/bin/activate
# 如果虚拟环境在系统卷：source /root/videorag_env/bin/activate

# 如果使用 --system-site-packages 创建虚拟环境，系统PyTorch已可用，只需安装numpy和其他依赖
# 如果使用完全隔离的虚拟环境，需要安装PyTorch

# 安装核心数值计算库
# 注意：numpy 1.26.4在某些源不可用，使用1.26.3
pip install numpy==1.26.3

# 检查PyTorch是否可用（如果使用--system-site-packages应该可用）
python -c "import torch; print(f'PyTorch {torch.__version__} from {torch.__file__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || echo "PyTorch not found, need to install"

# 如果PyTorch不可用（完全隔离的虚拟环境），需要安装：
# 注意：官方文档要求torchvision==0.16.2，但系统预装的是0.17.0
# 选项1：使用系统PyTorch（推荐）- 使用 --system-site-packages 创建虚拟环境
# 选项2：安装官方要求的版本（如果必须隔离环境）
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# 选项3：安装与系统相同的版本（2.2.0+cu121，但torchvision版本不匹配）
# pip install torch==2.2.0 torchvision==0.16.2 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 安装加速和量化库
pip install accelerate==0.30.1
pip install bitsandbytes==0.43.1

# 视频处理工具
pip install moviepy==1.0.3
pip install git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git@3fcf5c9039de97f6ff5528ee4a9dce903c5979b3

# 多模态和视觉库
pip install timm ftfy regex einops fvcore eva-decord==0.6.1 iopath matplotlib types-regex cartopy

# 音频处理和向量数据库
pip install ctranslate2==4.4.0 faster_whisper==1.0.3 neo4j hnswlib xxhash nano-vectordb

# 语言模型和工具
pip install transformers==4.37.1
pip install tiktoken openai tenacity
pip install ollama==0.5.3
```

### 步骤 6: 检查磁盘空间（使用网络卷）

```bash
# 检查网络卷空间（模型文件应该下载到网络卷）
df -h /workspace

# 检查系统卷空间（虚拟环境可能在这里）
df -h /

# 清理pip缓存以释放系统空间（可选，但推荐）
source /workspace/videorag_env/bin/activate
pip cache purge

# 注意：所有模型文件将下载到网络卷 /workspace，不会占用系统卷空间
```

### 步骤 7: 下载模型检查点（到网络卷）

```bash
# 确保在VideoRAG-algorithm目录下（应该在网络卷上）
cd /workspace/VideoRAG/VideoRAG-algorithm

# 验证当前在正确的目录和网络卷上
pwd  # 应该显示 /workspace/VideoRAG/VideoRAG-algorithm
df -h .  # 应该显示网络卷的空间信息

# 注意：模型文件很大，但网络卷通常有充足空间（几百GB到TB级别）
# 如果下载失败，检查网络卷空间：df -h /workspace

# 下载MiniCPM-V模型（使用标准git clone，git-lfs会自动处理大文件）
git clone https://huggingface.co/openbmb/MiniCPM-V-2_6-int4

# 下载Whisper模型
git clone https://huggingface.co/Systran/faster-distil-whisper-large-v3

# 下载ImageBind检查点
mkdir -p .checkpoints
cd .checkpoints
wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
cd ..
```

### 步骤 8: 验证安装

```bash
# 激活虚拟环境（根据步骤3的选择使用对应路径）
source /workspace/videorag_env/bin/activate
# 如果虚拟环境在系统卷：source /root/videorag_env/bin/activate

# 测试关键依赖导入
python3 << EOF
import torch
import numpy as np
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ NumPy {np.__version__}")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except:
    print("✗ Transformers not installed")

try:
    import moviepy
    print(f"✓ MoviePy installed")
except:
    print("✗ MoviePy not installed")

try:
    import faster_whisper
    print(f"✓ Faster Whisper installed")
except:
    print("✗ Faster Whisper not installed")
EOF
```

### 步骤 9: 验证目录结构

```bash
cd /workspace/VideoRAG/VideoRAG-algorithm
ls -la

# 应该看到以下结构：
# - .checkpoints/ (包含imagebind_huge.pth)
# - faster-distil-whisper-large-v3/
# - MiniCPM-V-2_6-int4/
# - videorag/
# - README.md
# 等文件
```

### 步骤 10: 测试VideoRAG导入

```bash
source /workspace/videorag_env/bin/activate
cd /workspace/VideoRAG/VideoRAG-algorithm

python3 << EOF
import sys
sys.path.insert(0, '.')

try:
    from videorag import VideoRAG, QueryParam
    print("✓ VideoRAG模块导入成功")
except Exception as e:
    print(f"✗ VideoRAG模块导入失败: {e}")
EOF
```

## 注意事项

### PyTorch版本兼容性
- **系统已预装**: PyTorch 2.2.0+cu121, torchvision 0.17.0+cu121, torchaudio 2.2.0
- **官方文档要求**: PyTorch 2.1.2 + torchvision 0.16.2
- **重要**: torchvision版本兼容性很关键，VideoRAG可能依赖特定版本的torchvision API
- **推荐方案**: 
  1. **使用 `--system-site-packages` 创建虚拟环境**（推荐），复用系统PyTorch，节省空间
  2. 如果VideoRAG导入失败（torchvision版本问题），可以尝试：
     - 在虚拟环境中单独安装torchvision==0.16.2覆盖系统版本
     - 或降级到官方要求的完整版本
  3. 如果必须使用完全隔离环境，参考故障排查部分降级PyTorch

### Python版本
- 系统Python 3.10.12，文档要求3.11
- **建议**: 3.10应该兼容，如果遇到问题再考虑升级

### 环境变量设置
- 确保设置OPENAI_API_KEY（如果使用OpenAI API）：
  ```bash
  export OPENAI_API_KEY="your-api-key-here"
  ```

### 磁盘空间和存储位置
- **重要**: 模型检查点非常大，需要大量磁盘空间
  - MiniCPM-V-2_6-int4: 约5.7GB
  - faster-distil-whisper-large-v3: 约1.5GB
  - imagebind_huge.pth: 约2.5GB
  - **总计需要至少15-20GB可用空间**（不包括虚拟环境和代码）
- **必须使用网络卷存储模型**：
  - RunPod的网络卷通常挂载在 `/workspace`，有几百GB到TB级别的空间
  - **所有模型文件必须下载到网络卷**，不要放在系统卷（默认只有20GB）
  - 检查网络卷：`df -h /workspace`
  - 项目代码和模型都应放在 `/workspace` 目录下

## 快速启动脚本

创建一个快速启动脚本：

```bash
cat > /workspace/start_videorag.sh << 'EOF'
#!/bin/bash
source /workspace/videorag_env/bin/activate
cd /workspace/VideoRAG/VideoRAG-algorithm
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
python3 -c "from videorag import VideoRAG; print('VideoRAG ready!')"
EOF

chmod +x /workspace/start_videorag.sh
```

## 故障排查

### 如果遇到PyTorch/torchvision版本问题
```bash
# 如果VideoRAG导入失败，提示"No module named 'torchvision.transforms.functional_tensor'"
# 可能是torchvision版本不兼容（系统预装0.17.0，VideoRAG需要0.16.2）

# 方案1：在虚拟环境中覆盖安装torchvision 0.16.2（如果使用--system-site-packages）
# 虚拟环境中的包会优先于系统包
pip install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 方案2：如果方案1失败，降级到官方文档要求的完整版本
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 方案3：重新创建虚拟环境并使用--system-site-packages（推荐先尝试）
deactivate
rm -rf /workspace/videorag_env
python3 -m venv --system-site-packages /workspace/videorag_env
source /workspace/videorag_env/bin/activate
# 然后只安装torchvision==0.16.2覆盖系统版本
pip install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

### 如果git-lfs下载失败或磁盘空间不足
```bash
# 检查磁盘空间
df -h /

# 清理pip缓存
pip cache purge

# 检查git-lfs是否正确安装
git lfs version

# 如果磁盘空间不足，检查是否使用了网络卷：
df -h /workspace  # 检查网络卷空间
df -h /          # 检查系统卷空间

# 确保所有模型文件都在网络卷上：
cd /workspace/VideoRAG/VideoRAG-algorithm

# 如果git clone失败，可以尝试使用git lfs pull手动下载大文件：
cd faster-distil-whisper-large-v3
git lfs pull
```

### 如果依赖冲突
```bash
# 创建全新的虚拟环境重新安装
deactivate
rm -rf /workspace/videorag_env
# 或如果虚拟环境在系统卷：rm -rf /root/videorag_env

python3 -m venv /workspace/videorag_env
source /workspace/videorag_env/bin/activate
# 然后重新执行步骤5
```

## 下一步

安装完成后，可以：
1. 运行示例代码测试VideoRAG功能
2. 配置API密钥（OpenAI或其他LLM服务）
3. 开始处理视频文件

参考主README.md和VideoRAG-algorithm/README.md获取更多使用说明。

