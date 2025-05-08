# ⚙️ Instructions to Set Up Docker and Run the mAIstro Environment on Windows

This guide explains how to install Docker with GPU support on Windows and run the mAIstro environment using the provided Docker image and workspace.

---

## 📁 Files You Will Receive

You can request access to the required files from this Google Drive folder:
👉 **[Request Access to mAIstro Files](https://drive.google.com/drive/folders/1Nk72e4olzfZEVDjXj_Yi4S_q-xBXoL1b?usp=sharing)**

Click the link and press "Request access" using your Google account.

1. `maistro_env_v1.tar` – The prebuilt Docker image with all dependencies
2. `mAIstro_workspace.zip` – A compressed folder containing:
   - Jupyter notebook(s)
   - Subfolders with all necessary data for the experiments

---

## 🛠️ Step 1: Install Docker Desktop

1. Download Docker Desktop: https://www.docker.com/products/docker-desktop
2. During installation:
   - Enable **WSL 2** integration
   - Choose a Linux distribution (e.g., Ubuntu from Microsoft Store)
3. After installation:
   - Open Docker Desktop
   - Go to **Settings → Resources → WSL Integration**
   - Enable integration for your chosen Linux distro

---

## ⚡ Step 2: Enable GPU Support for Docker (NVIDIA GPUs only)

1. Install the latest NVIDIA GPU driver for Windows:
   - https://www.nvidia.com/Download/index.aspx

2. Install the CUDA toolkit inside WSL2:

```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
```

3. Verify that the GPU is visible inside WSL:

```bash
nvidia-smi
```

You should see a table with your GPU info. If not, troubleshoot your WSL or driver setup.

4. (Recommended) Install NVIDIA Container Toolkit inside WSL2 to ensure full Docker GPU integration:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.list | \
  sed 's#https://#signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg & #' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt update
sudo apt install -y nvidia-container-toolkit
```

Then configure Docker runtime and restart Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 📦 Step 3: Import the Docker Image

```powershell
cd path\to\your\docker\image
docker load -i maistro_env_v1.tar
```

---

## 🗂️ Step 4: Prepare the Workspace

Extract `mAIstro_workspace.zip` to a folder, for example:

```powershell
C:\Users\yourname\Documents\mAIstro_workspace
```

---

## 🚀 Step 5: Run the Docker Container

```powershell
cd C:\Users\yourname\Documents\mAIstro_workspace

docker run -it --rm ^
  --gpus all ^
  -p 8888:8888 ^
  -v "%cd%":/workspace ^
  -e nnUNet_raw="/workspace/nnUNet_paths/nnUNet_raw" ^
  -e nnUNet_preprocessed="/workspace/nnUNet_paths/nnUNet_preprocessed" ^
  -e nnUNet_results="/workspace/nnUNet_paths/nnUNet_results" ^
  --shm-size=8g ^
  --name maistro_temp ^
  maistro_env:v1.0
```

---


## 📌 Notes

- To keep the container:
```powershell
docker run -it --gpus all -p 8888:8888 -v "%cd%":/workspace --name my_maistro_env maistro_env_v1_latest
```

Then later:
```bash
docker start -ai my_maistro_env
```
