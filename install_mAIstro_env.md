# ⚙️ Instructions to Set Up Docker and Run the mAIstro Environment 

This guide explains how to install Docker with GPU support on Windows and Linux, and how to run the mAIstro environment using the provided Docker image and workspace.


---
## 📁 Files You Will Receive

You can download the required files from this Google Drive folder:
👉 **[Download mAIstro Files](https://drive.google.com/drive/folders/1Nk72e4olzfZEVDjXj_Yi4S_q-xBXoL1b?usp=sharing)**

Click the link and download the files:

1. `maistro_env_v1.tar` – The prebuilt Docker image with all dependencies
2. `mAIstro_workspace.zip` – A compressed folder containing:

   * Jupyter notebook(s)
   * Subfolders with all necessary data for the experiments

---
## Windows Deployment Instructions

---
## 🛠️ Step 1: Install Docker Desktop

1. Download Docker Desktop: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. During installation:

   * Enable **WSL 2** integration
   * Choose a Linux distribution (e.g., Ubuntu from Microsoft Store)
3. After installation:

   * Open Docker Desktop
   * Go to **Settings → Resources → WSL Integration**
   * Enable integration for your chosen Linux distro

---

## ⚡ Step 2: Enable GPU Support for Docker (NVIDIA GPUs only)

1. Install the latest NVIDIA GPU driver for Windows:

   * [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)

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

This step is critical if Docker inside WSL is having trouble accessing your GPU.

---

## 📦 Step 3: Import the Docker Image

1. Open PowerShell or Windows Terminal
2. Navigate to the folder where `maistro_env_v1.tar` is located:

```powershell
cd path\to\your\docker\image
```

3. Load the Docker image:

```bash
docker load -i maistro_env_v1.tar
```

---

## 🗂️ Step 4: Prepare the Workspace

1. Extract the contents of `mAIstro_workspace.zip` into a folder, for example:

```powershell
C:\Users\yourname\Documents\mAIstro_workspace
```

This folder will be used as the **working directory** in Docker.

**Important:** If all data and notebooks remain inside this directory, no additional volume mounts are needed. Everything will be accessible through `/workspace` in the container.

---

## 🚀 Step 5: Run the Docker Container

1. Open a terminal (PowerShell or Command Prompt)
2. Navigate to the extracted workspace directory:

```powershell
cd C:\Users\yourname\Documents\mAIstro_workspace
```

3. Run the container:
   
Command for Command Prompt

```powershell
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
Command for PowerShell

```powershell
docker run -it --rm `
  --gpus all `
  -p 8888:8888 `
  -v "${PWD}:/workspace" `
  -e nnUNet_raw="/workspace/nnUNet_paths/nnUNet_raw" `
  -e nnUNet_preprocessed="/workspace/nnUNet_paths/nnUNet_preprocessed" `
  -e nnUNet_results="/workspace/nnUNet_paths/nnUNet_results" `
  --shm-size=8g `
  --name maistro_temp `
  maistro_env:v1.0
```
* `--gpus all`: Enables GPU access
* `-v "%cd%":/workspace`: Mounts your workspace into the container
* `-e ...`: Sets paths to nnUNet data if it's inside the workspace folder
* `--shm-size=8g`: Allocates shared memory for better performance with large models

Once the container starts, it will print a URL like:

```
http://127.0.0.1:8888/?token=...
```

Copy and paste this into your browser to access the Jupyter environment.


---

## 🐧 Linux Deployment Instructions

The steps below assume you're using a Linux system (e.g., Ubuntu, Pop!\_OS) and want to run the mAIstro environment with GPU support using Docker.

### ✅ Step 1: Install Docker from Terminal (Linux)

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Enable and start Docker:

```bash
sudo systemctl enable docker
sudo systemctl start docker
```

To run Docker without `sudo`, add your user to the docker group:

```bash
sudo usermod -aG docker $USER
# Log out and log back in for the change to take effect
```

### ✅ Step 2: (Optional) Install NVIDIA Drivers and Toolkit for GPU Support

> 📝 Only follow this step if your system does **not already have** the NVIDIA drivers and container toolkit. Make sure to use the appropriate version for your GPU — replace `535` below if needed.

Install the driver:

```bash
sudo apt install -y nvidia-driver-535
sudo reboot
```

Install the NVIDIA Container Toolkit:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.list | \
  sed 's#https://#signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg & #' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt update
sudo apt install -y nvidia-container-toolkit
```

Configure Docker runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 📦 Step 3: Extract the Workspace

```bash
unzip mAIstro_workspace.zip -d ~/Desktop/
cd ~/Desktop/mAIstro_workspace
```

### 🐋 Step 4: Load the Docker Image

```bash
docker load -i maistro_env_v1.tar
```

### 🚀 Step 5: Run the Docker Container with GPU Support

```bash
docker run -it --rm \
  --gpus all \
  -p 8888:8888 \
  -v "$PWD:/workspace" \
  -e nnUNet_raw="/workspace/nnUNet_paths/nnUNet_raw" \
  -e nnUNet_preprocessed="/workspace/nnUNet_paths/nnUNet_preprocessed" \
  -e nnUNet_results="/workspace/nnUNet_paths/nnUNet_results" \
  --shm-size=8g \
  --name maistro_temp \
  maistro_env:v1.0
```

Once launched, you'll see a URL printed like:

```
http://127.0.0.1:8888/?token=...
```

Open it in your browser to access Jupyter.

## 📌 Notes

* If you want to reuse the container without deleting it each time, remove the `--rm` flag and add `--name my_maistro_env`:

```powershell
docker run -it --gpus all -p 8888:8888 -v "%cd%":/workspace --name my_maistro_env maistro_env_v1_latest
```

Then later you can restart it with:

```bash
docker start -ai my_maistro_env
```

---
