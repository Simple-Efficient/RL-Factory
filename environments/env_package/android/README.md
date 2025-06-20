# 环境搭建指南

## 1.1 查看硬件是否满足需求

查看系统架构是否为 64 位 Linux

```bash
uname -m
```

查看是否使用了 GLIBC 2.31 或以上

```bash
ldd --version
```

查看内存大小（RAM）总内存 ≥ 32Gi  推荐 64Gi 更稳

```bash
free -h
```

查看磁盘空间（推荐有 ≥ 32GB 的空间）

```bash
df -h /
```

查看 CPU 型号 + 支持 VT-x/AMD-V（虚拟化）

如果含有 `VT-x` 或 `AMD-V`，表示已启用虚拟化

查看 CPU 是否支持虚拟化（VT-x 或 AMD-V）

```bash
egrep -c '(vmx|svm)' /proc/cpuinfo
```

查看/dev/kvm是否可见

```bash
ls -l /dev/kvm
lscpu | grep -E 'Model name|Virtualization'
```

## 1.2 安装命令行版本的 Android SDK

下载linux版本呢sdk

https://developer.android.com/studio?hl=zh-cn#command-tools

commandlinetools-linux-13114758_latest.zip

解压

```bash
unzip /your_data/commandlinetools-linux-13114758_latest.zip -d cmdline-tools
```

**配置环境变量**

找到`.bashrc` 是否存在

```bash
ls -la ~/.bashrc
```

打开文件

```bash
vim ~/.bashrc
```

设置环境变量

```Bash
# Android SDK 环境变量设置
export ANDROID_SDK_ROOT=/your_data/android-sdk
export PATH=$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$PATH
```

让 `.bashrc` 立即生效：

```bash
source ~/.bashrc
```

验证是否成功

```bash
sdkmanager --list
avdmanager list devices
```

安装java

下载jdk安装包，解压:

```bash
mkdir -p /tmp/jdk-17
tar -xzf /your_data/OpenJDK17U-jdk_x64_linux_hotspot_17.0.5_8.tar.gz -C /tmp/jdk-17 --strip-components=1
cp -r /tmp/jdk-17 /your_data/jdk-17
```

配置环境变量

```bash
vim ~/.bashrc
export JAVA_HOME=/your_data/jdk-17
export PATH=$JAVA_HOME/bin:$PATH
source ~/.bashrc
```

验证

```Python
java -version
javac -version
```

## 1.3 创建 Android 虚拟设备（AVD）和安卓模拟器

安装 SDK Platform 和 System Image（Tiramisu = API 33）

查看硬件设备的信息

```bash
avdmanager list devices
```

满足要求的硬件

```Python
id: 36 or "pixel_6"
    Name: Pixel 6
    OEM : Google
```

命令行创建avd

在 **Linux 系统上使用 AVD 时**，每个 **AVD（Android 虚拟设备）实例只能被一个模拟器进程使用一次**。也就是说，**只能**被一个 emulator 使用。

创建8个设备（适配batch_size）

名字从 `AndroidWorldAvd_0` 到 `AndroidWorldAvd_17`

设备型号为 `pixel_6`

系统镜像为 `android-33`、`google_apis`、`x86_64`

SD 卡大小为 512`M`

一键启动：

```bash
bash adv.sh
```

tmux一键启动8个模拟器:

```bash
bash start_emulators.sh
```

查看是否成功

```bash
avdmanager list avd
```

端口

```bash
启动 AndroidWorldAvd_0 | grpc=8554 | telnet=5554
启动 AndroidWorldAvd_1 | grpc=8555 | telnet=5556
启动 AndroidWorldAvd_2 | grpc=8556 | telnet=5558
启动 AndroidWorldAvd_3 | grpc=8557 | telnet=5560
启动 AndroidWorldAvd_4 | grpc=8558 | telnet=5562
启动 AndroidWorldAvd_5 | grpc=8559 | telnet=5564
启动 AndroidWorldAvd_6 | grpc=8560 | telnet=5566
```

# 2. 对外暴露请求接口

## redis配置

1. Redis 服务端（Server）

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install redis-server

# CentOS / RHEL
sudo yum install redis

#启动redis
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

2. Python Redis 客户端（库）

安装redis

```bash
pip install redis
```

3. 启动之前先清除重新添加端口

   执行py脚本

   ```bash
   python redis_idle.py
   ```

   看到Redis idle port list reset: [5554, 5556, 5558, 5560, 5562, 5564, 5566, 5568]

   验证是否生效

   

   ```bash
   redis-cli
   > LRANGE android:idle_ports 0 -1
   ```

​     

## 3. 启动fastapi端口，在8000端口

/home/ipa_sudo/conda/conda3/envs/android_world/bin/换成自己的地址

```bash
 tmux new-session -d -s pool_api "PYTHONPATH=. exec /home/ipa_sudo/conda/conda3/envs/android_world/bin/gunicorn pool_api:app \
  -k uvicorn.workers.UvicornWorker \
  -w 1 --threads 1 \
  --bind 0.0.0.0:8000 --timeout 500 \
  --access-logfile - --error-logfile - \
  --graceful-timeout 300 \
  2>&1 | tee /var/log/api_server.log"
```