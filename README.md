# facerec-grpc

基于 **SeetaFace 6.0** 的人脸识别 **gRPC 服务**，采用「C++ 服务端 + Go 客户端」架构，提供人脸特征提取与人脸相似度比对能力。

- **服务端**（C++）：封装 SeetaFace 的「人脸检测 + 特征点 + 识别」三个引擎，使用引擎池（`EnginePool`）实现并发安全的复用，对外暴露 gRPC 接口。
- **客户端**（Go）：通过 gRPC 调用 `Extract`（提取特征）与 `Compare`（比对相似度），并提供 `extract` / `register` / `identify` / `compare` / `bench` 五个开箱即用的子命令。

> **说明**：`client` 部分本质上是一个**示例**，主要用于验证 gRPC 通信是否正常；核心业务逻辑集中在 `server` 端。由于目前仅集成了 Linux 相关依赖（`seeta-linux/` 为 Linux 版 SDK），因此只能在 Linux 环境下运行，建议通过 Docker 部署运行。

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [安装步骤](#安装步骤)
- [配置方法](#配置方法)
- [启动与运行](#启动与运行)
- [GPU 部署](#gpu-部署)
- [命令使用说明](#命令使用说明)
- [参数详细说明](#参数详细说明)
- [gRPC 接口说明](#grpc-接口说明)
- [分步教程](#分步教程)
- [常见问题与故障排查](#常见问题与故障排查)
- [目录结构](#目录结构)
- [性能与资源占用](#性能与资源占用)

---

## 项目简介

本项目把人脸识别封装为标准的 gRPC 服务，客户端无需关心底层 SeetaFace 的 C++ 细节，只需发送图片或特征向量即可完成：

1. **特征提取**（`Extract`）：输入图片，输出图中人脸的特征向量（默认 1024 维，由识别模型决定）。
2. **相似度比对**（`Compare`）：输入两组特征向量，输出 `[0, 1]` 区间的余弦相似度，越大表示越像。

在此之上，Go 客户端封装了典型业务能力：

| 子命令 | 能力 | 对应接口 |
|--------|------|----------|
| `extract` | 提取单张图片的人脸特征，可保存为 JSON | `Extract` |
| `register` | 注册人脸到底库（本地 JSON 文件） | `Extract` |
| `identify` | 1:N 识别：在底库中找出最相似的人 | `Extract` + `Compare` |
| `compare` | 1:1 比对：计算两张图片的相似度 | `Extract` + `Compare` |
| `bench` | 压测服务端（QPS / 延迟分位数） | `Extract` |

### 核心特性

- **并发安全**：SeetaFace 对象非线程安全，服务端用引擎池串行化每个引擎的访问，支持多路并发请求。
- **自动调优**：启动时探测容器的物理核数与 CPU 配额，自动分配引擎池大小与每引擎线程数（保持「总线程 ≤ 物理核数」），无需手写参数。
- **镜像精简**：多阶段构建，运行镜像约 **369 MB**，仅保留 3 个必需模型与 5 个必需动态库。
- **跨语言**：gRPC + Protocol Buffers，客户端可用 Go 之外的任意语言重写。
- **灵活配置**：设备（CPU/GPU）、引擎池大小、每引擎线程数、日志级别、监听地址、模型路径均可通过命令行或环境变量配置。

---

## 系统架构

```
宿主机（本机 / WSL）                        Docker 容器 facerec
┌──────────────────────────┐               ┌─────────────────────────────┐
│ Go 客户端 (client/)       │               │ C++ 服务端 (server)          │
│  extract / register / ... │   gRPC        │  引擎池 EnginePool           │
│  faceclient 封装包         │ ───────────▶ │   ├ 检测 FaceDetector        │
│                           │  :50051       │   ├ 特征点 FaceLandmarker    │
│                           │  (insecure)   │   └ 识别 FaceRecognizer      │
└──────────────────────────┘               │  模型: model/*.csta          │
                                           │  动态库: seeta/lib64/*.so    │
                                           └─────────────────────────────┘
```

- **通信**：gRPC（默认明文 `insecure` 传输，监听 `50051` 端口）。
- **数据流**：客户端把图片解码为 BGR888 像素 → 发送给服务端 → 服务端检测人脸、标记特征点、提取特征 → 返回特征向量。
- **消息上限**：收发均为 64 MB，满足大图传输需求。

---

## 快速开始

```bash
# 1. 克隆项目（模型文件使用 Git LFS 托管，见下方“安装步骤”）
git clone https://github.com/ubadly/facerec-grpc facerec-grpc
cd facerec-grpc

# 2. 构建镜像（首次需数分钟）
docker build -t facerec-grpc:latest .

# 3. 启动服务端容器
docker run -d --name facerec -p 50051:50051 facerec-grpc:latest

# 4. 查看日志确认监听成功
docker logs facerec
#   输出示例（引擎池与线程数按 CPU 自动配置）：
#   [info] CPU 拓扑: 物理核=6 逻辑核=12（已考虑容器 CPU 配额）
#   [info] 配置: 设备=cpu 引擎池=3（自动） 每引擎线程数=2（自动） 监听地址=0.0.0.0:50051

# 5. 构建 Go 客户端
cd client
go build -o facerec .

# 6. 体验一次 1:1 比对
./facerec compare -img1 a.jpg -img2 b.jpg
```

---

## 安装步骤

### 0. 获取源码（克隆项目）

模型文件中的 `model/face_recognizer.csta`（约 98 MB）使用 **Git LFS** 托管，克隆前需先安装 git-lfs：

```bash
# 安装 git-lfs（首次需要）
sudo apt-get install -y git-lfs    # Ubuntu / Debian
# brew install git-lfs             # macOS

git lfs install                    # 初始化 LFS
git clone https://github.com/ubadly/facerec-grpc facerec-grpc    # 克隆仓库（会自动拉取 LFS 文件）
cd facerec-grpc
```

> 若克隆时未正确拉取 LFS 文件，可进入目录后执行 `git lfs pull` 手动补全。

### 1. 环境要求

| 组件 | 版本要求 | 用途 |
|------|----------|------|
| Docker | 20.10+（推荐 24+） | 构建并运行 C++ 服务端 |
| Go | 1.26+ | 编译客户端（仅客户端需要，服务端不需要） |
| 磁盘空间 | ≥ 2 GB | 构建上下文包含模型文件（约 300 MB）+ 镜像 |

> 说明：本机 **无需安装** SeetaFace SDK、gRPC、protobuf 等 C++ 依赖，这些全部在 Docker 镜像内完成。Go 仅用于编译客户端，容器内不需要 Go。

### 2. 构建服务端镜像

```bash
cd facerec-grpc
docker build -t facerec-grpc:latest .
```

Dockerfile 为多阶段构建：

- **阶段一（build）**：基于 `ubuntu:24.04`，安装 `g++`、`protoc`、`libgrpc++-dev` 等编译工具，用容器内 protoc 重新生成 pb 代码，再编译 `server.cpp`，最后收集运行所需的最小动态库集合。
- **阶段二（runtime）**：仅安装 `libgomp1`（SeetaFace 底层 TenniS 库的 OpenMP 运行时）和 `libgrpc++1.51t64`（gRPC 运行时，自动拉入 protobuf/absl 依赖），拷贝 `server`、动态库和 3 个模型。

> **国内加速**：Dockerfile 已内置把 apt 源切换到阿里云镜像（`mirrors.aliyun.com`），`apt-get` 安装依赖不再走国际源。若拉取基础镜像 `ubuntu:24.04` 本身较慢，可给 Docker 配置国内 registry mirror（如 `docker.m.daocloud.io`），或将 `FROM` 改为对应镜像源的 `ubuntu:24.04`。

### 3. 构建 Go 客户端

```bash
cd facerec-grpc/client
go mod download      # 下载依赖（google.golang.org/grpc 等）
go build -o facerec .
```

构建产物为 `client/facerec` 可执行文件。

> 也可以不编译，直接用 `go run . <command> ...` 运行。

> **国内加速**：首次 `go mod download` 建议先设置 Go 模块代理：`go env -w GOPROXY=https://goproxy.cn,direct`。

---

## 配置方法

项目配置分为**服务端配置**与**客户端配置**两部分。

### 服务端配置

服务端配置通过**命令行参数**或**环境变量**传入，命令行优先级更高。

**默认无需任何参数**：引擎池大小与每引擎线程数会根据容器可用的物理核数自动配置（见 [自动配置](#自动配置)）。

**方式一：命令行参数**（推荐，见 [服务端参数表](#服务端参数)）

```bash
./server --device cpu --addr 0.0.0.0:50051          # 全自动
./server --device cpu --pool 3 --threads 2          # 手动指定
```

**方式二：环境变量**

| 环境变量 | 说明 | 取值 | 默认 |
|----------|------|------|------|
| `SEETA_DEVICE` | 计算设备 | `cpu` / `gpu` / `auto` | `cpu` |
| `SEETA_POOL` | 推理引擎实例数 | ≥1 | 按 CPU 自动 |
| `SEETA_THREADS` | 每引擎计算线程数 | 1~4 | 按 CPU 自动 |
| `SEETA_LOG_LEVEL` | 日志详细程度 | `0` 仅启动与告警 / `1` 每请求摘要 / `2` 逐人脸明细 | `1` |

优先级：`命令行` > `环境变量` > `自动配置`。

在 Docker 中修改配置最方便的方式是覆盖启动命令：

```bash
docker run -d --name facerec -p 50051:50051 \
  facerec-grpc:latest ./server --device cpu --pool 3 --threads 2
```

或通过 `-e` 设置环境变量：

```bash
docker run -d --name facerec -p 50051:50051 -e SEETA_THREADS=2 facerec-grpc:latest
```

### 自动配置

服务端启动时会探测容器的 CPU 拓扑（包括 cgroup 的 CPU 配额），据此决定
**引擎池大小 × 每引擎线程数**。启动日志会打印探测结果，例如：

```
CPU 拓扑: 物理核=6 逻辑核=12（已考虑容器 CPU 配额）
配置: 设备=cpu 引擎池=3（自动） 每引擎线程数=2（自动） 监听地址=0.0.0.0:50051
推理并行度: 3 x 2 = 6 线程 / 6 物理核
```

**为什么需要这个**：SeetaFace 默认给每个检测/识别引擎开 4 个计算线程。若照搬默认
`--pool 6`，就会有 6 个引擎 × 4 线程 = **24 个线程争抢 12 个核**。实测这会让同一张
640px 图片的检测耗时从 42ms 膨胀到 **1.7 秒**，并发吞吐反而比单并发还低。

自动配置遵循两条规则：

1. **引擎池 × 每引擎线程数 ≤ 物理核数**（不是逻辑核数，超线程对浮点密集型推理收益很小）；
2. **每个引擎尽量分到 2 个线程**（单引擎收益最大的一段），且不超过 4 个——
   实测 3~4 线程是单请求的饱和点，12 线程反而慢 3~10 倍。
   只有 1~3 核这种连两个引擎都养不起的机器才会出现 1/3 线程。

| 物理核数 | 引擎池 | 每引擎线程数 | 合计线程 |
|----------|--------|--------------|----------|
| 1        | 1      | 1            | 1        |
| 2        | 1      | 2            | 2        |
| 3        | 1      | 3            | 3        |
| 4        | 2      | 2            | 4        |
| 5-16     | 核数/2 | 2            | 核数(偶数) / 核数-1(奇数) |
| >16      | 8      | 核数/8（≤4） | ≤ 核数   |

> 每个引擎实例会复制一份模型，约占用 500MB 内存，因此引擎池上限为 8。
> 若手动指定的 `--pool`/`--threads` 导致总线程数超过物理核数，启动日志会给出告警。

### 客户端配置

客户端通过 `-addr` 指定服务端地址，通过各子命令的 flag 指定其余配置（见 [客户端参数表](#客户端参数)）。默认服务端地址为 `localhost:50051`。

---

## 启动与运行

### 启动服务端（Docker）

```bash
# 前台运行（调试用，Ctrl+C 停止）
docker run --rm -p 50051:50051 facerec-grpc:latest

# 后台运行（生产用）
docker run -d --name facerec -p 50051:50051 facerec-grpc:latest
```

### 查看运行状态

```bash
docker ps                 # 查看容器是否在运行
docker logs facerec       # 查看服务端日志
docker logs -f facerec    # 持续跟踪日志
```

服务端会在 stdout 输出详细日志（`docker logs` 直接可见），包括：启动配置、引擎池逐实例初始化耗时、监听地址与实际端口、健康检查启用状态；以及每次 `Extract` 请求的来源 `peer`、图像尺寸、**排队/检测/总耗时**、异常兜底与优雅关闭日志。日志由后台线程异步写出，不阻塞请求。日志时间默认按容器时区（UTC）输出，比北京时间慢 8 小时。日志详细程度见 [Q11](#q11服务端日志都输出哪些内容)。

### 停止 / 重启 / 删除

```bash
docker stop facerec       # 停止
docker start facerec       # 重启
docker rm -f facerec       # 删除容器
```

### 运行客户端

```bash
cd facerec-grpc/client
./facerec <command> [flags]     # 已编译
# 或
go run . <command> [flags]      # 未编译
```

不带任何参数运行可查看帮助：

```bash
./facerec
```

---

## GPU 部署（可选）

> 说明：以下内容仅提供 GPU 部署思路。由于本项目开发环境没有 GPU 设备，相关内容未经实际测试验证，具体细节请以 GPU 版 SDK 官方文档为准。

服务端代码已支持 `--device gpu` / `--gpu-id` 参数，但当前仓库集成的 `seeta-linux/` 为 **CPU 版 SDK**，无法直接启用 GPU。如需 GPU 加速，需完成以下改造。

### 1. 环境要求

- NVIDIA GPU 及驱动（版本需匹配所选 CUDA 版本）
- CUDA / cuDNN（与 GPU 版 SDK 匹配的版本）
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)（使 Docker 能访问宿主机 GPU）

### 2. 替换为 GPU 版 SDK

1. 用 **GPU 版 SeetaFace SDK** 替换 `seeta-linux/` 下的 CPU 版动态库（可从 [SeetaFace6OpenBinary](https://github.com/ViewFaceCore/SeetaFace6OpenBinary) 获取 GPU 版二进制）；
2. 修改 `Dockerfile`：
   - 基础镜像改用带 CUDA 的镜像（如 `nvidia/cuda`）；
   - 编译与运行阶段改为链接 GPU 版引擎库，并在运行镜像中安装对应的 CUDA 运行时；
   - 将 GPU 版 `.so` 拷贝到 `seeta/lib64`。

### 3. 以 GPU 模式运行

```bash
docker run -d --name facerec --gpus all -p 50051:50051 \
  facerec-grpc:gpu ./server --device gpu --gpu-id 0
```

- `--gpus all`：将宿主机全部 GPU 暴露给容器（也可用 `--gpus '"device=0"'` 指定单卡）；
- `--device gpu --gpu-id 0`：让 SeetaFace 使用 GPU 推理，并选择 0 号显卡；
- `--pool` 建议根据 GPU 并发能力设置，通常可小于 CPU 场景。

> 注意：模型文件（`model/*.csta`）通常 CPU/GPU 通用，但具体以 GPU 版 SDK 说明为准。

---

## 命令使用说明

### `extract` — 提取人脸特征

从图片中提取第一张人脸的特征向量，可选保存为 JSON 文件。

```bash
./facerec extract -img photo.jpg                       # 只打印特征
./facerec extract -img photo.jpg -out feat.json        # 保存特征到文件
./facerec extract -img photo.jpg -out feat.json -name 张三  # 带人名标签
```

输出示例：

```
图片尺寸: 1080x1440
特征提取成功，维度=1024
前 10 个分量: 0.1234 -0.0456 ...
特征已保存到 feat.json
```

### `register` — 注册人脸到底库

提取图片特征并写入本地底库 JSON 文件（`face_db.json`）。同名人物会被覆盖更新。

```bash
./facerec register -name 张三 -img zhang3.jpg
./facerec register -name 李四 -img li4.jpg -db my_db.json   # 自定义底库文件
```

输出示例：

```
已注册: 张三 <- zhang3.jpg（特征维度 1024，底库当前 1 人）
```

### `identify` — 1:N 底库识别

将待识别图片与底库中所有人比对，按相似度降序输出，最高分超过阈值即判定匹配。

```bash
./facerec identify -img unknown.jpg
./facerec identify -img unknown.jpg -db my_db.json -threshold 0.65
```

输出示例：

```
底库匹配结果（按相似度降序）：
  1. 张三  相似度=0.7231
  2. 李四  相似度=0.4102

识别结果：张三（相似度 0.7231）
```

### `compare` — 1:1 图片比对

计算两张图片中人脸的相似度。

```bash
./facerec compare -img1 a.jpg -img2 b.jpg
```

输出示例：

```
[a.jpg] 特征维度=1024  解码=18ms  提取=71ms
[b.jpg] 特征维度=1024  解码=19ms  提取=69ms
相似度=0.7231  比对=420µs
```

> `比对` 是纯点积计算，与图片大小无关，通常在几百微秒（其中绝大部分是网络往返）。

### `bench` — 压测服务端

使用指定图片对服务端做并发压测，输出 QPS 与延迟分位数（解码时间已排除，只测服务端推理 + 网络）。

```bash
./facerec bench -img photo.jpg -c 10 -n 100
```

输出示例：

```
测试图片: photo.jpg (640x853)
并发数=4  总请求数=100  预热=5

===== 压测结果 =====
总耗时: 4.53s
成功: 100  失败: 0
QPS: 22.08
平均延迟: 178ms
最小: 68ms  最大: 261ms
P50: 173ms
P90: 224ms
P95: 233ms
P99: 253ms
```

> 延迟分位数明显高于单请求耗时（640px 约 83ms），差额是**排队时间**——并发数超过引擎池容量时请求会在服务端排队。
> 这是刻意保留的背压：与其让 24 个线程互相抢占把每次推理拖到 1.9 秒，不如按物理核数排队。
> 需要更低延迟就调大 `--pool`（同时按 `--pool × --threads ≤ 物理核数` 等比减小 `--threads`）。
> 服务端日志里的 `排队=` 字段就是这段等待时间，可直接据此判断是否需要加引擎。

---

## 参数详细说明

### 服务端参数

服务端同时支持 `--flag value` 与 `--flag=value` 两种写法；`--device` 另支持短别名 `-d`。

| 参数 | 类型 | 默认值 | 是否必填 | 说明 | 示例 |
|------|------|--------|----------|------|------|
| `--device` | `string`（`cpu`/`gpu`/`auto`） | `cpu` | 否 | 计算设备。`gpu` 需 GPU 版 SDK 与 NVIDIA GPU | `--device gpu` |
| `--gpu-id` | `int` | `0` | 否 | GPU 显卡编号，仅 `--device gpu` 时生效 | `--gpu-id 0` |
| `--pool` | `int` | 按 CPU 自动 | 否 | 推理引擎实例数（每个约 500MB 内存）；小于 1 时自动置 1 | `--pool 3` |
| `--threads` | `int` | 按 CPU 自动 | 否 | 每引擎计算线程数，1~4，超过 4 无收益 | `--threads 2` |
| `--log-level` | `int` | `1` | 否 | `0`=仅启动与告警（无每请求日志）/ `1`=每请求一行摘要 / `2`=含逐人脸明细 | `--log-level 0` |
| `--addr` | `string` | `0.0.0.0:50051` | 否 | gRPC 监听地址（`ip:port`） | `--addr 0.0.0.0:50051` |
| `--det` | `string` | `model/face_detector.csta` | 否 | 人脸检测模型路径 | `--det model/face_detector.csta` |
| `--lm` | `string` | `model/face_landmarker_pts5.csta` | 否 | 特征点模型路径 | `--lm model/face_landmarker_pts5.csta` |
| `--fr` | `string` | `model/face_recognizer.csta` | 否 | 人脸识别模型路径 | `--fr model/face_recognizer.csta` |

> 说明：模型路径是相对**工作目录 `/app`**（容器内）的相对路径；容器内已通过 `LD_LIBRARY_PATH=/app/seeta/lib64` 设置动态库搜索路径。

### 客户端参数

所有子命令均支持 `-addr` 指定服务端地址（默认 `localhost:50051`）。其余参数按子命令分组：

**通用**

| 参数 | 类型 | 默认值 | 是否必填 | 说明 |
|------|------|--------|----------|------|
| `-addr` | `string` | `localhost:50051` | 否 | gRPC 服务端地址 |

**`extract`**

| 参数 | 类型 | 默认值 | 是否必填 | 说明 | 示例 |
|------|------|--------|----------|------|------|
| `-img` | `string` | `""` | **是** | 图片路径 | `-img a.jpg` |
| `-out` | `string` | `""` | 否 | 特征保存为 JSON 的文件路径 | `-out feat.json` |
| `-name` | `string` | `""` | 否 | 人名标签，随特征一起保存 | `-name 张三` |

**`register`**

| 参数 | 类型 | 默认值 | 是否必填 | 说明 | 示例 |
|------|------|--------|----------|------|------|
| `-db` | `string` | `face_db.json` | 否 | 底库 JSON 文件路径 | `-db my_db.json` |
| `-name` | `string` | `""` | **是** | 人名 | `-name 张三` |
| `-img` | `string` | `""` | **是** | 图片路径 | `-img a.jpg` |

**`identify`**

| 参数 | 类型 | 默认值 | 是否必填 | 说明 | 示例 |
|------|------|--------|----------|------|------|
| `-db` | `string` | `face_db.json` | 否 | 底库 JSON 文件路径 | `-db my_db.json` |
| `-img` | `string` | `""` | **是** | 待识别图片路径 | `-img b.jpg` |
| `-threshold` | `float64` | `0.6` | 否 | 判定阈值，最高相似度高于此值才判定匹配 | `-threshold 0.65` |
| `-workers` | `int` | `8` | 否 | 并发比对底库的协程数（底库大时提速明显） | `-workers 32` |

**`compare`**

| 参数 | 类型 | 默认值 | 是否必填 | 说明 | 示例 |
|------|------|--------|----------|------|------|
| `-img1` | `string` | `a.jpg` | 否 | 第一张图片路径 | `-img1 a.jpg` |
| `-img2` | `string` | `b.jpg` | 否 | 第二张图片路径 | `-img2 b.jpg` |

**`bench`**

| 参数 | 类型 | 默认值 | 是否必填 | 说明 | 示例 |
|------|------|--------|----------|------|------|
| `-img` | `string` | `""` | **是** | 测试图片路径 | `-img a.jpg` |
| `-c` | `int` | `10` | 否 | 并发数 | `-c 10` |
| `-n` | `int` | `100` | 否 | 总请求数 | `-n 100` |
| `-warmup` | `int` | `5` | 否 | 预热请求数（不计入统计，避免首次建连/惰性初始化污染分位数） | `-warmup 10` |

---

## gRPC 接口说明

接口定义位于 `proto/facerec.proto`（proto3 语法）。

```proto
service FaceRecognition {
  rpc Extract(ExtractRequest) returns (ExtractResponse);
  rpc Compare(CompareRequest) returns (CompareResponse);
}
```

### Extract

- **输入** `ExtractRequest`：包含 `ImageData`（`data` 原始像素字节 + `width`/`height`/`channels`）。
- **输出** `ExtractResponse`：`repeated Face faces`，每张人脸含边框 `x/y/width/height`、检测置信度 `score` 和特征向量 `feature`（`repeated float`）。

### Compare

- **输入** `CompareRequest`：两组特征向量 `feature1` / `feature2`。
- **输出** `CompareResponse`：`similarity`（`float`，`[0,1]` 区间）。

> 客户端 `ExtractBGR` 约定输入为 BGR888 三通道像素（SeetaFace 要求），`decode.go` 负责把 jpeg/png 解码为 BGR。

### 健康检查（grpc.health.v1.Health）

服务端已内置 gRPC 标准健康检查服务 `grpc.health.v1.Health`，用于轻量级探活（无需发送图片），服务启动后即返回 `SERVING`。

```bash
# grpc_health_probe 方式（返回码 0 表示健康）
grpc_health_probe -addr=<host>:50051

# grpcurl 方式
grpcurl -plaintext <host>:50051 grpc.health.v1.Health/Check
# 正常输出：{"status":"SERVING"}
```

主要用于区分「网络不通」与「服务未就绪」：若探活超时/拒绝，说明端口未放行或服务未监听；若返回 `SERVING` 而业务仍失败，则问题在客户端连接配置或推理超时。

---

## 分步教程

以下教程假设已完成 [安装步骤](#安装步骤)，服务端容器 `facerec` 已启动、客户端已编译为 `client/facerec`。

### 场景一：提取单张图片的人脸特征

```bash
cd facerec-grpc/client
./facerec extract -img /path/to/photo.jpg -out feature.json -name 张三
```

然后查看保存的特征：

```bash
cat feature.json
# {"feature": [0.1234, -0.0456, ...], "name": "张三"}
```

### 场景二：注册底库 + 1:N 识别（典型人脸门禁流程）

**Step 1：注册多个人到底库**

```bash
./facerec register -name 张三 -img zhang3.jpg
./facerec register -name 李四 -img li4.jpg
./facerec register -name 王五 -img wang5.jpg
```

此时生成 `face_db.json`，结构为：

```json
{
  "persons": [
    { "name": "张三", "feature": [0.1, 0.2, ...] },
    { "name": "李四", "feature": [0.3, 0.4, ...] },
    { "name": "王五", "feature": [0.5, 0.6, ...] }
  ]
}
```

**Step 2：识别一张未知照片**

```bash
./facerec identify -img unknown.jpg
```

根据输出判定是否匹配；若不匹配可调整阈值：

```bash
./facerec identify -img unknown.jpg -threshold 0.7
```

### 场景三：1:1 人脸比对（验证是否为同一人）

```bash
./facerec compare -img1 id_card.jpg -img2 selfie.jpg
```

- 相似度越接近 1 越可能是同一人；本项目实测同一人约 `0.70~0.77`，不同人 `0.32~0.33`，建议阈值 `0.6` 左右。

### 场景四：服务端性能压测

```bash
./facerec bench -img photo.jpg -c 10 -n 100
```

通过调整 `-c`（并发）与 `-n`（总请求数）观察 QPS 与延迟变化，判断引擎池大小是否需要调整。

### 场景五：自定义服务端配置（换引擎池 / 换端口）

```bash
# 先停掉旧容器
docker rm -f facerec

# 自定义参数重新启动（4 个引擎 x 2 线程、监听 50052）
# 注意保持 pool x threads <= 物理核数，否则并发下反而更慢
docker run -d --name facerec -p 50052:50052 \
  facerec-grpc:latest ./server --device cpu --pool 4 --threads 2 --addr 0.0.0.0:50052

# 客户端指向新端口
./facerec identify -img unknown.jpg -addr localhost:50052
```

---

## 常见问题与故障排查

### Q1：启动容器后 `docker logs facerec` 报 `libgrpc++.so.1.51 not found` 等库缺失错误

**原因**：运行镜像缺少 gRPC 运行时库。

**解决**：该问题已在当前 Dockerfile 修复——运行阶段安装了 `libgrpc++1.51t64`。若使用旧镜像，请重新 `docker build`。相关缺失库包括 `libgrpc++.so.1.51`、`libgpr.so.29`、`libabsl_synchronization.so`、`libprotobuf.so.32`。

### Q2：`docker logs facerec` 报 `libgomp.so.1 not found`

**原因**：SeetaFace 底层 `libtennis.so` 依赖 OpenMP 运行时 `libgomp1`。

**解决**：运行镜像需安装 `libgomp1`（已写入 Dockerfile）。若手动部署，执行 `apt-get install -y libgomp1`。

### Q3：客户端报 `connection refused` / 连接失败

**排查步骤**：

1. 确认服务端容器在运行：`docker ps`；
2. 确认端口映射正确：`-p 50051:50051`；
3. 确认日志显示监听成功：`docker logs facerec`；
4. 客户端 `-addr` 是否指向正确地址（非本机 Docker 需用宿主机 IP 而非 `localhost`）。

### Q4：客户端报 `no face detected`

**原因**：图片中未检测到人脸。

**排查**：确认图片确有人脸且清晰、大小适中、非过度侧脸/遮挡。极小的脸或纯色背景图会导致检测失败。

### Q5：识别结果不理想（误判 / 漏判）

**排查**：

1. 调整 `-threshold` 阈值：调高减少误判（更严格），调低减少漏判（更宽松）；
2. 底库照片与待识别照片光照、角度尽量一致；
3. 注册时使用清晰正脸照。

### Q6：推理速度慢

**先看是哪一类慢**：服务端日志的每请求摘要已经拆开了排队、检测、总耗时，可直接定位：

```
Extract peer=... 图像=640x853 排队=0.0ms 检测=42ms 输出特征=1 个 总耗时=70ms
```

| 现象 | 原因 | 处理 |
|------|------|------|
| 单请求就慢（检测耗时高） | 图片太大，检测耗时随像素面积线性增长 | 上传前把图片缩到 640px 宽左右（1280px 的 249ms 可降到 83ms，相似度几乎不变） |
| 排队耗时高 | 并发超过引擎池容量 | 调大 `--pool`，同时确认 `--pool × --threads ≤ 物理核数` |
| 检测耗时在并发时暴涨 | 线程超配（旧版本固定 pool=6 × 4 线程 = 24 线程） | 升级到当前版本，或手动 `--pool 3 --threads 2` |
| 整体都慢 | CPU 推理、识别模型 98 MB | 部署 GPU 版 SDK（需 NVIDIA GPU + CUDA） |

> 注意：`Compare` 是纯点积计算（服务端 <1µs，端到端约 0.5ms 主要是网络往返），
> **不占用推理引擎**，不会与 `Extract` 抢资源。
> 若 1:N 检索慢，调大客户端的 `identify -workers`（200 人底库实测可从 3.5s 降到 0.3s）。

### Q7：`./facerec` 提示 `no such file or directory` 或无法执行

**原因**：客户端尚未编译。

**解决**：`cd client && go build -o facerec .`，或改用 `go run . <command> ...`。

### Q8：`go build` 报依赖下载失败

**解决**：设置 Go 模块代理后重试：

```bash
go env -w GOPROXY=https://goproxy.cn,direct
go mod download
```

### Q9：修改了 `facerec.proto` 后需要做什么

需要重新生成 pb 代码并重新编译两端：

```bash
# 重新生成 C++ pb（在构建容器内由 Dockerfile 自动完成，直接重建镜像即可）
docker build -t facerec-grpc:latest .

# 重新生成 Go pb（在宿主机执行，需安装 protoc 与 grpc 插件）
protoc -I proto --go_out=. --go-grpc_out=. proto/facerec.proto
cd client && go build -o facerec .
```

### Q10：后端调用报 `DeadlineExceeded` / `received context error while waiting for new LB policy update`

**原因**：这是 gRPC 客户端侧错误，表示客户端在超时前未能建立到服务端的连接，请求大概率根本没到达服务端。通常是网络层问题（云安全组未放行端口、防火墙、端口映射错误），而非推理慢。

**排查**：
1. 先做健康探活（见上文「健康检查」）：
   - `grpc_health_probe -addr=<host>:50051` 超时/拒绝 → 端口不通，检查安全组/防火墙/端口映射；
   - 返回 `SERVING` → 服务正常，问题在客户端连接配置或推理超时。
2. 查看 `docker logs -f facerec`：
   - 停在「引擎池初始化」→ 模型加载卡住或授权校验失败；
   - 显示「监听成功」但始终无「Extract 请求到达」→ 请求没进来，网络不通；
   - 有「Extract 请求到达」→ 请求已到达，问题在处理耗时或客户端超时配置。

### Q11：服务端日志都输出哪些内容

服务端会在 stdout 输出详细日志（`docker logs` 直接可见），主要包括：

- 启动配置、模型文件存在性检查、引擎池逐实例初始化耗时、监听地址与实际绑定端口、健康检查启用状态；
- 每次 `Extract` 请求一行摘要：来源 `peer`、图像尺寸、**排队耗时**、检测耗时、输出人脸数、总耗时；
- SeetaFace 异常兜底日志，以及 `SIGINT`/`SIGTERM` 优雅关闭日志（退出码 0）。

日志量用 `--log-level` 控制（也可用 `SEETA_LOG_LEVEL`）：

| 级别 | 内容 | 适用场景 |
|------|------|----------|
| `0` | 仅启动信息、告警与错误 | 极高 QPS，不愿保留请求日志时使用 |
| `1` | 每请求一行摘要（默认） | 常规线上运行，可按行统计延迟 |
| `2` | 追加逐张人脸的检测/特征点/识别细节 | 排障 |

> 日志由后台线程异步写出，请求线程只做一次入队，正常量级下开销很小。
> 队列超过 65536 条时会丢弃最旧的并打一条告警（宁可丢日志也不拖慢请求）。

> 注意：容器默认时区为 UTC，日志时间比北京时间慢 8 小时，对比业务日志时需 +8 小时（或为容器设置 `TZ=Asia/Shanghai`）。

---

## 目录结构

```
facerec-grpc/
├── Dockerfile                   # 多阶段构建脚本（编译 + 精简运行）
├── .dockerignore                # 构建上下文忽略规则
├── README.md                    # 本文件
├── proto/
│   └── facerec.proto            # gRPC 接口定义（proto3）
├── server/
│   └── server.cpp               # C++ gRPC 服务端（唯一服务端源码）
├── seeta-linux/                 # Linux 版 SeetaFace 6.0 SDK
│   ├── include/                 #   C++ 头文件（seeta/*.h）
│   └── lib64/                   #   动态库（.so）与静态库（.a）
├── model/                       # 推理模型（构建时仅需 3 个）
│   ├── face_detector.csta         #   人脸检测模型
│   ├── face_landmarker_pts5.csta  #   5 点特征点模型
│   └── face_recognizer.csta       #   人脸识别模型（特征提取）
└── client/                      # Go 客户端
    ├── go.mod / go.sum          #   Go 依赖管理
    ├── main.go                  #   单一入口（extract/register/identify/compare/bench）
    ├── faceclient/              #   gRPC 客户端封装包
    │   ├── client.go            #     gRPC 连接与 Extract/Compare 调用
    │   ├── db.go                #     底库读写、注册与识别逻辑
    │   └── decode.go            #     图片解码为 BGR 像素
    └── facerec/                 #   由 protoc 生成的 pb 代码
        ├── facerec.pb.go
        └── facerec_grpc.pb.go
```

> 说明：`face_db.json`（底库）、`feature.json`（特征）等文件由客户端在运行时生成，不属于源码。

---

## 性能与资源占用

实测环境：Intel i5-10400（6 核 12 线程，WSL2），Docker 无 CPU 限制，客户端与服务端同机。

| 指标 | 数值 | 说明 |
|------|------|------|
| 运行镜像大小 | 约 369 MB | 精简后（原 812 MB） |
| 单请求延迟（320px） | 约 44 ms | 无并发、无排队 |
| 单请求延迟（640px） | 约 83 ms | 无并发、无排队 |
| 单请求延迟（1280px） | 约 249 ms | 无并发、无排队 |
| 并发吞吐（640px，c=4） | 约 22 QPS | 自动配置 3 引擎 × 2 线程 |
| 并发吞吐（320px，c=8） | 约 18 QPS | 受单请求检测耗时支配 |
| `Compare` 延迟 | 约 0.5 ms / 次 | 纯点积本身 <1 µs，其余是网络往返 |
| 特征维度 | 默认 1024 | 由 `face_recognizer.csta` 决定 |
| 引擎实例内存 | 约 500 MB | 模型副本，引擎池越大占用越高 |
| gRPC 消息上限 | 64 MB | 收发均 64 MB |
| 默认监听端口 | 50051 | 可通过 `--addr` 修改 |

各分辨率与线程数下的单请求 P50（SeetaFace 直调，无 gRPC 开销）：

| 每引擎线程数 | 320px | 640px | 1280px |
|--------------|-------|-------|--------|
| 1 | 80 ms | 118 ms | 318 ms |
| 2 | 49 ms | 82 ms | 269 ms |
| **3** | **39 ms** | **80 ms** | **251 ms** |
| 4 | 37 ms | 80 ms | 256 ms |
| 8 | 43 ms | 94 ms | 281 ms |
| 12 | 210 ms | 761 ms | 432 ms |

> **延迟主要来自人脸检测**，且随像素面积近似线性增长。若业务允许，把图片缩放到
> 640px 宽再上传，延迟可从 249ms 降到 83ms，相似度几乎不变（同图缩放后比对
> 相似度仍在 0.99 以上）。
>
> 早期版本固定 `--pool 6` 且未限制每引擎线程数（默认 4），导致 24 线程争抢 12 核，
> 并发下 640px 检测耗时膨胀到 1.7 秒。现版本已自动规避，详见 [自动配置](#自动配置)。

### 关于 WSL2 下的长尾延迟

如果你在 **WSL2 里用 `localhost:50051` 压测**，可能会看到少量请求耗时 0.7s / 1.6s /
3.2s / 6.6s / 13.4s —— 这是 TCP 重传超时（RTO）的指数退避序列，发生在
**Windows↔WSL 的 localhost 端口转发**上，与服务端无关：

- 服务端日志显示这些请求之间存在几秒的到达空档；
- `ss -tin` 能看到 `cwnd:1 backoff:3~5 ... lost:34`；
- 换成容器 IP（`docker inspect facerec` 的 `IPAddress`）或宿主网卡 IP 压测，
  200 次请求 **0 次** 超过 500ms。

生产环境用真实网卡通信不受影响；本机调试想测准，请直连容器 IP。

---

## 相关参考

- SeetaFace 官方：人脸检测 `SeetaFaceDetector600`、特征点 `SeetaFaceLandmarker600`、识别 `SeetaFaceRecognizer610`。

---

## 鸣谢

- 模型与能力提供：<https://github.com/seetafaceengine/SeetaFace6>
- 编译后的二进制文件提供：<https://github.com/ViewFaceCore/SeetaFace6OpenBinary>
