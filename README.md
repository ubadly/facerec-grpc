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

- **并发安全**：SeetaFace 对象非线程安全，服务端用引擎池（默认 6 个实例）串行化每个引擎的访问，支持多路并发请求。
- **镜像精简**：多阶段构建，运行镜像约 **369 MB**，仅保留 3 个必需模型与 5 个必需动态库。
- **跨语言**：gRPC + Protocol Buffers，客户端可用 Go 之外的任意语言重写。
- **灵活配置**：设备（CPU/GPU）、引擎池大小、监听地址、模型路径均可通过命令行或环境变量配置。

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
#   输出示例：SeetaFace gRPC server 监听于 0.0.0.0:50051  设备=cpu  引擎池=6

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

### 3. 构建 Go 客户端

```bash
cd facerec-grpc/client
go mod download      # 下载依赖（google.golang.org/grpc 等）
go build -o facerec .
```

构建产物为 `client/facerec` 可执行文件。

> 也可以不编译，直接用 `go run . <command> ...` 运行。

---

## 配置方法

项目配置分为**服务端配置**与**客户端配置**两部分。

### 服务端配置

服务端配置通过**命令行参数**或**环境变量**传入，命令行优先级更高。

**方式一：命令行参数**（推荐，见 [服务端参数表](#服务端参数)）

```bash
./server --device cpu --pool 6 --addr 0.0.0.0:50051
```

**方式二：环境变量**

| 环境变量 | 说明 | 取值 | 默认 |
|----------|------|------|------|
| `SEETA_DEVICE` | 预设计算设备 | `cpu` / `gpu` / `auto` | `cpu` |

优先级：`命令行 --device` > `环境变量 SEETA_DEVICE` > 默认值 `cpu`。

在 Docker 中修改配置最方便的方式是覆盖启动命令：

```bash
docker run -d --name facerec -p 50051:50051 \
  facerec-grpc:latest ./server --device cpu --pool 8 --addr 0.0.0.0:50051
```

或通过 `-e` 设置环境变量：

```bash
docker run -d --name facerec -p 50051:50051 -e SEETA_DEVICE=cpu facerec-grpc:latest
```

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
[a.jpg] 特征维度=1024  解码=18ms  提取=2.7s
[b.jpg] 特征维度=1024  解码=19ms  提取=2.7s
相似度=0.7231  比对=12ms
```

### `bench` — 压测服务端

使用指定图片对服务端做并发压测，输出 QPS 与延迟分位数（解码时间已排除，只测服务端推理 + 网络）。

```bash
./facerec bench -img photo.jpg -c 10 -n 100
```

输出示例：

```
测试图片: photo.jpg (1080x1440)
并发数=10  总请求数=100

===== 压测结果 =====
总耗时: 29.3s
成功: 100  失败: 0
QPS: 3.41
平均延迟: 2.9s
最小: 2.6s  最大: 3.4s
P50: 2.9s  P90: 3.1s  P95: 3.2s  P99: 3.4s
```

---

## 参数详细说明

### 服务端参数

服务端同时支持 `--flag value` 与 `--flag=value` 两种写法；`--device` 另支持短别名 `-d`。

| 参数 | 类型 | 默认值 | 是否必填 | 说明 | 示例 |
|------|------|--------|----------|------|------|
| `--device` | `string`（`cpu`/`gpu`/`auto`） | `cpu` | 否 | 计算设备。`gpu` 需 GPU 版 SDK 与 NVIDIA GPU | `--device gpu` |
| `--gpu-id` | `int` | `0` | 否 | GPU 显卡编号，仅 `--device gpu` 时生效 | `--gpu-id 0` |
| `--pool` | `int` | `6` | 否 | 推理引擎实例数，建议设为物理核数；小于 1 时自动置 1 | `--pool 8` |
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

# 用自定义参数重新启动（8 个引擎、监听 50052）
docker run -d --name facerec -p 50052:50052 \
  facerec-grpc:latest ./server --device cpu --pool 8 --addr 0.0.0.0:50052

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

### Q6：推理速度慢（约 2.7 秒/次）

**原因**：当前为 **CPU 推理**，且识别模型较大（98 MB）。CPU 纯推理约 `2.7s/次`（QPS≈2）。

**优化方向**：

- 增大 `--pool` 提升并发吞吐（不降低单次延迟）；
- 使用 GPU 部署：需要 **GPU 版 SeetaFace SDK** 与 NVIDIA GPU + CUDA 驱动（当前 `seeta-linux/` 为 CPU 版）。

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

| 指标 | 数值 | 说明 |
|------|------|------|
| 运行镜像大小 | 约 369 MB | 精简后（原 812 MB） |
| 单次 CPU 推理延迟 | 约 2.7 s | 含检测 + 特征点 + 识别 |
| CPU QPS | 约 2 | 单实例串行，`--pool` 可提升并发吞吐 |
| 特征维度 | 默认 1024 | 由 `face_recognizer.csta` 决定 |
| gRPC 消息上限 | 64 MB | 收发均 64 MB |
| 默认监听端口 | 50051 | 可通过 `--addr` 修改 |

> 延迟主要受限于无 GPU 与较大的识别模型（98 MB），与镜像精简无关。如需低延迟，请部署 GPU 版 SDK。

---

## 相关参考

- SeetaFace 官方：人脸检测 `SeetaFaceDetector600`、特征点 `SeetaFaceLandmarker600`、识别 `SeetaFaceRecognizer610`。

---

## 鸣谢

- 模型与能力提供：<https://github.com/seetafaceengine/SeetaFace6>
- 编译后的二进制文件提供：<https://github.com/ViewFaceCore/SeetaFace6OpenBinary>
