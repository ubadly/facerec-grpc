# ===== 阶段一：编译 gRPC 服务端 =====
FROM ubuntu:24.04 AS build

# 安装编译工具链：g++ / protoc / grpc 插件 / grpc 与 protobuf 开发库
# libgomp1 是 SeetaFace 底层 TenniS 库（libtennis.so）的 OpenMP 运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        protobuf-compiler \
        protobuf-compiler-grpc \
        libgrpc++-dev \
        libprotobuf-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 只拷贝编译所需文件（尽量精简构建上下文）
COPY proto/facerec.proto ./proto/
COPY server/server.cpp ./
COPY seeta-linux/include ./seeta/include
COPY seeta-linux/lib64 ./seeta/lib64

# 用容器内的 protoc 重新生成 pb 代码，保证与链接库版本一致
RUN protoc -I proto --cpp_out=. proto/facerec.proto \
 && protoc -I proto --grpc_out=. --plugin=protoc-gen-grpc=/usr/bin/grpc_cpp_plugin proto/facerec.proto

# 编译服务端
# 注意：SeetaFace 的 .so 引用 ts_*(TenniS) 与 orz::*(ORZ) 符号，
# 现代链接器默认不跟随 .so 的传递依赖，必须显式链接这些底层库，且顺序要放在后面。
RUN g++ -std=c++17 -O2 -Wall \
        -I seeta/include \
        server.cpp facerec.pb.cc facerec.grpc.pb.cc \
        -o server \
        -L seeta/lib64 \
        -lSeetaFaceDetector600 -lSeetaFaceLandmarker600 -lSeetaFaceRecognizer610 \
        -ltennis -lSeetaAuthorize -lORZ_static \
        $(pkg-config --cflags --libs grpc++) \
        -lprotobuf \
        -lpthread

# 收集运行所需的最小动态库集合（剔除 SDK 中与识别无关的扩展库）
RUN mkdir -p /build/runlibs && cp \
        seeta/lib64/libSeetaFaceDetector600.so \
        seeta/lib64/libSeetaFaceLandmarker600.so \
        seeta/lib64/libSeetaFaceRecognizer610.so \
        seeta/lib64/libtennis.so \
        seeta/lib64/libSeetaAuthorize.so \
        /build/runlibs/

# ===== 阶段二：精简运行镜像 =====
FROM ubuntu:24.04

# 运行所需运行时库：
# - libgomp1: SeetaFace 底层 TenniS 库（libtennis.so）的 OpenMP 运行时
# - libgrpc++1.51t64: gRPC 运行时（自动拉入 libgrpc29t64/libprotobuf32t64/libabsl）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgrpc++1.51t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /build/server ./server
COPY --from=build /build/runlibs ./seeta/lib64
# 只拷贝服务端实际用到的 3 个模型（face_detector / pts5 特征点 / face_recognizer）
COPY model/face_detector.csta model/face_landmarker_pts5.csta model/face_recognizer.csta ./model/

EXPOSE 50051

# 让动态链接器能找到 SeetaFace 的 .so（含 libtennis.so / libSeetaAuthorize.so）
ENV LD_LIBRARY_PATH=/app/seeta/lib64

# 默认 CPU 推理；需要 GPU 时改为：--device gpu --gpu-id 0
CMD ["./server", "--addr", "0.0.0.0:50051", "--device", "cpu", "--pool", "6"]
