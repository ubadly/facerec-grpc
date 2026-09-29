#include <grpcpp/grpcpp.h>

#include "facerec.grpc.pb.h"

#include "seeta/FaceDetector.h"
#include "seeta/FaceLandmarker.h"
#include "seeta/FaceRecognizer.h"

#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using facerec::CompareRequest;
using facerec::CompareResponse;
using facerec::ExtractRequest;
using facerec::ExtractResponse;
using facerec::FaceRecognition;

// 单个推理引擎：检测 + 特征点 + 识别。
// 注意：SeetaFace 对象不是线程安全的，同一时刻只能被一个请求使用。
class Engine {
 public:
  Engine(const SeetaModelSetting& det, const SeetaModelSetting& lm,
         const SeetaModelSetting& fr)
      : detector_(det), landmarker_(lm), recognizer_(fr) {}

  seeta::FaceDetector detector_;
  seeta::FaceLandmarker landmarker_;
  seeta::FaceRecognizer recognizer_;
};

// 引擎池：多个 Engine 实例 + 空闲队列，实现并发安全复用。
class EnginePool {
 public:
  EnginePool(int size, const SeetaModelSetting& det,
             const SeetaModelSetting& lm, const SeetaModelSetting& fr) {
    for (int i = 0; i < size; ++i) {
      engines_.push_back(std::make_unique<Engine>(det, lm, fr));
      idle_.push(engines_.back().get());
    }
  }

  // 租借一个引擎；返回的 Lease 析构时自动归还。
  class Lease {
   public:
    Lease(EnginePool& pool, Engine* e) : pool_(pool), engine_(e) {}
    ~Lease() {
      if (engine_) pool_.release(engine_);
    }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& o) noexcept : pool_(o.pool_), engine_(o.engine_) {
      o.engine_ = nullptr;
    }
    Engine* operator->() const { return engine_; }

   private:
    EnginePool& pool_;
    Engine* engine_;
  };

  Lease acquire() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this] { return !idle_.empty(); });
    Engine* e = idle_.front();
    idle_.pop();
    return Lease(*this, e);
  }

 private:
  void release(Engine* e) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      idle_.push(e);
    }
    cv_.notify_one();
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<Engine*> idle_;
  std::vector<std::unique_ptr<Engine>> engines_;
};

class FaceRecognitionServiceImpl final : public FaceRecognition::Service {
 public:
  explicit FaceRecognitionServiceImpl(EnginePool& pool) : pool_(pool) {}

  Status Extract(ServerContext* /*ctx*/, const ExtractRequest* req,
                 ExtractResponse* resp) override {
    auto engine = pool_.acquire();

    const auto& img = req->image();
    auto* p = reinterpret_cast<const unsigned char*>(img.data().data());
    SeetaImageData s{img.width(), img.height(), img.channels(),
                     const_cast<unsigned char*>(p)};

    SeetaFaceInfoArray faces = engine->detector_.detect(s);
    for (int i = 0; i < faces.size; ++i) {
      SeetaFaceInfo& f = faces.data[i];

      std::vector<SeetaPointF> pts(engine->landmarker_.number());
      engine->landmarker_.mark(s, f.pos, pts.data());

      std::vector<float> feat(engine->recognizer_.GetExtractFeatureSize());
      if (!engine->recognizer_.Extract(s, pts.data(), feat.data())) continue;

      auto* out = resp->add_faces();
      out->set_x(f.pos.x);
      out->set_y(f.pos.y);
      out->set_width(f.pos.width);
      out->set_height(f.pos.height);
      out->set_score(f.score);
      out->mutable_feature()->Add(feat.begin(), feat.end());
    }
    return Status::OK;
  }

  Status Compare(ServerContext* /*ctx*/, const CompareRequest* req,
                 CompareResponse* resp) override {
    const auto& f1 = req->feature1();
    const auto& f2 = req->feature2();
    if (f1.empty() || f1.size() != f2.size()) {
      return Status(grpc::INVALID_ARGUMENT, "feature size mismatch");
    }
    auto engine = pool_.acquire();
    resp->set_similarity(
        engine->recognizer_.CalculateSimilarity(f1.data(), f2.data()));
    return Status::OK;
  }

 private:
  EnginePool& pool_;
};

// ---- 设备类型解析 ----

static SeetaDevice parse_device(const std::string& s) {
  if (s == "cpu" || s == "CPU") return SEETA_DEVICE_CPU;
  if (s == "gpu" || s == "GPU") return SEETA_DEVICE_GPU;
  return SEETA_DEVICE_AUTO;
}

static const char* device_name(SeetaDevice d) {
  if (d == SEETA_DEVICE_CPU) return "cpu";
  if (d == SEETA_DEVICE_GPU) return "gpu";
  return "auto";
}

static void print_usage(const char* prog) {
  std::cerr << "用法: " << prog << " [选项]\n"
            << "  --device <cpu|gpu|auto>   计算设备 (默认 cpu)\n"
            << "  --gpu-id <N>              GPU 显卡编号, 仅 device=gpu 时生效 (默认 0)\n"
            << "  --pool <N>                推理引擎实例数, 建议=物理核数 (默认 6)\n"
            << "  --addr <ip:port>          监听地址 (默认 0.0.0.0:50051)\n"
            << "  --det <path>              人脸检测模型 (默认 model/face_detector.csta)\n"
            << "  --lm  <path>              特征点模型 (默认 model/face_landmarker_pts5.csta)\n"
            << "  --fr  <path>              人脸识别模型 (默认 model/face_recognizer.csta)\n"
            << "\n环境变量: SEETA_DEVICE 可预设设备 (命令行 --device 优先级更高)\n";
}

int main(int argc, char** argv) {
  std::string det_path = "model/face_detector.csta";
  std::string lm_path = "model/face_landmarker_pts5.csta";
  std::string fr_path = "model/face_recognizer.csta";
  std::string addr = "0.0.0.0:50051";
  int gpu_id = 0;
  int pool_size = 6;

  // 设备优先级: 命令行 --device > 环境变量 SEETA_DEVICE > 默认 cpu
  std::string device_str = "cpu";
  if (const char* env = std::getenv("SEETA_DEVICE")) {
    if (*env) device_str = env;
  }

  // 解析命令行参数 (同时支持 --flag value 和 --flag=value 两种写法)
  auto value_of = [&](int& i) -> std::string {
    return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
  };

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--device" || a == "-d") {
      device_str = value_of(i);
    } else if (a.rfind("--device=", 0) == 0) {
      device_str = a.substr(9);
    } else if (a == "--gpu-id") {
      gpu_id = std::stoi(value_of(i));
    } else if (a.rfind("--gpu-id=", 0) == 0) {
      gpu_id = std::stoi(a.substr(9));
    } else if (a == "--pool") {
      pool_size = std::stoi(value_of(i));
    } else if (a.rfind("--pool=", 0) == 0) {
      pool_size = std::stoi(a.substr(7));
    } else if (a == "--addr") {
      addr = value_of(i);
    } else if (a.rfind("--addr=", 0) == 0) {
      addr = a.substr(7);
    } else if (a == "--det") {
      det_path = value_of(i);
    } else if (a.rfind("--det=", 0) == 0) {
      det_path = a.substr(6);
    } else if (a == "--lm") {
      lm_path = value_of(i);
    } else if (a.rfind("--lm=", 0) == 0) {
      lm_path = a.substr(5);
    } else if (a == "--fr") {
      fr_path = value_of(i);
    } else if (a.rfind("--fr=", 0) == 0) {
      fr_path = a.substr(5);
    } else {
      std::cerr << "未知参数: " << a << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (pool_size < 1) pool_size = 1;

  const SeetaDevice dev = parse_device(device_str);
  const int device_id = (dev == SEETA_DEVICE_GPU) ? gpu_id : 0;

  const char* det_models[] = {det_path.c_str(), nullptr};
  SeetaModelSetting det_setting{dev, device_id, det_models};

  const char* lm_models[] = {lm_path.c_str(), nullptr};
  SeetaModelSetting lm_setting{dev, device_id, lm_models};

  const char* fr_models[] = {fr_path.c_str(), nullptr};
  SeetaModelSetting fr_setting{dev, device_id, fr_models};

  EnginePool pool(pool_size, det_setting, lm_setting, fr_setting);
  FaceRecognitionServiceImpl service(pool);

  ServerBuilder builder;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
  builder.SetMaxSendMessageSize(64 * 1024 * 1024);
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  if (!server) {
    std::cerr << "启动失败: 无法监听 " << addr << std::endl;
    return 1;
  }

  std::cout << "SeetaFace gRPC server 监听于 " << addr
            << "  设备=" << device_name(dev)
            << (dev == SEETA_DEVICE_GPU ? (" id=" + std::to_string(gpu_id)) : "")
            << "  引擎池=" << pool_size << std::endl;
  server->Wait();
  return 0;
}
