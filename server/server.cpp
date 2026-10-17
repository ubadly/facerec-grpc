#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "facerec.grpc.pb.h"

#include "seeta/FaceDetector.h"
#include "seeta/FaceLandmarker.h"
#include "seeta/FaceRecognizer.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
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

// ====================== 日志工具（输出到 stdout，docker logs 直接可见） ======================

static std::mutex g_log_mu;

// 当前本地时间字符串，形如 2026-10-17 22:32:56.672
static std::string now_str() {
  using namespace std::chrono;
  auto now = system_clock::now();
  auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
  std::time_t t = system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
  std::ostringstream os;
  os << buf << "." << std::setfill('0') << std::setw(3) << ms.count();
  return os.str();
}

// 自 start 起经过的毫秒数
static long elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

static void log(const std::string& level, const std::string& msg) {
  std::lock_guard<std::mutex> lk(g_log_mu);
  std::cout << "[" << now_str() << "][" << level << "] " << msg << std::endl;
}
static void log_info(const std::string& msg) { log("info", msg); }
static void log_warn(const std::string& msg) { log("warn", msg); }
static void log_error(const std::string& msg) { log("error", msg); }

static bool file_exists(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

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
    log_info("引擎池初始化开始，实例数=" + std::to_string(size));
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < size; ++i) {
      auto ti = std::chrono::steady_clock::now();
      engines_.push_back(std::make_unique<Engine>(det, lm, fr));
      idle_.push(engines_.back().get());
      log_info("  引擎实例 " + std::to_string(i + 1) + "/" + std::to_string(size) +
               " 初始化完成，耗时=" + std::to_string(elapsed_ms(ti)) + "ms");
    }
    log_info("引擎池初始化完成，共 " + std::to_string(size) + " 个实例，总耗时=" +
             std::to_string(elapsed_ms(t0)) + "ms");
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

  Status Extract(ServerContext* ctx, const ExtractRequest* req,
                 ExtractResponse* resp) override {
    auto t0 = std::chrono::steady_clock::now();
    std::string peer = ctx ? ctx->peer() : "unknown";
    log_info("Extract 请求到达 peer=" + peer);

    const auto& img = req->image();
    log_info("Extract peer=" + peer + " 图像=" + std::to_string(img.width()) + "x" +
             std::to_string(img.height()) + " 通道=" + std::to_string(img.channels()) +
             " 字节数=" + std::to_string(img.data().size()));

    if (img.data().empty() || img.width() <= 0 || img.height() <= 0 ||
        img.channels() <= 0) {
      log_error("Extract peer=" + peer + " 非法图像参数，拒绝请求");
      return Status(grpc::INVALID_ARGUMENT,
                    "invalid image: empty data or non-positive dimensions");
    }

    auto t_acquire = std::chrono::steady_clock::now();
    auto engine = pool_.acquire();
    log_info("Extract peer=" + peer + " 引擎租借耗时=" +
             std::to_string(elapsed_ms(t_acquire)) + "ms（含等待空闲引擎）");

    try {
      auto* p = reinterpret_cast<const unsigned char*>(img.data().data());
      SeetaImageData s{img.width(), img.height(), img.channels(),
                       const_cast<unsigned char*>(p)};

      auto t_detect = std::chrono::steady_clock::now();
      SeetaFaceInfoArray faces = engine->detector_.detect(s);
      log_info("Extract peer=" + peer + " 人脸检测耗时=" +
               std::to_string(elapsed_ms(t_detect)) + "ms 检测到人脸=" +
               std::to_string(faces.size) + " 张");

      int extracted = 0;
      for (int i = 0; i < faces.size; ++i) {
        SeetaFaceInfo& f = faces.data[i];

        auto t_lm = std::chrono::steady_clock::now();
        std::vector<SeetaPointF> pts(engine->landmarker_.number());
        engine->landmarker_.mark(s, f.pos, pts.data());
        long lm_ms = elapsed_ms(t_lm);

        auto t_feat = std::chrono::steady_clock::now();
        std::vector<float> feat(engine->recognizer_.GetExtractFeatureSize());
        bool ok = engine->recognizer_.Extract(s, pts.data(), feat.data());
        long feat_ms = elapsed_ms(t_feat);

        if (!ok) {
          log_warn("Extract peer=" + peer + " 第 " + std::to_string(i) +
                   " 张人脸特征提取失败（Extract 返回 false），跳过");
          continue;
        }

        auto* out = resp->add_faces();
        out->set_x(f.pos.x);
        out->set_y(f.pos.y);
        out->set_width(f.pos.width);
        out->set_height(f.pos.height);
        out->set_score(f.score);
        out->mutable_feature()->Add(feat.begin(), feat.end());
        ++extracted;

        log_info("Extract peer=" + peer + " 第 " + std::to_string(i) +
                 " 张人脸: 特征点=" + std::to_string(lm_ms) + "ms 特征提取=" +
                 std::to_string(feat_ms) + "ms 特征维度=" +
                 std::to_string(feat.size()));
      }

      log_info("Extract peer=" + peer + " 完成，输出特征=" + std::to_string(extracted) +
               " 个，总耗时=" + std::to_string(elapsed_ms(t0)) + "ms");
      return Status::OK;
    } catch (const std::exception& e) {
      log_error("Extract peer=" + peer + " 异常: " + e.what());
      return Status(grpc::INTERNAL, std::string("seeta exception: ") + e.what());
    } catch (...) {
      log_error("Extract peer=" + peer + " 未知异常");
      return Status(grpc::INTERNAL, "unknown seeta exception");
    }
  }

  Status Compare(ServerContext* ctx, const CompareRequest* req,
                 CompareResponse* resp) override {
    auto t0 = std::chrono::steady_clock::now();
    std::string peer = ctx ? ctx->peer() : "unknown";
    log_info("Compare 请求到达 peer=" + peer + " 特征1维度=" +
             std::to_string(req->feature1_size()) + " 特征2维度=" +
             std::to_string(req->feature2_size()));

    const auto& f1 = req->feature1();
    const auto& f2 = req->feature2();
    if (f1.empty() || f1.size() != f2.size()) {
      log_error("Compare peer=" + peer + " 特征维度不匹配（f1=" +
                std::to_string(f1.size()) + ", f2=" + std::to_string(f2.size()) +
                "），拒绝请求");
      return Status(grpc::INVALID_ARGUMENT, "feature size mismatch");
    }

    auto t_acquire = std::chrono::steady_clock::now();
    auto engine = pool_.acquire();
    log_info("Compare peer=" + peer + " 引擎租借耗时=" +
             std::to_string(elapsed_ms(t_acquire)) + "ms");

    try {
      auto t_calc = std::chrono::steady_clock::now();
      float sim = engine->recognizer_.CalculateSimilarity(f1.data(), f2.data());
      resp->set_similarity(sim);
      log_info("Compare peer=" + peer + " 相似度=" + std::to_string(sim) +
               " 计算耗时=" + std::to_string(elapsed_ms(t_calc)) + "ms 总耗时=" +
               std::to_string(elapsed_ms(t0)) + "ms");
      return Status::OK;
    } catch (const std::exception& e) {
      log_error("Compare peer=" + peer + " 异常: " + e.what());
      return Status(grpc::INTERNAL, std::string("seeta exception: ") + e.what());
    } catch (...) {
      log_error("Compare peer=" + peer + " 未知异常");
      return Status(grpc::INTERNAL, "unknown seeta exception");
    }
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

// 全局 server 指针，供信号处理器触发优雅关闭
static std::unique_ptr<grpc::Server> g_server;
static std::atomic<int> g_signal{0};

static void handle_signal(int sig) {
  g_signal.store(sig);
  if (g_server) {
    g_server->Shutdown();  // grpc::Server::Shutdown 可从任意线程调用
  }
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

  log_info("==== facerec-grpc 服务端启动 ====");
  log_info("配置: 设备=" + std::string(device_name(dev)) +
           " 引擎池=" + std::to_string(pool_size) + " 监听地址=" + addr);
  log_info("模型路径: det=" + det_path + " lm=" + lm_path + " fr=" + fr_path);

  if (!file_exists(det_path)) log_warn("检测模型文件不存在: " + det_path);
  if (!file_exists(lm_path)) log_warn("特征点模型文件不存在: " + lm_path);
  if (!file_exists(fr_path)) log_warn("识别模型文件不存在: " + fr_path);

  const char* det_models[] = {det_path.c_str(), nullptr};
  SeetaModelSetting det_setting{dev, device_id, det_models};

  const char* lm_models[] = {lm_path.c_str(), nullptr};
  SeetaModelSetting lm_setting{dev, device_id, lm_models};

  const char* fr_models[] = {fr_path.c_str(), nullptr};
  SeetaModelSetting fr_setting{dev, device_id, fr_models};

  std::unique_ptr<EnginePool> pool;
  try {
    pool = std::make_unique<EnginePool>(pool_size, det_setting, lm_setting,
                                        fr_setting);
  } catch (const std::exception& e) {
    log_error("引擎池初始化失败（很可能是模型缺失/损坏，或 SeetaFace 授权失败）: " +
              std::string(e.what()));
    return 1;
  }
  FaceRecognitionServiceImpl service(*pool);

  ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials(), &selected_port);
  builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
  builder.SetMaxSendMessageSize(64 * 1024 * 1024);
  builder.RegisterService(&service);
  // 启用标准 gRPC 健康检查服务 grpc.health.v1.Health（用于 grpc_health_probe / grpcurl / 负载均衡探活）
  grpc::EnableDefaultHealthCheckService(true);

  log_info("健康检查服务 grpc.health.v1.Health 已启用");
  log_info("正在构建并启动 gRPC 服务，监听地址=" + addr + " ...");
  g_server = builder.BuildAndStart();
  if (!g_server) {
    log_error("启动失败: 无法监听 " + addr + "，请检查端口是否被占用或地址是否合法");
    return 1;
  }

  log_info("监听成功: " + addr + "（实际绑定端口=" + std::to_string(selected_port) +
           "）");
  log_info("服务就绪，等待客户端连接...");

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  g_server->Wait();

  if (g_signal.load() != 0) {
    log_warn("收到信号 " + std::to_string(g_signal.load()) + "，服务已优雅关闭");
  } else {
    log_info("服务已停止");
  }
  return 0;
}
