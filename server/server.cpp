#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "facerec.grpc.pb.h"

#include "seeta/FaceDetector.h"
#include "seeta/FaceLandmarker.h"
#include "seeta/FaceRecognizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <pthread.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

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

// 日志详细级别：
//   0 = 仅启动信息与告警/错误（不打每请求日志，开销最低）
//   1 = 每请求一行摘要（默认）
//   2 = 额外打印逐张人脸的明细
static std::atomic<int> g_log_level{1};

static bool log_enabled(int level) { return g_log_level.load() >= level; }

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

// 异步日志：请求线程只做一次入队（一把短锁 + 字符串），由单独的后台线程写 stdout。
//
// 直接在请求线程里加锁写 std::cout 时，所有请求线程会排在同一把锁上等 IO
// （stdout 被 docker 接管后是管道 + 落盘），吞吐随日志量线性下降。
// 改成异步后，日志开销与磁盘/管道速度解耦。
//
// 队列设上限：日志消费不过来时宁可丢弃并计数，也不让业务线程阻塞。
class AsyncLogger {
 public:
  static AsyncLogger& instance() {
    static AsyncLogger inst;
    return inst;
  }

  void write(const std::string& line) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (queue_.size() >= kMaxQueue) {  // 过载保护：丢弃最旧的
        queue_.pop_front();
        ++dropped_;
      }
      queue_.push_back(line);
    }
    cv_.notify_one();
  }

  // 停止并冲刷剩余日志（main 退出前调用）
  void shutdown() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (stop_) return;  // 幂等：静态析构时可能再调一次
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    std::cout.flush();
  }

 private:
  AsyncLogger() { worker_ = std::thread([this] { run(); }); }
  ~AsyncLogger() { shutdown(); }
  AsyncLogger(const AsyncLogger&) = delete;
  AsyncLogger& operator=(const AsyncLogger&) = delete;

  void run() {
    std::deque<std::string> batch;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
        if (queue_.empty()) {
          if (stop_) return;
          continue;
        }
        batch.swap(queue_);
        if (dropped_ > 0) {
          batch.push_front("[" + now_str() + "][warn] 日志过载，已丢弃 " +
                           std::to_string(dropped_) + " 条");
          dropped_ = 0;
        }
      }
      // 锁外写：把一批日志拼成一次 std::cout 写入，减少系统调用
      std::string out;
      for (const auto& l : batch) {
        out += l;
        out += '\n';
      }
      std::cout << out << std::flush;
      batch.clear();
    }
  }

  static constexpr size_t kMaxQueue = 65536;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::string> queue_;
  std::thread worker_;
  size_t dropped_ = 0;
  bool stop_ = false;
};

static void log(const std::string& level, const std::string& msg) {
  AsyncLogger::instance().write("[" + now_str() + "][" + level + "] " + msg);
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
//
// threads 是每个引擎内部的计算线程数。SeetaFace 默认给检测/识别各 4 个线程，
// 而引擎池默认 6 个实例 => 30 个计算线程争抢 12 个逻辑核，实测并发吞吐反而
// 比单并发低 3 倍以上（检测耗时从 40ms 膨胀到 1.6s）。这里统一收敛为
// 「引擎池大小 x 每引擎线程数 <= 物理核数」。
class Engine {
 public:
  Engine(const SeetaModelSetting& det, const SeetaModelSetting& lm,
         const SeetaModelSetting& fr, int threads)
      : detector_(det), landmarker_(lm), recognizer_(fr) {
    detector_.set(seeta::FaceDetector::PROPERTY_NUMBER_THREADS, threads);
    recognizer_.set(seeta::FaceRecognizer::PROPERTY_NUMBER_THREADS, threads);
  }

  seeta::FaceDetector detector_;
  seeta::FaceLandmarker landmarker_;
  seeta::FaceRecognizer recognizer_;
};

// 引擎池：多个 Engine 实例 + 空闲队列，实现并发安全复用。
class EnginePool {
 public:
  EnginePool(int size, int threads, const SeetaModelSetting& det,
             const SeetaModelSetting& lm, const SeetaModelSetting& fr)
      : size_(size), threads_(threads) {
    log_info("引擎池初始化开始，实例数=" + std::to_string(size) +
             " 每实例线程数=" + std::to_string(threads) + "（合计 " +
             std::to_string(size * threads) + " 个计算线程）");
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < size; ++i) {
      auto ti = std::chrono::steady_clock::now();
      engines_.push_back(std::make_unique<Engine>(det, lm, fr, threads));
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
    Lease(EnginePool& pool, Engine* e, long wait_us)
        : pool_(pool), engine_(e), wait_us_(wait_us) {}
    ~Lease() {
      if (engine_) pool_.release(engine_);
    }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& o) noexcept
        : pool_(o.pool_), engine_(o.engine_), wait_us_(o.wait_us_) {
      o.engine_ = nullptr;
    }
    Engine* operator->() const { return engine_; }
    // 在空闲队列上等待的时间（微秒），即真正的排队耗时
    long wait_us() const { return wait_us_; }

   private:
    EnginePool& pool_;
    Engine* engine_;
    long wait_us_;
  };

  Lease acquire() {
    auto t = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this] { return !idle_.empty(); });
    Engine* e = idle_.front();
    idle_.pop();
    long wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - t)
                       .count();
    return Lease(*this, e, wait_us);
  }

  int size() const { return size_; }
  int threads() const { return threads_; }

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
  int size_;
  int threads_;
};

class FaceRecognitionServiceImpl final : public FaceRecognition::Service {
 public:
  explicit FaceRecognitionServiceImpl(EnginePool& pool) : pool_(pool) {}

  Status Extract(ServerContext* ctx, const ExtractRequest* req,
                 ExtractResponse* resp) override {
    auto t0 = std::chrono::steady_clock::now();
    std::string peer = ctx ? ctx->peer() : "unknown";
    if (log_enabled(2)) log_info("Extract 请求到达 peer=" + peer);

    const auto& img = req->image();
    if (log_enabled(2)) {
      log_info("Extract peer=" + peer + " 图像=" + std::to_string(img.width()) + "x" +
               std::to_string(img.height()) + " 通道=" + std::to_string(img.channels()) +
               " 字节数=" + std::to_string(img.data().size()));
    }

    if (img.data().empty() || img.width() <= 0 || img.height() <= 0 ||
        img.channels() <= 0) {
      log_error("Extract peer=" + peer + " 非法图像参数，拒绝请求");
      return Status(grpc::INVALID_ARGUMENT,
                    "invalid image: empty data or non-positive dimensions");
    }

    auto engine = pool_.acquire();
    if (log_enabled(2)) {
      log_info("Extract peer=" + peer + " 排队等待引擎=" +
               std::to_string(engine.wait_us() / 1000.0) + "ms");
    }

    try {
      auto* p = reinterpret_cast<const unsigned char*>(img.data().data());
      SeetaImageData s{img.width(), img.height(), img.channels(),
                       const_cast<unsigned char*>(p)};

      auto t_detect = std::chrono::steady_clock::now();
      SeetaFaceInfoArray faces = engine->detector_.detect(s);
      long detect_ms = elapsed_ms(t_detect);

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

        if (log_enabled(2)) {
          log_info("Extract peer=" + peer + " 第 " + std::to_string(i) +
                   " 张人脸: 特征点=" + std::to_string(lm_ms) + "ms 特征提取=" +
                   std::to_string(feat_ms) + "ms 特征维度=" +
                   std::to_string(feat.size()));
        }
      }

      // 一行摘要，适合 production 下观察延迟：拆开排队/检测/总耗时便于定位。
      // level 0 时整条日志（含字符串拼接）都跳过，压测/高吞吐场景可不产生任何 IO。
      if (log_enabled(1)) {
        log_info("Extract peer=" + peer + " 图像=" + std::to_string(img.width()) +
                 "x" + std::to_string(img.height()) + " 排队=" +
                 std::to_string(engine.wait_us() / 1000.0) + "ms 检测=" +
                 std::to_string(detect_ms) + "ms 输出特征=" +
                 std::to_string(extracted) + " 个 总耗时=" +
                 std::to_string(elapsed_ms(t0)) + "ms");
      }
      return Status::OK;
    } catch (const std::exception& e) {
      log_error("Extract peer=" + peer + " 异常: " + e.what());
      return Status(grpc::INTERNAL, std::string("seeta exception: ") + e.what());
    } catch (...) {
      log_error("Extract peer=" + peer + " 未知异常");
      return Status(grpc::INTERNAL, "unknown seeta exception");
    }
  }

  // 相似度比对。
  // SeetaFace 的 CalculateSimilarity 是纯函数（无状态、无内部缓冲），特征向量
  // 已是 L2 归一化，故它等价于点积 == 余弦相似度。实测 8 线程并发调用零错误，
  // 单次约 0.1us。因此这里**不再占用推理引擎**——否则一个 0.1us 的点积要和
  // 一次 100ms 的推理抢锁，1:N 检索（客户端对底库逐人调用 Compare）会被拖垮。
  // 为避免依赖第三方库并发调用的未定义行为，这里直接用点积自行实现。
  static float dot_similarity(const float* a, const float* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(sum);
  }

  Status Compare(ServerContext* ctx, const CompareRequest* req,
                 CompareResponse* resp) override {
    auto t0 = std::chrono::steady_clock::now();
    std::string peer = ctx ? ctx->peer() : "unknown";

    const auto& f1 = req->feature1();
    const auto& f2 = req->feature2();
    if (f1.empty() || f1.size() != f2.size()) {
      log_error("Compare peer=" + peer + " 特征维度不匹配（f1=" +
                std::to_string(f1.size()) + ", f2=" + std::to_string(f2.size()) +
                "），拒绝请求");
      return Status(grpc::INVALID_ARGUMENT, "feature size mismatch");
    }

    const float sim = dot_similarity(f1.data(), f2.data(), f1.size());
    resp->set_similarity(sim);
    if (log_enabled(2)) {
      log_info("Compare peer=" + peer + " 维度=" + std::to_string(f1.size()) +
               " 相似度=" + std::to_string(sim) + " 总耗时=" +
               std::to_string(elapsed_ms(t0)) + "ms");
    }
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

// ---- CPU 拓扑探测 ----
//
// 自动配置的核心约束：引擎池大小 x 每引擎线程数 <= 物理核数。
// 两者相乘就是同一时刻真正跑推理的线程数；超过物理核数只会带来上下文切换
// 和缓存抖动（实测超配 2.5 倍时并发吞吐反而下降 3 倍以上）。
//
// 之所以按「物理核」而不是「逻辑核」，是因为 SeetaFace 底层 TenniS 的计算
// 是量密集型浮点运算，超线程带来的收益远小于它带来的争用。

struct CpuTopology {
  int logical = 1;   // 逻辑核（含超线程、含 cgroup 配额）
  int physical = 1;  // 物理核
};

static CpuTopology detect_cpu_topology() {
  CpuTopology t;
#if defined(__linux__)
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  if (n > 0) t.logical = static_cast<int>(n);

  // cgroup v2 / v1 的 CPU 配额可能小于 sysconf 报告的核数（容器被限流）
  auto read_int = [](const char* path) -> long {
    std::ifstream f(path);
    long v = -1;
    if (f.good()) f >> v;
    return v;
  };

  // 读取 cgroup 的 cpu.max（v2）或 cfs_quota/cfs_period（v1），换算成可用核数
  auto quota_cores = [&read_int]() -> int {
    std::ifstream f2("/sys/fs/cgroup/cpu.max");  // v2: "<quota> <period>"
    if (f2.good()) {
      std::string quota, period;
      f2 >> quota >> period;
      if (quota != "max") {
        long q = std::atol(quota.c_str());
        long p = std::atol(period.c_str());
        if (q > 0 && p > 0) {
          int c = static_cast<int>((q + p - 1) / p);
          return c > 0 ? c : 1;
        }
      }
      return -1;
    }
    long q = read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");      // v1
    long p = read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
    if (q > 0 && p > 0) return static_cast<int>((q + p - 1) / p);
    return -1;
  };
  int quota = quota_cores();
  if (quota > 0 && quota < t.logical) t.logical = quota;

  // 统计 /sys/devices/system/cpu/cpuN/topology/thread_siblings_list：
  // 每个物理核对应一组超线程兄弟，物理核数 = 独立兄弟组数
  int nproc = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
  if (nproc <= 0) nproc = t.logical;
  int groups = 0;
  std::string seen;  // 已统计过的兄弟组首元素
  for (int i = 0; i < nproc; ++i) {
    std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(i) +
                       "/topology/thread_siblings_list";
    std::ifstream f(base);
    if (!f.good()) continue;
    std::string line;
    std::getline(f, line);  // 形如 "0,6" 或 "0-1" 或 "0"
    if (line.empty()) continue;
    // 取该组的第一个 CPU 编号作为组的标识
    size_t dash = line.find('-');
    size_t comma = line.find(',');
    size_t end = std::min(dash == std::string::npos ? line.size() : dash,
                          comma == std::string::npos ? line.size() : comma);
    std::string first = line.substr(0, end);
    std::string key = "|" + first + "|";
    if (seen.find(key) == std::string::npos) {
      seen += key;
      ++groups;
    }
  }
  if (groups > 0) t.physical = std::min(groups, t.logical);
  else t.physical = t.logical;
  // 离线（offline）的核不计入，这里用 std::max(1,..) 兜底
  if (t.physical < 1) t.physical = 1;
  if (t.logical < t.physical) t.logical = t.physical;
#else
  unsigned hc = std::thread::hardware_concurrency();
  t.logical = t.physical = hc > 0 ? static_cast<int>(hc) : 1;
#endif
  return t;
}

// 根据物理核数给出默认的 (引擎池大小, 每引擎线程数)。
//
// 实测单请求 P50（SeetaFace 默认给检测/识别各 4 线程，这是超配的根源）：
//
//   线程/引擎    320px    640px    1280px
//      1         80ms    118ms    318ms
//      2         49ms     82ms    269ms
//      3         39ms     80ms    251ms
//      4         37ms     80ms    256ms
//      6         38ms     75ms    265ms
//      8         43ms     94ms    281ms
//     12        210ms    761ms    432ms   <-- 灾难
//
// 结论一：3-4 线程是单请求的饱和点，12 线程会慢 3~10 倍。
// 结论二：收益递减明显——从 1 线程到 3 线程只快 2 倍，不是 3 倍。
//         识别阶段基本只有 1.5~2 倍并行度（26ms 就到底了），小图检测也是。
//         所以「每引擎 4 线程」在物理核不够时很浪费。
//
// 并发吞吐实测（640px，pool x threads 的所有组合）：
//   1x1=8.3   2x1=14.2  4x1=18.6  6x1=18.3  8x1=17.7  (QPS)
//   2x2=12.8  3x2=14.9  4x2=17.5  6x2=17.1
//   1x4=11.7  2x4=15.7  3x4=1.7（严重退化）
// 即：总线程数决定吞吐，最佳区间是「总线程 ≈ 物理核数」；
//     同时每个引擎至少分到 2 线程，否则池内排队会很严重。
//
// 每个引擎实例要复制一份模型（约 500MB），所以池上限取 8。
// 不变式：pool x threads <= 物理核数。
static void auto_tune(int physical_cores, int* pool, int* threads) {
  const int kMaxPool = 8;
  const int kMaxThreads = 4;  // 超过 4 无收益，12 更是灾难
  const int kMinThreads = 2;  // 每引擎至少 2 线程，否则单引擎太慢

  if (physical_cores <= 1) {
    *pool = 1;
    *threads = 1;  // 单核机器加线程只会互相抢，直接串行
    return;
  }

  if (physical_cores == 2) {
    *pool = 1;
    *threads = 2;
    return;
  }

  // 物理核 3：只够一个 3 线程引擎。若拆成 2 个引擎各 2 线程就是 4 线程超配，
  // 反而触发下面要避免的争用，所以宁可单引擎略慢也不超配。
  if (physical_cores == 3) {
    *pool = 1;
    *threads = 3;
    return;
  }

  // 物理核 4：正好两个 2 线程引擎
  if (physical_cores == 4) {
    *pool = 2;
    *threads = 2;
    return;
  }

  // 物理核 >= 5：优先让每个引擎拿到 2 线程（收益最大的一段），
  // 池大小 = 核数/2，多余核数再加到每个引擎上（封顶 4）
  *threads = 2;
  *pool = physical_cores / 2;
  if (*pool > kMaxPool) {
    *pool = kMaxPool;
    *threads = physical_cores / kMaxPool;
  }
  if (*threads > kMaxThreads) *threads = kMaxThreads;
  if (*threads < kMinThreads) *threads = kMinThreads;
  if (*pool < 1) *pool = 1;
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
            << "  --pool <N>                推理引擎实例数 (默认按 CPU 自动配置)\n"
            << "  --threads <N>             每引擎计算线程数, 1-4 (默认按 CPU 自动配置)\n"
            << "  --addr <ip:port>          监听地址 (默认 0.0.0.0:50051)\n"
            << "  --log-level <N>           日志级别 0=摘要 1=正常 2=详细 (默认 1)\n"
            << "  --det <path>              人脸检测模型 (默认 model/face_detector.csta)\n"
            << "  --lm  <path>              特征点模型 (默认 model/face_landmarker_pts5.csta)\n"
            << "  --fr  <path>              人脸识别模型 (默认 model/face_recognizer.csta)\n"
            << "\n自动配置原则：引擎池大小 x 每引擎线程数 <= 物理核数，\n"
            << "二者相乘即同时运行的推理线程数，超过物理核数会因争用而变慢。\n"
            << "\n环境变量: SEETA_DEVICE(计算设备) SEETA_POOL(引擎池)\n"
            << "         SEETA_THREADS(每引擎线程数) SEETA_LOG_LEVEL(日志级别)\n"
            << "命令行参数优先级高于环境变量。\n";
}

// 全局 server 指针，供信号线程触发优雅关闭
static std::unique_ptr<grpc::Server> g_server;

// 信号处理：分两步。
//
// 1) block_shutdown_signals() 必须在 main 的最开头调用——在任何线程（日志线程、
//    gRPC 线程）创建之前。pthread_sigmask 只影响调用线程，新线程继承创建者的
//    掩码；若晚于 gRPC 建线程，那几个线程没屏蔽，进程级信号会被投递给它们，
//    直接按默认动作终止进程（退出码 143），优雅关闭逻辑根本跑不到。
// 2) spawn_signal_thread() 在服务起来之后启动一个 sigwait 线程同步等待。
//
// 不能在信号处理函数里直接调用 grpc::Server::Shutdown()：信号到达时主线程
// 正停在 Server::Wait() 内部、持有 absl 的互斥锁，从信号处理函数重入
// Shutdown 会再次去拿同一把锁，触发 absl 的锁顺序环检测，进程直接 abort
// （日志里表现为 "dying due to potential deadlock"，退出码 134 而不是 0）。
static void block_shutdown_signals() {
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGINT);
  sigaddset(&set, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &set, nullptr);
}

static void spawn_signal_thread() {
  std::thread([] {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    int sig = 0;
    if (sigwait(&set, &sig) == 0) {
      log_warn("收到信号 " + std::to_string(sig) + "，正在优雅关闭...");
      if (g_server) g_server->Shutdown();  // 从普通线程调用是安全的
    }
  }).detach();
}

int main(int argc, char** argv) {
  // 必须最先执行：此后创建的线程都会继承这个掩码
  block_shutdown_signals();

  std::string det_path = "model/face_detector.csta";
  std::string lm_path = "model/face_landmarker_pts5.csta";
  std::string fr_path = "model/face_recognizer.csta";
  std::string addr = "0.0.0.0:50051";
  int gpu_id = 0;
  int pool_size = 0;    // 0 = 未指定，稍后按 CPU 自动配置
  int threads = 0;      // 同上

  // 优先级: 命令行 > 环境变量 > 自动配置
  auto env_int = [](const char* name) -> int {
    if (const char* v = std::getenv(name)) {
      if (*v) {
        char* end = nullptr;
        long n = std::strtol(v, &end, 10);
        if (end != v) return static_cast<int>(n);
      }
    }
    return 0;
  };
  pool_size = env_int("SEETA_POOL");
  threads = env_int("SEETA_THREADS");
  // 日志级别：SEETA_LOG_LEVEL 未设置时保持默认 1；显式设 0 表示只看摘要
  g_log_level.store(1);
  if (const char* v = std::getenv("SEETA_LOG_LEVEL")) {
    if (*v) {
      char* end = nullptr;
      long n = std::strtol(v, &end, 10);
      if (end != v && n >= 0 && n <= 2) g_log_level.store(static_cast<int>(n));
    }
  }

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
    } else if (a == "--threads") {
      threads = std::stoi(value_of(i));
    } else if (a.rfind("--threads=", 0) == 0) {
      threads = std::stoi(a.substr(10));
    } else if (a == "--log-level") {
      g_log_level.store(std::stoi(value_of(i)));
    } else if (a.rfind("--log-level=", 0) == 0) {
      g_log_level.store(std::stoi(a.substr(12)));
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

  const SeetaDevice dev = parse_device(device_str);
  const int device_id = (dev == SEETA_DEVICE_GPU) ? gpu_id : 0;

  // ---- 自动配置 ----
  const CpuTopology cpu = detect_cpu_topology();
  int auto_pool = 0, auto_threads = 0;
  auto_tune(cpu.physical, &auto_pool, &auto_threads);
  const bool pool_is_auto = pool_size <= 0;
  const bool threads_is_auto = threads <= 0;
  if (pool_is_auto) pool_size = auto_pool;
  if (threads_is_auto) threads = auto_threads;
  if (pool_size < 1) pool_size = 1;
  if (threads < 1) threads = 1;

  log_info("==== facerec-grpc 服务端启动 ====");
  log_info("CPU 拓扑: 物理核=" + std::to_string(cpu.physical) + " 逻辑核=" +
           std::to_string(cpu.logical) + "（已考虑容器 CPU 配额）");
  log_info("配置: 设备=" + std::string(device_name(dev)) + " 引擎池=" +
           std::to_string(pool_size) +
           (pool_is_auto ? "（自动）" : "（手动指定）") + " 每引擎线程数=" +
           std::to_string(threads) +
           (threads_is_auto ? "（自动）" : "（手动指定）") + " 监听地址=" + addr);
  log_info("推理并行度: " + std::to_string(pool_size) + " x " +
           std::to_string(threads) + " = " +
           std::to_string(pool_size * threads) + " 线程 / " +
           std::to_string(cpu.physical) + " 物理核");
  if (pool_size * threads > cpu.physical) {
    log_warn("推理线程数(" + std::to_string(pool_size * threads) +
             ") 超过物理核数(" + std::to_string(cpu.physical) +
             ")，并发下会因线程争用而变慢；建议减小 --pool 或 --threads");
  }
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
    pool = std::make_unique<EnginePool>(pool_size, threads, det_setting,
                                        lm_setting, fr_setting);
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

  spawn_signal_thread();

  g_server->Wait();
  log_info("服务已停止");
  AsyncLogger::instance().shutdown();  // 冲刷队列中剩余的日志
  return 0;
}
