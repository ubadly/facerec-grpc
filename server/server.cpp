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
#include <cstdint>
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

// 微秒转可读时长。小于 1ms 用 us 表示，避免出现 "0.0ms" 这种没有信息量的值。
static std::string fmt_us(long us) {
  if (us < 0) us = 0;
  std::ostringstream os;
  if (us < 1000) {
    os << us << "us";
  } else if (us < 1000000) {
    os << std::fixed << std::setprecision(1) << (us / 1000.0) << "ms";
  } else {
    os << std::fixed << std::setprecision(2) << (us / 1000000.0) << "s";
  }
  return os.str();
}

// 进程当前常驻内存（KB）。用于启动后确认「引擎池 x 225MB」的实际开销，
// 也便于对照 README 里的内存表判断 --pool 是否设得过大。
static long rss_kb() {
#if defined(__linux__)
  std::ifstream f("/proc/self/status");
  std::string key;
  while (f >> key) {
    if (key == "VmRSS:") {
      long v = 0;
      f >> v;
      return v;
    }
    std::string rest;
    std::getline(f, rest);
  }
#endif
  return -1;
}

// 异步日志：请求线程只做一次入队（一把短锁 + 字符串），由单独的后台线程写 stdout。
//
// 直接在请求线程里加锁写 std::cout 时，所有请求线程会排在同一把锁上等 IO
// （stdout 被 docker 接管后是管道 + 落盘），吞吐随日志量线性下降。
// 改成异步后，日志开销与磁盘/管道速度解耦。
//
// 队列设上限：日志消费不过来时宁可丢弃并计数，也不让业务线程阻塞。
//
// 错误日志是例外：走 write_sync 直接落盘。进程随后可能 abort/被 kill，
// 排在队列里的错误信息会一起丢掉，而错误恰恰是最需要留下的那一条。
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

  // 同步写：跳过队列，立刻输出。只给 error 用（调用频率低，不怕阻塞）。
  void write_sync(const std::string& line) {
    std::lock_guard<std::mutex> lk(out_mu_);
    std::cout << line << '\n' << std::flush;
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
      std::lock_guard<std::mutex> lk(out_mu_);
      std::cout << out << std::flush;
      batch.clear();
    }
  }

  static constexpr size_t kMaxQueue = 65536;
  std::mutex out_mu_;  // 保护 std::cout：异步批写与 write_sync 可能同时发生
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
// error 同步落盘：紧接着可能就是 abort/被 kill，排队中的错误会丢
static void log_error(const std::string& msg) {
  AsyncLogger::instance().write_sync("[" + now_str() + "][error] " + msg);
}

// 进程内的短标识。多个容器/实例的日志汇总到一起时，用来区分是谁打的。
// 取 hostname（容器里是 container id 前 12 位），取不到就退化成 "node"。
static const std::string& node_id() {
  static const std::string id = [] {
#if defined(__linux__) || defined(__APPLE__)
    char buf[128] = {0};
    if (gethostname(buf, sizeof(buf) - 1) == 0 && buf[0]) {
      std::string h(buf);
      if (h.size() > 12) h.resize(12);
      return h;
    }
#endif
    return std::string("node");
  }();
  return id;
}

// ---- 运行期统计 ----
//
// 每 --stats-interval 秒打一行汇总（默认 60s，0 = 关闭）。
// 请求级日志适合排查单次问题；而「排队是不是整体在涨」「池有没有打满」
// 这类趋势只能看汇总——逐条日志刷屏时反而看不出来。
struct ServiceStats {
  static constexpr int kBuckets = 13;
  // 总耗时直方图的桶上界（ms），最后一桶是 >= 2000ms。
  // 60~200ms 是这套模型最密集的区间（320~640px 单张人脸），分得细一点，
  // 否则 P50 只能报成 "<=120ms" 这种量级，看不出优化有没有效果。
  static constexpr int kBounds[kBuckets] = {10, 20,  30,  40,  60,  80, 100,
                                            150, 200, 300, 500, 1000, 2000};

  std::atomic<uint64_t> extract_ok{0};
  std::atomic<uint64_t> extract_fail{0};
  std::atomic<uint64_t> compare_ok{0};
  std::atomic<uint64_t> compare_fail{0};
  std::atomic<uint64_t> faces{0};
  std::atomic<uint64_t> queue_us_sum{0};
  std::atomic<uint64_t> detect_ms_sum{0};
  std::atomic<uint64_t> infer_ms_sum{0};  // 特征点 + 识别
  std::atomic<uint64_t> total_ms_sum{0};
  std::atomic<uint64_t> busy_sum{0};      // 每次取引擎时的在用实例数之和
  std::atomic<uint64_t> hist[kBuckets + 1];

  ServiceStats() {
    for (auto& h : hist) h.store(0);
  }

  void on_extract(long queue_us, long detect_ms, long infer_ms, long total_ms,
                  int nfaces, int busy) {
    extract_ok.fetch_add(1);
    faces.fetch_add(nfaces);
    queue_us_sum.fetch_add(queue_us > 0 ? queue_us : 0);
    detect_ms_sum.fetch_add(detect_ms > 0 ? detect_ms : 0);
    infer_ms_sum.fetch_add(infer_ms > 0 ? infer_ms : 0);
    total_ms_sum.fetch_add(total_ms > 0 ? total_ms : 0);
    busy_sum.fetch_add(busy > 0 ? busy : 0);
    int i = 0;
    while (i < kBuckets && total_ms >= kBounds[i]) ++i;
    hist[i].fetch_add(1);
  }

  // 取快照并把累计值清零，返回可直接打印的一行
  std::string take_snapshot_line(double seconds, int pool_size) {
    const uint64_t ok = extract_ok.exchange(0);
    const uint64_t fail = extract_fail.exchange(0);
    const uint64_t cok = compare_ok.exchange(0);
    const uint64_t cfail = compare_fail.exchange(0);
    const uint64_t nf = faces.exchange(0);
    const uint64_t q = queue_us_sum.exchange(0);
    const uint64_t det = detect_ms_sum.exchange(0);
    const uint64_t inf = infer_ms_sum.exchange(0);
    const uint64_t tot = total_ms_sum.exchange(0);
    const uint64_t busy = busy_sum.exchange(0);
    uint64_t h[kBuckets + 1];
    for (int i = 0; i <= kBuckets; ++i) h[i] = hist[i].exchange(0);

    // 整段空闲就不打日志：服务闲一整天时每分钟刷一行全 0 只是噪声，
    // 反而会淹掉真正需要看的告警。
    if (ok == 0 && fail == 0 && cok == 0 && cfail == 0) return "";

    std::ostringstream os;
    os << "统计 Extract=" << ok << " 失败=" << fail << " 人脸=" << nf
       << " Compare=" << cok << " 失败=" << cfail;
    if (ok > 0) {
      const double n = static_cast<double>(ok);
      // seconds<=0 表示进程退出时的收尾快照，没有可靠的统计区间，不报 QPS
      if (seconds > 0) {
        os << " QPS=" << std::fixed << std::setprecision(1) << n / seconds;
      }
      os << " 均排队=" << fmt_us(static_cast<long>(q / ok))
         << " 均检测=" << fmt_us(static_cast<long>(det * 1000 / ok))
         << " 均识别=" << fmt_us(static_cast<long>(inf * 1000 / ok))
         << " 均总耗时=" << fmt_us(static_cast<long>(tot * 1000 / ok))
         << " " << pctl(h, ok, 50) << " " << pctl(h, ok, 95);
      if (pool_size > 0) {
        os << " 池占用=" << std::fixed << std::setprecision(2)
           << static_cast<double>(busy) / n << "/" << pool_size;
      }
    }
    return os.str();
  }

 private:
  // 从直方图估分位数。桶内不知道具体分布，报桶下界，形如 P50<=60ms
  static std::string pctl(const uint64_t* h, uint64_t total, int p) {
    const uint64_t target = (total * p + 99) / 100;
    uint64_t acc = 0;
    for (int i = 0; i <= kBuckets; ++i) {
      acc += h[i];
      if (acc >= target) {
        std::ostringstream os;
        // 落到最后一桶说明超过表内最大值，只能报下界
        const int bound = (i < kBuckets) ? kBounds[i] : kBounds[kBuckets - 1];
        os << "P" << p << (i < kBuckets ? "<=" : ">=") << bound << "ms";
        return os.str();
      }
    }
    return "";
  }
};

static ServiceStats g_stats;
static std::atomic<bool> g_stats_stop{false};

// 统计汇总线程。间隔取 1s 粒度轮询，这样关闭时能及时退出，不必等满一个周期。
static void spawn_stats_thread(int interval_s, int pool_size) {
  if (interval_s <= 0) return;
  std::thread([interval_s, pool_size] {
    using clock = std::chrono::steady_clock;
    auto next = clock::now() + std::chrono::seconds(interval_s);
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      if (g_stats_stop.load()) return;
      auto now = clock::now();
      if (now < next) continue;
      const auto period = std::chrono::seconds(interval_s);
      const double elapsed =
          std::chrono::duration<double>(now - (next - period)).count();
      next = now + std::chrono::seconds(interval_s);
      std::string line = g_stats.take_snapshot_line(elapsed, pool_size);
      if (!line.empty()) log_info(line);
    }
  }).detach();
}

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
    Lease(EnginePool& pool, Engine* e, long wait_us, int busy)
        : pool_(pool), engine_(e), wait_us_(wait_us), busy_(busy) {}
    ~Lease() {
      if (engine_) pool_.release(engine_);
    }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& o) noexcept
        : pool_(o.pool_), engine_(o.engine_), wait_us_(o.wait_us_),
          busy_(o.busy_) {
      o.engine_ = nullptr;
    }
    Engine* operator->() const { return engine_; }
    // 在空闲队列上等待的时间（微秒），即真正的排队耗时
    long wait_us() const { return wait_us_; }
    // 拿到这个引擎时，池里同时在用的引擎数。等于池大小说明已经打满，
    // 下个请求必然排队——这一项能从日志里直接看出容量瓶颈。
    int busy() const { return busy_; }

   private:
    EnginePool& pool_;
    Engine* engine_;
    long wait_us_;
    int busy_;
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
    const int busy = size_ - static_cast<int>(idle_.size());
    return Lease(*this, e, wait_us, busy);
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
      g_stats.extract_fail.fetch_add(1);
      log_error("Extract peer=" + peer + " 非法图像参数（" +
                std::to_string(img.width()) + "x" + std::to_string(img.height()) +
                "/" + std::to_string(img.channels()) + "ch 字节数=" +
                std::to_string(img.data().size()) + "），拒绝请求");
      return Status(grpc::INVALID_ARGUMENT,
                    "invalid image: empty data or non-positive dimensions");
    }

    auto engine = pool_.acquire();
    const long queue_us = engine.wait_us();
    const int busy = engine.busy();
    if (log_enabled(2)) {
      log_info("Extract peer=" + peer + " 排队等待引擎=" + fmt_us(queue_us) +
               " 池占用=" + std::to_string(busy) + "/" +
               std::to_string(pool_.size()));
    }

    try {
      auto* p = reinterpret_cast<const unsigned char*>(img.data().data());
      SeetaImageData s{img.width(), img.height(), img.channels(),
                       const_cast<unsigned char*>(p)};

      auto t_detect = std::chrono::steady_clock::now();
      SeetaFaceInfoArray faces = engine->detector_.detect(s);
      long detect_ms = elapsed_ms(t_detect);

      long infer_ms = 0;  // 特征点 + 识别，逐张累加
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
        infer_ms += lm_ms + feat_ms;

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
                   " 张人脸: 框=" + std::to_string(f.pos.width) + "x" +
                   std::to_string(f.pos.height) + " 置信度=" +
                   std::to_string(static_cast<int>(f.score * 100)) + "% 特征点=" +
                   std::to_string(lm_ms) + "ms 特征提取=" +
                   std::to_string(feat_ms) + "ms 特征维度=" +
                   std::to_string(feat.size()));
        }
      }

      const long total_ms = elapsed_ms(t0);
      g_stats.on_extract(queue_us, detect_ms, infer_ms, total_ms, extracted,
                         busy);

      // 一行摘要，适合 production 下观察延迟。排队 + 检测 + 识别应当接近总耗时，
      // 差值是 gRPC 收发、反序列化和日志自身的开销；没检出人脸时识别段为 0，
      // 此时差值就是纯框架开销。池=a/b 表示拿引擎那一刻池里有 a 个在用、
      // 共 b 个——a==b 说明已经打满，再加并发只会排队。
      // level 0 时整条日志（含字符串拼接）都跳过，压测/高吞吐场景不产生 IO。
      if (log_enabled(1)) {
        std::ostringstream os;
        os << "Extract peer=" << peer << " 图像=" << img.width() << "x"
           << img.height() << "/" << img.channels() << "ch 排队="
           << fmt_us(queue_us) << " 检测=" << detect_ms << "ms 识别="
           << infer_ms << "ms 人脸=" << extracted << " 池=" << busy << "/"
           << pool_.size() << " 总耗时=" << total_ms << "ms";
        log_info(os.str());
      }
      return Status::OK;
    } catch (const std::exception& e) {
      g_stats.extract_fail.fetch_add(1);
      log_error("Extract peer=" + peer + " 异常: " + e.what());
      return Status(grpc::INTERNAL, std::string("seeta exception: ") + e.what());
    } catch (...) {
      g_stats.extract_fail.fetch_add(1);
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
      g_stats.compare_fail.fetch_add(1);
      log_error("Compare peer=" + peer + " 特征维度不匹配（f1=" +
                std::to_string(f1.size()) + ", f2=" + std::to_string(f2.size()) +
                "），拒绝请求");
      return Status(grpc::INVALID_ARGUMENT, "feature size mismatch");
    }

    const float sim = dot_similarity(f1.data(), f2.data(), f1.size());
    resp->set_similarity(sim);
    g_stats.compare_ok.fetch_add(1);
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
  int available = 1; // 进程实际可调度的 CPU 数（cpuset / 亲和性），自动配置按它算
};

static CpuTopology detect_cpu_topology() {
  CpuTopology t;
#if defined(__linux__)
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  if (n > 0) t.logical = static_cast<int>(n);

  // 进程实际能调度到哪些 CPU：cpuset（sched_setaffinity）不会改
  // sysconf(_SC_NPROCESSORS_ONLN)，也不会改 cgroup 的 cpu.max，所以必须
  // 显式读亲和性掩码。踩过的坑：docker --cpuset-cpus 0-3 时容器里只有
  // 2 个物理核，但 sysconf 仍报 12，自动配置算出 3x2=6 线程，实测吞吐
  // 只有正确配置的 1/3。
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  bool have_mask = sched_getaffinity(0, sizeof(allowed), &allowed) == 0;
  auto cpu_allowed = [&](int i) -> bool {
    return !have_mask || CPU_ISSET(i, &allowed);
  };
  int avail = 0;
  if (have_mask) {
    for (int i = 0; i < CPU_SETSIZE; ++i) {
      if (CPU_ISSET(i, &allowed)) ++avail;
    }
  }
  t.available = avail > 0 ? avail : t.logical;

  // cgroup v2 / v1 的 CPU 配额可能小于亲和性掩码（容器被限流）
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
  if (quota > 0 && quota < t.available) t.available = quota;

  // 统计 /sys/devices/system/cpu/cpuN/topology/thread_siblings_list：
  // 每个物理核对应一组超线程兄弟，物理核数 = 独立兄弟组数。
  // 只统计亲和性掩码内的 CPU，否则会把容器根本用不到的核算进来。
  int nproc = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
  if (nproc <= 0) nproc = t.logical;
  int groups = 0;
  std::string seen;  // 已统计过的兄弟组首元素
  for (int i = 0; i < nproc; ++i) {
    if (!cpu_allowed(i)) continue;
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
  if (groups > 0) t.physical = std::min(groups, t.available);
  else t.physical = t.available;
  // 离线（offline）的核不计入，这里用 std::max(1,..) 兜底
  if (t.physical < 1) t.physical = 1;
  if (t.logical < t.physical) t.logical = t.physical;
  if (t.available < 1) t.available = 1;
#else
  unsigned hc = std::thread::hardware_concurrency();
  t.logical = t.physical = t.available = hc > 0 ? static_cast<int>(hc) : 1;
#endif
  return t;
}

// 根据「可调度 CPU 数」和「物理核数」给出默认的 (引擎池大小, 每引擎线程数)。
//
// 实测单请求 P50（12 逻辑核 / 6 物理核的机器，检测 320px / 640px / 1280px）：
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
//
// ---- 小机器上的坑（256px 检测耗时，单请求，多次取中位）----
//   cpuset 可用 CPU 数    threads=1    threads=2
//        1                  12ms         12ms
//        2                   6ms         35ms
//        3                   6ms         53ms
//        4                   6ms         40~65ms
//        5                   6ms          5ms
//       12                   5ms          5ms
// 即：可用 CPU <= 4 时，每引擎 2 线程会让检测慢 7~10 倍（识别阶段不受影响，
// 始终 45~55ms）。libgomp 的 OMP_WAIT_POLICY / GOMP_SPINCOUNT /
// OMP_PROC_BIND 都改不动它，因为真正并行的是 TenniS 自己的线程池
// （ts_Workbench_set_computing_thread_number），不是 OpenMP。
//
// 同一个 2 物理核机器（可用 4 CPU，256px，threads=1）的池大小对比：
//   pool   内存      c=1        c=4            c=8
//     1   311MB  14.8 QPS   14.7/268ms   14.8/539ms   <-- 并发再高吞吐也不涨
//     2   536MB  14.6 QPS   27.6/142ms   26.7/294ms
//     3   760MB  14.3 QPS   37.5/ 98ms   35.3/224ms
//     4   948MB  14.2 QPS   41.3/ 92ms   41.9/188ms   <-- 当前默认
// 单请求延迟与池大小无关（都是约 68ms），池大小只影响并发时的排队。
// pool=1 时所有请求挤同一个引擎排队，这就是生产日志里排队 242ms 的来源。
// 内存吃紧可以用 --pool 2：一半内存换约 2/3 吞吐。
//
// 可用 CPU >= 5 时总线程数决定吞吐，最佳区间是「总线程 ≈ 物理核数」，
// pool x threads 的实测（640px）：4x1=18.6  6x1=18.3  3x2=14.9  4x2=17.5
// 因此大机器上仍按「每引擎 2 线程、池大小 = 物理核/2」配置，内存更省，
// 单请求延迟也更好（6 物理核：3x2 单并发 26.5 QPS，6x1 只有 16.7 QPS）。
//
// 注意 available_cpus 是「实际可调度 CPU 数」，没有任何 cpuset/cgroup 限制时
// 它就等于 nproc。所以一台原生 2 物理核/4 逻辑核的机器（不跑容器也一样）
// 会得到 4x1 + 约 950MB 常驻内存——这是刻意的，见上面的实测数据。
//
// 每个引擎实例要复制一份模型（约 225MB 常驻，进程基线约 310MB），池上限取 8。
static void auto_tune(int available_cpus, int physical_cores, int* pool,
                      int* threads) {
  const int kMaxPool = 8;
  const int kMaxThreads = 4;  // 超过 4 无收益，12 更是灾难

  if (available_cpus <= 1) {
    *pool = 1;
    *threads = 1;  // 单核机器加线程只会互相抢，直接串行
    return;
  }

  if (available_cpus <= 4) {
    // 小机器：每引擎只给 1 线程，靠多实例撑并发。
    // 线程数必须 <= 可用 CPU 数——2 线程在 4 CPU 上会让检测慢 10 倍。
    // 代价是内存：4 核机器 4 个引擎约 950MB 常驻。内存吃紧可用 --pool 2。
    *threads = 1;
    *pool = available_cpus;
    if (*pool > kMaxPool) *pool = kMaxPool;
    return;
  }

  // 可用 CPU >= 5：每引擎 2 线程（收益最大的一段），池大小 = 物理核/2
  *threads = 2;
  *pool = physical_cores / 2;
  if (*pool < 2) *pool = 2;  // 再小的机器也别退化成单引擎串行
  if (*pool > kMaxPool) {
    *pool = kMaxPool;
    *threads = physical_cores / kMaxPool;
  }
  if (*threads > kMaxThreads) *threads = kMaxThreads;
  if (*threads < 1) *threads = 1;
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
            << "  --log-level <N>           日志级别 0=仅启动/告警 1=每请求摘要 2=详细 (默认 1)\n"
            << "  --stats-interval <N>      运行统计汇总间隔秒数, 0=关闭 (默认 60)\n"
            << "  --det <path>              人脸检测模型 (默认 model/face_detector.csta)\n"
            << "  --lm  <path>              特征点模型 (默认 model/face_landmarker_pts5.csta)\n"
            << "  --fr  <path>              人脸识别模型 (默认 model/face_recognizer.csta)\n"
            << "\n自动配置原则：可调度 CPU <= 4 时按「1 线程 x N 个引擎」，\n"
            << "否则按「2 线程 x (物理核/2) 个引擎」，总线程数 ≈ 物理核数。\n"
            << "可调度 CPU 会同时考虑 cpuset(--cpuset-cpus) 与 cgroup 配额(--cpus)。\n"
            << "小机器上别手动把 --threads 调到 2 以上，检测会慢 7~10 倍。\n"
            << "\n环境变量: SEETA_DEVICE(计算设备) SEETA_POOL(引擎池)\n"
            << "         SEETA_THREADS(每引擎线程数) SEETA_LOG_LEVEL(日志级别)\n"
            << "         SEETA_STATS_INTERVAL(统计汇总间隔秒数, 0=关闭)\n"
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
  int stats_interval = 60;  // 运行期统计汇总间隔（秒），0 = 关闭

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
  if (const char* v = std::getenv("SEETA_STATS_INTERVAL")) {
    if (*v) {
      char* end = nullptr;
      long n = std::strtol(v, &end, 10);
      if (end != v && n >= 0) stats_interval = static_cast<int>(n);
    }
  }
  // 日志级别：SEETA_LOG_LEVEL 未设置时保持默认 1；显式设 0 表示只留启动与告警
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
    } else if (a == "--stats-interval") {
      stats_interval = std::stoi(value_of(i));
    } else if (a.rfind("--stats-interval=", 0) == 0) {
      stats_interval = std::stoi(a.substr(17));
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
  auto_tune(cpu.available, cpu.physical, &auto_pool, &auto_threads);
  const bool pool_is_auto = pool_size <= 0;
  const bool threads_is_auto = threads <= 0;
  if (pool_is_auto) pool_size = auto_pool;
  if (threads_is_auto) threads = auto_threads;
  if (pool_size < 1) pool_size = 1;
  if (threads < 1) threads = 1;

  log_info("==== facerec-grpc 服务端启动 [" + node_id() + "] ====");
  log_info("CPU 拓扑: 物理核=" + std::to_string(cpu.physical) + " 逻辑核=" +
           std::to_string(cpu.logical) + " 可调度 CPU=" +
           std::to_string(cpu.available) + "（取 sysconf/亲和性/cgroup 配额的最小值）");
  log_info("配置: 设备=" + std::string(device_name(dev)) + " 引擎池=" +
           std::to_string(pool_size) +
           (pool_is_auto ? "（自动）" : "（手动指定）") + " 每引擎线程数=" +
           std::to_string(threads) +
           (threads_is_auto ? "（自动）" : "（手动指定）") + " 监听地址=" + addr);
  log_info("推理并行度: " + std::to_string(pool_size) + " x " +
           std::to_string(threads) + " = " +
           std::to_string(pool_size * threads) + " 线程 / " +
           std::to_string(cpu.physical) + " 物理核");
  // 小机器上每引擎多线程的代价远大于收益：可用 CPU <= 4 时 threads=2
  // 会让检测慢 7~10 倍（见 auto_tune 的实测表）。这里显式警告手动配置踩坑。
  // 注意小机器的池大小是按「可调度 CPU 数」配的（4 核就配 4 个引擎），
  // 引擎数超过物理核数是刻意的：每引擎 1 线程，引擎之间是阻塞租借而非
  // 自旋竞争，实测 4x1 在 2 物理核上 c=8 有 40 QPS，远超 1x2 的 9 QPS。
  if (threads > 1 && cpu.available <= 4) {
    log_warn("可调度 CPU 只有 " + std::to_string(cpu.available) +
             " 个，每引擎 " + std::to_string(threads) +
             " 线程会让检测慢 7~10 倍；建议 --threads 1 --pool " +
             std::to_string(std::min(cpu.available, 8)));
  } else if (pool_size * threads > cpu.available) {
    log_warn("推理线程数(" + std::to_string(pool_size * threads) +
             ") 超过可调度 CPU 数(" + std::to_string(cpu.available) +
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

  const long rss = rss_kb();
  log_info("日志级别=" + std::to_string(g_log_level.load()) +
           (stats_interval > 0 ? " 统计汇总间隔=" +
                                     std::to_string(stats_interval) + "s"
                               : " 统计汇总已关闭") +
           (rss > 0 ? " 常驻内存=" + std::to_string(rss / 1024) + "MB"
                    : std::string()));
  log_info("服务就绪，等待客户端连接...");

  spawn_stats_thread(stats_interval, pool_size);
  spawn_signal_thread();

  g_server->Wait();

  // 关闭时把最后一个不完整周期的统计也打出来，避免丢失最后几十秒的数据
  g_stats_stop.store(true);
  {
    std::string tail = g_stats.take_snapshot_line(0.0, pool_size);
    if (!tail.empty()) log_info("退出前统计（本周期）: " + tail);
  }
  log_info("服务已停止");
  AsyncLogger::instance().shutdown();  // 冲刷队列中剩余的日志
  return 0;
}
