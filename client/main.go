package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"facerec/faceclient"
)

const usageText = `用法: facerec <command> [flags]

命令:
  extract   从图片提取人脸特征
  register  注册人脸到底库
  identify  在底库中识别人脸
  compare   比对两张图片的相似度
  bench     压测服务端

全局 flag:
  -addr     服务端地址（默认 localhost:50051）

示例:
  facerec extract -img a.jpg -out feat.json -name 张三
  facerec register -name 张三 -img a.jpg
  facerec identify -img b.jpg
  facerec compare -img1 a.jpg -img2 b.jpg
  facerec bench -img a.jpg -c 10 -n 100
`

func main() {
	if len(os.Args) < 2 {
		fmt.Print(usageText)
		os.Exit(1)
	}
	switch os.Args[1] {
	case "extract":
		cmdExtract(os.Args[2:])
	case "register":
		cmdRegister(os.Args[2:])
	case "identify":
		cmdIdentify(os.Args[2:])
	case "compare":
		cmdCompare(os.Args[2:])
	case "bench":
		cmdBench(os.Args[2:])
	case "help", "-h", "--help":
		fmt.Print(usageText)
	default:
		fmt.Fprintf(os.Stderr, "未知命令: %s\n\n", os.Args[1])
		fmt.Print(usageText)
		os.Exit(1)
	}
}

// ===== extract：提取人脸特征 =====
func cmdExtract(args []string) {
	fs := flag.NewFlagSet("extract", flag.ExitOnError)
	addr := fs.String("addr", "localhost:50051", "gRPC 服务端地址")
	img := fs.String("img", "", "图片路径（必填）")
	out := fs.String("out", "", "可选：把特征保存为 JSON 文件")
	name := fs.String("name", "", "可选：人名标签，随特征一起保存（建底库用）")
	fs.Parse(args)

	if *img == "" {
		log.Fatal("请用 -img 指定图片路径")
	}

	c, err := faceclient.New(*addr)
	if err != nil {
		log.Fatalf("连接服务端失败: %v", err)
	}
	defer c.Close()

	// 读图 + 解码成 BGR 像素
	data, err := os.ReadFile(*img)
	if err != nil {
		log.Fatalf("读图失败: %v", err)
	}
	bgr, w, h, err := faceclient.DecodeBGR(data)
	if err != nil {
		log.Fatalf("解码失败: %v", err)
	}
	fmt.Printf("图片尺寸: %dx%d\n", w, h)

	// 提取人脸特征
	feat, err := c.ExtractBGR(bgr, w, h, 3)
	if err != nil {
		log.Fatalf("提取特征失败: %v", err)
	}

	fmt.Printf("特征提取成功，维度=%d\n", len(feat))
	fmt.Print("前 10 个分量: ")
	for i := 0; i < 10 && i < len(feat); i++ {
		fmt.Printf("%.4f ", feat[i])
	}
	fmt.Println()

	// 可选：保存到文件
	if *out != "" {
		record := map[string]any{"feature": feat}
		if *name != "" {
			record["name"] = *name
		}
		b, err := json.MarshalIndent(record, "", "  ")
		if err != nil {
			log.Fatalf("序列化失败: %v", err)
		}
		if err := os.WriteFile(*out, b, 0644); err != nil {
			log.Fatalf("保存失败: %v", err)
		}
		fmt.Printf("特征已保存到 %s\n", *out)
	}
}

// ===== register：注册人脸到底库 =====
func cmdRegister(args []string) {
	fs := flag.NewFlagSet("register", flag.ExitOnError)
	addr := fs.String("addr", "localhost:50051", "gRPC 服务端地址")
	dbPath := fs.String("db", "face_db.json", "底库文件路径")
	name := fs.String("name", "", "人名（必填）")
	img := fs.String("img", "", "图片路径（必填）")
	fs.Parse(args)

	if *name == "" || *img == "" {
		log.Fatal("请用 -name 指定人名、-img 指定图片路径")
	}

	c, err := faceclient.New(*addr)
	if err != nil {
		log.Fatalf("连接失败: %v", err)
	}
	defer c.Close()

	// 提取特征
	data, err := os.ReadFile(*img)
	if err != nil {
		log.Fatalf("读图失败: %v", err)
	}
	feat, err := c.GetFeature(data)
	if err != nil {
		log.Fatalf("提取特征失败: %v", err)
	}

	// 写入底库
	db, err := faceclient.LoadDB(*dbPath)
	if err != nil {
		log.Fatalf("读取底库失败: %v", err)
	}
	db.Upsert(*name, feat)
	if err := db.SaveDB(*dbPath); err != nil {
		log.Fatalf("保存底库失败: %v", err)
	}

	fmt.Printf("已注册: %s <- %s（特征维度 %d，底库当前 %d 人）\n",
		*name, *img, len(feat), len(db.Persons))
}

// ===== identify：在底库中识别人脸 =====
func cmdIdentify(args []string) {
	fs := flag.NewFlagSet("identify", flag.ExitOnError)
	addr := fs.String("addr", "localhost:50051", "gRPC 服务端地址")
	dbPath := fs.String("db", "face_db.json", "底库文件路径")
	img := fs.String("img", "", "待识别图片路径（必填）")
	threshold := fs.Float64("threshold", 0.6, "判定阈值，相似度高于此值才判定匹配")
	fs.Parse(args)

	if *img == "" {
		log.Fatal("请用 -img 指定待识别图片")
	}

	c, err := faceclient.New(*addr)
	if err != nil {
		log.Fatalf("连接失败: %v", err)
	}
	defer c.Close()

	db, err := faceclient.LoadDB(*dbPath)
	if err != nil {
		log.Fatalf("读取底库失败: %v", err)
	}
	if len(db.Persons) == 0 {
		log.Fatal("底库为空，请先用 register 命令注册")
	}

	// 提取特征
	data, err := os.ReadFile(*img)
	if err != nil {
		log.Fatalf("读图失败: %v", err)
	}
	feat, err := c.GetFeature(data)
	if err != nil {
		log.Fatalf("提取特征失败: %v", err)
	}

	// 与底库所有人比对
	matches, err := db.Identify(c, feat)
	if err != nil {
		log.Fatalf("识别失败: %v", err)
	}

	fmt.Println("底库匹配结果（按相似度降序）：")
	for i, m := range matches {
		fmt.Printf("  %d. %s  相似度=%.4f\n", i+1, m.Name, m.Similarity)
	}

	best := matches[0]
	if float64(best.Similarity) >= *threshold {
		fmt.Printf("\n识别结果：%s（相似度 %.4f）\n", best.Name, best.Similarity)
	} else {
		fmt.Printf("\n未识别：最高相似度 %.4f 低于阈值 %.2f\n", best.Similarity, *threshold)
	}
}

// ===== compare：比对两张图片的相似度 =====
func cmdCompare(args []string) {
	fs := flag.NewFlagSet("compare", flag.ExitOnError)
	addr := fs.String("addr", "localhost:50051", "gRPC server address")
	img1 := fs.String("img1", "a.jpg", "first image path")
	img2 := fs.String("img2", "b.jpg", "second image path")
	fs.Parse(args)

	c, err := faceclient.New(*addr)
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer c.Close()

	f1, dec1, ext1 := process(c, *img1)
	fmt.Printf("[%s] 特征维度=%d  解码=%v  提取=%v\n", *img1, len(f1), dec1, ext1)

	f2, dec2, ext2 := process(c, *img2)
	fmt.Printf("[%s] 特征维度=%d  解码=%v  提取=%v\n", *img2, len(f2), dec2, ext2)

	t := time.Now()
	sim, err := c.Compare(f1, f2)
	compare := time.Since(t)
	if err != nil {
		log.Fatalf("compare: %v", err)
	}
	fmt.Printf("相似度=%.4f  比对=%v\n", sim, compare)
}

func process(c *faceclient.Client, path string) (feat []float32, decode, extract time.Duration) {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("read %s: %v", path, err)
	}

	t0 := time.Now()
	bgr, w, h, err := faceclient.DecodeBGR(data)
	decode = time.Since(t0)
	if err != nil {
		log.Fatalf("decode %s: %v", path, err)
	}

	t1 := time.Now()
	feat, err = c.ExtractBGR(bgr, w, h, 3)
	extract = time.Since(t1)
	if err != nil {
		log.Fatalf("extract %s: %v", path, err)
	}
	return feat, decode, extract
}

// ===== bench：压测服务端 =====
func cmdBench(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	addr := fs.String("addr", "localhost:50051", "gRPC 服务端地址")
	img := fs.String("img", "", "测试图片路径（必填）")
	concurrency := fs.Int("c", 10, "并发数")
	total := fs.Int("n", 100, "总请求数")
	fs.Parse(args)

	if *img == "" {
		log.Fatal("请用 -img 指定测试图片")
	}

	c, err := faceclient.New(*addr)
	if err != nil {
		log.Fatalf("连接失败: %v", err)
	}
	defer c.Close()

	// 预解码一次，把解码时间排除在测量之外（只测服务端推理 + 网络）
	data, err := os.ReadFile(*img)
	if err != nil {
		log.Fatalf("读图失败: %v", err)
	}
	bgr, w, h, err := faceclient.DecodeBGR(data)
	if err != nil {
		log.Fatalf("解码失败: %v", err)
	}
	fmt.Printf("测试图片: %s (%dx%d)\n", *img, w, h)
	fmt.Printf("并发数=%d  总请求数=%d\n", *concurrency, *total)

	latencies := make([]time.Duration, *total)
	var okCount, failCount int64
	var idx int64 = -1
	var wg sync.WaitGroup

	start := time.Now()
	for i := 0; i < *concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				n := atomic.AddInt64(&idx, 1)
				if n >= int64(*total) {
					return
				}
				t0 := time.Now()
				_, err := c.ExtractBGR(bgr, w, h, 3)
				latencies[n] = time.Since(t0)
				if err != nil {
					atomic.AddInt64(&failCount, 1)
				} else {
					atomic.AddInt64(&okCount, 1)
				}
			}
		}()
	}
	wg.Wait()
	elapsed := time.Since(start)

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	fmt.Println("\n===== 压测结果 =====")
	fmt.Printf("总耗时: %v\n", elapsed)
	fmt.Printf("成功: %d  失败: %d\n", okCount, failCount)
	fmt.Printf("QPS: %.2f\n", float64(*total)/elapsed.Seconds())
	fmt.Printf("平均延迟: %v\n", avg(latencies))
	fmt.Printf("最小: %v  最大: %v\n", latencies[0], latencies[len(latencies)-1])
	fmt.Printf("P50: %v\n", percentile(latencies, 50))
	fmt.Printf("P90: %v\n", percentile(latencies, 90))
	fmt.Printf("P95: %v\n", percentile(latencies, 95))
	fmt.Printf("P99: %v\n", percentile(latencies, 99))
}

func avg(ds []time.Duration) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	var sum time.Duration
	for _, d := range ds {
		sum += d
	}
	return sum / time.Duration(len(ds))
}

func percentile(sorted []time.Duration, p int) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	i := (len(sorted) - 1) * p / 100
	return sorted[i]
}
