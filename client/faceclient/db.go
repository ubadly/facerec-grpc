package faceclient

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"
)

// Person 底库中的一个人。
type Person struct {
	Name    string    `json:"name"`
	Feature []float32 `json:"feature"`
}

// Database 人脸底库。
type Database struct {
	Persons []Person `json:"persons"`
}

// LoadDB 从 JSON 文件读取底库；文件不存在时返回空库。
func LoadDB(path string) (*Database, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &Database{}, nil
		}
		return nil, err
	}
	var db Database
	if err := json.Unmarshal(b, &db); err != nil {
		return nil, fmt.Errorf("解析底库 %s: %w", path, err)
	}
	return &db, nil
}

// SaveDB 保存底库到 JSON 文件。
func (db *Database) SaveDB(path string) error {
	b, err := json.MarshalIndent(db, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0644)
}

// Upsert 添加一个人，或更新同名人的特征。
func (db *Database) Upsert(name string, feat []float32) {
	for i := range db.Persons {
		if db.Persons[i].Name == name {
			db.Persons[i].Feature = feat
			return
		}
	}
	db.Persons = append(db.Persons, Person{Name: name, Feature: feat})
}

// Match 一条识别结果。
type Match struct {
	Name       string
	Similarity float32
}

// Identify 在底库中找与 feat 最相似的人，返回按相似度降序的结果。
// 逐人串行调用 Compare，底库较大时较慢；推荐用 IdentifyConcurrent。
func (db *Database) Identify(c *Client, feat []float32) ([]Match, error) {
	if len(db.Persons) == 0 {
		return nil, fmt.Errorf("底库为空")
	}
	matches := make([]Match, 0, len(db.Persons))
	for _, p := range db.Persons {
		sim, err := c.Compare(feat, p.Feature)
		if err != nil {
			return nil, fmt.Errorf("比对 %s 失败: %w", p.Name, err)
		}
		matches = append(matches, Match{Name: p.Name, Similarity: sim})
	}
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Similarity > matches[j].Similarity
	})
	return matches, nil
}

// IdentifyConcurrent 并发地在底库中检索，workers 为并发协程数。
// Compare 在服务端是纯点积计算（不占推理引擎），因此并发调用能线性加速。
func (db *Database) IdentifyConcurrent(c *Client, feat []float32, workers int) ([]Match, error) {
	if len(db.Persons) == 0 {
		return nil, fmt.Errorf("底库为空")
	}
	if workers < 1 {
		workers = 1
	}
	if workers > len(db.Persons) {
		workers = len(db.Persons)
	}

	type result struct {
		idx int
		sim float32
		err error
	}
	jobs := make(chan int)
	results := make(chan result, len(db.Persons))

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				sim, err := c.Compare(feat, db.Persons[i].Feature)
				results <- result{idx: i, sim: sim, err: err}
			}
		}()
	}
	go func() {
		for i := range db.Persons {
			jobs <- i
		}
		close(jobs)
	}()
	wg.Wait()
	close(results)

	matches := make([]Match, 0, len(db.Persons))
	for r := range results {
		if r.err != nil {
			return nil, fmt.Errorf("比对 %s 失败: %w", db.Persons[r.idx].Name, r.err)
		}
		matches = append(matches, Match{Name: db.Persons[r.idx].Name, Similarity: r.sim})
	}
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Similarity > matches[j].Similarity
	})
	return matches, nil
}
