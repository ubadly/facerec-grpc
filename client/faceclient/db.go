package faceclient

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
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
