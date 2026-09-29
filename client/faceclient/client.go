package faceclient

import (
	"context"
	"fmt"
	"time"

	pb "facerec/facerec"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const maxMsgSize = 64 * 1024 * 1024

// Client 是人脸识别 gRPC 服务的封装。
type Client struct {
	conn *grpc.ClientConn
	stub pb.FaceRecognitionClient
}

// New 连接人脸识别服务端，addr 形如 "localhost:50051"。
func New(addr string) (*Client, error) {
	conn, err := grpc.Dial(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxMsgSize),
			grpc.MaxCallSendMsgSize(maxMsgSize),
		),
	)
	if err != nil {
		return nil, err
	}
	return &Client{conn: conn, stub: pb.NewFaceRecognitionClient(conn)}, nil
}

// Close 释放连接。
func (c *Client) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// ExtractBGR 输入 BGR 原始像素，返回第一张人脸的特征向量。
func (c *Client) ExtractBGR(bgr []byte, width, height, channels int) ([]float32, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	resp, err := c.stub.Extract(ctx, &pb.ExtractRequest{
		Image: &pb.ImageData{Data: bgr, Width: int32(width), Height: int32(height), Channels: int32(channels)},
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Faces) == 0 {
		return nil, fmt.Errorf("no face detected")
	}
	return resp.Faces[0].Feature, nil
}

// GetFeature 把图片原始字节（jpeg/png）转成第一张人脸的特征向量。
func (c *Client) GetFeature(imageBytes []byte) ([]float32, error) {
	bgr, w, h, err := DecodeBGR(imageBytes)
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return c.ExtractBGR(bgr, w, h, 3)
}

// Compare 计算两个特征向量的相似度（0~1，越大越像）。
func (c *Client) Compare(f1, f2 []float32) (float32, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	resp, err := c.stub.Compare(ctx, &pb.CompareRequest{Feature1: f1, Feature2: f2})
	if err != nil {
		return 0, err
	}
	return resp.Similarity, nil
}
