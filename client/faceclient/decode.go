package faceclient

import (
	"bytes"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
)

// DecodeBGR 把图片原始字节解码为 BGR 三通道像素（SeetaFace 要求 BGR888）。
func DecodeBGR(data []byte) (bgr []byte, width, height int, err error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, err
	}
	b := img.Bounds()
	width, height = b.Dx(), b.Dy()
	bgr = make([]byte, width*height*3)
	i := 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, _ := img.At(x, y).RGBA() // 0..65535
			bgr[i], bgr[i+1], bgr[i+2] = byte(bl>>8), byte(g>>8), byte(r>>8)
			i += 3
		}
	}
	return bgr, width, height, nil
}
