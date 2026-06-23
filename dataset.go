package main

import (
	"encoding/binary"
	"os"
)

func loadImages(path string) ([][]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	// magic uint32, num uint32, rows uint32, cols uint32, then pixels
	num := int(binary.BigEndian.Uint32(data[4:8]))
	rows := int(binary.BigEndian.Uint32(data[8:12]))
	cols := int(binary.BigEndian.Uint32(data[12:16]))
	pixels := rows * cols

	images := make([][]float32, num)
	for i := 0; i < num; i++ {
		start := 16 + i*pixels
		img := make([]float32, pixels)
		for j := 0; j < pixels; j++ {
			img[j] = float32(data[start+j]) / 255.0
		}
		images[i] = img
	}
	return images, nil
}

func loadLabels(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	num := int(binary.BigEndian.Uint32(data[4:8]))
	return data[8 : 8+num], nil
}
