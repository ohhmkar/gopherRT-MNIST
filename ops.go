package main

import "math"

func matMul(a, b *Tensor) *Tensor {
	m := a.Shape[0]
	n := b.Shape[1]
	k := b.Shape[0]

	data := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for l := 0; l < k; l++ {
				sum += a.Data[i*k+l] * b.Data[l*n+j]
			}
			data[i*n+j] = sum
		}
	}
	return &Tensor{Shape: []int{m, n}, Data: data}
}

func add(a, b *Tensor) *Tensor {
	m := a.Shape[0]
	n := a.Shape[1]

	data := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			data[i*n+j] = a.Data[i*n+j] + b.Data[j]
		}
	}
	return &Tensor{Shape: []int{m, n}, Data: data}
}

func relu(a *Tensor) *Tensor {
	data := make([]float32, len(a.Data))
	for i, v := range a.Data {
		if v > 0 {
			data[i] = v
		}
	}
	return &Tensor{Shape: a.Shape, Data: data}
}

func softmax(a *Tensor) *Tensor {
	n := a.Shape[1]
	data := make([]float32, n)
	//finding max
	maxVal := a.Data[0]
	for _, val := range a.Data {
		if val > maxVal {
			maxVal = val
		}
	}
	var sum float32
	for i, v := range a.Data {
		data[i] = float32(math.Exp(float64(v - maxVal)))
		sum += data[i]
	}
	for i := range data {
		data[i] /= sum
	}
	return &Tensor{Shape: a.Shape, Data: data}
}
