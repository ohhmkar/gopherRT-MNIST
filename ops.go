package main

import "math"

func matMul(a, b *Tensor) *Tensor {
	m := a.Shape[0]
	n := b.Shape[1]
	k := b.Shape[0]

	data := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for l := 0; l < k; l++ {
			aVal := a.Data[i*k+l]
			for j := 0; j < n; j++ {
				data[i*n+j] += aVal * b.Data[l*n+j]
			}
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

func Dense(input, weight, bias *Tensor) *Tensor {
	m := input.Shape[0]
	k := input.Shape[1]
	n := weight.Shape[1]
	data := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for l := 0; l < k; l++ {
			aVal := input.Data[i*k+l]
			for j := 0; j < n; j++ {
				data[i*n+j] += aVal * weight.Data[l*n+j]
			}
		}
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			data[i*n+j] += bias.Data[j]
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

func argmax(data []float32) int {
	idx := 0
	for i, v := range data {
		if v > data[idx] {
			idx = i
		}
	}
	return idx
}
