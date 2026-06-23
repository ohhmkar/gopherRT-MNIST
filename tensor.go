package main

type Tensor struct {
	Shape []int
	Data  []float32
}

func newTensor(shape []int, data []float32) *Tensor {
	return &Tensor{Shape: shape, Data: data}
}
