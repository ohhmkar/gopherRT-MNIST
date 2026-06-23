package main

type Tensor struct {
	Shape []int32
	Data  []float32
}

func newTensor(shape []int32, data []float32) *Tensor {
	return &Tensor{Shape: shape, Data: data}
}
