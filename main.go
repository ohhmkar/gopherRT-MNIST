package main

import (
	"fmt"
	"os"

	"github.com/omkargajare/gopherRT/onnx"
	"google.golang.org/protobuf/proto"
)

func main() {
	data, _ := os.ReadFile("mnist.onnx")
	model := &onnx.ModelProto{}
	err := proto.Unmarshal(data, model)
	if err != nil {
		fmt.Println(err)
	}
	//populating value map with initializers
	values := make(map[string]*Tensor)
	for _, init := range model.GetGraph().GetInitializer() {
		floats, _ := tensorToFloats(init)
		dims := init.GetDims()
		shape := make([]int, len(dims))
		for i, d := range dims {
			shape[i] = int(d)
		}
		values[init.GetName()] = &Tensor{Shape: shape, Data: floats}
	}

	//Input
	values["input_layer"] = &Tensor{Shape: []int{1, 784}, Data: make([]float32, 784)}

	for _, node := range model.GetGraph().GetNode() {
		in := values[node.GetInput()[0]]
		var out *Tensor
		switch node.GetOpType() {
		case "MatMul":
			w := values[node.GetInput()[1]]
			out = matMul(in, w)
		case "Add":
			b := values[node.GetInput()[1]]
			out = add(in, b)

		case "Relu":
			out = relu(in)
		case "Softmax":
			out = softmax(in)
		}
		values[node.GetOutput()[0]] = out
	}

	for _, out := range model.GetGraph().GetOutput() {
		fmt.Println(values[out.GetName()].Data)
	}
}
