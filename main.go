package main

import (
	"fmt"
	"os"
	"time"

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
	images, _ := loadImages("data/t10k-images-idx3-ubyte")
	labels, _ := loadLabels("data/t10k-labels-idx1-ubyte")

	start := time.Now()
	correct := 0
	n := len(images)

	w1 := values["StatefulPartitionedCall/sequential_1/dense_1/Cast/ReadVariableOp:0"]
	b1 := values["StatefulPartitionedCall/sequential_1/dense_1/BiasAdd/ReadVariableOp:0"]
	w2 := values["StatefulPartitionedCall/sequential_1/dense_1_2/Cast/ReadVariableOp:0"]
	b2 := values["StatefulPartitionedCall/sequential_1/dense_1_2/BiasAdd/ReadVariableOp:0"]
	w3 := values["StatefulPartitionedCall/sequential_1/dense_2_1/Cast/ReadVariableOp:0"]
	b3 := values["StatefulPartitionedCall/sequential_1/dense_2_1/BiasAdd/ReadVariableOp:0"]

	// for i := 0; i < n; i++ {
	// 	values["input_layer"] = &Tensor{Shape: []int{1, 784}, Data: images[i]}

	// 	for _, node := range model.GetGraph().GetNode() {
	// 		in := values[node.GetInput()[0]]
	// 		var out *Tensor
	// 		switch node.GetOpType() {
	// 		case "MatMul":
	// 			lastMMOutput = node.GetOutput()[0]
	// 			lastMMInput = node.GetInput()[0]
	// 			lastMMWeight = node.GetInput()[1]
	// 			out = nil // fused Add will handle it
	// 		case "Add":
	// 			if node.GetInput()[0] == lastMMOutput {
	// 				// Fuse: use Dense instead
	// 				w := values[lastMMWeight]
	// 				b := values[node.GetInput()[1]]
	// 				out = Dense(values[lastMMInput], w, b)
	// 			} else {
	// 				b := values[node.GetInput()[1]]
	// 				out = add(in, b)
	// 			}
	// 		case "Relu":
	// 			out = relu(in)
	// 		case "Softmax":
	// 			out = softmax(in)
	// 		}
	// 		values[node.GetOutput()[0]] = out
	// 	}

	// 	pred := argmax(values[model.GetGraph().GetOutput()[0].GetName()].Data)
	// 	if pred == int(labels[i]) {
	// 		correct++
	// 	}
	// }

	for i := 0; i < n; i++ {
		values["input_layer"] = &Tensor{Shape: []int{1, 784}, Data: images[i]}
		h1 := relu(Dense(values["input_layer"], w1, b1))
		h2 := relu(Dense(h1, w2, b2))
		out := softmax(Dense(h2, w3, b3))
		pred := argmax(out.Data)
		if pred == int(labels[i]) {
			correct++
		}
	}

	end := time.Since(start)
	fmt.Printf("Correct: %d\n", correct)
	fmt.Printf("Images processed: %d\n", n)
	fmt.Printf("Accuracy: %.3f\n", float64(correct)/float64(n))
	fmt.Printf("Time taken: %d ms\n", (end.Milliseconds()))
	fmt.Printf("Time taken per image: %.4f ms\n", float64(end.Milliseconds())/float64(n))
}
