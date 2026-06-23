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

	// weights := make(map[string][]float32)
	// bias := make(map[string][]float32)

	for _, init := range model.GetGraph().GetInitializer() {
		data, _ := tensorToFloats(init)
		name := init.GetName()

		fmt.Printf("%s shape = %v len=%d\n", name, init.GetDims(), len(data))
	}
}
