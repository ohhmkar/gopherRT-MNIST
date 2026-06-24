package main

import (
	"fmt"
	"log"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/omkargajare/gopherRT/onnx"
	"google.golang.org/protobuf/proto"
)

func main() {
	if err := ensureDataset("data"); err != nil {
		log.Fatalf("dataset: %v", err)
	}

	modelData, err := os.ReadFile("mnist.onnx")
	if err != nil {
		log.Fatalf("reading model: %v", err)
	}
	model := &onnx.ModelProto{}
	if err := proto.Unmarshal(modelData, model); err != nil {
		log.Fatalf("parsing model: %v", err)
	}

	values := make(map[string]*Tensor)
	for _, init := range model.GetGraph().GetInitializer() {
		floats, err := tensorToFloats(init)
		if err != nil {
			log.Fatalf("reading initializer %s: %v", init.GetName(), err)
		}
		dims := init.GetDims()
		shape := make([]int, len(dims))
		for i, d := range dims {
			shape[i] = int(d)
		}
		values[init.GetName()] = &Tensor{Shape: shape, Data: floats}
	}

	images, err := loadImages("data/t10k-images-idx3-ubyte")
	if err != nil {
		log.Fatalf("loading images: %v", err)
	}
	labels, err := loadLabels("data/t10k-labels-idx1-ubyte")
	if err != nil {
		log.Fatalf("loading labels: %v", err)
	}

	w1 := values["StatefulPartitionedCall/sequential_1/dense_1/Cast/ReadVariableOp:0"]
	b1 := values["StatefulPartitionedCall/sequential_1/dense_1/BiasAdd/ReadVariableOp:0"]
	w2 := values["StatefulPartitionedCall/sequential_1/dense_1_2/Cast/ReadVariableOp:0"]
	b2 := values["StatefulPartitionedCall/sequential_1/dense_1_2/BiasAdd/ReadVariableOp:0"]
	w3 := values["StatefulPartitionedCall/sequential_1/dense_2_1/Cast/ReadVariableOp:0"]
	b3 := values["StatefulPartitionedCall/sequential_1/dense_2_1/BiasAdd/ReadVariableOp:0"]

	numCPU := runtime.NumCPU()
	chunkSize := (len(images) + numCPU - 1) / numCPU

	start := time.Now()
	var mu sync.Mutex
	var correct int

	var wg sync.WaitGroup
	for c := 0; c < numCPU; c++ {
		lo := c * chunkSize
		hi := lo + chunkSize
		if hi > len(images) {
			hi = len(images)
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			localCorrect := 0
			for i := lo; i < hi; i++ {
				input := &Tensor{Shape: []int{1, 784}, Data: images[i]}
				h1 := relu(dense(input, w1, b1))
				h2 := relu(dense(h1, w2, b2))
				out := softmax(dense(h2, w3, b3))
				if argmax(out.Data) == int(labels[i]) {
					localCorrect++
				}
			}
			mu.Lock()
			correct += localCorrect
			mu.Unlock()
		}(lo, hi)
	}
	wg.Wait()

	end := time.Since(start)
	n := len(images)
	fmt.Printf("Correct:  %d\n", correct)
	fmt.Printf("Total:    %d\n", n)
	fmt.Printf("Accuracy: %.3f\n", float64(correct)/float64(n))
	fmt.Printf("Time:     %d ms (%6.4f ms / image)\n", end.Milliseconds(), float64(end.Milliseconds())/float64(n))
}
