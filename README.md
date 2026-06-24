# gopherRT

From-scratch ONNX inference engine for MNIST, written in Go. No CGO, no ONNX Runtime

### Model

A 3-layer MLP with ReLU activations and a softmax output layer:

```
input (784) -> dense(128) -> ReLU -> dense(64) -> ReLU -> dense(10) -> softmax
```

Weights and biases are loaded from `mnist.onnx` via generated protobuf bindings. The graph is hardcoded for this specific architecture.

## Dataset

The MNIST test set (10,000 images) is downloaded automatically from `yann.lecun.com/exdb/mnist/` on first run. Images are normalized to `[0, 1]`.

## Usage

```bash
go run .
```

On first run the dataset is fetched and cached in `data/`. The model file `mnist.onnx` is tracked in the repository.

## Performance

| Metric                     | Value                  |
| -------------------------- | ---------------------- |
| Accuracy (10k test images) | 97.41%                 |
| Total inference time       | 98 ms                  |
| Throughput                 | ~102,000 images/sec    |
| Hardware                   | Apple M4 Air (8 cores) |

Inference is parallelized across all available CPU cores using goroutines. Each image is processed independently with no batching.

## Dependencies

- `google.golang.org/protobuf` -- protobuf deserialization for ONNX format
- `golang.org/x/net` -- required by protobuf (transitive, vendored)

Everything else is the Go standard library.
