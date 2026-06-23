package main

import (
	"bytes"
	"encoding/binary"

	"github.com/omkargajare/gopherRT/onnx"
)

func tensorToFloats(t *onnx.TensorProto) ([]float32, error) {
	//checking raw data
	if raw := t.GetRawData(); len(raw) > 0 {
		floats := make([]float32, len(raw)/4)
		err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, &floats)
		return floats, err
	}
	return t.GetFloatData(), nil
}
