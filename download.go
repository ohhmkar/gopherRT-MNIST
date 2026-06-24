package main

import (
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const datasetURL = "https://yann.lecun.com/exdb/mnist"

var datasetFiles = []struct {
	gzName string
	outName string
}{
	{"t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"},
	{"t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte"},
}

func ensureDataset(dir string) error {
	for _, f := range datasetFiles {
		outPath := filepath.Join(dir, f.outName)
		if _, err := os.Stat(outPath); err == nil {
			continue
		}

		gzPath := filepath.Join(dir, f.gzName)
		url := datasetURL + "/" + f.gzName

		fmt.Printf("Downloading %s ...\n", url)
		if err := downloadFile(gzPath, url); err != nil {
			return fmt.Errorf("downloading %s: %w", f.gzName, err)
		}

		fmt.Printf("Decompressing %s ...\n", f.gzName)
		if err := gunzip(gzPath, outPath); err != nil {
			return fmt.Errorf("decompressing %s: %w", f.gzName, err)
		}
		os.Remove(gzPath)
	}
	return nil
}

func downloadFile(path, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func gunzip(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	r, err := gzip.NewReader(in)
	if err != nil {
		return err
	}
	defer r.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, r)
	return err
}
