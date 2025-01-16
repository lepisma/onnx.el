release: onnx-core.so onnx-ml-utils.so

CC := gcc
CFLAGS := -O2 -march=native -fPIC -pthread

onnx-core.so: onnx-core.c
	$(CC) $(CFLAGS) -I/usr/include/onnxruntime -shared onnx-core.c -o $@ -lonnxruntime

onnx-ml-utils.so: onnx-ml-utils.c
	$(CC) $(CFLAGS) -shared onnx-ml-utils.c -o $@

clean:
	rm -f *.so
