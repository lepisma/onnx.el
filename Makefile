all: onnx-core.so onnx-ml-utils.so

CC := gcc

onnx-core.so: onnx-core.c
	$(CC) -I/usr/include/onnxruntime -fPIC -pthread -shared onnx-core.c -o $@ -lonnxruntime

onnx-ml-utils.so: onnx-ml-utils.c
	$(CC) -fPIC -pthread -shared onnx-ml-utils.c -o $@

clean:
	rm -f *.so
