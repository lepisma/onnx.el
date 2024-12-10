all: onnx-core.so

CFLAGS := -I/usr/include/onnxruntime
LDFLAGS := -lonnxruntime

onnx-core.so: $(wildcard *.c) $(wildcard *.h)
	gcc $(CFLAGS) -fPIC -pthread -shared $(wildcard *.c) -o $@ $(LDFLAGS)

clean:
	rm -f *.so
