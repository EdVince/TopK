ROOT_DIR=$1
if [ -z "$ROOT_DIR" ]; then
        ROOT_DIR=$(cd $(dirname $0); pwd)
fi

nvcc -x cu -Xcompiler -mavx -Xcompiler -msse2 -Xcompiler -fopenmp \
        $ROOT_DIR/src/main.cpp \
        $ROOT_DIR/src/topk.cu \
        -o $ROOT_DIR/bin/query_doc_scoring \
        -I $ROOT_DIR/src -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas -O3
echo "build success"