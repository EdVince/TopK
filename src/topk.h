#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <cuda.h>
#include <chrono>
#include <thread>
#include <numeric>
#include "omp.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include "cub/cub.cuh"
#include "cub/util_type.cuh"
#include "cub/util_allocator.cuh"
#include "cub/device/device_radix_sort.cuh"
#include <limits>
#include <cublas_v2.h>
#include <queue>
#include <random>
#include <unordered_map>

#define DEBUG

#define MAX_DOC_SIZE 128
#define MAX_QUERY_SIZE 128
#define N_THREADS_IN_ONE_BLOCK 512
#define TOPK 100
#define grouptopk_size 16384

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &query,
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices);
