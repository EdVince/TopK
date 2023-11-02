#include "topk.h"
#include "ThreadPool.h"

// https://github.com/facebookincubator/AITemplate/blob/ab1cfbcefcfe9639255a7ed8a2ff4bb522ebf61e/python/aitemplate/backend/common/tensor/topk_common.py
const int32_t kThreadsNumPerBlock = 256;
const int32_t kMaxBlocksNum = 8192;

#define GPU_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

inline size_t GetAlignedSize(size_t size) {
  const size_t kAlignSize = 512;
  return (size + kAlignSize - 1) / kAlignSize * kAlignSize;
}

class MultiplyFunctor final {
 public:
  MultiplyFunctor(int32_t num_col) : num_col_(num_col) {}
  __host__ __device__ __forceinline__ int32_t operator()(int32_t idx) const {
    return idx * num_col_;
  }

 private:
  int32_t num_col_;
};

template <typename KeyType, typename ValueType>
size_t InferTempStorageForSortPairsDescending(
    int32_t num_row,
    int32_t num_col) {
  using SegmentOffsetIter = cub::TransformInputIterator<
      int32_t,
      MultiplyFunctor,
      cub::CountingInputIterator<int32_t>>;

  cub::CountingInputIterator<int32_t> counting_iter(0);
  MultiplyFunctor multiply_functor(num_col);
  SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

  size_t temp_storage_bytes = 0;
  auto err = cub::DeviceSegmentedRadixSort::
      SortPairsDescending<KeyType, ValueType, SegmentOffsetIter>(
          /* d_temp_storage */ nullptr,
          /* temp_storage_bytes */ temp_storage_bytes,
          /* d_keys_in */ nullptr,
          /* d_keys_out */ nullptr,
          /* d_values_in */ nullptr,
          /* d_values_out */ nullptr,
          /* num_items */ num_row * num_col,
          /* num_segments */ num_row,
          /* d_begin_offsets */ segment_offset_iter,
          /* d_end_offsets */ segment_offset_iter + 1,
          /* begin_bit */ 0,
          /* end_bit */ sizeof(KeyType) * 8,
          /* stream */ 0);

  return temp_storage_bytes;
}

template <typename KeyType, typename ValueType>
void SortPairsDescending(
    const KeyType* keys_ptr,
    const ValueType* values_ptr,
    int32_t num_row,
    int32_t num_col,
    void* temp_storage_ptr,
    int32_t temp_storage_bytes,
    KeyType* sorted_keys_ptr,
    ValueType* sorted_values_ptr,
    cudaStream_t stream) {
  size_t rt_inferred_temp_storage_bytes =
      InferTempStorageForSortPairsDescending<KeyType, ValueType>(
          num_row, num_col);

  using SegmentOffsetIter = cub::TransformInputIterator<
      int32_t,
      MultiplyFunctor,
      cub::CountingInputIterator<int32_t>>;

  cub::CountingInputIterator<int32_t> counting_iter(0);
  MultiplyFunctor multiply_functor(num_col);
  SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

  auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      /* d_temp_storage */ temp_storage_ptr,
      /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
      /* d_keys_in */ keys_ptr,
      /* d_keys_out */ sorted_keys_ptr,
      /* d_values_in */ values_ptr,
      /* d_values_out */ sorted_values_ptr,
      /* num_items */ num_row * num_col,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offset_iter,
      /* d_end_offsets */ segment_offset_iter + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(KeyType) * 8,
      /* stream */ stream);
}

template <typename T>
class TmpBufferManager final {
 public:
  TmpBufferManager(int32_t capacity, void* ptr, const int32_t N)
      : capacity_{capacity},
        sorted_in_elem_cnt_{N},
        indices_elem_cnt_{sorted_in_elem_cnt_},
        sorted_indices_elem_cnt_{sorted_in_elem_cnt_} {
    const int32_t sorted_in_aligned_bytes =
        GetAlignedSize(sorted_in_elem_cnt_ * sizeof(T));
    const int32_t indices_aligned_bytes =
        GetAlignedSize(indices_elem_cnt_ * sizeof(int32_t));
    const int32_t sorted_indices_aligned_bytes = indices_aligned_bytes;
    sorted_in_ptr_ = reinterpret_cast<T*>(ptr);
    indices_ptr_ = reinterpret_cast<int32_t*>(
        reinterpret_cast<char*>(sorted_in_ptr_) + sorted_in_aligned_bytes);
    sorted_indices_ptr_ = reinterpret_cast<int32_t*>(
        reinterpret_cast<char*>(indices_ptr_) + indices_aligned_bytes);
    temp_storage_ptr_ = reinterpret_cast<void*>(
        reinterpret_cast<char*>(sorted_indices_ptr_) +
        sorted_indices_aligned_bytes);
    temp_storage_bytes_ = capacity_ - sorted_in_aligned_bytes -
        indices_aligned_bytes - sorted_indices_aligned_bytes;
  }
  ~TmpBufferManager() = default;

  T* SortedInPtr() const {
    return sorted_in_ptr_;
  }
  int32_t* IndicesPtr() const {
    return indices_ptr_;
  }
  int32_t* SortedIndicesPtr() const {
    return sorted_indices_ptr_;
  }
  void* TempStoragePtr() const {
    return temp_storage_ptr_;
  }

  int32_t TempStorageBytes() const {
    return temp_storage_bytes_;
  }

 private:
  int32_t capacity_;

  T* sorted_in_ptr_;
  int32_t* indices_ptr_;
  int32_t* sorted_indices_ptr_;
  void* temp_storage_ptr_;

  int32_t sorted_in_elem_cnt_;
  int32_t indices_elem_cnt_;
  int32_t sorted_indices_elem_cnt_;
  int32_t temp_storage_bytes_;
};

__global__ void InitializeIndices(
    int32_t elem_cnt,
    int32_t* indices_ptr,
    int32_t instance_size) {
  GPU_KERNEL_LOOP(i, elem_cnt) {
    indices_ptr[i] = i % instance_size;
  };
}

// ALIGNPTR
int32_t* alignPtr(int32_t* ptr, uintptr_t to) {
  uintptr_t addr = (uintptr_t)ptr;
  if (addr % to) {
    addr += to - addr % to;
  }
  return (int32_t*)addr;
}

inline int32_t BlocksNum4ThreadsNum(const int32_t n) {
  return std::min(
      (n + kThreadsNumPerBlock - 1) / kThreadsNumPerBlock,
      kMaxBlocksNum);
}

template <typename T>
void topk_launcher(
    cudaStream_t stream,
    const int elem_cnt,
    const int instance_size,
    const int instance_num,
    const int top_k,
    const void* input,
    void* workspace,
    void* output_index,
    void* output_value) {
  const int32_t k = std::min(top_k, instance_size);
  const uintptr_t ALIGNMENT = 32;
  int32_t* vworkspace = alignPtr((int32_t*)workspace, ALIGNMENT);
  T* tmp_buffer = (T*)vworkspace;

  TmpBufferManager<T> buf_manager(
      static_cast<int32_t>(elem_cnt), tmp_buffer, elem_cnt);

  InitializeIndices<<<
      BlocksNum4ThreadsNum(elem_cnt),
      kThreadsNumPerBlock,
      0,
      stream>>>(elem_cnt, buf_manager.IndicesPtr(), instance_size);

  SortPairsDescending(
      (const T*)input,
      buf_manager.IndicesPtr(),
      instance_num,
      instance_size,
      buf_manager.TempStoragePtr(),
      buf_manager.TempStorageBytes(),
      buf_manager.SortedInPtr(),
      buf_manager.SortedIndicesPtr(),
      stream);

  cudaMemcpy2DAsync(
      (int32_t*)output_index,
      k * sizeof(int32_t),
      buf_manager.SortedIndicesPtr(),
      instance_size * sizeof(int32_t),
      k * sizeof(int32_t),
      instance_num,
      cudaMemcpyDefault,
      stream);

  cudaMemcpy2DAsync(
      (T*)output_value,
      k * sizeof(T),
      buf_manager.SortedInPtr(),
      instance_size * sizeof(T),
      k * sizeof(T),
      instance_num,
      cudaMemcpyDefault,
      stream);
}

typedef uint4 group_t; // uint32_t

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
        const __restrict__ uint16_t *docs, 
        const uint16_t *doc_lens, const size_t n_docs, 
        uint16_t *query, const int query_len, uint16_t *scores, uint8_t *dict) {
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x, tnum = gridDim.x * blockDim.x;
    if (tid >= n_docs)
        return;

#pragma unroll
    for (auto i = threadIdx.x; i < query_len; i += blockDim.x) {
        dict[query[i]] = 1;
    }
    __syncthreads();

    for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
        register int32_t tmp_score = 0.;
        register bool no_more_load = false;
        for (auto i = 0; i < MAX_DOC_SIZE / (sizeof(group_t) / sizeof(uint16_t)); i++) {
            if (no_more_load) break;
            register group_t loaded = ((group_t *)docs)[i * n_docs + doc_id]; // tid
            register uint16_t *doc_segment = (uint16_t*)(&loaded);
            for (auto j = 0; j < sizeof(group_t) / sizeof(uint16_t); j++) {
                if (doc_segment[j] == 0) {
                    no_more_load = true;
                    break;
                }
                if (dict[doc_segment[j]]) {
                    tmp_score += 1;
                }
            }
            __syncwarp();
        }
        scores[doc_id] = 16384 * tmp_score / max(query_len, doc_lens[doc_id]);
    }
}

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices //shape [querys.size(), TOPK]
    ) {

    int grouptopk_size = 8192;
    int n_docs = docs.size();
    n_docs = ((n_docs - 1) / grouptopk_size + 1) * grouptopk_size;
    for(int i = 0; i < (n_docs-docs.size()); i++)
        lens.emplace_back();
    int grouptopk_batch = n_docs / grouptopk_size;

    uint16_t* scores = nullptr;
    uint16_t* d_scores = nullptr;
    uint16_t* d_docs = nullptr;
    uint16_t* d_doc_lens = nullptr;
    uint16_t* d_query = nullptr;
    uint8_t* dict = nullptr;
    uint16_t* grouptopk_val = nullptr;
    int32_t* grouptopk_idx = nullptr;
    int32_t* grouptopk_workspace = nullptr;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // 子线程
    uint16_t *h_docs;
    std::thread convert_format([&]() {
        std::chrono::high_resolution_clock::time_point d1 = std::chrono::high_resolution_clock::now();
        h_docs = (uint16_t*)calloc(MAX_DOC_SIZE*n_docs,sizeof(uint16_t));
        #pragma omp parallel for
        for (int i = 0; i < docs.size(); i++) {
            for (int j = 0; j < lens[i]; j++) {
                auto group_sz = sizeof(group_t) / sizeof(uint16_t);
                auto layer_0_offset = j / group_sz;
                auto layer_0_stride = n_docs * group_sz;
                auto layer_1_offset = i;
                auto layer_1_stride = group_sz;
                auto layer_2_offset = j % group_sz;
                auto final_offset = layer_0_offset * layer_0_stride + layer_1_offset * layer_1_stride + layer_2_offset;
                h_docs[final_offset] = docs[i][j];
            }
        }

        cudaMalloc(&d_doc_lens, sizeof(uint16_t) * n_docs);
        cudaMalloc(&d_query, sizeof(uint16_t) * MAX_QUERY_SIZE);
        cudaMalloc(&d_scores, sizeof(uint16_t) * n_docs);
        cudaMalloc(&dict, sizeof(uint8_t) * 50000);
        cudaMemcpy(d_doc_lens, lens.data(), sizeof(uint16_t) * n_docs, cudaMemcpyHostToDevice);
        std::chrono::high_resolution_clock::time_point d2 = std::chrono::high_resolution_clock::now();
        std::cout << "[CUDA] convert: " << std::chrono::duration_cast<std::chrono::milliseconds>(d2 - d1).count() << " ms " << std::endl;
    });

    // 主线程
    std::chrono::high_resolution_clock::time_point d1 = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
    cudaMalloc(&grouptopk_workspace, n_docs * (TOPK+5) * sizeof(int32_t));
    cudaMallocHost(&scores, n_docs * sizeof(uint16_t));
    cudaMallocHost(&grouptopk_val, grouptopk_size * TOPK * sizeof(uint16_t));
    cudaMallocHost(&grouptopk_idx, grouptopk_size * TOPK * sizeof(int32_t));
    ThreadPool pool(1);
    auto result = pool.enqueue([&](){});
    std::chrono::high_resolution_clock::time_point d2 = std::chrono::high_resolution_clock::now();
    std::cout << "[CUDA] malloc: " << std::chrono::duration_cast<std::chrono::milliseconds>(d2 - d1).count() << " ms " << std::endl;

    convert_format.join();

    // 非常耗时
    cudaMemcpy(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, cudaMemcpyHostToDevice);

    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    cudaSetDevice(0);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    bool needWait = false;
    for(auto& query : querys) {

        // host-to-device
        const size_t query_len = query.size();
        cudaMemcpy(d_query, query.data(), sizeof(uint16_t) * query_len, cudaMemcpyHostToDevice);
        // launch kernel
        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (n_docs + block - 1) / block;
        cudaMemset(dict, 0, sizeof(uint8_t) * 50000);
        docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block>>>(d_docs, d_doc_lens, n_docs, d_query, query_len, d_scores, dict);
        cudaDeviceSynchronize();

        if (needWait) result.wait();
        else needWait = true;

        // std::chrono::high_resolution_clock::time_point d1 = std::chrono::high_resolution_clock::now();
        // launch topk
        topk_launcher<uint16_t>(0,
            n_docs,           // elem_cnt
            grouptopk_size,   // instance_size
            grouptopk_batch,  // instance_num
            TOPK,             // top_k
            d_scores,grouptopk_workspace,grouptopk_idx,grouptopk_val);
        cudaDeviceSynchronize();

        result = pool.enqueue([&]() {
            std::vector<int> top(grouptopk_batch,0);
            std::vector<int> topk(TOPK);
            for (int i = 0; i < TOPK; i++) {
                int idx = -1;
                int16_t val = -1;
                for (int j = 0; j < grouptopk_batch; j++) {
                    if (grouptopk_val[j*TOPK+top[j]] > val) {
                        val = grouptopk_val[j*TOPK+top[j]];
                        idx = grouptopk_idx[j*TOPK+top[j]] + j*grouptopk_size;
                    }
                }
                topk[i] = idx;
                top[idx/grouptopk_size]++;
            }
            indices.push_back(topk);
        });
        // std::chrono::high_resolution_clock::time_point d2 = std::chrono::high_resolution_clock::now();
        // std::cout << "[topk] " << std::chrono::duration_cast<std::chrono::microseconds>(d2 - d1).count() << " us " << std::endl;
    }

    // deallocation
    cudaFree(dict);
    cudaFree(d_docs);
    cudaFree(d_query);
    cudaFree(d_scores);
    cudaFree(d_doc_lens);
    cudaFreeHost(scores);
    result.wait();
    cudaFreeHost(grouptopk_val);
    cudaFreeHost(grouptopk_idx);
    cudaFree(grouptopk_workspace);
    free(h_docs);
    
    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    std::cout << "[CUDA] preprocess: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms " << std::endl;
    std::cout << "[CUDA] process: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms " << std::endl;
}