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

    cudaDeviceSynchronize();

    cudaMemcpy2D(
        (int32_t*)output_index,
        k * sizeof(int32_t),
        buf_manager.SortedIndicesPtr(),
        instance_size * sizeof(int32_t),
        k * sizeof(int32_t),
        instance_num,
        cudaMemcpyDefault);

    cudaMemcpy2D(
        (T*)output_value,
        k * sizeof(T),
        buf_manager.SortedInPtr(),
        instance_size * sizeof(T),
        k * sizeof(T),
        instance_num,
        cudaMemcpyDefault);
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

class CUDAInit {
public:
    CUDAInit(size_t N) {
        std::cout<<"------------- CUDA Init -------------"<<std::endl;
        N = ((N - 1) / grouptopk_size + 1) * grouptopk_size;
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cudaFree(0);

        cudaMallocHost(&h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * N);
        cudaMallocHost(&grouptopk_val, sizeof(uint16_t) * grouptopk_size * TOPK);
        cudaMallocHost(&grouptopk_idx, sizeof(int32_t) * grouptopk_size * TOPK);

        cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * N);
        cudaMalloc(&d_query, sizeof(uint16_t) * MAX_QUERY_SIZE);
        cudaMalloc(&d_doc_lens, sizeof(uint16_t) * N);
        cudaMalloc(&d_scores, sizeof(uint16_t) * N);
        cudaMalloc(&dict, sizeof(uint8_t) * 50000);
        {
            size_t GLOBAL_WORKSPACE_SIZE = 0;
            int elem_cnt = N;
            int instance_size = grouptopk_size;
            int instance_num = elem_cnt / instance_size;
            const int32_t sorted_in_aligned_bytes = GetAlignedSize(elem_cnt * sizeof(uint16_t));
            const int32_t indices_aligned_bytes = GetAlignedSize(elem_cnt * sizeof(int32_t));
            const int32_t sorted_indices_aligned_bytes = indices_aligned_bytes;
            int32_t temp_storage_bytes = InferTempStorageForSortPairsDescending<uint16_t, int32_t>(instance_size, instance_num);
            GLOBAL_WORKSPACE_SIZE = GetAlignedSize(sorted_in_aligned_bytes + indices_aligned_bytes + sorted_indices_aligned_bytes + temp_storage_bytes);
            cudaMalloc(&grouptopk_workspace, GLOBAL_WORKSPACE_SIZE * sizeof(uint8_t));
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::cout<<"Init with N="<<N<<", cost:"<<std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()<<"ms"<<std::endl;
        std::cout<<"------------- CUDA Init -------------"<<std::endl;
    }

    ~CUDAInit() {
        cudaFree(dict);
        cudaFree(d_docs);
        cudaFree(d_query);
        cudaFree(d_scores);
        cudaFree(d_doc_lens);
        cudaFree(grouptopk_workspace);
        cudaFreeHost(grouptopk_val);
        cudaFreeHost(grouptopk_idx);
        cudaFreeHost(h_docs);
    }

    uint16_t *h_docs = nullptr;
    uint16_t* grouptopk_val = nullptr;
    int32_t* grouptopk_idx = nullptr;

    uint16_t* d_docs = nullptr;
    uint16_t* d_query = nullptr;
    uint16_t* d_doc_lens = nullptr;
    uint16_t* d_scores = nullptr;
    uint8_t* dict = nullptr;
    uint8_t* grouptopk_workspace = nullptr;
};
CUDAInit cudaInit(8500000); // 850万

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices //shape [querys.size(), TOPK]
    ) {

    int n_docs = docs.size();
    n_docs = ((n_docs - 1) / grouptopk_size + 1) * grouptopk_size;
    for(int i = 0; i < (n_docs-docs.size()); i++)
        lens.emplace_back();
    int grouptopk_batch = n_docs / grouptopk_size;

    
    std::chrono::high_resolution_clock::time_point h1, h2;
    int64_t transT = 0;
    int64_t convertT = 0;
    int64_t kernelT = 0;
    int64_t topkT = 0;
    int64_t cpuT = 0;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    h1 = std::chrono::high_resolution_clock::now();
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
            cudaInit.h_docs[final_offset] = docs[i][j];
        }
    }
    h2 = std::chrono::high_resolution_clock::now();
    convertT = std::chrono::duration_cast<std::chrono::milliseconds>(h2 - h1).count();

    h1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(cudaInit.d_doc_lens, lens.data(), sizeof(uint16_t) * n_docs, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaInit.d_docs, cudaInit.h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, cudaMemcpyHostToDevice);
    h2 = std::chrono::high_resolution_clock::now();
    transT = std::chrono::duration_cast<std::chrono::milliseconds>(h2-h1).count();

    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    cudaSetDevice(0);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    for(auto& query : querys) {

        h1 = std::chrono::high_resolution_clock::now();
        // host-to-device
        const size_t query_len = query.size();
        cudaMemcpy(cudaInit.d_query, query.data(), sizeof(uint16_t) * query_len, cudaMemcpyHostToDevice);
        // launch kernel
        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (n_docs + block - 1) / block;
        cudaMemset(cudaInit.dict, 0, sizeof(uint8_t) * 50000);
        docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block>>>(
            cudaInit.d_docs, 
            cudaInit.d_doc_lens, 
            n_docs, 
            cudaInit.d_query, 
            query_len, 
            cudaInit.d_scores, 
            cudaInit.dict);
        cudaDeviceSynchronize();
        h2 = std::chrono::high_resolution_clock::now();
        kernelT += std::chrono::duration_cast<std::chrono::microseconds>(h2-h1).count();

        // launch topk
        h1 = std::chrono::high_resolution_clock::now();
        topk_launcher<uint16_t>(0,
            n_docs,           // elem_cnt
            grouptopk_size,   // instance_size
            grouptopk_batch,  // instance_num
            TOPK,             // top_k
            cudaInit.d_scores,
            cudaInit.grouptopk_workspace,
            cudaInit.grouptopk_idx,
            cudaInit.grouptopk_val);
        h2 = std::chrono::high_resolution_clock::now();
        topkT += std::chrono::duration_cast<std::chrono::microseconds>(h2-h1).count();

        h1 = std::chrono::high_resolution_clock::now();
        std::vector<int> top(grouptopk_batch,0);
        std::vector<int> topk(TOPK);
        for (int i = 0; i < TOPK; i++) {
            int idx = -1;
            int16_t val = -1;
            for (int j = 0; j < grouptopk_batch; j++) {
                if (cudaInit.grouptopk_val[j*TOPK+top[j]] > val) {
                    val = cudaInit.grouptopk_val[j*TOPK+top[j]];
                    idx = cudaInit.grouptopk_idx[j*TOPK+top[j]] + j*grouptopk_size;
                }
            }
            topk[i] = idx;
            top[idx/grouptopk_size]++;
        }
        h2 = std::chrono::high_resolution_clock::now();
        cpuT += std::chrono::duration_cast<std::chrono::microseconds>(h2-h1).count();
        indices.push_back(topk);
    }
    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

    auto pret = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count(); // 预处理总时间
    auto prot = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count(); // query总时间
    float queryT = (float)prot / querys.size(); // 单次query时间us(<2ms)
    kernelT = kernelT / querys.size(); // 单次kernel时间us(1000~1200us)
    topkT = topkT / querys.size(); // 单次topk时间us(300~500us)
    cpuT = cpuT / querys.size(); // 单次cpu时间us(<100us)

    printf("[TIME] preprocess time: %ldms, process time: %ldms\n",pret,prot);
    printf("[TIME] convert time: %ldms, trans time: %ldms\n",convertT,transT);
    printf("[TIME] query time: %.2fms, kernel time: %ldus, topk time: %ldus, cpu time: %ldus\n",queryT,kernelT,topkT,cpuT);
}