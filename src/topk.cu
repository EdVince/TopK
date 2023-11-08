#include "topk.h"
#include "cub/cub.cuh"

const int32_t kThreadsNumPerBlock = 256;
const int32_t kMaxBlocksNum = 8192;
const int32_t grouptopk_size = 16384;

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

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernelBatch8(
        const __restrict__ uint16_t *docs, const uint16_t *doc_lens, const size_t n_docs, 
        const uint16_t *query_lens, 
        uint16_t *scores, uint8_t *dict) {
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x, tnum = gridDim.x * blockDim.x;
    if (tid >= n_docs)
        return;

    for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
        register int32_t tmp_score[8] = {0};
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
                register uint8_t flag = dict[doc_segment[j]];
                tmp_score[0] += flag&0b00000001; flag >>= 1;
                tmp_score[1] += flag&0b00000001; flag >>= 1;
                tmp_score[2] += flag&0b00000001; flag >>= 1;
                tmp_score[3] += flag&0b00000001; flag >>= 1;
                tmp_score[4] += flag&0b00000001; flag >>= 1;
                tmp_score[5] += flag&0b00000001; flag >>= 1;
                tmp_score[6] += flag&0b00000001; flag >>= 1;
                tmp_score[7] += flag&0b00000001; flag >>= 1;
            }
            __syncwarp();
        }
        scores[0*n_docs+doc_id] = 16384 * tmp_score[0] / max(query_lens[0], doc_lens[doc_id]);
        scores[1*n_docs+doc_id] = 16384 * tmp_score[1] / max(query_lens[1], doc_lens[doc_id]);
        scores[2*n_docs+doc_id] = 16384 * tmp_score[2] / max(query_lens[2], doc_lens[doc_id]);
        scores[3*n_docs+doc_id] = 16384 * tmp_score[3] / max(query_lens[3], doc_lens[doc_id]);
        scores[4*n_docs+doc_id] = 16384 * tmp_score[4] / max(query_lens[4], doc_lens[doc_id]);
        scores[5*n_docs+doc_id] = 16384 * tmp_score[5] / max(query_lens[5], doc_lens[doc_id]);
        scores[6*n_docs+doc_id] = 16384 * tmp_score[6] / max(query_lens[6], doc_lens[doc_id]);
        scores[7*n_docs+doc_id] = 16384 * tmp_score[7] / max(query_lens[7], doc_lens[doc_id]);
    }
}

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernelBatch1(
        const __restrict__ uint16_t *docs, const uint16_t *doc_lens, const size_t n_docs, 
        const uint16_t query_len, 
        uint16_t *scores, uint8_t *dict) {
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x, tnum = gridDim.x * blockDim.x;
    if (tid >= n_docs)
        return;

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
        N = ((N - 1) / grouptopk_size + 1) * grouptopk_size;

        cudaFree(0);

        cudaMallocHost(&h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * N);
        cudaMallocHost(&h_query_lens, sizeof(uint16_t) * 8);
        cudaMallocHost(&h_dict, sizeof(uint8_t) * 50000);
        cudaMallocHost(&h_grouptopk_val, sizeof(uint16_t) * grouptopk_size * TOPK * 8);
        cudaMallocHost(&h_grouptopk_idx, sizeof(int32_t) * grouptopk_size * TOPK * 8);
        
        cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * N);
        cudaMalloc(&d_query_lens, sizeof(uint16_t) * 8);
        cudaMalloc(&d_doc_lens, sizeof(uint16_t) * N);
        cudaMalloc(&d_scores, sizeof(uint16_t) * N * 8);
        cudaMalloc(&d_dict, sizeof(uint8_t) * 50000);
        {
            size_t GLOBAL_WORKSPACE_SIZE = 0;
            int elem_cnt = N * 8;
            int instance_size = grouptopk_size;
            int instance_num = elem_cnt / instance_size;
            const int32_t sorted_in_aligned_bytes = GetAlignedSize(elem_cnt * sizeof(uint16_t));
            const int32_t indices_aligned_bytes = GetAlignedSize(elem_cnt * sizeof(int32_t));
            const int32_t sorted_indices_aligned_bytes = indices_aligned_bytes;
            int32_t temp_storage_bytes = InferTempStorageForSortPairsDescending<uint16_t, int32_t>(instance_size, instance_num);
            GLOBAL_WORKSPACE_SIZE = GetAlignedSize(sorted_in_aligned_bytes + indices_aligned_bytes + sorted_indices_aligned_bytes + temp_storage_bytes);
            cudaMalloc(&d_grouptopk_workspace, GLOBAL_WORKSPACE_SIZE * sizeof(uint8_t));
        }
    }

    ~CUDAInit() {
        cudaFreeHost(h_docs);
        cudaFreeHost(h_query_lens);
        cudaFreeHost(h_dict);
        cudaFreeHost(h_grouptopk_val);
        cudaFreeHost(h_grouptopk_idx);
        
        cudaFree(d_docs);
        cudaFree(d_query_lens);
        cudaFree(d_doc_lens);
        cudaFree(d_scores);
        cudaFree(d_dict);
        cudaFree(d_grouptopk_workspace);
    }

    uint16_t* h_docs = nullptr;
    uint16_t* h_query_lens = nullptr;
    uint8_t*  h_dict = nullptr;
    uint16_t* h_grouptopk_val = nullptr;
    int32_t*  h_grouptopk_idx = nullptr;
    
    uint16_t* d_docs = nullptr;
    uint16_t* d_query_lens = nullptr;
    uint16_t* d_doc_lens = nullptr;
    uint16_t* d_scores = nullptr;
    uint8_t*  d_dict = nullptr;
    uint8_t*  d_grouptopk_workspace = nullptr;
};
CUDAInit cudaInit(8500000); // 850万

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices //shape [querys.size(), TOPK]
    ) 
{

    int n_docs = docs.size();
    n_docs = ((n_docs - 1) / grouptopk_size + 1) * grouptopk_size;
    for(int i = 0; i < (n_docs-docs.size()); i++)
        lens.emplace_back();
    int grouptopk_batch = n_docs / grouptopk_size;

    std::chrono::high_resolution_clock::time_point h1, h2;

    int64_t cT = 0;
    h1 = std::chrono::high_resolution_clock::now();
    int numThreads = std::thread::hardware_concurrency();
    int docsSize = docs.size();
    int docsPerThread = docsSize / numThreads;
    std::vector<std::thread> threads;
    for (int q = 0; q < numThreads; q++) {
        auto start = q * docsPerThread;
        auto end = start + docsPerThread;
        if (q == numThreads-1) end = docsSize;
        threads.emplace_back([&](int s, int e){
            for (int i = s; i < e; i++) {
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
        },start,end);
    }
    for (auto& thread : threads) thread.join();
    h2 = std::chrono::high_resolution_clock::now();
    cT = std::chrono::duration_cast<std::chrono::milliseconds>(h2 - h1).count();

    int64_t tT = 0;
    h1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(cudaInit.d_doc_lens, lens.data(), sizeof(uint16_t) * n_docs, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaInit.d_docs, cudaInit.h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, cudaMemcpyHostToDevice);
    h2 = std::chrono::high_resolution_clock::now();
    tT = std::chrono::duration_cast<std::chrono::milliseconds>(h2 - h1).count();

    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    cudaSetDevice(0);
    int block = N_THREADS_IN_ONE_BLOCK;
    int grid = (n_docs + block - 1) / block;
    int cur_pos = 0;

    double b8T = 0;
    h1 = std::chrono::high_resolution_clock::now();
    for (; cur_pos + 8 < querys.size(); cur_pos+=8) {
        memset(cudaInit.h_dict, 0, sizeof(uint8_t) * 50000);
        for(int i = 0; i < 8; i++) {
            for(int j = 0; j < querys[cur_pos+i].size(); j++)
                cudaInit.h_dict[querys[cur_pos+i][j]] |= 1<<i;
            cudaInit.h_query_lens[i] = querys[cur_pos+i].size();
        }
        cudaMemcpy(cudaInit.d_dict, cudaInit.h_dict, sizeof(uint8_t) * 50000, cudaMemcpyHostToDevice);
        cudaMemcpy(cudaInit.d_query_lens, cudaInit.h_query_lens, sizeof(uint16_t) * 8, cudaMemcpyHostToDevice);
        docQueryScoringCoalescedMemoryAccessSampleKernelBatch8<<<grid, block>>>(
            cudaInit.d_docs, cudaInit.d_doc_lens, n_docs, 
            cudaInit.d_query_lens, 
            cudaInit.d_scores, cudaInit.d_dict);
        cudaDeviceSynchronize();

        topk_launcher<uint16_t>(0,
            n_docs*8,           // elem_cnt
            grouptopk_size,     // instance_size
            grouptopk_batch*8,  // instance_num
            TOPK,               // top_k
            cudaInit.d_scores,
            cudaInit.d_grouptopk_workspace,
            cudaInit.h_grouptopk_idx,
            cudaInit.h_grouptopk_val);

        for(int bb = 0; bb < 8; bb++) {
            std::vector<int> top(grouptopk_batch,0);
            std::vector<int> topk(TOPK);
            for (int i = 0; i < TOPK; i++) {
                int idx = -1;
                int16_t val = -1;
                for (int j = 0; j < grouptopk_batch; j++) {
                    if (cudaInit.h_grouptopk_val[bb*TOPK*grouptopk_batch+j*TOPK+top[j]] > val) {
                        val = cudaInit.h_grouptopk_val[bb*TOPK*grouptopk_batch+j*TOPK+top[j]];
                        idx = cudaInit.h_grouptopk_idx[bb*TOPK*grouptopk_batch+j*TOPK+top[j]] + j*grouptopk_size;
                    }
                }
                topk[i] = idx;
                top[idx/grouptopk_size]++;
            }
            indices.emplace_back(topk);
        }
    }
    h2 = std::chrono::high_resolution_clock::now();
    b8T = (double)std::chrono::duration_cast<std::chrono::milliseconds>(h2 - h1).count() / cur_pos;

    double b1T = 0;
    h1 = std::chrono::high_resolution_clock::now();
    for (; cur_pos < querys.size(); cur_pos++) {
        auto& query = querys[cur_pos];

        const size_t query_len = query.size();
        memset(cudaInit.h_dict, 0, sizeof(uint8_t) * 50000);
        for(int i = 0; i < query_len; i++)
            cudaInit.h_dict[query[i]] = 1;
        cudaMemcpy(cudaInit.d_dict, cudaInit.h_dict, sizeof(uint8_t) * 50000, cudaMemcpyHostToDevice);
        docQueryScoringCoalescedMemoryAccessSampleKernelBatch1<<<grid, block>>>(
            cudaInit.d_docs, cudaInit.d_doc_lens, n_docs, 
            query_len, 
            cudaInit.d_scores, cudaInit.d_dict);
        cudaDeviceSynchronize();

        topk_launcher<uint16_t>(0,
            n_docs,           // elem_cnt
            grouptopk_size,   // instance_size
            grouptopk_batch,  // instance_num
            TOPK,             // top_k
            cudaInit.d_scores,
            cudaInit.d_grouptopk_workspace,
            cudaInit.h_grouptopk_idx,
            cudaInit.h_grouptopk_val);

        std::vector<int> top(grouptopk_batch,0);
        std::vector<int> topk(TOPK);
        for (int i = 0; i < TOPK; i++) {
            int idx = -1;
            int16_t val = -1;
            for (int j = 0; j < grouptopk_batch; j++) {
                if (cudaInit.h_grouptopk_val[j*TOPK+top[j]] > val) {
                    val = cudaInit.h_grouptopk_val[j*TOPK+top[j]];
                    idx = cudaInit.h_grouptopk_idx[j*TOPK+top[j]] + j*grouptopk_size;
                }
            }
            topk[i] = idx;
            top[idx/grouptopk_size]++;
        }
        indices.emplace_back(topk);
    }
    h2 = std::chrono::high_resolution_clock::now();
    b1T = (double)std::chrono::duration_cast<std::chrono::milliseconds>(h2 - h1).count() / (querys.size()%8);

    printf("[TIME] convert:%ldms, transfer:%ldms\n",cT,tT);
    printf("[TIME] Batch8:%.2lfms, Batch1:%.2lfms\n",b8T,b1T);
}