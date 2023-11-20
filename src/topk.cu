#include "topk.h"
#include "cub/cub.cuh"

const int32_t kThreadsNumPerBlock = 256;
const int32_t kMaxBlocksNum = 8192;
const int32_t grouptopk_size = 65536;
const int32_t N = 8000000;

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
        GetAlignedSize(indices_elem_cnt_ * sizeof(uint16_t));
    const int32_t sorted_indices_aligned_bytes = indices_aligned_bytes;
    sorted_in_ptr_ = reinterpret_cast<T*>(ptr);
    indices_ptr_ = reinterpret_cast<uint16_t*>(
        reinterpret_cast<char*>(sorted_in_ptr_) + sorted_in_aligned_bytes);
    sorted_indices_ptr_ = reinterpret_cast<uint16_t*>(
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
    uint16_t* IndicesPtr() const {
        return indices_ptr_;
    }
    uint16_t* SortedIndicesPtr() const {
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
    uint16_t* indices_ptr_;
    uint16_t* sorted_indices_ptr_;
    void* temp_storage_ptr_;

    int32_t sorted_in_elem_cnt_;
    int32_t indices_elem_cnt_;
    int32_t sorted_indices_elem_cnt_;
    int32_t temp_storage_bytes_;
};

__global__ void InitializeIndices(
        int32_t elem_cnt,
        uint16_t* indices_ptr,
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
        (uint16_t*)output_index,
        k * sizeof(uint16_t),
        buf_manager.SortedIndicesPtr(),
        instance_size * sizeof(uint16_t),
        k * sizeof(uint16_t),
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

__global__ void permute102_kernel(uint16_t* output,
                                const uint16_t *input,
                                const int n_docs) {
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_docs)
        return;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        ((group_t *)output)[i*n_docs+tid] = ((group_t *)input)[tid*16+i];
    }
}

template <typename T, const int len>
void __global__ docQueryScoringCoalescedMemoryAccessSampleKernelBatchN(
        const __restrict__ uint16_t *docs, const uint16_t *doc_lens, const size_t n_docs, 
        const uint16_t *query_lens, 
        uint16_t *scores, T *dict) {
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x, tnum = gridDim.x * blockDim.x;
    if (tid >= n_docs)
        return;

    for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
        register int32_t tmp_score[len] = {0};
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
                register T flag = dict[doc_segment[j]];
                #pragma unroll
                for (auto l = 0; l < len; l++) {
                    tmp_score[l] += flag&0x1; flag >>= 1;
                }
            }
            __syncwarp();
        }
        #pragma unroll
        for (auto l = 0; l < len; l++) {
            register uint16_t score = 16384 * tmp_score[l] / max(query_lens[l], doc_lens[doc_id]);
            scores[l*n_docs+doc_id] = score;
        }
    }
}

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
    int block = N_THREADS_IN_ONE_BLOCK;
    int grid = (n_docs + block - 1) / block;



    uint16_t* h_docs_T = nullptr;
    uint16_t* h_query_lens = nullptr;
    uint8_t * h_dict8 = nullptr;
    uint16_t* h_grouptopk_val = nullptr;
    uint16_t * h_grouptopk_idx = nullptr;
    
    uint16_t* d_docs = nullptr;
    uint16_t* d_docs_T = nullptr;
    uint16_t* d_query_lens = nullptr;
    uint16_t* d_doc_lens = nullptr;
    uint16_t* d_scores = nullptr;
    uint8_t * d_dict8 = nullptr;
    uint8_t * d_grouptopk_workspace = nullptr;


    std::chrono::high_resolution_clock::time_point h1, h2;


    h1 = std::chrono::high_resolution_clock::now();
    cudaSetDevice(0);
    cudaFree(0);
    h2 = std::chrono::high_resolution_clock::now();
    int initT = std::chrono::duration_cast<std::chrono::milliseconds>(h2 - h1).count();


    h1 = std::chrono::high_resolution_clock::now();
    h_docs_T = (uint16_t*)calloc(MAX_DOC_SIZE * N, sizeof(uint16_t)); // 2*128*8000000=2GB
    h_query_lens = (uint16_t*)calloc(8, sizeof(uint16_t)); // 2*8=16B
    h_dict8 = (uint8_t*)calloc(50000, sizeof(uint8_t)); // 1*5000=5KB
    h_grouptopk_val = (uint16_t*)calloc(grouptopk_batch * TOPK * 8, sizeof(uint16_t)); // 2*(8000000/65536)*100*8=200MB
    h_grouptopk_idx = (uint16_t*)calloc(grouptopk_batch * TOPK * 8, sizeof(uint16_t)); // 4*(8000000/65536)*100*8=400MB
    indices.resize(querys.size(), std::vector<int>(TOPK));
    h2 = std::chrono::high_resolution_clock::now();
    int newHT = std::chrono::duration_cast<std::chrono::milliseconds>(h2 - h1).count();


    h1 = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * N); // 2*128*8000000=2GB
    cudaMalloc(&d_docs_T, sizeof(uint16_t) * MAX_DOC_SIZE * N); // 2*128*8000000=2GB
    cudaMalloc(&d_query_lens, sizeof(uint16_t) * 8); // 2*8=16B
    cudaMalloc(&d_doc_lens, sizeof(uint16_t) * N); // 2*8000000=16MB
    cudaMalloc(&d_scores, sizeof(uint16_t) * N * 8); // 2*8000000*8=128MB
    cudaMalloc(&d_dict8, sizeof(uint8_t) * 50000); // 1*5000=5KB
    {
        int64_t GLOBAL_WORKSPACE_SIZE = 0;
        int64_t elem_cnt = N * 8;
        int64_t instance_size = grouptopk_size;
        int64_t instance_num = elem_cnt / instance_size;
        int64_t sorted_in_aligned_bytes = GetAlignedSize(elem_cnt * sizeof(uint16_t));
        int64_t indices_aligned_bytes = GetAlignedSize(elem_cnt * sizeof(uint16_t));
        int64_t sorted_indices_aligned_bytes = indices_aligned_bytes;
        int64_t temp_storage_bytes = InferTempStorageForSortPairsDescending<uint16_t, uint16_t>(instance_size, instance_num);
        GLOBAL_WORKSPACE_SIZE = GetAlignedSize(sorted_in_aligned_bytes + indices_aligned_bytes + sorted_indices_aligned_bytes + temp_storage_bytes);
        cudaMalloc(&d_grouptopk_workspace, GLOBAL_WORKSPACE_SIZE * sizeof(uint8_t)); // 1*1031799296=1GB
    }
    h2 = std::chrono::high_resolution_clock::now();
    int newDT = std::chrono::duration_cast<std::chrono::milliseconds>(h2 - h1).count();

    


    int64_t rtT = 0;
    h1 = std::chrono::high_resolution_clock::now();
    cudaStream_t memcpyStream;
    cudaStreamCreate(&memcpyStream);
    int numPart = 4;
    int numThreads = std::thread::hardware_concurrency() / 4;
    int docsSize = docs.size();
    int docsPerThread = docsSize / numThreads / numPart;
    cudaMemcpy(d_doc_lens, lens.data(), sizeof(uint16_t) * n_docs, cudaMemcpyHostToDevice);
    bool memcpyShot = false;
    for (auto p = 0; p < numPart; p++) {
        std::vector<std::thread> threads;
        for (auto t = 0; t < numThreads; t++) {
            auto start = (p*numThreads+t) * docsPerThread;
            auto end = start + docsPerThread;
            if (p == numPart-1 && t == numThreads-1) end = docsSize;
            threads.emplace_back([&](int s, int e){
                for (int i = s; i < e; i++) {
                    for (int j = 0; j < lens[i]; j++) {
                        h_docs_T[i*MAX_DOC_SIZE+j] = docs[i][j];
                    }
                }
            },start,end);
        }
        if (memcpyShot) {
            auto start = (p-1) * docsPerThread * numThreads;
            auto end = start + docsPerThread * numThreads;
            auto len = end - start;
            cudaMemcpyAsync(d_docs_T + start * MAX_DOC_SIZE, h_docs_T + start * MAX_DOC_SIZE, sizeof(uint16_t) * MAX_DOC_SIZE * len, cudaMemcpyHostToDevice, memcpyStream);
        }
        for (auto& thread : threads) thread.join();
        if (memcpyShot) cudaStreamSynchronize(memcpyStream);
        if (!memcpyShot) memcpyShot = true;
    }
    {
        auto start = (numPart-1) * docsPerThread * numThreads;
        auto len = docsSize - start;
        cudaMemcpy(d_docs_T + start * MAX_DOC_SIZE, h_docs_T + start * MAX_DOC_SIZE, sizeof(uint16_t) * MAX_DOC_SIZE * len, cudaMemcpyHostToDevice);
    }
    h2 = std::chrono::high_resolution_clock::now();
    rtT = std::chrono::duration_cast<std::chrono::milliseconds>(h2 - h1).count();



    int64_t cT = 0;
    h1 = std::chrono::high_resolution_clock::now();
    permute102_kernel<<<grid, block>>>(d_docs,d_docs_T,n_docs);
    cudaDeviceSynchronize();
    h2 = std::chrono::high_resolution_clock::now();
    cT = std::chrono::duration_cast<std::chrono::milliseconds>(h2 - h1).count();





    
    int cur_pos = 0;

    int64_t kernelT = 0, topkT = 0, start_pos = 0;
    std::chrono::high_resolution_clock::time_point tt1, tt2, tt3;


    double b8T = 0;
    kernelT = 0;
    topkT = 0;
    start_pos = cur_pos;
    h1 = std::chrono::high_resolution_clock::now();
    for (; cur_pos+8 < querys.size(); cur_pos+=8) {
        tt1 = std::chrono::high_resolution_clock::now();

        memset(h_dict8, 0, sizeof(uint8_t) * 50000);
        for(int i = 0; i < 8; i++) {
            for(int j = 0; j < querys[cur_pos+i].size(); j++)
                h_dict8[querys[cur_pos+i][j]] |= 1<<i;
            h_query_lens[i] = querys[cur_pos+i].size();
        }
        cudaMemcpy(d_dict8, h_dict8, sizeof(uint8_t) * 50000, cudaMemcpyHostToDevice);
        cudaMemcpy(d_query_lens, h_query_lens, sizeof(uint16_t) * 8, cudaMemcpyHostToDevice);
        docQueryScoringCoalescedMemoryAccessSampleKernelBatchN<uint8_t,8><<<grid, block>>>(
            d_docs, d_doc_lens, n_docs, 
            d_query_lens, 
            d_scores, d_dict8);
        cudaDeviceSynchronize();

        tt2 = std::chrono::high_resolution_clock::now();

        topk_launcher<uint16_t>(0,
            n_docs*8,           // elem_cnt
            grouptopk_size,     // instance_size
            grouptopk_batch*8,  // instance_num
            TOPK,               // top_k
            d_scores,
            d_grouptopk_workspace,
            h_grouptopk_idx,
            h_grouptopk_val);
            



        for(int bb = 0; bb < 8; bb++) {
            std::vector<int> top(grouptopk_batch,0);
            for (int i = 0; i < TOPK; i++) {
                int idx = -1;
                int16_t val = -1;
                for (int j = 0; j < grouptopk_batch; j++) {
                    if (h_grouptopk_val[bb*TOPK*grouptopk_batch+j*TOPK+top[j]] > val) {
                        val = h_grouptopk_val[bb*TOPK*grouptopk_batch+j*TOPK+top[j]];
                        idx = h_grouptopk_idx[bb*TOPK*grouptopk_batch+j*TOPK+top[j]] + j*grouptopk_size;
                    }
                }
                indices[cur_pos+bb][i] = idx;
                top[idx/grouptopk_size]++;
            }
        }

        tt3 = std::chrono::high_resolution_clock::now();

        kernelT += std::chrono::duration_cast<std::chrono::microseconds>(tt2-tt1).count();
        topkT += std::chrono::duration_cast<std::chrono::microseconds>(tt3-tt2).count();
    }

    h2 = std::chrono::high_resolution_clock::now();
    b8T = (double)std::chrono::duration_cast<std::chrono::microseconds>(h2 - h1).count() / (cur_pos-start_pos);

    kernelT /= (cur_pos-start_pos);
    topkT /= (cur_pos-start_pos);
    printf("[TIME] [B8] num:%d, kernel:%ldus, topk:%ldus\n",cur_pos-start_pos,kernelT,topkT);



    double b1T = 0;
    h1 = std::chrono::high_resolution_clock::now();
    if (cur_pos < querys.size()) {
        kernelT = 0;
        topkT = 0;
        start_pos = cur_pos;
        h1 = std::chrono::high_resolution_clock::now();

        tt1 = std::chrono::high_resolution_clock::now();

        int remain = querys.size()-cur_pos;

        memset(h_dict8, 0, sizeof(uint8_t) * 50000);
        for(int i = 0; i < remain; i++) {
            for(int j = 0; j < querys[cur_pos+i].size(); j++)
                h_dict8[querys[cur_pos+i][j]] |= 1<<i;
            h_query_lens[i] = querys[cur_pos+i].size();
        }
        cudaMemcpy(d_dict8, h_dict8, sizeof(uint8_t) * 50000, cudaMemcpyHostToDevice);
        cudaMemcpy(d_query_lens, h_query_lens, sizeof(uint16_t) * remain, cudaMemcpyHostToDevice);
        docQueryScoringCoalescedMemoryAccessSampleKernelBatchN<uint8_t,8><<<grid, block>>>(
            d_docs, d_doc_lens, n_docs, 
            d_query_lens, 
            d_scores, d_dict8);
        cudaDeviceSynchronize();

        tt2 = std::chrono::high_resolution_clock::now();

        topk_launcher<uint16_t>(0,
            n_docs*remain,          // elem_cnt
            grouptopk_size,         // instance_size
            grouptopk_batch*remain, // instance_num
            TOPK,                   // top_k
            d_scores,
            d_grouptopk_workspace,
            h_grouptopk_idx,
            h_grouptopk_val);
            
        for(int bb = 0; bb < remain; bb++) {
            std::vector<int> top(grouptopk_batch,0);
            for (int i = 0; i < TOPK; i++) {
                int idx = -1;
                int16_t val = -1;
                for (int j = 0; j < grouptopk_batch; j++) {
                    if (h_grouptopk_val[bb*TOPK*grouptopk_batch+j*TOPK+top[j]] > val) {
                        val = h_grouptopk_val[bb*TOPK*grouptopk_batch+j*TOPK+top[j]];
                        idx = h_grouptopk_idx[bb*TOPK*grouptopk_batch+j*TOPK+top[j]] + j*grouptopk_size;
                    }
                }
                indices[cur_pos+bb][i] = idx;
                top[idx/grouptopk_size]++;
            }
        }

        cur_pos += remain;

        tt3 = std::chrono::high_resolution_clock::now();
        kernelT += std::chrono::duration_cast<std::chrono::microseconds>(tt2-tt1).count();
        topkT += std::chrono::duration_cast<std::chrono::microseconds>(tt3-tt2).count();

        h2 = std::chrono::high_resolution_clock::now();
        b1T = (double)std::chrono::duration_cast<std::chrono::microseconds>(h2 - h1).count() / (cur_pos-start_pos);

        kernelT /= (cur_pos-start_pos);
        topkT /= (cur_pos-start_pos);
        printf("[TIME] [B1] num:%d, kernel:%ldus, topk:%ldus\n",cur_pos-start_pos,kernelT,topkT);

    }


  





    h1 = std::chrono::high_resolution_clock::now();
    free(h_docs_T);
    free(h_query_lens);
    free(h_dict8);
    free(h_grouptopk_val);
    free(h_grouptopk_idx);
    h2 = std::chrono::high_resolution_clock::now();
    int freeHT = std::chrono::duration_cast<std::chrono::milliseconds>(h2 - h1).count();

    h1 = std::chrono::high_resolution_clock::now();
    cudaFree(d_docs);
    cudaFree(d_docs_T);
    cudaFree(d_query_lens);
    cudaFree(d_doc_lens);
    cudaFree(d_scores);
    cudaFree(d_dict8);
    cudaFree(d_grouptopk_workspace);
    h2 = std::chrono::high_resolution_clock::now();
    int freeDT = std::chrono::duration_cast<std::chrono::milliseconds>(h2 - h1).count();


    printf("\n");
    printf("[TIME] cuda init : %dms\n",initT);
    printf("[TIME] new host : %dms\n",newHT);
    printf("[TIME] new device : %dms\n",newDT);
    printf("[TIME] read&transfer : %ldms\n",rtT);
    printf("[TIME] convert : %ldms\n",cT);
    printf("[TIME] free host : %dms\n",freeHT);
    printf("[TIME] free device : %dms\n",freeDT);
    printf("\n");
    printf("[TIME] Batch8 :%.4lfms\n",b8T/1000);
    printf("[TIME] Batch1 :%.4lfms\n",b1T/1000);  
}