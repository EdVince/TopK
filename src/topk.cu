#include "topk.h"

typedef uint4 group_t; // uint32_t

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
        const __restrict__ uint16_t *docs, 
        const uint16_t *doc_lens, const size_t n_docs, 
        uint16_t *query, const int query_len, int16_t *scores, uint8_t *dict) {
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x, tnum = gridDim.x * blockDim.x;
    if (tid >= n_docs)
        return;

// #pragma unroll
    // for (auto i = threadIdx.x; i < query_len; i += blockDim.x) {
    //     dict[query[i]/8] += 1<<(query[i]%8);
    // }
    if (threadIdx.x == 0) {
        for (auto i = 0; i < query_len; i++) {
            dict[query[i]/8] |= 1<<(query[i]%8);
        }
    }
    __syncthreads();

    for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
        register uint32_t tmp_score = 0.;
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
                if ((dict[doc_segment[j]/8] >> (doc_segment[j]%8)) & 1) {
                    tmp_score += 1;
                }
            }
            __syncwarp();
        }
        scores[doc_id] = 16384 * tmp_score / max(query_len, doc_lens[doc_id]); // tid
    }
}

void __global__ groupArgMax(const int16_t *scores, const int32_t size, int32_t* max_idx) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;
    register int32_t _max_val = -1;
    register int32_t _max_idx = -1;
    for(auto i = 0; i < 512; i++) {
        if(scores[tid*512+i] > _max_val) {
            _max_val = scores[tid*512+i];
            _max_idx = i;
        }
    }
    max_idx[tid] = _max_idx;
}

struct BlockInfo {
    int32_t block_id;
    int32_t max_index;
    int32_t max_value;
};

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices //shape [querys.size(), TOPK]
    ) {

    int n_docs = docs.size();
    n_docs = ((n_docs - 1) / 512 + 1) * 512;
    for(int i = 0; i < (n_docs-docs.size()); i++)
        lens.emplace_back();

    int16_t* scores[2] = {nullptr,nullptr};
    int16_t* d_scores[2] = {nullptr,nullptr};
    uint16_t* d_docs = nullptr;
    uint16_t* d_doc_lens = nullptr;
    uint16_t* d_query = nullptr;
    int32_t* d_group_max_index[2] = {nullptr,nullptr};
    int32_t* group_max_index = nullptr;
    uint8_t* dict = nullptr;

    cudaStream_t kernelStream, memcpyStream;

    int group_max_block_size = 512;
    int group_max_block_num = n_docs / group_max_block_size;

    auto comparator = [](const BlockInfo& a, const BlockInfo& b) {
        if (a.max_value != b.max_value)
            return a.max_value < b.max_value;
        return a.max_index > b.max_index;
    };

    // cuda第一次启动要创建context，很慢且无法避免

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
        cudaMalloc(&d_scores[0], sizeof(int16_t) * n_docs);
        cudaMalloc(&d_scores[1], sizeof(int16_t) * n_docs);
        cudaMalloc(&dict, sizeof(uint8_t) * 6250);
        cudaMemcpy(d_doc_lens, lens.data(), sizeof(uint16_t) * n_docs, cudaMemcpyHostToDevice);

        std::chrono::high_resolution_clock::time_point d2 = std::chrono::high_resolution_clock::now();
        std::cout << "[CUDA] convert: " << std::chrono::duration_cast<std::chrono::milliseconds>(d2 - d1).count() << " ms " << std::endl;
    });

    // 主线程
    std::chrono::high_resolution_clock::time_point d1 = std::chrono::high_resolution_clock::now();
    cudaMallocHost(&scores[0], n_docs * sizeof(int16_t));
    cudaMallocHost(&scores[1], n_docs * sizeof(int16_t));
    cudaMallocHost(&group_max_index, group_max_block_num * sizeof(int32_t));
    cudaMalloc(&d_group_max_index[0], group_max_block_num * sizeof(int32_t));
    cudaMalloc(&d_group_max_index[1], group_max_block_num * sizeof(int32_t));
    cudaStreamCreate(&kernelStream);
    cudaStreamCreate(&memcpyStream);
    cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
    std::chrono::high_resolution_clock::time_point d2 = std::chrono::high_resolution_clock::now();
    std::cout << "[CUDA] malloc: " << std::chrono::duration_cast<std::chrono::milliseconds>(d2 - d1).count() << " ms " << std::endl;

    convert_format.join();

    // 非常耗时
    cudaMemcpy(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, cudaMemcpyHostToDevice);

    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    cudaSetDevice(0);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    bool step = 0;
    bool needWait = false;
    std::thread sort_thread;
    for(auto& query : querys) {

        cudaMemset(dict, 0, sizeof(uint8_t) * 6250);

        // host-to-device
        const size_t query_len = query.size();
        cudaMemcpy(d_query, query.data(), sizeof(uint16_t) * query_len, cudaMemcpyHostToDevice);
        // launch kernel
        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (n_docs + block - 1) / block;
        docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0, kernelStream>>>(d_docs, d_doc_lens, n_docs, d_query, query_len, d_scores[step], dict);
        // cudaDeviceSynchronize();
        cudaStreamSynchronize(kernelStream);
        groupArgMax<<<group_max_block_num/512+1,512,0,kernelStream>>>(d_scores[step],n_docs,d_group_max_index[step]);
        cudaStreamSynchronize(kernelStream);

        if (needWait) {
            sort_thread.join();
        }
        else
            needWait = true;

        // sort scores
        if (needWait) {
            sort_thread = std::thread([&](int cur_step) {
                cudaMemcpyAsync(scores[cur_step], d_scores[cur_step], sizeof(int16_t) * n_docs, cudaMemcpyDeviceToHost, memcpyStream);
                cudaStreamSynchronize(memcpyStream);
                cudaMemcpyAsync(group_max_index, d_group_max_index[cur_step], group_max_block_num * sizeof(int32_t), cudaMemcpyDeviceToHost, memcpyStream);
                cudaStreamSynchronize(memcpyStream);
                int16_t* cur_scores = scores[cur_step];
                std::priority_queue<BlockInfo, std::vector<BlockInfo>, decltype(comparator)> max_record(comparator);
                for(int i = 0; i < group_max_block_num; i++)
                    max_record.push({i,group_max_index[i]+i*512,cur_scores[group_max_index[i]+i*512]});
                std::vector<int> topk(TOPK);
                for(int i = 0; i < TOPK; i++) {
                    BlockInfo max_info = max_record.top();
                    max_record.pop();
                    topk[i] = max_info.max_index;
                    cur_scores[max_info.max_index] = -1; // 一定要清到小于0，不然会跟原始的0值发生冲突
                    int max_index = std::max_element(cur_scores+max_info.block_id*512, cur_scores+max_info.block_id*512+512) - (cur_scores+max_info.block_id*512);
                    max_info.max_index = max_index+max_info.block_id*512;
                    max_info.max_value = cur_scores[max_info.block_id*512+max_index];
                    max_record.push(max_info);
                }
                indices.push_back(topk); // 存储结果
            }, step);
            // sort_thread.join();
        }

        step = !step;
    }

    sort_thread.join();

    // deallocation
    cudaFree(dict);
    cudaFree(d_docs);
    cudaFree(d_query);
    cudaFree(d_scores[0]);
    cudaFree(d_scores[1]);
    cudaFree(d_doc_lens);
    cudaFree(d_group_max_index[0]);
    cudaFree(d_group_max_index[1]);
    free(h_docs);
    cudaFreeHost(scores[0]);
    cudaFreeHost(scores[1]);
    cudaFreeHost(group_max_index);
    cudaStreamDestroy(kernelStream);
    cudaStreamDestroy(memcpyStream);

    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    std::cout << "[CUDA] preprocess: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms " << std::endl;
    std::cout << "[CUDA] process: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms " << std::endl;
}