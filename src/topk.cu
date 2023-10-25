
#include "topk.h"

typedef uint4 group_t; // uint32_t

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
        const __restrict__ uint16_t *docs, 
        const uint16_t *doc_lens, const size_t n_docs, 
        uint16_t *query, const int query_len, float *scores) {
    // each thread process one doc-query pair scoring task
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x, tnum = gridDim.x * blockDim.x;

    if (tid >= n_docs) {
        return;
    }

    __shared__ uint16_t query_on_shm[MAX_QUERY_SIZE];
#pragma unroll
    for (auto i = threadIdx.x; i < query_len; i += blockDim.x) {
        query_on_shm[i] = query[i]; // not very efficient query loading temporally, as assuming its not hotspot
    }

    __syncthreads();

    for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
        register int query_idx = 0;

        register float tmp_score = 0.;

        register bool no_more_load = false;

        for (auto i = 0; i < MAX_DOC_SIZE / (sizeof(group_t) / sizeof(uint16_t)); i++) {
            if (no_more_load) {
                break;
            }
            register group_t loaded = ((group_t *)docs)[i * n_docs + doc_id]; // tid
            register uint16_t *doc_segment = (uint16_t*)(&loaded);
            for (auto j = 0; j < sizeof(group_t) / sizeof(uint16_t); j++) {
                if (doc_segment[j] == 0) {
                    no_more_load = true;
                    break;
                }
                while (query_idx < query_len && query_on_shm[query_idx] < doc_segment[j]) {
                    ++query_idx;
                }
                if (query_idx < query_len) {
                    tmp_score += (query_on_shm[query_idx] == doc_segment[j]);
                }
            }
            // __syncwarp();
        }
        scores[doc_id] = tmp_score / max(query_len, doc_lens[doc_id]); // tid
    }
}

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices //shape [querys.size(), TOPK]
    ) {

    auto n_docs = docs.size();
    float* scores[2] = {nullptr,nullptr};
    float *d_scores[2] = {nullptr,nullptr};
    uint16_t *d_docs = nullptr;
    uint16_t *d_doc_lens = nullptr;
    uint16_t *d_query = nullptr;
    cudaStream_t kernelStream, memcpyStream;

    // cuda第一次启动要创建context，很慢且无法避免

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // 子线程
    uint16_t *h_docs;
    std::thread convert_format([&]() {
        std::chrono::high_resolution_clock::time_point d1 = std::chrono::high_resolution_clock::now();
        h_docs = (uint16_t*)calloc(MAX_DOC_SIZE*n_docs,sizeof(uint16_t));
        #pragma omp parallel for
        for (int i = 0; i < lens.size(); i++) {
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
        cudaMalloc(&d_scores[0], sizeof(float) * n_docs);
        cudaMalloc(&d_scores[1], sizeof(float) * n_docs);
        cudaMemcpy(d_doc_lens, lens.data(), sizeof(uint16_t) * n_docs, cudaMemcpyHostToDevice);

        std::chrono::high_resolution_clock::time_point d2 = std::chrono::high_resolution_clock::now();
        std::cout << "[CUDA] convert: " << std::chrono::duration_cast<std::chrono::milliseconds>(d2 - d1).count() << " ms " << std::endl;
    });

    // 主线程
    std::chrono::high_resolution_clock::time_point d1 = std::chrono::high_resolution_clock::now();
    cudaMallocHost(&scores[0], n_docs * sizeof(float));
    cudaMallocHost(&scores[1], n_docs * sizeof(float));
    cudaStreamCreate(&kernelStream);
    cudaStreamCreate(&memcpyStream);
    std::vector<int> s_indices(n_docs);
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
    uint8_t memcpyWait = 1;
    uint8_t sortWait = 2;
    bool step = false;
    std::thread sort_thread;
    // number: 0 1 2 3 4 5 6 7 8 9
    // kernel: 0 1 0 1 0 1 0 1
    // memcpy:   0 1 0 1 0 1 0 1
    // sorted:     0 1 0 1 0 1 0 1
    for(auto& query : querys) {

        // host-to-device
        const size_t query_len = query.size();
        cudaMemcpy(d_query, query.data(), sizeof(uint16_t) * query_len, cudaMemcpyHostToDevice);
        // launch kernel
        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (n_docs + block - 1) / block;
        docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0, kernelStream>>>(d_docs, d_doc_lens, n_docs, d_query, query_len, d_scores[step]);

        // device-to-host
        if (memcpyWait == 0)
            cudaMemcpyAsync(scores[!step], d_scores[!step], sizeof(float) * n_docs, cudaMemcpyDeviceToHost, memcpyStream);

        // sort scores
        if (sortWait == 0) {
            sort_thread = std::thread([&]() {
                float* cur_scores = scores[step];

                // for (int i = 0; i < n_docs; ++i) s_indices[i] = i;
                // std::partial_sort(s_indices.begin(), s_indices.begin() + TOPK, s_indices.end(),
                //                 [&cur_scores](const int& a, const int& b) {
                //                     if (cur_scores[a] != cur_scores[b])
                //                         return cur_scores[a] > cur_scores[b];  // 按照分数降序排序
                //                     return a < b;  // 如果分数相同，按索引从小到大排序
                //                 });
                // std::vector<int> s_ans(s_indices.begin(), s_indices.begin() + TOPK);
                // indices.push_back(s_ans);

                // 初始下标
                std::iota(s_indices.begin(), s_indices.end(), 0);
                // 多线程分块topk排序
                const int NUM_THREADS = 8;
                int chunk_size = s_indices.size() / NUM_THREADS;
                std::vector<std::thread> threads;
                std::vector<int> merged_results(NUM_THREADS * TOPK);
                for (int i = 0; i < NUM_THREADS; ++i) {
                    threads.emplace_back([&s_indices, &cur_scores, chunk_size, i, &merged_results]() {
                        int start = i * chunk_size;
                        int end = (i == NUM_THREADS - 1) ? s_indices.size() : start + chunk_size;
                        std::partial_sort(s_indices.begin() + start, s_indices.begin() + start + TOPK, s_indices.begin() + end,
                                        [&cur_scores](const int& a, const int& b) {
                                            if (cur_scores[a] != cur_scores[b])
                                                return cur_scores[a] > cur_scores[b];  // 按照分数降序排序
                                            return a < b;  // 如果分数相同，按索引从小到大排序
                                        });
                        std::copy(s_indices.begin() + start, s_indices.begin() + start + TOPK, merged_results.begin() + i * TOPK);
                    });
                }
                for (auto& thread : threads) {
                    thread.join();
                }
                // 合并块topk排序
                std::partial_sort(merged_results.begin(), merged_results.begin() + TOPK, merged_results.end(),
                                [&cur_scores](const int& a, const int& b) {
                                    if (cur_scores[a] != cur_scores[b])
                                        return cur_scores[a] > cur_scores[b];  // 按照分数降序排序
                                    return a < b;  // 如果分数相同，按索引从小到大排序
                                });
                std::vector<int> s_ans(merged_results.begin(), merged_results.begin() + TOPK);
                indices.push_back(s_ans);
            });
        }

        // 全部同步
        cudaStreamSynchronize(kernelStream);
        cudaStreamSynchronize(memcpyStream);
        if (sortWait == 0)
            sort_thread.join();

        // pipeline调度
        if (memcpyWait > 0) memcpyWait--;
        if (sortWait > 0) sortWait--;

        // 切换到另一个buffer
        step = !step;
    }
    
    cudaMemcpyAsync(scores[!step], d_scores[!step], sizeof(float) * n_docs, cudaMemcpyDeviceToHost, memcpyStream);
    sort_thread = std::thread([&]() {
        float* cur_scores = scores[step];
        // 初始下标
        std::iota(s_indices.begin(), s_indices.end(), 0);
        // 多线程分块topk排序
        const int NUM_THREADS = 8;
        int chunk_size = s_indices.size() / NUM_THREADS;
        std::vector<std::thread> threads;
        std::vector<int> merged_results(NUM_THREADS * TOPK);
        for (int i = 0; i < NUM_THREADS; ++i) {
            threads.emplace_back([&s_indices, &cur_scores, chunk_size, i, &merged_results]() {
                int start = i * chunk_size;
                int end = (i == NUM_THREADS - 1) ? s_indices.size() : start + chunk_size;
                std::partial_sort(s_indices.begin() + start, s_indices.begin() + start + TOPK, s_indices.begin() + end,
                                [&cur_scores](const int& a, const int& b) {
                                    if (cur_scores[a] != cur_scores[b])
                                        return cur_scores[a] > cur_scores[b];  // 按照分数降序排序
                                    return a < b;  // 如果分数相同，按索引从小到大排序
                                });
                std::copy(s_indices.begin() + start, s_indices.begin() + start + TOPK, merged_results.begin() + i * TOPK);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        // 合并块topk排序
        std::partial_sort(merged_results.begin(), merged_results.begin() + TOPK, merged_results.end(),
                        [&cur_scores](const int& a, const int& b) {
                            if (cur_scores[a] != cur_scores[b])
                                return cur_scores[a] > cur_scores[b];  // 按照分数降序排序
                            return a < b;  // 如果分数相同，按索引从小到大排序
                        });
        std::vector<int> s_ans(merged_results.begin(), merged_results.begin() + TOPK);
        indices.push_back(s_ans);
    });

    sort_thread.join();
    cudaStreamSynchronize(memcpyStream);

    sort_thread = std::thread([&]() {
        float* cur_scores = scores[!step];
        // 初始下标
        std::iota(s_indices.begin(), s_indices.end(), 0);
        // 多线程分块topk排序
        const int NUM_THREADS = 8;
        int chunk_size = s_indices.size() / NUM_THREADS;
        std::vector<std::thread> threads;
        std::vector<int> merged_results(NUM_THREADS * TOPK);
        for (int i = 0; i < NUM_THREADS; ++i) {
            threads.emplace_back([&s_indices, &cur_scores, chunk_size, i, &merged_results]() {
                int start = i * chunk_size;
                int end = (i == NUM_THREADS - 1) ? s_indices.size() : start + chunk_size;
                std::partial_sort(s_indices.begin() + start, s_indices.begin() + start + TOPK, s_indices.begin() + end,
                                [&cur_scores](const int& a, const int& b) {
                                    if (cur_scores[a] != cur_scores[b])
                                        return cur_scores[a] > cur_scores[b];  // 按照分数降序排序
                                    return a < b;  // 如果分数相同，按索引从小到大排序
                                });
                std::copy(s_indices.begin() + start, s_indices.begin() + start + TOPK, merged_results.begin() + i * TOPK);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        // 合并块topk排序
        std::partial_sort(merged_results.begin(), merged_results.begin() + TOPK, merged_results.end(),
                        [&cur_scores](const int& a, const int& b) {
                            if (cur_scores[a] != cur_scores[b])
                                return cur_scores[a] > cur_scores[b];  // 按照分数降序排序
                            return a < b;  // 如果分数相同，按索引从小到大排序
                        });
        std::vector<int> s_ans(merged_results.begin(), merged_results.begin() + TOPK);
        indices.push_back(s_ans);
    });

    // deallocation
    cudaFree(d_docs);
    cudaFree(d_query);
    cudaFree(d_scores[0]);
    cudaFree(d_scores[1]);
    cudaFree(d_doc_lens);
    free(h_docs);

    cudaStreamDestroy(kernelStream);
    cudaStreamDestroy(memcpyStream);

    sort_thread.join();

    cudaFreeHost(scores[0]);
    cudaFreeHost(scores[1]);

    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    std::cout << "[CUDA] preprocess: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms " << std::endl;
    std::cout << "[CUDA] process: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms " << std::endl;
}