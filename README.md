# TopK

## 问题
从profile来看，41的实现，cpu的sort时间是固定的，在长query的情况下能完全cover住sort，但在短query的情况下要等待cpu
新的实现sort要用占用一部分gpu
考虑的可能是线上以长query为主，新sort占用了一部分的gpu，导致kernel不能完美衔接

## 优化技巧
1. 用nsys profile bash run.sh做性能profile分析瓶颈
2. 用sudo /usr/local/cuda-11.8/nsight-compute-2022.3.0/nv-nsight-cu-cli --set detailed -o hash45 ./bin/query_doc_scoring translate/docs.txt translate/querys translate/res/result.txt做kernel profile分别瓶颈
3. 使用cudaMallocHost在cpu上发分配锁页内存：malloc变慢，host2device变快，但malloc更慢，得不偿失

## 提交记录
1. 0.00022分：convert format和malloc overlap
2. 0.00023分：kernel+memcpy和sort overlap
3. 0.00027分：calloc替换new+memeset，openmp并行化convert的for，碎片malloc合并到convert线程
4. 0.00035分：最后的超长topk sort用多线程分块实现，最后再跑一次sort合并
5. 0.00041分：用双stream把kernel和memcpy给overlap起来，总的是kernel、memcpy、sort三个overlap
6. 0.00045分(最高0.0005分)：kernel用超大数组做哈希map，score量化到uint16_t减少io开销
7. 0.0005分：哈希数组压缩到8bit，提高访存效率
8. 9.22509分：全局CUDA预分配，cub的topk算法

## CPU优化(不算分)
1. 多线程分块读取大txt文件