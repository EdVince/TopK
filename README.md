# TopK

## 技巧
1. 用```nsys profile bash run.sh```做完整pipeline的性能profile分析瓶颈
2. 用```sudo /usr/local/cuda-11.8/nsight-compute-2022.3.0/nv-nsight-cu-cli --set detailed -o kernel ./bin/query_doc_scoring translate/docs.txt translate/querys translate/res/result.txt```做kernel profile分别瓶颈
3. 使用cudaMallocHost在cpu上发分配锁页内存：malloc变慢，host2device变快，但malloc更慢，得不偿失
4. 提交打包：```zip -r result.zip src build.sh run.sh```

## 提交记录(不稳定)
1. 0.00022分：convert format和malloc overlap
2. 0.00023分：kernel+memcpy和sort overlap
3. 0.00027分：calloc替换new+memeset，openmp并行化convert的for，碎片malloc合并到convert线程
4. 0.00035分：最后的超长topk sort用多线程分块实现，最后再跑一次sort合并
5. 0.00041分：用双stream把kernel和memcpy给overlap起来，总的是kernel、memcpy、sort三个overlap
6. 0.00045分(0.0005分)：kernel用超大数组做哈希map，score量化到uint16_t减少io开销
7. 0.0005分：哈希数组压缩到8bit，提高访存效率
8. 9.22509分：全局CUDA预分配，cub的topk算法
9. 10.77255分：实现了batch8
10. 17.56147分(13.64522分)：convert从openmp换成了手动多线程
11. 12.69956分：kernel实现了batch32、batch16、batch8、batchN的全覆盖，topk是batch16、batch8、batchN
12. 20.28398分(19.93734分、17.95793分)：convert线程设置为系统最大线程数量的1/4，测评机内存性能较低
13. 15.49358分(19.29438分、20.66726分)：convert改成read，并跟transfer做4分块的overlap
14. 20.28398分(17.17791分)：read+transpose，但是不overlap了

## CPU优化(不算分)
1. 多线程分块读取大txt文件