# TopK

## 技巧
1. 用nsys profile bash run.sh做性能profile分析瓶颈
2. 使用cudaMallocHost在cpu上发分配锁页内存：malloc变慢，host2device变快，但malloc更慢，得不偿失

## 提交
1. 0.00022分：convert format和malloc overlap
2. 0.00023分：kernel+memcpy和sort overlap
3. 0.00027分：calloc替换new+memeset，openmp并行化convert的for，碎片malloc合并到convert线程
4. 0.00035分：最后的超长topk sort用多线程分块实现，最后再跑一次sort合并