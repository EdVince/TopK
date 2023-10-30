# TopK

## 问题
从profile来看，41的实现，cpu的sort时间是固定的，在长query的情况下能完全cover住sort，但在短query的情况下要等待cpu
新的实现sort要用占用一部分gpu
考虑的可能是线上以长query为主，新sort占用了一部分的gpu，导致kernel不能完美衔接

数量大于10，平均长度大于64，存在
数量大于100，平均长度大于64，不存在
数量大于100，平均长度大于48，存在
数量大于100，平均长度大于56，存在
数量大于100，平均长度大于60，不存在
数量大于1000，平均长度大于64，不存在
数量大于50，存在
数量大于100，存在
数量大于200，不存在


## 优化技巧
1. 用nsys profile bash run.sh做性能profile分析瓶颈
2. 使用cudaMallocHost在cpu上发分配锁页内存：malloc变慢，host2device变快，但malloc更慢，得不偿失

## 提交记录
1. 0.00022分：convert format和malloc overlap
2. 0.00023分：kernel+memcpy和sort overlap
3. 0.00027分：calloc替换new+memeset，openmp并行化convert的for，碎片malloc合并到convert线程
4. 0.00035分：最后的超长topk sort用多线程分块实现，最后再跑一次sort合并
5. 0.00041分：用双stream把kernel和memcpy给overlap起来，总的是kernel、memcpy、sort三个overlap

## CPU优化(不算分)
1. 多线程分块读取大txt文件