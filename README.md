# TopK

1. 使用cudaMallocHost在cpu上发分配锁页内存：malloc变慢，host2device变快，但malloc更慢，得不偿失
2. convert format和malloc overlap起来：0.00022分
3. kernel+memcpy和sort overlap起来：0.00023分