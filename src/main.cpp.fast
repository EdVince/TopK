#include <cuda.h>
#include <chrono>
#include <vector>
#include <random>
#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <sys/time.h>
#include <stdio.h>
#include <dirent.h>
#include "topk.h"
#include <sys/stat.h>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>

const int threadCount = 4;
std::streamsize loadsize = 256*1024*1024; // 512M
char* loadedFile[2];
std::vector<std::vector<uint16_t>> doc_lens_set(threadCount);
std::vector<std::vector<std::vector<uint16_t>>> docs_set(threadCount);

// 文件获取临界不截断的真正大小
bool words[128];
std::streamsize inline getRealSize(
    std::ifstream* file,
    std::streamoff start,
    std::streamsize size) {

    file->seekg(start+size);
    while (words[file->get()])
        ++size;
    return size;
}

// 内存截断检查
std::streamsize inline getBlockSize(
    int step,
    std::streamoff start,
    std::streamsize size) {

    char* p = loadedFile[step] + start + size;
    while (words[*p]){
        ++size;
        ++p;
    }
    return size;
}

// 文件读入到堆
void inline readLoad(
    int step,
    std::ifstream* file,
    std::streamoff start,
    std::streamsize size) {

    file->seekg(start);
    file->read(loadedFile[step],size);
}

void readBlock(
    int step,
    int id,
    std::streamoff start,
    std::streamsize size) {

    // std::stringstream ss(loadedFile[step]);
    std::stringstream ss;
    ss.rdbuf()->pubsetbuf(loadedFile[step]+start,size);
    std::stringstream line_ss;
    std::string line;
    std::string number;
    while (std::getline(ss, line, '\n')) {
        std::vector<uint16_t> numbers;
        line_ss.clear();
        line_ss << line;
        while (std::getline(line_ss, number, ',')) {
            numbers.push_back(std::stoi(number));
        }
        if (numbers.size() == 0)
            continue;
        docs_set[id].emplace_back(numbers);
        doc_lens_set[id].emplace_back(numbers.size());
    }
}

std::vector<std::string> getFilesInDirectory(const std::string& directory)
{
    std::vector<std::string> files;
    DIR* dirp = opendir(directory.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        struct stat path_stat;
        stat((directory + "/" + dp->d_name).c_str(), &path_stat);
        if (S_ISREG(path_stat.st_mode)) // Check if it's a regular file - not a directory
            files.push_back(dp->d_name);
    }
    closedir(dirp);
    std::sort(files.begin(), files.end()); // sort the files
    return files;
}

struct UserSpecifiedInput
{
    int n_docs;
    std::vector<std::vector<uint16_t>> querys;
    std::vector<std::vector<uint16_t>> docs_ref;
    std::vector<std::vector<uint16_t>> docs;
    std::vector<uint16_t> doc_lens_ref;
    std::vector<uint16_t> doc_lens;

    UserSpecifiedInput(std::string qf, std::string df) {
        load(qf, df);
    }

    void load(std::string query_file_dir, std::string docs_file_name) {

        {
            std::stringstream ss;
            std::string tmp_str;
            std::string tmp_index_str;

            std::vector<std::string> files = getFilesInDirectory(query_file_dir);
            for(const auto& query_file_name: files)
            {
                std::vector<uint16_t> single_query;

                std::ifstream query_file(query_file_dir + "/" + query_file_name);
                while (std::getline(query_file, tmp_str)) {
                    ss.clear();
                    ss << tmp_str;
                    std::cout << query_file_name << ":" << tmp_str << std::endl;
                    while (std::getline(ss, tmp_index_str, ',')) {
                        single_query.emplace_back(std::stoi(tmp_index_str));
                    }
                }
                query_file.close();
                ss.clear();
                std::sort(single_query.begin(), single_query.end()); // pre-sort the query
                querys.emplace_back(single_query);
            }
            std::cout << "query_size: " << querys.size() << std::endl;
        }

        { // 多线程分块读取：https://www.cnblogs.com/lekko/p/3370825.html
            // fix bug
            doc_lens.emplace_back(0);
            docs.emplace_back();

            std::ios::sync_with_stdio(false);
            std::ifstream file;
            file.open(docs_file_name,std::ios::binary|std::ios::in);
            
            std::streamoff start = 0;
            file.seekg(0,std::ios::end);
            std::streamoff size = file.tellg();
            // std::cout<< "total size: " << size << std::endl;

            memset(words,true,128);
            words[10] = false;
            std::streamsize maxsize = loadsize+1024;
            loadedFile[0] = new char[maxsize];
            loadedFile[1] = new char[maxsize];
            std::streamsize realsize;
            std::streamoff index,part;
            bool step = 0;
            bool needWait = false;

            std::streamoff len;
            std::thread* threads = new std::thread[threadCount];

            while(size) {
                // 读取
                realsize = (size>maxsize) ? getRealSize(&file,start,loadsize) : size;
                memset(loadedFile[step],0,maxsize);
                readLoad(step,&file,start,realsize);
                // std::cout<< "read size: " << realsize << std::endl;
                start += realsize;
                size -= realsize;
                // 等待计算线程
                if(needWait) 
                    for (int i=0;i<threadCount;++i) threads[i].join();
                else
                    needWait = true;
                // 合并数据
                for (int i = 0; i < threadCount; i++) {
                    std::copy(doc_lens_set[i].begin(), doc_lens_set[i].end(), std::back_inserter(doc_lens));
                    std::copy(docs_set[i].begin(), docs_set[i].end(), std::back_inserter(docs));
                    doc_lens_set[i].clear();
                    docs_set[i].clear();
                }
                // 开启计算线程
                index=0, part = realsize/threadCount;
                for (int i=0;i<threadCount-1;++i)
                {
                    len = getBlockSize(step,index,part);
                    threads[i] = std::thread(readBlock,step,i,index,len);
                    index += len;
                }
                threads[threadCount-1] = std::thread(readBlock,step,threadCount-1,index,realsize-index);
                // buffer切换
                step = !step;
            }
            // 等待最后一次计算完成
            for (int i=0;i<threadCount;++i) {
                threads[i].join();
            }
            file.close();
            // 合并数据
            for (int i = 0; i < threadCount; i++) {
                std::copy(doc_lens_set[i].begin(), doc_lens_set[i].end(), std::back_inserter(doc_lens));
                std::copy(docs_set[i].begin(), docs_set[i].end(), std::back_inserter(docs));
            }

            n_docs = docs.size();
            std::cout << "doc_size: " << docs.size() << std::endl;
        }



        // {
        //     {
        //         std::stringstream ss;
        //         std::string tmp_str;
        //         std::string tmp_index_str;
                
        //         std::ifstream docs_file(docs_file_name);
        //         while (std::getline(docs_file, tmp_str)) {
        //             std::vector<uint16_t> next_doc;
        //             ss.clear();
        //             ss << tmp_str;
        //             while (std::getline(ss, tmp_index_str, ',')) {
        //                 next_doc.emplace_back(std::stoi(tmp_index_str));
        //             }
        //             docs_ref.emplace_back(next_doc);
        //             doc_lens_ref.emplace_back(next_doc.size());
        //         }
        //         docs_file.close();
        //         ss.clear();
        //         n_docs = docs_ref.size();

        //         std::cout<< "ref size: " << docs_ref.size() << std::endl;
        //     }

        //     int len = docs.size();
        //     int i = 0;
        //     for (i = 0; i < len; i++) {
        //         // 长度检查
        //         if (doc_lens[i] != doc_lens_ref[i]) {
        //             std::cout<<"len diff: "<<i<<","<<doc_lens[i]<<","<<doc_lens_ref[i]<<std::endl;
        //         }
        //         // 长度检查
        //         if (docs[i].size() != docs_ref[i].size()) {
        //             std::cout<<"doc diff: "<<i<<","<<docs[i].size()<<","<<docs_ref[i].size()<<std::endl;
        //         }
        //         // 数据检查
        //         bool cont = false;
        //         int j = 0;
        //         for (j = 0; j < docs[i].size(); j++) {
        //             if (docs[i][j] != docs_ref[i][j]) {
        //                 std::cout<<"cot diff: "<<i<<","<<docs[i][j]<<","<<docs_ref[i][j]<<std::endl;
        //                 cont = true;
        //             }
        //         }
        //         if (cont)
        //             break;
        //     }
        // }
    }
};

int main(int argc, char *argv[])
{    
    if (argc != 4) {
        std::cout << "Usage: query_doc_scoring.bin <doc_file_name> <query_file_name> <output_filename>" << std::endl;
        return -1;
    }
    std::string doc_file_name = argv[1];;
    std::string query_file_dir = argv[2];;
    std::string output_file = argv[3];

    std::cout << "start get topk" << std::endl;

    // �文件
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    UserSpecifiedInput inputs(query_file_dir, doc_file_name);
    std::vector<std::vector<int>> indices;
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "read file cost " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms " << std::endl;

    // 计算得分
    doc_query_scoring_gpu_function(inputs.querys, inputs.docs, inputs.doc_lens, indices);
    
    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    std::cout << "topk cost " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms " << std::endl;

    // get total time
    std::chrono::milliseconds total_time = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1);
    // write result data
    std::ofstream ofs;
    ofs.open(output_file, std::ios::out);
    // first line topk cost time in ms
    ofs << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()  << std::endl;
    // topk index
    for (auto& s_indices : indices) { //makesure indices.size() == querys.size()
        for(size_t i = 0; i < s_indices.size(); ++i)
        {
            ofs << s_indices[i];
            if(i != s_indices.size() - 1) // if not the last element
                ofs << "\t";
        }
        ofs << "\n";
    }
    ofs.close();

    std::cout << "all cost " << total_time.count() << " ms " << std::endl;
    std::cout << "end get topk" << std::endl;
    return 0;
}