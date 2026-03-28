#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>

std::vector<float> load_fvecs(const std::string& path, size_t& num_vectors, size_t& dim);
std::vector<int> load_ivecs(const std::string& filename, size_t& n, size_t& k);
#endif
