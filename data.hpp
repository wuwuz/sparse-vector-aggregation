#ifndef DATA_H
#define DATA_H

#include "json.hpp"
#include "utils.hpp"
#include <vector>
#include <map>

using Json = nlohmann::json;
using std::vector;
using KVMap = std::unordered_map<int, double>;
//typedef std::unordered_map Map;

class ClientData {
  private:
    vector<KVMap> input_vec;
    vector<double> real_mean;
    int n; // client num
    int d; // vector dim
    int k; // vector sparsity
    //Json args;
  public: 
    ClientData(Json &args);
    KVMap& get_kv_map(int idx);
    vector<double> get_vec(int idx);
    vector<double>& get_mean();
    void print_info();
};


#endif