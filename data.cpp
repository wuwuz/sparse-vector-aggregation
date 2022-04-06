#include "data.hpp"
#include <random>
#include <iostream>
#include <string>
#include <fstream>
#include <map>


ClientData::ClientData(Json &args) {
    //std::random_device rd;
    std::mt19937 gen(0);

    n = args["client"];
    d = args["dim"];
    k = args["sparsity"];

    if (args["type"] == "real") {
        std::map<int, int> flag;
        std::string file_name = args["file_name"];
        std::ifstream input(file_name, std::ios::in);
        input >> n >> d >> k;
        std::cout << "n d k" << n << " " << d << " " << k << std::endl;
        int client_id, key;
        double val;
        for (int i = 0; i < n; i++)
            input_vec.push_back(std::move(KVMap()));
        int j = 0;
        int new_d;
        while (input >> client_id) {
            input >> key >> val;
            if (flag.find(key) == flag.end()) {
                flag[key] = new_d;
                ++new_d;
            }
            key = flag[key];
            if (val < -1)  
                val = -1;
            if (val > 1)
                val = 1;
            input_vec[client_id][key] = val;
            ++j;
            if (j % 100000 == 0) 
                std::cout << j << std::endl;
        }
        input.close();
        //d = new_d;
        std::cout << "new_d " << d << std::endl;
        std::cout << "finished reading" << std::endl;
    } else if (args["type"] == "synthesis") {
        for (int i = 0; i < n; i++) {
            std::uniform_int_distribution<> uniform(0, d - 1);
            std::normal_distribution<> gauss(1, 0.3);
            KVMap kv_map;

            for (int j = 0; j < k; j++) {
                int idx = zipf(gen, 1.4, d) - 1;
                //int idx = uniform(gen);
                //int idx = j;
                //double val = gauss(gen);
                double val = 1;
                val = clip(val, -1, 1);
                if (idx % 3 == 0) {
                    val = -val;
                }
                kv_map[idx] = val;
            }
            input_vec.push_back(std::move(kv_map));
        }
    } else if (args["type"] == "supersparse") {
        for (int i = 0; i < n; i++) {
            //std::uniform_int_distribution<> uniform(0, d - 1);
            std::poisson_distribution<> poisson(k / 2);
            std::normal_distribution<> gauss(1, 0.3);
            KVMap kv_map;

            for (int j = 0; j < k - 1; j++) {
                //int idx = zipf(gen, 1.4, d) - 1;
                int idx = int(poisson(gen));
                double val = gauss(gen);
                val = clip(val, -1, 1);
                if (idx % 3 == 0) {
                    val = -val;
                }
                kv_map[idx] = val;
            }
            kv_map[0] = 1;
            input_vec.push_back(std::move(kv_map));
        }
    } else {
        std::cout << "Undefined data type." << std::endl;
    }

    // computing the real mean

    //real_mean = std::move(vector<double>(d, 0));
    if (args["type"] == "supersparse") {
        int top = 2 * int(args["sparsity"]);
        real_mean = vector<double>(top, 0);

        for (int i = 0; i < n; i++) {
            for (auto it = input_vec[i].begin(); it != input_vec[i].end(); ++it) {
                if (it -> first >= d) {
                    std::cout << "false" << std::endl;
                    continue;
                }
                if (it -> first < top) 
                    real_mean[it -> first] += it -> second;
            }
        }

        for (int i = 0; i < top; i++)
            real_mean[i] /= double(n);
    } else {
        real_mean = vector<double>(d, 0);

        for (int i = 0; i < n; i++) {
            for (auto it = input_vec[i].begin(); it != input_vec[i].end(); ++it) {
                if (it -> first >= d) {
                    std::cout << "false " << it-> first << std::endl;
                    continue;
                }
                real_mean[it -> first] += it -> second;
            }
        }

        for (int i = 0; i < d; i++)
            real_mean[i] /= double(n);
    }

}

KVMap& ClientData::get_kv_map(int idx) {
    return input_vec[idx];
}

vector<double> ClientData::get_vec(int idx) {
    vector<double> vec(d);
    KVMap& kv = get_kv_map(idx);
    for (auto it = kv.begin(); it != kv.end(); ++it) {
        vec[it -> first] = it -> second;
    }
    return vec;
}

vector<double>& ClientData::get_mean() {
    return real_mean;
}

void ClientData::print_info() {
    std::cout << "Client num: " << n << std::endl;
    std::cout << "Dimension: " << d << std::endl;
    std::cout << "Sparsity: " << k << std::endl;

    if (d <= 50) {
        /*
        std::cout << "Client 0 input: " << std::endl;
        auto vec = get_vec(0);
        for (auto v: vec) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
        */

        std::cout << "=========Real mean:========== " << std::endl;
        auto mean = get_mean();
        for (auto v: mean) {
            std::cout << v << " ";
        }
        std::cout << "============================= " << std::endl;
    } else {

        auto mean = get_mean();
        int i = 0;
        std::cout << "=========Real mean(0-99):========== " << std::endl;
        for (auto v: mean) {
            i++;
            std::cout << v << " ";
            if (i == 100) break;
        }
        std::cout << std::endl;
        std::cout << "=================================== " << std::endl;
    }
}
