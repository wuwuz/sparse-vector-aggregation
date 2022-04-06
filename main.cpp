#include <cstdio>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <thread>
#include <string>
#define ITEM_LEVEL true
#define USER_LEVEL false

#include "json.hpp"
using Json = nlohmann::json;

#include "data.hpp"
#include "utils.hpp"
#include "estimator.hpp"
#include <map>

template<class Estimator> 
std::map<string, double> run_test(Json &args, ClientData &data) {
    Estimator est(args);
    est.estimate(data);

    auto name = est.get_name();

    auto real_mean = data.get_mean();

    int test_time = 10;

    double avg_square = 0;
    double avg_abs = 0;
    double comm_cost = 0;

    for (int i = 0; i < test_time; i++) {
        auto est_mean = est.get_mean();
        double square_err = Metric::mean_square_error(real_mean, est_mean);
        double abs_err = Metric::absolute_error(real_mean, est_mean);
        avg_square += square_err / test_time;
        avg_abs += abs_err / test_time;
        comm_cost = est.get_comm_cost();
    }


    std::cout << "====" << name << "====" << std::endl;
    std::cout << "Absolute Error: " << avg_abs << std::endl;
    std::cout << "Mean Square Error: " << avg_square << std::endl;
    std::cout << "Comm. Cost per Client: " << comm_cost << std::endl;
    //auto est_mean = est.get_mean();
    //for (auto it : est_mean) 
    //    std::cout << it << " ";
    //std::cout << std::endl;
    std::cout << "==============" << std::endl << std::endl;

    std::map<string, double> res;
    res["abs_err"] = avg_abs;
    res["square_err"] = avg_square;
    res["comm_cost"] = comm_cost;
    return res;
}

template<class Estimator> 
std::map<string, double> run_test_super_sparse(Json &args, ClientData &data, int top) {
    Estimator est(args);
    est.estimate_range(data, top);

    auto name = est.get_name();

    auto real_mean = data.get_mean();

    int test_time = 10;

    double avg_square = 0;
    double avg_abs = 0;
    double comm_cost = 0;

    for (int i = 0; i < test_time; i++) {
        auto est_mean = est.get_mean();
        double square_err = Metric::mean_square_error(real_mean, est_mean);
        double abs_err = Metric::absolute_error(real_mean, est_mean);
        avg_square += square_err / test_time;
        avg_abs += abs_err / test_time;
        comm_cost = est.get_comm_cost();
    }


    std::cout << "====" << name << "====" << std::endl;
    std::cout << "Absolute Error: " << avg_abs << std::endl;
    std::cout << "Mean Square Error: " << avg_square << std::endl;
    std::cout << "Comm. Cost per Client: " << comm_cost << std::endl;
    //auto est_mean = est.get_mean();
    //for (auto it : est_mean) 
    //    std::cout << it << " ";
    //std::cout << std::endl;
    std::cout << "==============" << std::endl << std::endl;

    std::map<string, double> res;
    res["abs_err"] = avg_abs;
    res["square_err"] = avg_square;
    res["comm_cost"] = comm_cost;
    return res;
}


void sparsity_test(bool dp_level = USER_LEVEL) {
    Json args = {
        {"client", 100000},
        {"dim", 4096},
        {"sparsity", 1},
        {"eps", 1.0},
        {"neighbor_dist", -1.0},
        {"delta", 0.000001},
        {"type", "synthesis"}
    };
    //int n = args["client"];
    int d = args["dim"];

    string of1, of2, of3;

    if (dp_level == ITEM_LEVEL) {
        args["neighbor_dist"] = 2.0;
        of1 = "large_k_abs_item.csv";
        of2 = "large_k_mse_item.csv";
        of3 = "large_k_comm_item.csv";
    } else {
        args["neighbor_dist"] = -1.0;
        of1 = "large_k_abs.csv";
        of2 = "large_k_mse.csv";
        of3 = "large_k_comm.csv";
    }
    std::ofstream out1(of1, std::ios_base::app);
    std::ofstream out2(of2, std::ios_base::app);
    std::ofstream out3(of3, std::ios_base::app);

    out1 << std::fixed << std::setprecision(6);
    out1 << "k, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;
    out2 << std::fixed << std::setprecision(6);
    out2 << "k, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;
    out3 << std::fixed << std::setprecision(6);
    out3 << "k, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;

    for (int k = 2; k * k <= d; k += 2) {
        args["sparsity"] = k * k;
        ClientData data(args);
        data.print_info();

        auto svme = run_test<SVME>(args, data);
        auto harmony = run_test<Harmony>(args, data);
        auto gm = run_test<GaussianMechanism>(args, data);
        auto pckv = run_test<PCKV>(args, data);
        auto hyb = run_test<Hybrid>(args, data);
        auto stm = run_test<Strawman>(args, data);

        string met = "abs_err";
        out1 << std::fixed << std::setprecision(6) << args["sparsity"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
        met = "square_err";
        out2 << std::fixed << std::setprecision(6) << args["sparsity"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
        met = "comm_cost";
        out3 << std::fixed << std::setprecision(6) << args["sparsity"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
    }
}


void neighbor_dist_test() {
    Json args = {
        {"client", 100000},
        {"dim", 4096},
        {"sparsity", 64},
        {"eps", 1.0},
        {"neighbor_dist", -1},
        {"delta", 0.000001},
        {"type", "synthesis"}
    };
    //int n = args["client"];
    //int d = args["dim"];

    std::ofstream out1("vary_L_abs.csv", std::ios_base::app);
    out1 << std::fixed << std::setprecision(6);
    out1 << "L, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;
    std::ofstream out2("vary_L_mse.csv", std::ios_base::app);
    out2 << std::fixed << std::setprecision(6);
    out2 << "L, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;
    std::ofstream out3("vary_L_comm.csv", std::ios_base::app);
    out3 << std::fixed << std::setprecision(6);
    out3 << "L, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;

    //for (int L = 2; L * L <= d; L += 2) {
    for (int L = 34; L <= 2 * 64; L += 4) {
        args["neighbor_dist"] = double(L);
        ClientData data(args);
        data.print_info();

        auto svme = run_test<SVME>(args, data);
        auto harmony = run_test<Harmony>(args, data);
        auto gm = run_test<GaussianMechanism>(args, data);
        auto pckv = run_test<PCKV>(args, data);
        auto hyb = run_test<Hybrid>(args, data);
        auto stm = run_test<Strawman>(args, data);

        string met = "abs_err";
        out1 << std::fixed << std::setprecision(6) << args["neighbor_dist"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
        met = "square_err";
        out2 << std::fixed << std::setprecision(6) << args["neighbor_dist"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
        met = "comm_cost";
        out3 << std::fixed << std::setprecision(6) << args["neighbor_dist"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
    }
}

void vary_eps_test(bool dp_level = USER_LEVEL) {
    Json args = {
        {"client", 100000},
        {"dim", 100000},
        {"sparsity", 64},
        {"eps", 1.0},
        {"neighbor_dist", -1},
        {"delta", 0.000001},
        {"type", "supersparse"}
    };
    //int n = args["client"];
    //int d = args["dim"];
    int k = args["sparsity"];

    string of1, of2, of3;

    if (dp_level == ITEM_LEVEL) {
        args["neighbor_dist"] = 2.0;
        of1 = "vary_e_abs_item.csv";
        of2 = "vary_e_mse_item.csv";
        of3 = "vary_e_comm_item.csv";
    } else {
        args["neighbor_dist"] = -1.0;
        of1 = "vary_e_abs.csv";
        of2 = "vary_e_mse.csv";
        of3 = "vary_e_comm.csv";
    }
    std::ofstream out1(of1, std::ios_base::app);
    std::ofstream out2(of2, std::ios_base::app);
    std::ofstream out3(of3, std::ios_base::app);

    out1 << std::fixed << std::setprecision(6);
    out1 << "e, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;
    out2 << std::fixed << std::setprecision(6);
    out2 << "e, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;
    out3 << std::fixed << std::setprecision(6);
    out3 << "e, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;

    for (double e = 0.5; e <= 3.5; e += 0.5) {
        args["eps"] = double(e);
        ClientData data(args);
        data.print_info();

        auto svme = run_test_super_sparse<SVME>(args, data, 2 * k);
        auto harmony = run_test_super_sparse<Harmony>(args, data, 2 * k);
        auto gm = run_test_super_sparse<GaussianMechanism>(args, data, 2 * k);
        auto pckv = run_test_super_sparse<PCKV>(args, data, 2 * k);
        auto hyb = run_test_super_sparse<Hybrid>(args, data, 2 * k);
        auto stm = run_test_super_sparse<Strawman>(args, data, 2 * k);

        /*
        auto svme = run_test<SVME>(args, data);
        auto harmony = run_test<Harmony>(args, data);
        auto gm = run_test<GaussianMechanism>(args, data);
        auto pckv = run_test<PCKV>(args, data);
        auto hyb = run_test<Hybrid>(args, data);
        auto stm = run_test<Strawman>(args, data);
        */

        string met = "abs_err";
        out1 << std::fixed << std::setprecision(6) << args["eps"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
        met = "square_err";
        out2 << std::fixed << std::setprecision(6) << args["eps"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
        met = "comm_cost";
        out3 << std::fixed << std::setprecision(6) << args["eps"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
    }
}


void super_sparse_test(bool dp_level = USER_LEVEL) {
    Json args = {
        {"client", 100000},
        {"dim", 100},
        {"sparsity", 64},
        {"eps", 1.0},
        {"neighbor_dist", -1},
        {"delta", 0.000001},
        {"type", "supersparse"}
    };
    //int n = args["client"];
    //int d = args["dim"];
    int k = args["sparsity"];

    string of1, of2, of3;

    if (dp_level == ITEM_LEVEL) {
        args["neighbor_dist"] = 2.0;
        of1 = "vary_d_abs_item.csv";
        of2 = "vary_d_mse_item.csv";
        of3 = "vary_d_comm_item.csv";
    } else {
        args["neighbor_dist"] = -1.0;
        of1 = "vary_d_abs.csv";
        of2 = "vary_d_mse.csv";
        of3 = "vary_d_comm.csv";
    }
    std::ofstream out1(of1, std::ios_base::app);
    std::ofstream out2(of2, std::ios_base::app);
    std::ofstream out3(of3, std::ios_base::app);

    out1 << std::fixed << std::setprecision(6);
    out1 << "d, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;
    out2 << std::fixed << std::setprecision(6);
    out2 << "d, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;
    out3 << std::fixed << std::setprecision(6);
    out3 << "d, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;

    //for (int k = 2; k * k <= d; k += 2) {
    //for (double e = 0.5; e <= 3.5; e += 0.5) {
    for (int d = 1 << 20; d <= 1e8; d <<= 1) {
    //for (int d = int(args["dim"]); d <= 1e8; d <<= 1) {
    //for (int d = 100; d <= )
    //for (int i = 10; i <= 200; i += 10) {
    //    args["dim"] = i * i;
        args["dim"] = d;
        ClientData data(args);
        data.print_info();

        auto svme = run_test_super_sparse<SVME>(args, data, 2 * k);
        auto harmony = run_test_super_sparse<Harmony>(args, data, 2 * k);
        auto gm = run_test_super_sparse<GaussianMechanism>(args, data, 2 * k);
        auto pckv = run_test_super_sparse<PCKV>(args, data, 2 * k);
        auto hyb = run_test_super_sparse<Hybrid>(args, data, 2 * k);
        auto stm = run_test_super_sparse<Strawman>(args, data, 2 * k);

        string met = "abs_err";
        out1 << std::fixed << std::setprecision(6) << args["dim"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
        met = "square_err";
        out2 << std::fixed << std::setprecision(6) << args["dim"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
        met = "comm_cost";
        out3 << std::fixed << std::setprecision(6) << args["dim"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
    }
}

void real_test(string file_name, bool dp_level = USER_LEVEL) {
    Json args = {
        {"client", 100000},
        {"dim", 100},
        {"sparsity", 64},
        {"eps", 1.0},
        {"neighbor_dist", -1},
        {"delta", 0.000001},
        {"type", "real"},
        {"file_name", ""}
    };

    int n, d, k;
    std::ifstream input(file_name, std::ios::in);
    input >> n >> d >> k;
    input.close();

    args["client"] = n;
    args["dim"] = d;
    args["sparsity"] = k;
    args["file_name"] = file_name;

    string of1, of2, of3;

    string dataset_name = file_name.substr(0, file_name.size() - 4);
    std::cout << dataset_name << std::endl;

    if (dp_level == ITEM_LEVEL) {
        args["neighbor_dist"] = 2.0;
        of1 = dataset_name + "_abs_item.csv";
        of2 = dataset_name + "_mse_item.csv";
        of3 = dataset_name + "_comm_item.csv";
    } else {
        args["neighbor_dist"] = -1.0;
        of1 = dataset_name + "_abs.csv";
        of2 = dataset_name + "_mse.csv";
        of3 = dataset_name + "_comm.csv";
    }
    std::ofstream out1(of1, std::ios_base::app);
    std::ofstream out2(of2, std::ios_base::app);
    std::ofstream out3(of3, std::ios_base::app);

    out1 << std::fixed << std::setprecision(6);
    out1 << "e, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;
    out2 << std::fixed << std::setprecision(6);
    out2 << "e, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;
    out3 << std::fixed << std::setprecision(6);
    out3 << "e, SVME, Harmony, Gaussian, PCKV, Hybrid, Strawman" << std::endl;

    for (double e = 1; e <= 1; e += 1.0) {
        args["eps"] = double(e);
        ClientData data(args);
        data.print_info();

        auto svme = run_test<SVME>(args, data);
        auto harmony = run_test<Harmony>(args, data);
        auto gm = run_test<GaussianMechanism>(args, data);
        auto pckv = run_test<PCKV>(args, data);
        auto hyb = run_test<Hybrid>(args, data);
        auto stm = run_test<Strawman>(args, data);

        string met = "abs_err";
        out1 << std::fixed << std::setprecision(6) << args["eps"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
        met = "square_err";
        out2 << std::fixed << std::setprecision(6) << args["eps"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
        met = "comm_cost";
        out3 << std::fixed << std::setprecision(6) << args["eps"] << ", " << svme[met] << ", " << harmony[met] << ", " << gm[met] << ", " << pckv[met] << ", " << hyb[met] << ", " << stm[met] <<std::endl;
    }
}


int main() {

    //Single configuration test
    //Configuration is set in config.json
    //arg parser
    std::ifstream input("config.json");
    Json args;
    input >> args;

    ClientData data(args);
    data.print_info();
    run_test<SVME>(args, data);
    run_test<Hybrid>(args, data);
    run_test<Strawman>(args, data);
    run_test<PCKV>(args, data);
    run_test<GaussianMechanism>(args, data);
    run_test<Harmony>(args, data);
    //return 0;

    // All the test we showed in the papers.
    bool dp_level = USER_LEVEL;
    std::vector<std::thread> th;
    //th.push_back(std::thread(vary_eps_test, dp_level));
    //th.push_back(std::thread(vary_eps_test, !dp_level));
    //th.push_back(std::thread(sparsity_test, dp_level));
    //th.push_back(std::thread(vary_eps_test, dp_level));
    //th.push_back(std::thread(super_sparse_test, dp_level));
    //th.push_back(std::thread(super_sparse_test, !dp_level));
    //th.push_back(std::thread(neighbor_dist_test));
    //th.push_back(std::thread(real_test, "rent.txt", dp_level));
    //th.push_back(std::thread(real_test, "cloth.txt", dp_level));
    //th.push_back(std::thread(real_test, "rent.txt", !dp_level));
    //th.push_back(std::thread(real_test, "cloth.txt", !dp_level));
    //th.push_back(std::thread(real_test, "movie100.txt", dp_level));
    //th.push_back(std::thread(real_test, "movie100.txt", !dp_level));

    for (auto &t: th) {
        t.join();
    }

    return 0;
}
