#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>

#include "utils.hpp"
#include "data.hpp"
#include "json.hpp"
using Json = nlohmann::json;

#include "estimator.hpp"


/*
 * SVME: our algorithm in the paper
 * Implementing SVME class
 */

SVME::SVME(Json &_args) {
    args = _args;
    n = args["client"];
    d = args["dim"];
    k = args["sparsity"];
    eps = args["eps"];
    delta = args["delta"];
    if (args["neighbor_dist"] == -1) {
        L = 2.0 * k;
    } else {
        L = args["neighbor_dist"];
        L = std::min(L, 2.0 * k);
    }

}

string SVME::get_name() { return "SVME"; }
vector<double>& SVME::get_mean() { return mean_vec; }
double SVME::get_comm_cost() { return comm_cost; } 
double SVME::predicted_error() {
    return std::max(sqrt(k / bucket), noise_param);
}

int SVME::sampling(int x, int seed, double p) {
    int t = HashUtil::MurmurHash32(x ^ seed ^ 0x7765439) & 0xfffffff;
    return int(double(t) < p * 0xfffffff);
}

int SVME::hash_bin(int x, int seed, int bucket) {
    int t = HashUtil::MurmurHash32(x ^ seed ^ 0x1998765) % bucket;
    return t;
}

double SVME::random_sign(int x, int seed) {
    int t = HashUtil::MurmurHash32(x ^ seed ^ 0x18733ff) % 2;
    if (t == 1) {
        return 1;
    } else {
        return -1;
    }
}

double SVME::rr(double v, double eps, double range, std::mt19937 &gen) {    
    //randomized response
    // rr sometimes has better constant error than Laplacian mechanism
    v = v / range;
    double e_eps = exp(eps);
    double c_eps = (e_eps + 1) / (e_eps - 1);
    double p = (v * (e_eps - 1) + e_eps + 1) / (2 * (1 + e_eps));
    double noisy_v = Harmony::bern(gen, p) ? c_eps : -c_eps;
    noisy_v *= range;

    return noisy_v;
}

void SVME::estimate_range(ClientData &data, int top) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> uniform(0, 0x7fffffff - 1);

    vector<int> seed(n);
    vector<vector<float> > T; // the randomization output

    int bucket;
    double clip_r;
    double noise_param;

    // This is a simplified version of the algorithm
    // We only set the noise param to 
    //
    // L / eps
    // or 
    // 2 * clipping range / eps
    //
    // In the paper, there is another setting that noise param
    // 3sqrt(2bL log(2b/delta))

    if (L < sqrt(k)) {
        bucket = std::min(int(k * 1.0 / L / L), k);
        noise_param = L;
        clip_r = k;
    } else {
        bucket = 1;
        // We set a smaller clipping range for better error constant
        // Notice that this won't affect privacy
        // We generally think that directly setting clip_r = sqrt(k) could match the lower bound
        // The actual performance is even better :)
        clip_r = sqrt(k * log(n)) / 2.5; 
        noise_param = std::min(2 * clip_r, L);
    }

    if (L / bucket > log(2 * bucket / delta)) {
        noise_param = std::min(noise_param, 3 * sqrt(bucket * L * log(2 * bucket / delta)));
    }

    std::cout << "#bucket  = " << bucket << std::endl;
    std::cout << "Clipping Range = " << clip_r << std::endl;
    std::cout << "Lap Noise Magnitude = " << noise_param << std::endl;

    noise_param /= eps;

    int tot_clipped = 0;

    for (int i = 0; i < n; i++) {
        // local randomization process
        seed[i] = uniform(gen);
        vector<float> T_i(bucket, 0);
        auto kv = data.get_kv_map(i);

        vector<float> ns(bucket, 0);
        for (auto const &it : kv) {
            int j = it.first;
            double v = it.second;
            int hash_j = SVME::hash_bin(j, seed[i], bucket);
            double sign_j = SVME::random_sign(j, seed[i]);
            T_i[hash_j] += sign_j * v;
        }

        for (int j = 0; j < bucket; j++) {

            T_i[j] = clip(T_i[j], -clip_r, clip_r);
            if (noise_param >= clip_r) {
                T_i[j] = SVME::rr(T_i[j], eps, clip_r, gen);
            } else {
                T_i[j] += SVME::laplacian_noise(gen, noise_param);
            }
        }
        T.push_back(std::move(T_i));
    }

    //record communication cost
    comm_cost = sizeof(seed[0]) + sizeof(T[0][0]) * T[0].size();

    //aggregation
    mean_vec = vector<double>(top, 0);
    auto cnt_vec = vector<double>(top, 0);
    auto noise_vec = vector<double>(top, 0);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < top; j++) {
            int hash_j = SVME::hash_bin(j, seed[i], bucket);
            double sign_j = SVME::random_sign(j, seed[i]);

            cnt_vec[j] += 1;
            mean_vec[j] += T[i][hash_j] * sign_j;
        }

    for (int j = 0; j < top; j++) {
        mean_vec[j] /= cnt_vec[j];
        mean_vec[j] = clip(mean_vec[j], -1, 1);
    }
}


double SVME::laplacian_noise(std::mt19937 &gen, double noise_param) {
    if (noise_param == 0) 
        return 0;
    std::exponential_distribution<> exp_dist(1 / noise_param);
    std::bernoulli_distribution bern(0.5);
    double t = exp_dist(gen);
    if (bern(gen)) {
        return t;
    } else {
        return -t;
    }
}


void SVME::estimate(ClientData &data) {
    estimate_range(data, d);
}


GaussianMechanism::GaussianMechanism(Json &_args) {
    args = _args;
    n = args["client"];
    d = args["dim"];
    k = args["sparsity"];
    eps = args["eps"];
    delta = args["delta"];
    if (args["neighbor_dist"] == -1) {
        L = 2.0 * k;
    } else {
        L = args["neighbor_dist"];
        L = std::min(L, 2.0 * k);
    }

    noise_param = sqrt(2 * log(1.25 / delta) * 2 * L) / eps;
}

string GaussianMechanism::get_name() { return "GaussianMechanism"; }
vector<double>& GaussianMechanism::get_mean() { return mean_vec; }
double GaussianMechanism::get_comm_cost() { return comm_cost; } 

double GaussianMechanism::gaussian_noise(std::mt19937 &gen, double noise_param) {
    std::normal_distribution<> d(0, noise_param);
    double t = d(gen);
    return t;
}

void GaussianMechanism::estimate_range(ClientData &data, int top) {
    std::random_device rd;
    std::mt19937 gen(rd());

    mean_vec = vector<double>(top, 0);

    for (int i = 0; i < n; i++) {
        // local randomization process
        //auto vec = data.get_vec(i);
        auto kv = data.get_kv_map(i);
        auto vec = vector<double>(top, 0);
        for (auto it = kv.begin(); it != kv.end(); ++it) {
            int j = it -> first;
            double v = it -> second;
            if (j < top) {
                vec[j] = v;
            }
        }
        for (int j = 0; j < top; j++) {
            //double v = 0;
            //if (kv.find(j) != kv.end()) 
            //    v = kv[j];
            // local randomization 
            vec[j] += GaussianMechanism::gaussian_noise(gen, noise_param);
            mean_vec[j] += vec[j];
        }

        //aggregation
        for (int j = 0; j < top; j++) {
        }
    }

    for (int j = 0; j < top; j++) {
        mean_vec[j] /= n;
        mean_vec[j] = clip(mean_vec[j], -1, 1);
    }

    comm_cost = sizeof(float) * d;
}

void GaussianMechanism::estimate(ClientData &data) {
    GaussianMechanism::estimate_range(data, d);
}

PCKV::PCKV(Json &_args) {
    args = _args;
    n = args["client"];
    d = args["dim"];
    k = args["sparsity"];
    eps = args["eps"];
    delta = args["delta"];

    a = (k * (exp(eps) - 1) + 2) / (k * (exp(eps) - 1) + 2 * d);
    b = (1 - a) / (d - 1);
    p = (k * (exp(eps) - 1) + 1) / (k * (exp(eps) - 1) + 2);
}

string PCKV::get_name() { return "PCKV"; }
vector<double>& PCKV::get_mean() { return mean_vec; }
double PCKV::get_comm_cost() { return comm_cost; } 

int PCKV::bern(std::mt19937 &gen, double p) {
    std::bernoulli_distribution d(p);
    bool t = d(gen);
    return t;
}

void PCKV::estimate_range(ClientData &data, int top) {
    std::random_device rd;
    std::mt19937 gen(rd());
    //std::mt19937 gen(0);

    mean_vec = vector<double>(top, 0);

    vector<int> pos(top, 0);
    vector<int> neg(top, 0);

    for (int i = 0; i < n; i++) {
        // local randomization process
        auto kv = data.get_kv_map(i);
        std::uniform_int_distribution<> uniform(0, k - 1);
        int selected_idx = uniform(gen);
        int j = ((gen() % d) + d) % d;
        double v = 0;
        if (selected_idx < kv.size()) {
            for (auto it = kv.begin(); it != kv.end(); ++it) {
                if (selected_idx == 0) {
                    j = it -> first;
                    v = it -> second;
                    break;
                }
                selected_idx --;
            }
        }

        int disc_v = 0;
        if (PCKV::bern(gen, (1 + v) / 2) == 1) 
            disc_v = 1;
        else disc_v = -1;

        int noisy_j = j, noisy_v;

        if (PCKV::bern(gen, a) == 1) {
            if (PCKV::bern(gen, p) == 1) {
                noisy_v = disc_v;
            } else noisy_v = -disc_v;
        } else { 
            noisy_j = ((gen() % d) + d) % d;
            while (noisy_j == j) 
                noisy_j = ((gen() % d) + d) % d;
            if (gen() % 2 == 1) noisy_v = 1; else noisy_v = -1;
        }

        comm_cost = sizeof(noisy_j) + sizeof(noisy_v);

        //aggregation
        if (noisy_j < top) {
            if (noisy_v == 1) {
                pos[noisy_j] += 1;
            } else neg[noisy_j] += 1;
        }
    }

    for (int j = 0; j < top; j++)  {
        int n1 = pos[j], n2 = neg[j];
        double freq = (double(n1 + n2) / n - b) / (a - b) * k;
        double sum = (n1 - n2) * (a - b) / (a * (2 * p - 1) * (n1 + n2 - n * b));
        mean_vec[j] = freq * sum;
        mean_vec[j] = clip(mean_vec[j], -1, 1);
    }
}

void PCKV::estimate(ClientData &data) {
    PCKV::estimate_range(data, d);
}


Harmony::Harmony(Json &_args) {
    args = _args;
    n = args["client"];
    d = args["dim"];
    k = args["sparsity"];
    eps = args["eps"];
    delta = args["delta"];
}

string Harmony::get_name() { return "Harmony"; }
vector<double>& Harmony::get_mean() { return mean_vec; }
double Harmony::get_comm_cost() { return comm_cost; } 

int Harmony::bern(std::mt19937 &gen, double p) {
    std::bernoulli_distribution d(p);
    bool t = d(gen);
    return t;
}

void Harmony::estimate_range(ClientData &data, int top) {
    std::random_device rd;
    std::mt19937 gen(rd());
    //std::mt19937 gen(0);

    mean_vec = vector<double>(top, 0);

    for (int i = 0; i < n; i++) {
        // local randomization process
        auto kv = data.get_kv_map(i);
        int j = ((gen() % d) + d) % d;
        double v = 0;
        if (kv.find(j) != kv.end())
            v = kv[j];

        double e_eps = exp(eps);
        double c_eps = (e_eps + 1) / (e_eps - 1);
        double p = (v * (e_eps - 1) + e_eps + 1) / (2 * (1 + e_eps));
        //double noisy_v;
        int noisy_v = Harmony::bern(gen, p);

        comm_cost = sizeof(j) + sizeof(noisy_v);

        //aggregation
        if (j < top) {
            if (noisy_v == 1) {
                mean_vec[j] += d * c_eps;
            } else mean_vec[j] += -d * c_eps;
        }
    }

    for (int j = 0; j < top; j++)  {
        mean_vec[j] /= n;
        mean_vec[j] = clip(mean_vec[j], -1, 1);
    }
}

void Harmony::estimate(ClientData &data) {
    Harmony::estimate_range(data, d);
}


/*
 * Hybrid is an old deprecated version.
 * Just ignore it.
 *
 */


Hybrid::Hybrid(Json &_args): svme_estimator(_args) {
    args = _args;
    n = args["client"];
    d = args["dim"];
    k = args["sparsity"];
    eps = args["eps"];
    delta = args["delta"];
    L = args["neighbor_dist"];
    if (L == -1) L = 2 * k;
}

string Hybrid::get_name() { return "Hybrid"; }
vector<double>& Hybrid::get_mean() { return mean_vec; }
double Hybrid::get_comm_cost() { return comm_cost; } 
double Hybrid::rr(double v, double eps, double range, std::mt19937 &gen) {
    //randomized response
    v = v / range;
    double e_eps = exp(eps);
    double c_eps = (e_eps + 1) / (e_eps - 1);
    double p = (v * (e_eps - 1) + e_eps + 1) / (2 * (1 + e_eps));
    double noisy_v = Harmony::bern(gen, p) ? c_eps : -c_eps;
    noisy_v *= range;

    return noisy_v;
}

void Hybrid::estimate_range(ClientData &data, int top) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> uniform(0, 0x7fffffff - 1);

    vector<int> seed(n);
    vector<vector<float> > T; // the randomization output
    vector<vector<float> > noise; // the randomization output

    //double exc_p = 1.0 / n / b / ;
    int bucket;
    double clip_r;
    double noise_param;

    if (L < sqrt(k)) {
        bucket = std::min(int(k * 1.0 / L / L), k);
        noise_param = L / eps;
        clip_r = k;
    } else {
        bucket = 1;
        clip_r = sqrt(k * log(n / 1)) / 2.5;
        //clip_r = sqrt(k) * 2;
        //clip_r = sqrt(k);
        noise_param = std::min(2 * clip_r, L) / eps;
    }
    int tot_clipped = 0;

    for (int i = 0; i < n; i++) {
        // local randomization process
        seed[i] = uniform(gen);
        vector<float> T_i(bucket, 0);
        auto kv = data.get_kv_map(i);

        vector<float> ns(bucket, 0);
        for (auto const &it : kv) {
            int j = it.first;
            double v = it.second;
            int hash_j = SVME::hash_bin(j, seed[i], bucket);
            double sign_j = SVME::random_sign(j, seed[i]);
            T_i[hash_j] += sign_j * v;
        }

        for (int j = 0; j < bucket; j++) {
            if (T_i[j] > clip_r || T_i[j] < -clip_r) {
                tot_clipped ++;
            }
            if (T_i[j] <= -clip_r) {
                ns[j] = clip_r - T_i[j];
            } else if (T_i[j] >= clip_r) {
                ns[j] = T_i[j] - clip_r;
            }

            T_i[j] = clip(T_i[j], -clip_r, clip_r);
            if (noise_param >= clip_r) {
                T_i[j] = Hybrid::rr(T_i[j], eps, clip_r, gen);
            } else {
                T_i[j] += SVME::laplacian_noise(gen, noise_param);
            }
            //ns[j] = lap_noise;
        }

        T.push_back(std::move(T_i));
        noise.push_back(std::move(ns));
    }

    //record communication cost
    comm_cost = sizeof(seed[0]) + sizeof(T[0][0]) * T[0].size();

    //aggregation
    mean_vec = vector<double>(top, 0);
    auto cnt_vec = vector<double>(top, 0);
    auto noise_vec = vector<double>(top, 0);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < top; j++) {
            int hash_j = SVME::hash_bin(j, seed[i], bucket);
            double sign_j = SVME::random_sign(j, seed[i]);

            //if (std::fabs(T[i][hash_j]) <= sqrt(k * bucket * log(2 * bucket / delta))) {
            cnt_vec[j] += 1;
            mean_vec[j] += T[i][hash_j] * sign_j;

            noise_vec[j] += noise[i][hash_j] * sign_j;
        }

    double max_noise = 0;
    double avg_noise = 0;

    for (int j = 0; j < top; j++) {
        //mean_vec[j] /= (sample_prob * n);
        mean_vec[j] /= cnt_vec[j];
        //mean_vec[j] /= double(trunc) / k;
        mean_vec[j] = clip(mean_vec[j], -1, 1);

        noise_vec[j] /= cnt_vec[j];
        max_noise = std::max(noise_vec[j], max_noise);
        avg_noise += noise_vec[j];
    }

    avg_noise /= top;
}


void Hybrid::estimate(ClientData &data) {
    Hybrid::estimate_range(data, d);
}


Strawman::Strawman(Json &_args) {
    args = _args;
    n = args["client"];
    d = args["dim"];
    k = args["sparsity"];
    eps = args["eps"];
    delta = args["delta"];
    L = args["neighbor_dist"];
    if (L == -1) L = 2 * k;
}

string Strawman::get_name() { return "Strawmann"; }
vector<double>& Strawman::get_mean() { return mean_vec; }
double Strawman::get_comm_cost() { return comm_cost; } 
double Strawman::rr(double v, double eps, double range, std::mt19937 &gen) {
    //randomized response
    v = v / range;
    double e_eps = exp(eps);
    double c_eps = (e_eps + 1) / (e_eps - 1);
    double p = (v * (e_eps - 1) + e_eps + 1) / (2 * (1 + e_eps));
    double noisy_v = Harmony::bern(gen, p) ? c_eps : -c_eps;
    noisy_v *= range;

    return noisy_v;
}

void Strawman::estimate_range(ClientData &data, int top) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> uniform(0, 0x7fffffff - 1);

    vector<int> seed(n);
    vector<vector<float> > T; // the randomization output
    //vector<vector<float> > noise; // the randomization output

    for (int i = 0; i < n; i++) {
        // local randomization process
        seed[i] = uniform(gen);
        auto kv = data.get_kv_map(i);

        if (L <= 2) {
            vector<float> T_i;
            for (auto it = kv.begin(); it != kv.end(); ++it) {
                int j = it -> first;
                double v = it -> second;
                v = v * SVME::random_sign(j, seed[i]);
                double noisy_v = Strawman::rr(v, eps, 1, gen);

                T_i.push_back(noisy_v);
            }

            while (T_i.size() < k) {
                T_i.push_back(Strawman::rr(0, eps, 1, gen));
            }

            T.push_back(std::move(T_i));
        } else {
            std::uniform_int_distribution<> uniform(0, k - 1);
            int selected_idx = uniform(gen);
            int j = (gen() % d + d) % d;
            double v = 0;
            if (selected_idx < kv.size()) {
                for (auto it = kv.begin(); it != kv.end(); ++it) {
                    if (selected_idx == 0) {
                        j = it -> first;
                        v = it -> second;
                        break;  
                    }
                    selected_idx --;
                }
            } 

            v = v * SVME::random_sign(j, seed[i]);
            //T_i[j] = Strawman::rr(T_i[j], eps, clip_r, gen);
            auto noisy_v = Strawman::rr(v, eps, 1, gen) * k;

            vector<float> T_i;
            T_i.push_back(noisy_v);
            T.push_back(std::move(T_i));

        }
        //noise.push_back(std::move(ns));
    }

    //record communication cost
    comm_cost = sizeof(seed[0]) + sizeof(T[0][0]) * T[0].size();

    //aggregation
    mean_vec = vector<double>(top, 0);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < top; j++) 
            for (auto it = T[i].begin(); it != T[i].end(); ++it) {
                double sign_j = SVME::random_sign(j, seed[i]);
                mean_vec[j] += (*it) * sign_j;
            }

    double max_noise = 0;

    for (int j = 0; j < top; j++) {
        //mean_vec[j] /= (sample_prob * n);
        mean_vec[j] /= n;
        //mean_vec[j] /= double(trunc) / k;
        mean_vec[j] = clip(mean_vec[j], -1, 1);
    }
}

void Strawman::estimate(ClientData &data) {
    Strawman::estimate_range(data, d);
}

