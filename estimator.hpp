#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <string>
#include <random>
#include "utils.hpp"
#include "data.hpp"
#include "json.hpp"
using Json = nlohmann::json;
using std::string;

class EstimatorBase {
  public:
    virtual string get_name() = 0;
    virtual vector<double>& get_mean() = 0;
    virtual double get_comm_cost() = 0;
    virtual void estimate_range(ClientData &data, int top) = 0;
    virtual void estimate(ClientData &data) = 0;
};

class SVME: public EstimatorBase {
  private:
    Json args;
    double comm_cost;

    int n, d, k, trunc;
    double eps, delta, L;
    int bucket;
    double sample_prob;
    double noise_param;
    vector<double> mean_vec;
  public:
    SVME(Json &_args);
    string get_name();
    vector<double>& get_mean();
    double get_comm_cost();
    void estimate(ClientData &data);
    void estimate_range(ClientData &data, int top);
    double predicted_error();

    static int sampling(int x, int seed, double p);
    static int hash_bin(int x, int seed, int bucket);
    static double random_sign(int x, int seed);
    static double laplacian_noise(std::mt19937 &gen, double noise_param);
    double rr(double v, double eps, double range, std::mt19937 &gen);
};

class GaussianMechanism: public EstimatorBase {
  private:
    Json args;
    double comm_cost;

    int n, d, k;
    double eps, delta, L;
    double noise_param;
    vector<double> mean_vec;
  public:
    GaussianMechanism(Json &_args);
    string get_name();
    vector<double>& get_mean();
    double get_comm_cost();
    void estimate(ClientData &data);
    void estimate_range(ClientData &data, int top);

    static double gaussian_noise(std::mt19937 &gen, double noise_param);
};

class PCKV: public EstimatorBase {
  private:
    Json args;
    double comm_cost;

    int n, d, k;
    double eps, delta;
    double a, b, p;
    vector<double> mean_vec;
  public:
    PCKV(Json &_args);
    string get_name();
    vector<double>& get_mean();
    double get_comm_cost();
    void estimate(ClientData &data);
    void estimate_range(ClientData &data, int top);
    static int bern(std::mt19937 &gen, double p);
};

class Harmony: public EstimatorBase {
  private:
    Json args;
    double comm_cost;

    int n, d, k;
    double eps, delta;
    vector<double> mean_vec;
  public:
    Harmony(Json &_args);
    string get_name();
    vector<double>& get_mean();
    double get_comm_cost();
    void estimate(ClientData &data);
    void estimate_range(ClientData &data, int top);
    static int bern(std::mt19937 &gen, double p);
};

class Hybrid: public EstimatorBase {
  private:
    Json args;
    double comm_cost;

    int n, d, k, trunc;
    double eps, delta, L;
    int bucket;
    double sample_prob;
    double noise_param;
    vector<double> mean_vec;
    
    SVME svme_estimator;

  public:
    Hybrid(Json &_args);
    string get_name();
    vector<double>& get_mean();
    double get_comm_cost();
    void estimate(ClientData &data);
    void estimate_range(ClientData &data, int top);
    double rr(double v, double eps, double range, std::mt19937 &gen);
};

class Strawman: public EstimatorBase {
  private:
    Json args;
    double comm_cost;

    int n, d, k, trunc;
    double eps, delta, L;
    int bucket;
    double sample_prob;
    double noise_param;
    vector<double> mean_vec;

  public:
    Strawman(Json &_args);
    string get_name();
    vector<double>& get_mean();
    double get_comm_cost();
    void estimate(ClientData &data);
    void estimate_range(ClientData &data, int top);
    double rr(double v, double eps, double range, std::mt19937 &gen);
};


#endif
