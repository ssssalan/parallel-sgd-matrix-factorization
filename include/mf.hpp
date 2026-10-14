#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Rating {
    int user = 0;
    int item = 0;
    double value = 0.0;
};

struct Dataset {
    std::vector<Rating> ratings;
    int num_users = 0;
    int num_items = 0;
};

struct DataSplit {
    std::vector<Rating> train;
    std::vector<Rating> validation;
    std::vector<Rating> test;
};

struct MFModel {
    int num_users = 0;
    int num_items = 0;
    int rank = 32;
    double global_mean = 0.0;
    std::vector<double> user_factors;
    std::vector<double> item_factors;
    std::vector<double> user_bias;
    std::vector<double> item_bias;
};

struct TrainConfig {
    int epochs = 30;
    int threads = 1;
    int rank = 32;
    int patience = 5;
    std::uint32_t seed = 42;
    double learning_rate = 0.012;
    double decay = 0.015;
    double regularization = 0.015;
    double min_delta = 1e-5;
};

struct EpochMetric {
    int epoch = 0;
    double validation_rmse = 0.0;
    double epoch_seconds = 0.0;
    double cumulative_seconds = 0.0;
};

Dataset load_movielens_csv(const std::string& path);
DataSplit split_dataset(const Dataset& dataset, std::uint32_t seed,
                        double train_ratio = 0.8, double validation_ratio = 0.1);
MFModel initialize_model(const std::vector<Rating>& train, int num_users, int num_items,
                         int rank, std::uint32_t seed);

double predict(const MFModel& model, int user, int item);
double rmse(const MFModel& model, const std::vector<Rating>& ratings);
void write_metrics_csv(const std::string& path, const std::string& method,
                       const TrainConfig& config,
                       const std::vector<EpochMetric>& metrics,
                       double final_test_rmse);
