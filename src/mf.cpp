#include "mf.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <omp.h>

Dataset load_movielens_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open dataset: " + path);
    }

    Dataset dataset;
    std::unordered_map<int, int> user_map;
    std::unordered_map<int, int> item_map;
    std::string line;

    std::getline(in, line); // MovieLens header: userId,movieId,rating,timestamp
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string user_s, item_s, rating_s;
        if (!std::getline(ss, user_s, ',')) continue;
        if (!std::getline(ss, item_s, ',')) continue;
        if (!std::getline(ss, rating_s, ',')) continue;

        try {
            const int raw_user = std::stoi(user_s);
            const int raw_item = std::stoi(item_s);
            const double value = std::stod(rating_s);

            auto [uit, uinserted] = user_map.emplace(raw_user, static_cast<int>(user_map.size()));
            auto [iit, iinserted] = item_map.emplace(raw_item, static_cast<int>(item_map.size()));
            (void)uinserted;
            (void)iinserted;
            dataset.ratings.push_back({uit->second, iit->second, value});
        } catch (const std::exception&) {
            // Ignore malformed rows in the public demo pipeline.
        }
    }

    dataset.num_users = static_cast<int>(user_map.size());
    dataset.num_items = static_cast<int>(item_map.size());
    if (dataset.ratings.empty()) {
        throw std::runtime_error("No valid ratings were parsed from: " + path);
    }
    return dataset;
}

DataSplit split_dataset(const Dataset& dataset, std::uint32_t seed,
                        double train_ratio, double validation_ratio) {
    if (train_ratio <= 0.0 || validation_ratio < 0.0 ||
        train_ratio + validation_ratio >= 1.0) {
        throw std::invalid_argument("Invalid train/validation ratios.");
    }

    std::vector<Rating> shuffled = dataset.ratings;
    std::mt19937 rng(seed);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);

    const std::size_t n_train = static_cast<std::size_t>(shuffled.size() * train_ratio);
    const std::size_t n_val = static_cast<std::size_t>(shuffled.size() * validation_ratio);

    DataSplit split;
    split.train.assign(shuffled.begin(), shuffled.begin() + static_cast<std::ptrdiff_t>(n_train));
    split.validation.assign(shuffled.begin() + static_cast<std::ptrdiff_t>(n_train),
                            shuffled.begin() + static_cast<std::ptrdiff_t>(n_train + n_val));
    split.test.assign(shuffled.begin() + static_cast<std::ptrdiff_t>(n_train + n_val), shuffled.end());
    return split;
}

MFModel initialize_model(const std::vector<Rating>& train, int num_users, int num_items,
                         int rank, std::uint32_t seed) {
    MFModel model;
    model.num_users = num_users;
    model.num_items = num_items;
    model.rank = rank;
    model.user_factors.resize(static_cast<std::size_t>(num_users) * rank);
    model.item_factors.resize(static_cast<std::size_t>(num_items) * rank);
    model.user_bias.assign(num_users, 0.0);
    model.item_bias.assign(num_items, 0.0);

    double sum = 0.0;
    for (const auto& r : train) sum += r.value;
    model.global_mean = sum / static_cast<double>(train.size());

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    for (double& x : model.user_factors) x = dist(rng);
    for (double& x : model.item_factors) x = dist(rng);
    return model;
}

double predict(const MFModel& model, int user, int item) {
    double p = model.global_mean + model.user_bias[user] + model.item_bias[item];
    const std::size_t uoff = static_cast<std::size_t>(user) * model.rank;
    const std::size_t ioff = static_cast<std::size_t>(item) * model.rank;
    for (int k = 0; k < model.rank; ++k) {
        p += model.user_factors[uoff + k] * model.item_factors[ioff + k];
    }
    return p;
}

double rmse(const MFModel& model, const std::vector<Rating>& ratings) {
    if (ratings.empty()) return 0.0;
    double squared_error = 0.0;
    #pragma omp parallel for reduction(+:squared_error)
    for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(ratings.size()); ++idx) {
        const auto& r = ratings[static_cast<std::size_t>(idx)];
        double p = predict(model, r.user, r.item);
        p = std::clamp(p, 0.5, 5.0);
        const double e = r.value - p;
        squared_error += e * e;
    }
    return std::sqrt(squared_error / static_cast<double>(ratings.size()));
}

void write_metrics_csv(const std::string& path, const std::string& method,
                       const TrainConfig& config,
                       const std::vector<EpochMetric>& metrics,
                       double final_test_rmse) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write metrics to: " + path);
    out << "method,threads,rank,epoch,validation_rmse,epoch_seconds,cumulative_seconds,final_test_rmse\n";
    out << std::fixed << std::setprecision(8);
    for (const auto& m : metrics) {
        out << method << ',' << config.threads << ',' << config.rank << ',' << m.epoch << ','
            << m.validation_rmse << ',' << m.epoch_seconds << ',' << m.cumulative_seconds << ','
            << final_test_rmse << '\n';
    }
}
