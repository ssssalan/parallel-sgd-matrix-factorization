#include "optimizers.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <omp.h>

namespace {
bool should_stop(const std::vector<EpochMetric>& metrics, const TrainConfig& cfg) {
    if (static_cast<int>(metrics.size()) <= cfg.patience) return false;
    int stagnant = 0;
    for (std::size_t i = metrics.size() - cfg.patience; i < metrics.size(); ++i) {
        if (std::abs(metrics[i - 1].validation_rmse - metrics[i].validation_rmse) < cfg.min_delta) ++stagnant;
    }
    return stagnant >= cfg.patience;
}
}

std::vector<EpochMetric> train_hogwild(MFModel& model,
                                       std::vector<Rating> train,
                                       const std::vector<Rating>& validation,
                                       const TrainConfig& cfg) {
    omp_set_num_threads(cfg.threads);
    std::mt19937 rng(cfg.seed);
    std::vector<EpochMetric> metrics;
    double cumulative = 0.0;

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        std::shuffle(train.begin(), train.end(), rng);
        const double lr = cfg.learning_rate / (1.0 + cfg.decay * epoch);
        const double start = omp_get_wtime();

        // Hogwild!: intentionally lock-free parameter updates.
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(train.size()); ++idx) {
            const auto& r = train[static_cast<std::size_t>(idx)];
            const std::size_t uoff = static_cast<std::size_t>(r.user) * model.rank;
            const std::size_t ioff = static_cast<std::size_t>(r.item) * model.rank;
            const double p = predict(model, r.user, r.item);
            if (!std::isfinite(p)) continue;
            const double err = r.value - p;

            model.user_bias[r.user] += lr * (err - cfg.regularization * model.user_bias[r.user]);
            model.item_bias[r.item] += lr * (err - cfg.regularization * model.item_bias[r.item]);
            for (int k = 0; k < model.rank; ++k) {
                const double u_old = model.user_factors[uoff + k];
                const double v_old = model.item_factors[ioff + k];
                model.user_factors[uoff + k] += lr * (err * v_old - cfg.regularization * u_old);
                model.item_factors[ioff + k] += lr * (err * u_old - cfg.regularization * v_old);
            }
        }

        const double elapsed = omp_get_wtime() - start;
        cumulative += elapsed;
        const double val_rmse = rmse(model, validation);
        metrics.push_back({epoch + 1, val_rmse, elapsed, cumulative});
        std::cout << "Epoch " << std::setw(2) << epoch + 1
                  << " | val RMSE=" << std::fixed << std::setprecision(5) << val_rmse
                  << " | train=" << std::setprecision(3) << elapsed << "s\n";
        if (should_stop(metrics, cfg)) break;
    }
    return metrics;
}
