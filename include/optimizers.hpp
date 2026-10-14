#pragma once

#include "mf.hpp"
#include <string>
#include <vector>

std::vector<EpochMetric> train_sgd(MFModel& model,
                                   std::vector<Rating> train,
                                   const std::vector<Rating>& validation,
                                   const TrainConfig& config);

std::vector<EpochMetric> train_hogwild(MFModel& model,
                                       std::vector<Rating> train,
                                       const std::vector<Rating>& validation,
                                       const TrainConfig& config);

std::vector<EpochMetric> train_locked_sgd(MFModel& model,
                                          std::vector<Rating> train,
                                          const std::vector<Rating>& validation,
                                          const TrainConfig& config);
