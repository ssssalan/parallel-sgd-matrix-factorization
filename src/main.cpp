#include "mf.hpp"
#include "optimizers.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Args {
    std::string data;
    std::string method = "hogwild";
    std::string output = "results/metrics.csv";
    TrainConfig cfg;
};

void print_help() {
    std::cout
        << "Matrix Factorization with parallel SGD baselines\n\n"
        << "Usage:\n"
        << "  mf_train --data PATH [options]\n\n"
        << "Options:\n"
        << "  --method sgd|hogwild|locked-sgd   Optimizer (default: hogwild)\n"
        << "  --threads N                       OpenMP threads (default: 1)\n"
        << "  --rank K                          Latent dimension (default: 32)\n"
        << "  --epochs N                        Maximum epochs (default: 30)\n"
        << "  --lr X                            Initial learning rate (default: 0.012)\n"
        << "  --reg X                           L2 regularization (default: 0.015)\n"
        << "  --decay X                         Learning-rate decay (default: 0.015)\n"
        << "  --patience N                      Early-stop window (default: 5)\n"
        << "  --seed N                          Random seed (default: 42)\n"
        << "  --output PATH                     Metrics CSV (default: results/metrics.csv)\n"
        << "  --help                            Show this message\n\n"
        << "Note: the unpublished A-ASGD research implementation is intentionally\n"
        << "not included in this public portfolio build.\n";
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else if (arg == "--data") a.data = need(arg);
        else if (arg == "--method") a.method = need(arg);
        else if (arg == "--threads") a.cfg.threads = std::stoi(need(arg));
        else if (arg == "--rank") a.cfg.rank = std::stoi(need(arg));
        else if (arg == "--epochs") a.cfg.epochs = std::stoi(need(arg));
        else if (arg == "--lr") a.cfg.learning_rate = std::stod(need(arg));
        else if (arg == "--reg") a.cfg.regularization = std::stod(need(arg));
        else if (arg == "--decay") a.cfg.decay = std::stod(need(arg));
        else if (arg == "--patience") a.cfg.patience = std::stoi(need(arg));
        else if (arg == "--seed") a.cfg.seed = static_cast<std::uint32_t>(std::stoul(need(arg)));
        else if (arg == "--output") a.output = need(arg);
        else throw std::runtime_error("Unknown argument: " + arg);
    }
    if (a.data.empty()) throw std::runtime_error("--data is required. Use --help for usage.");
    if (a.cfg.threads < 1 || a.cfg.rank < 1 || a.cfg.epochs < 1) {
        throw std::runtime_error("threads, rank and epochs must be positive.");
    }
    return a;
}
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        std::cout << "Loading: " << args.data << '\n';
        const Dataset dataset = load_movielens_csv(args.data);
        const DataSplit split = split_dataset(dataset, args.cfg.seed);
        std::cout << "ratings=" << dataset.ratings.size()
                  << " users=" << dataset.num_users
                  << " items=" << dataset.num_items << '\n'
                  << "train=" << split.train.size()
                  << " validation=" << split.validation.size()
                  << " test=" << split.test.size() << '\n';

        MFModel model = initialize_model(split.train, dataset.num_users, dataset.num_items,
                                         args.cfg.rank, args.cfg.seed);

        std::vector<EpochMetric> metrics;
        if (args.method == "sgd") {
            metrics = train_sgd(model, split.train, split.validation, args.cfg);
        } else if (args.method == "hogwild") {
            metrics = train_hogwild(model, split.train, split.validation, args.cfg);
        } else if (args.method == "locked-sgd") {
            metrics = train_locked_sgd(model, split.train, split.validation, args.cfg);
        } else if (args.method == "aasgd") {
            throw std::runtime_error(
                "A-ASGD is part of ongoing/unpublished research and is not included in the public build.");
        } else {
            throw std::runtime_error("Unknown method: " + args.method);
        }

        const double test_rmse = rmse(model, split.test);
        std::cout << "Final test RMSE: " << test_rmse << '\n';
        if (!std::filesystem::path(args.output).parent_path().empty()) {
            std::filesystem::create_directories(std::filesystem::path(args.output).parent_path());
        }
        write_metrics_csv(args.output, args.method, args.cfg, metrics, test_rmse);
        std::cout << "Metrics written to: " << args.output << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
