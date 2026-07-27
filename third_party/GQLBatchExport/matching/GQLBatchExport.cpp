//
// Batch candidate-space exporter for FlowSC-style GQL filtering.
//

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "FilterVertices.h"
#include "graph/graph.h"

namespace {

using Clock = std::chrono::high_resolution_clock;
constexpr size_t kLabelPairSketchBuckets = 64;

size_t label_pair_bucket(LabelID left, LabelID right) {
    uint64_t a = static_cast<uint64_t>(std::min(left, right));
    uint64_t b = static_cast<uint64_t>(std::max(left, right));
    uint64_t x = (a + 0x9e3779b97f4a7c15ULL) ^ (b + 0xbf58476d1ce4e5b9ULL + (a << 6) + (a >> 2));
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return static_cast<size_t>(x % kLabelPairSketchBuckets);
}

void release_query_heap() {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

double seconds_since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string require_arg(int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("Missing argument value for ") + argv[index]);
    }
    index += 1;
    return std::string(argv[index]);
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

void free_candidates(Graph* query_graph, ui** candidates, ui* candidates_count) {
    delete[] candidates_count;
    if (candidates != NULL) {
        for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
            delete[] candidates[i];
        }
        delete[] candidates;
    }
}

double safe_log(double value) {
    return std::log(std::max(value, 1e-12));
}

double safe_log1p(double value) {
    return std::log1p(std::max(value, 0.0));
}

double mean_or_zero(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double value : values) sum += value;
    return sum / static_cast<double>(values.size());
}

double min_or_zero(const std::vector<double>& values) {
    return values.empty() ? 0.0 : *std::min_element(values.begin(), values.end());
}

double max_or_zero(const std::vector<double>& values) {
    return values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
}

double std_or_zero(const std::vector<double>& values) {
    if (values.size() <= 1) return 0.0;
    const double center = mean_or_zero(values);
    double variance = 0.0;
    for (double value : values) {
        const double delta = value - center;
        variance += delta * delta;
    }
    variance /= static_cast<double>(values.size());
    return std::sqrt(std::max(variance, 0.0));
}

double normalized_entropy(const std::vector<double>& values) {
    if (values.size() <= 1) return 0.0;
    double sum = 0.0;
    for (double value : values) sum += std::max(value, 0.0);
    if (sum <= 0.0) return 0.0;
    double entropy = 0.0;
    for (double value : values) {
        double p = std::max(value, 0.0) / sum;
        if (p > 0.0) entropy -= p * std::log(p);
    }
    return entropy / std::log(static_cast<double>(values.size()));
}

double gini_coefficient(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    double sum = 0.0;
    double weighted_sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        const double value = std::max(values[i], 0.0);
        sum += value;
        weighted_sum += static_cast<double>(i + 1) * value;
    }
    if (sum <= 0.0) return 0.0;
    const double n = static_cast<double>(values.size());
    return (2.0 * weighted_sum) / (n * sum) - (n + 1.0) / n;
}

size_t sorted_intersection_count(const std::vector<ui>& left, const std::vector<ui>& right) {
    size_t i = 0;
    size_t j = 0;
    size_t count = 0;
    while (i < left.size() && j < right.size()) {
        if (left[i] == right[j]) {
            ++count;
            ++i;
            ++j;
        } else if (left[i] < right[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return count;
}

unsigned long long directed_candidate_pair_key(ui left, ui right) {
    return (static_cast<unsigned long long>(left) << 32) | static_cast<unsigned long long>(right);
}

double log_injection_bound(double candidate_count, double query_count) {
    if (query_count <= 0.0) return 0.0;
    if (candidate_count < query_count) return safe_log(0.0);
    double total = 0.0;
    const ui steps = static_cast<ui>(query_count);
    for (ui i = 0; i < steps; ++i) {
        total += safe_log(candidate_count - static_cast<double>(i));
    }
    return total;
}

void write_ui_array(std::ostream& out, const std::vector<ui>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << values[i];
    }
    out << ']';
}

void write_int_array(std::ostream& out, const std::vector<int>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << values[i];
    }
    out << ']';
}

void write_double_array(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << values[i];
    }
    out << ']';
}

void write_nested_ui_array(std::ostream& out, const std::vector<std::vector<ui>>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        write_ui_array(out, values[i]);
    }
    out << ']';
}

void write_nested_double_array(std::ostream& out, const std::vector<std::vector<double>>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        write_double_array(out, values[i]);
    }
    out << ']';
}

void write_pair_array(std::ostream& out, const std::vector<std::pair<int, int>>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << '[' << values[i].first << ',' << values[i].second << ']';
    }
    out << ']';
}

template <typename T>
void write_binary_value(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_binary_string(std::ostream& out, const std::string& value) {
    uint32_t length = static_cast<uint32_t>(value.size());
    write_binary_value(out, length);
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_binary_float_array(std::ostream& out, const std::vector<double>& values) {
    for (double value : values) {
        float converted = static_cast<float>(value);
        write_binary_value(out, converted);
    }
}

void write_binary_nested_float_array(std::ostream& out, const std::vector<std::vector<double>>& values) {
    for (const auto& row : values) {
        write_binary_float_array(out, row);
    }
}

struct QueryStats {
    std::vector<std::vector<ui>> candidate_nodes;
    std::vector<std::vector<ui>> components;
    std::vector<std::vector<ui>> candidate_neighbors;
    std::vector<double> candidate_node_degrees;
    std::vector<double> candidate_data_degrees;
    std::vector<std::pair<int, int>> query_edge_list;
    std::vector<std::unordered_set<unsigned long long>> query_edge_candidate_pair_sets;
    std::vector<double> query_edge_candidate_counts;
    std::vector<double> query_edge_densities;
    std::vector<double> query_edge_left_support_nonzero_frac;
    std::vector<double> query_edge_right_support_nonzero_frac;
    std::vector<double> query_edge_left_log_support_min;
    std::vector<double> query_edge_left_log_support_mean;
    std::vector<double> query_edge_left_log_support_std;
    std::vector<double> query_edge_left_log_support_max;
    std::vector<double> query_edge_right_log_support_min;
    std::vector<double> query_edge_right_log_support_mean;
    std::vector<double> query_edge_right_log_support_std;
    std::vector<double> query_edge_right_log_support_max;
    std::vector<double> tree_edge_mask;
    std::string candidate_tree_strategy = "mwst_low_density_edges";
    int tree_root = 0;
    std::vector<int> tree_parent;
    std::vector<int> tree_order;
    std::vector<int> tree_depth_by_node;
    std::vector<int> tree_child_count_by_node;
    std::vector<double> tree_leaf_mask;
    std::vector<double> tree_parent_log_density;
    double log_candidate_tree_count = 0.0;
    std::vector<std::pair<std::string, double>> tree_prior_variants;
    double tree_density_log_sum = 0.0;
    double tree_density_log_mean = 0.0;
    double tree_density_log_min = 0.0;
    double tree_density_log_max = 0.0;
    double tree_density_log_std = 0.0;
    double non_tree_density_log_mean = 0.0;
    double non_tree_density_log_min = 0.0;
    double non_tree_density_log_std = 0.0;
    double density_log_gap = 0.0;
    double tree_edge_log_count_mean = 0.0;
    double tree_edge_log_count_min = 0.0;
    double tree_edge_log_count_std = 0.0;
    double non_tree_edge_log_count_mean = 0.0;
    double non_tree_edge_log_count_min = 0.0;
    double non_tree_edge_log_count_std = 0.0;
    std::vector<double> label_pair_log_density_hash_sum;
    std::vector<double> label_pair_log_count_hash_sum;
    std::vector<double> label_pair_tree_log_density_hash_sum;
    std::vector<double> label_pair_non_tree_log_density_hash_sum;
    std::vector<double> label_pair_edge_hash_count;
    double candidate_degree_log_mean = 0.0;
    double candidate_degree_log_std = 0.0;
    double candidate_degree_log_max = 0.0;
    double candidate_degree_l2_log = 0.0;
    double candidate_edge_count = 0.0;
    int cycle_path_edge_count = 0;
    int cycle_path_budget_skipped = 0;
    double cycle_path_support_frac_mean = 0.0;
    double cycle_path_support_frac_min = 0.0;
    double cycle_path_support_frac_std = 0.0;
    double cycle_path_log_gap_mean = 0.0;
    double cycle_path_log_gap_max = 0.0;
    double cycle_path_local_edge_log_mean = 0.0;
    double cycle_path_supported_edge_log_mean = 0.0;
    double cycle_path_tree_pair_support_frac_mean = 0.0;
    double cycle_path_tree_pair_support_frac_min = 0.0;
    double cycle_path_tree_pair_support_frac_std = 0.0;
    double cycle_path_tree_pair_log_gap_mean = 0.0;
    double cycle_path_tree_pair_log_gap_max = 0.0;
    double cycle_path_tree_pair_log_mean = 0.0;
    int label_injectivity_group_count = 0;
    double label_injectivity_gap_sum = 0.0;
    double label_injectivity_gap_mean = 0.0;
    double label_injectivity_gap_max = 0.0;
    double label_injectivity_overlap_frac_mean = 0.0;
    double label_injectivity_overlap_frac_max = 0.0;
    double label_injectivity_min_union_slack = 0.0;
    double label_injectivity_product_log_sum = 0.0;
    double label_injectivity_distinct_bound_log_sum = 0.0;
    double label_injectivity_pair_collision_log1p = 0.0;
    int triangle_joint_probe_count = 0;
    int triangle_joint_budget_skipped = 0;
    int triangle_joint_capped = 0;
    double triangle_joint_exact_log_mean = 0.0;
    double triangle_joint_exact_log_min = 0.0;
    double triangle_joint_domain_to_exact_gap_mean = 0.0;
    double triangle_joint_domain_to_exact_gap_max = 0.0;
    double triangle_joint_zero_exact_frac = 0.0;
    int motif4_joint_candidate_count = 0;
    int motif4_joint_probe_count = 0;
    int motif4_joint_budget_skipped = 0;
    int motif4_joint_capped = 0;
    double motif4_joint_edge_count_mean = 0.0;
    double motif4_joint_exact_log_mean = 0.0;
    double motif4_joint_exact_log_min = 0.0;
    double motif4_joint_domain_to_exact_gap_mean = 0.0;
    double motif4_joint_domain_to_exact_gap_max = 0.0;
    double motif4_joint_zero_exact_frac = 0.0;
    int dense_clique_vertex_count = 0;
    int dense_clique_query_edge_count = 0;
    int dense_clique_budget_exhausted = 0;
    int dense_clique_count_capped = 0;
    double dense_clique_domain_log = 0.0;
    double dense_clique_exact_log_count = 0.0;
    double dense_clique_domain_to_exact_gap = 0.0;
    double dense_clique_tree_log_count = 0.0;
    double dense_clique_tree_to_exact_gap = 0.0;
    double dense_clique_corrected_tree_log = 0.0;
    int full_match_probe_found = 0;
    int full_match_probe_budget_exhausted = 0;
    int full_match_probe_count_capped = 0;
    int full_match_probe_max_depth = 0;
    double full_match_probe_search_nodes = 0.0;
    double full_match_probe_count = 0.0;
    double full_match_probe_log_count = 0.0;
    int core_outside_disabled = 0;
    std::string core_outside_policy = "cycle_repeat_top10";
    std::vector<int> core_outside_vertices;
    int core_outside_vertex_count = 0;
    int core_outside_truncated = 0;
    int core_outside_edge_count = 0;
    int core_outside_frontier_edge_count = 0;
    int core_outside_uncovered_non_tree_edge_count = 0;
    double core_exact_count = 0.0;
    double core_exact_log_count = 0.0;
    int core_exact_count_capped = 0;
    int core_exact_budget_exhausted = 0;
    double core_exact_search_nodes = 0.0;
    double core_outside_estimate_count = 0.0;
    double core_outside_estimate_log_count = 0.0;
    int core_outside_estimate_capped = 0;
    int tree_sample_trials = 0;
    int tree_sample_all_edge_success = 0;
    int tree_sample_injective_success = 0;
    int tree_sample_exact_success = 0;
    double tree_sample_all_edge_success_rate = 0.0;
    double tree_sample_injective_success_rate = 0.0;
    double tree_sample_exact_success_rate = 0.0;
    double tree_sample_log_success_gap = 0.0;
    double tree_sample_pred_log_count = 0.0;
    double tree_sample_importance_all_edge_success_rate = 0.0;
    double tree_sample_importance_injective_success_rate = 0.0;
    double tree_sample_importance_exact_success_rate = 0.0;
    double tree_sample_importance_pred_log_count = 0.0;
    double tree_sample_non_tree_edge_success_frac_mean = 0.0;
    double tree_sample_non_tree_edge_success_frac_min = 0.0;
    double tree_sample_non_tree_edge_success_frac_std = 0.0;
    int probe_fastest2_enabled = 0;
    int probe_fastest2_fallback_used = 0;
    int probe_fastest2_tree_success_threshold = 10;
    long long probe_fastest2_requested_budget = 0;
    long long probe_fastest2_budget_cap = 0;
    long long probe_fastest2_budget = 0;
    int probe_fastest2_used_samples = 0;
    int probe_fastest2_root = -1;
    double probe_fastest2_graph_estimate_count = 0.0;
    double probe_fastest2_pred_log_count = 0.0;
    double probe_fastest2_graph_sample_seconds = 0.0;
    int probe_fastest2_root_draws = 0;
    int probe_fastest2_roots_processed = 0;
    int probe_fastest2_positive_roots = 0;
    int probe_fastest2_budget_exhausted = 0;
    int probe_fastest2_dead_end_count = 0;
    int probe_fastest2_terminal_leaf_count = 0;
    int probe_fastest2_max_depth = 0;
    double probe_fastest2_root_coverage = 0.0;
    double probe_fastest2_positive_root_frac = 0.0;
    double probe_fastest2_root_estimate_max = 0.0;
    double probe_fastest2_root_estimate_std = 0.0;
    double probe_fastest2_observed_terminal_match_count = 0.0;
    double probe_fastest2_terminal_fanout_max = 0.0;
    int probe_fastest2_observed_match_capped = 0;
    int tree_dp_weighted_closure_edge_count = 0;
    int tree_dp_weighted_closure_disabled = 0;
    int tree_dp_weighted_closure_budget_skipped = 0;
    double tree_dp_weighted_closure_prob_mean = 0.0;
    double tree_dp_weighted_closure_prob_min = 0.0;
    double tree_dp_weighted_closure_prob_std = 0.0;
    double tree_dp_weighted_closure_log_gap_sum = 0.0;
    double tree_dp_weighted_closure_log_gap_mean = 0.0;
    double tree_dp_weighted_closure_log_gap_max = 0.0;
    double tree_dp_weighted_closure_pred_log_count = 0.0;
    int tree_root_positive_candidate_count = 0;
    double tree_root_positive_candidate_frac = 0.0;
    double tree_root_contrib_top1_frac = 0.0;
    double tree_root_contrib_top5_frac = 0.0;
    double tree_root_contrib_top10_frac = 0.0;
    double tree_root_contrib_entropy = 0.0;
    double tree_root_contrib_gini = 0.0;
    int factor_bp_iterations = 0;
    int factor_bp_disabled = 0;
    int factor_bp_budget_skipped = 0;
    double factor_bp_full_edge_log_count = 0.0;
    double factor_bp_tree_to_full_edge_gap = 0.0;
    int factor_bp_injective_iterations = 0;
    int factor_bp_injective_disabled = 0;
    int factor_bp_injective_budget_skipped = 0;
    double factor_bp_injective_log_count = 0.0;
    double factor_bp_tree_to_injective_gap = 0.0;
    int component_count_before_prune = 0;
    int component_count_after_prune = 0;
    int pruned_component_count = 0;
    int pruned_candidate_node_count = 0;
    int edge_fixpoint_iterations = 0;
    int edge_fixpoint_pruned_candidate_count = 0;
    int triangle_edges_checked = 0;
    int triangle_candidate_edges_removed = 0;
    int triangle_filter_skipped_by_budget = 0;
    ui num_candidates = 0;
    ui num_candidate_edges = 0;
    double timing_reindex_seconds = 0.0;
    double timing_candidate_adjacency_seconds = 0.0;
    double timing_component_prune_seconds = 0.0;
    double timing_edge_fixpoint_seconds = 0.0;
    double timing_triangle_prune_seconds = 0.0;
    double timing_degree_stats_seconds = 0.0;
    double timing_tree_setup_seconds = 0.0;
    double timing_tree_dp_seconds = 0.0;
    double timing_tree_weighted_closure_seconds = 0.0;
    double timing_tree_sampling_seconds = 0.0;
    double timing_probe_fastest2_seconds = 0.0;
    double timing_dense_clique_seconds = 0.0;
    double timing_density_summary_seconds = 0.0;
    double timing_cycle_path_seconds = 0.0;
    double timing_label_injectivity_seconds = 0.0;
    double timing_triangle_joint_seconds = 0.0;
    double timing_motif4_joint_seconds = 0.0;
};

void write_tree_sample_stats(std::ostream& out, const QueryStats& stats) {
    out << ",\"tree_sample_trials\":" << stats.tree_sample_trials
        << ",\"tree_sample_all_edge_success\":" << stats.tree_sample_all_edge_success
        << ",\"tree_sample_injective_success\":" << stats.tree_sample_injective_success
        << ",\"tree_sample_exact_success\":" << stats.tree_sample_exact_success
        << ",\"tree_sample_all_edge_success_rate\":" << stats.tree_sample_all_edge_success_rate
        << ",\"tree_sample_injective_success_rate\":" << stats.tree_sample_injective_success_rate
        << ",\"tree_sample_exact_success_rate\":" << stats.tree_sample_exact_success_rate
        << ",\"tree_sample_log_success_gap\":" << stats.tree_sample_log_success_gap
        << ",\"tree_sample_pred_log_count\":" << stats.tree_sample_pred_log_count
        << ",\"tree_sample_importance_all_edge_success_rate\":" << stats.tree_sample_importance_all_edge_success_rate
        << ",\"tree_sample_importance_injective_success_rate\":" << stats.tree_sample_importance_injective_success_rate
        << ",\"tree_sample_importance_exact_success_rate\":" << stats.tree_sample_importance_exact_success_rate
        << ",\"tree_sample_importance_pred_log_count\":" << stats.tree_sample_importance_pred_log_count
        << ",\"tree_sample_non_tree_edge_success_frac_mean\":" << stats.tree_sample_non_tree_edge_success_frac_mean
        << ",\"tree_sample_non_tree_edge_success_frac_min\":" << stats.tree_sample_non_tree_edge_success_frac_min
        << ",\"tree_sample_non_tree_edge_success_frac_std\":" << stats.tree_sample_non_tree_edge_success_frac_std;
}

void write_probe_fastest2_stats(std::ostream& out, const QueryStats& stats) {
    out << ",\"probe_fastest2_enabled\":" << stats.probe_fastest2_enabled
        << ",\"probe_fastest2_fallback_used\":" << stats.probe_fastest2_fallback_used
        << ",\"probe_fastest2_tree_success_threshold\":" << stats.probe_fastest2_tree_success_threshold
        << ",\"probe_fastest2_requested_budget\":" << stats.probe_fastest2_requested_budget
        << ",\"probe_fastest2_budget_cap\":" << stats.probe_fastest2_budget_cap
        << ",\"probe_fastest2_budget\":" << stats.probe_fastest2_budget
        << ",\"probe_fastest2_used_samples\":" << stats.probe_fastest2_used_samples
        << ",\"probe_fastest2_root\":" << stats.probe_fastest2_root
        << ",\"probe_fastest2_graph_estimate_count\":" << stats.probe_fastest2_graph_estimate_count
        << ",\"probe_fastest2_pred_log_count\":" << stats.probe_fastest2_pred_log_count
        << ",\"probe_fastest2_graph_sample_seconds\":" << stats.probe_fastest2_graph_sample_seconds
        << ",\"probe_fastest2_root_draws\":" << stats.probe_fastest2_root_draws
        << ",\"probe_fastest2_roots_processed\":" << stats.probe_fastest2_roots_processed
        << ",\"probe_fastest2_positive_roots\":" << stats.probe_fastest2_positive_roots
        << ",\"probe_fastest2_budget_exhausted\":" << stats.probe_fastest2_budget_exhausted
        << ",\"probe_fastest2_dead_end_count\":" << stats.probe_fastest2_dead_end_count
        << ",\"probe_fastest2_terminal_leaf_count\":" << stats.probe_fastest2_terminal_leaf_count
        << ",\"probe_fastest2_max_depth\":" << stats.probe_fastest2_max_depth
        << ",\"probe_fastest2_root_coverage\":" << stats.probe_fastest2_root_coverage
        << ",\"probe_fastest2_positive_root_frac\":" << stats.probe_fastest2_positive_root_frac
        << ",\"probe_fastest2_root_estimate_max\":" << stats.probe_fastest2_root_estimate_max
        << ",\"probe_fastest2_root_estimate_std\":" << stats.probe_fastest2_root_estimate_std
        << ",\"probe_fastest2_observed_terminal_match_count\":" << stats.probe_fastest2_observed_terminal_match_count
        << ",\"probe_fastest2_terminal_fanout_max\":" << stats.probe_fastest2_terminal_fanout_max
        << ",\"probe_fastest2_observed_match_capped\":" << stats.probe_fastest2_observed_match_capped
        << ",\"dense_clique_vertex_count\":" << stats.dense_clique_vertex_count
        << ",\"dense_clique_query_edge_count\":" << stats.dense_clique_query_edge_count
        << ",\"dense_clique_budget_exhausted\":" << stats.dense_clique_budget_exhausted
        << ",\"dense_clique_count_capped\":" << stats.dense_clique_count_capped
        << ",\"dense_clique_domain_log\":" << stats.dense_clique_domain_log
        << ",\"dense_clique_exact_log_count\":" << stats.dense_clique_exact_log_count
        << ",\"dense_clique_domain_to_exact_gap\":" << stats.dense_clique_domain_to_exact_gap
        << ",\"dense_clique_tree_log_count\":" << stats.dense_clique_tree_log_count
        << ",\"dense_clique_tree_to_exact_gap\":" << stats.dense_clique_tree_to_exact_gap
        << ",\"dense_clique_corrected_tree_log\":" << stats.dense_clique_corrected_tree_log;
}

void write_stage5_probe_stats(std::ostream& out, const QueryStats& stats) {
    out << "\"tree_sample_trials\":" << stats.tree_sample_trials
        << ",\"tree_sample_all_edge_success\":" << stats.tree_sample_all_edge_success
        << ",\"tree_sample_injective_success\":" << stats.tree_sample_injective_success
        << ",\"tree_sample_exact_success\":" << stats.tree_sample_exact_success
        << ",\"tree_sample_pred_log_count\":" << stats.tree_sample_pred_log_count
        << ",\"probe_fastest2_fallback_used\":" << stats.probe_fastest2_fallback_used
        << ",\"probe_fastest2_used_samples\":" << stats.probe_fastest2_used_samples
        << ",\"probe_fastest2_graph_estimate_count\":" << stats.probe_fastest2_graph_estimate_count
        << ",\"probe_fastest2_pred_log_count\":" << stats.probe_fastest2_pred_log_count
        << ",\"probe_fastest2_root_draws\":" << stats.probe_fastest2_root_draws
        << ",\"probe_fastest2_roots_processed\":" << stats.probe_fastest2_roots_processed
        << ",\"probe_fastest2_positive_roots\":" << stats.probe_fastest2_positive_roots
        << ",\"probe_fastest2_budget_exhausted\":" << stats.probe_fastest2_budget_exhausted
        << ",\"probe_fastest2_dead_end_count\":" << stats.probe_fastest2_dead_end_count
        << ",\"probe_fastest2_terminal_leaf_count\":" << stats.probe_fastest2_terminal_leaf_count
        << ",\"probe_fastest2_max_depth\":" << stats.probe_fastest2_max_depth
        << ",\"probe_fastest2_root_coverage\":" << stats.probe_fastest2_root_coverage
        << ",\"probe_fastest2_positive_root_frac\":" << stats.probe_fastest2_positive_root_frac
        << ",\"probe_fastest2_root_estimate_max\":" << stats.probe_fastest2_root_estimate_max
        << ",\"probe_fastest2_root_estimate_std\":" << stats.probe_fastest2_root_estimate_std
        << ",\"probe_fastest2_observed_terminal_match_count\":" << stats.probe_fastest2_observed_terminal_match_count
        << ",\"probe_fastest2_terminal_fanout_max\":" << stats.probe_fastest2_terminal_fanout_max
        << ",\"probe_fastest2_observed_match_capped\":" << stats.probe_fastest2_observed_match_capped
        << ",\"dense_clique_vertex_count\":" << stats.dense_clique_vertex_count
        << ",\"dense_clique_query_edge_count\":" << stats.dense_clique_query_edge_count
        << ",\"dense_clique_budget_exhausted\":" << stats.dense_clique_budget_exhausted
        << ",\"dense_clique_count_capped\":" << stats.dense_clique_count_capped
        << ",\"dense_clique_domain_log\":" << stats.dense_clique_domain_log
        << ",\"dense_clique_exact_log_count\":" << stats.dense_clique_exact_log_count
        << ",\"dense_clique_domain_to_exact_gap\":" << stats.dense_clique_domain_to_exact_gap
        << ",\"dense_clique_tree_log_count\":" << stats.dense_clique_tree_log_count
        << ",\"dense_clique_tree_to_exact_gap\":" << stats.dense_clique_tree_to_exact_gap
        << ",\"dense_clique_corrected_tree_log\":" << stats.dense_clique_corrected_tree_log;
}

void write_tree_dp_weighted_closure_stats(std::ostream& out, const QueryStats& stats) {
    out << ",\"tree_dp_weighted_closure_edge_count\":" << stats.tree_dp_weighted_closure_edge_count
        << ",\"tree_dp_weighted_closure_disabled\":" << stats.tree_dp_weighted_closure_disabled
        << ",\"tree_dp_weighted_closure_budget_skipped\":" << stats.tree_dp_weighted_closure_budget_skipped
        << ",\"tree_dp_weighted_closure_prob_mean\":" << stats.tree_dp_weighted_closure_prob_mean
        << ",\"tree_dp_weighted_closure_prob_min\":" << stats.tree_dp_weighted_closure_prob_min
        << ",\"tree_dp_weighted_closure_prob_std\":" << stats.tree_dp_weighted_closure_prob_std
        << ",\"tree_dp_weighted_closure_log_gap_sum\":" << stats.tree_dp_weighted_closure_log_gap_sum
        << ",\"tree_dp_weighted_closure_log_gap_mean\":" << stats.tree_dp_weighted_closure_log_gap_mean
        << ",\"tree_dp_weighted_closure_log_gap_max\":" << stats.tree_dp_weighted_closure_log_gap_max
        << ",\"tree_dp_weighted_closure_pred_log_count\":" << stats.tree_dp_weighted_closure_pred_log_count;
}

void write_tree_root_contribution_stats(std::ostream& out, const QueryStats& stats) {
    out << ",\"tree_root_positive_candidate_count\":" << stats.tree_root_positive_candidate_count
        << ",\"tree_root_positive_candidate_frac\":" << stats.tree_root_positive_candidate_frac
        << ",\"tree_root_contrib_top1_frac\":" << stats.tree_root_contrib_top1_frac
        << ",\"tree_root_contrib_top5_frac\":" << stats.tree_root_contrib_top5_frac
        << ",\"tree_root_contrib_top10_frac\":" << stats.tree_root_contrib_top10_frac
        << ",\"tree_root_contrib_entropy\":" << stats.tree_root_contrib_entropy
        << ",\"tree_root_contrib_gini\":" << stats.tree_root_contrib_gini;
}

void write_factor_bp_stats(std::ostream& out, const QueryStats& stats) {
    out << ",\"factor_bp_iterations\":" << stats.factor_bp_iterations
        << ",\"factor_bp_disabled\":" << stats.factor_bp_disabled
        << ",\"factor_bp_budget_skipped\":" << stats.factor_bp_budget_skipped
        << ",\"factor_bp_full_edge_log_count\":" << stats.factor_bp_full_edge_log_count
        << ",\"factor_bp_tree_to_full_edge_gap\":" << stats.factor_bp_tree_to_full_edge_gap
        << ",\"factor_bp_injective_iterations\":" << stats.factor_bp_injective_iterations
        << ",\"factor_bp_injective_disabled\":" << stats.factor_bp_injective_disabled
        << ",\"factor_bp_injective_budget_skipped\":" << stats.factor_bp_injective_budget_skipped
        << ",\"factor_bp_injective_log_count\":" << stats.factor_bp_injective_log_count
        << ",\"factor_bp_tree_to_injective_gap\":" << stats.factor_bp_tree_to_injective_gap;
}

std::vector<int> orient_tree_order(
    const ui qn,
    const ui root,
    const std::vector<std::vector<ui>>& tree_adjacency,
    std::vector<int>& parent
) {
    parent.assign(qn, -1);
    std::vector<int> order;
    std::vector<int> visited(qn, 0);
    std::queue<ui> queue;
    visited[root] = 1;
    queue.push(root);
    while (!queue.empty()) {
        ui u = queue.front();
        queue.pop();
        order.push_back(static_cast<int>(u));
        for (ui v : tree_adjacency[u]) {
            if (visited[v]) continue;
            visited[v] = 1;
            parent[v] = static_cast<int>(u);
            queue.push(v);
        }
    }
    return order;
}

double compute_tree_log_count(
    const std::vector<std::vector<ui>>& candidate_nodes,
    const std::vector<std::unordered_set<ui>>& candidate_adjacency,
    const std::vector<int>& tree_parent,
    const std::vector<int>& tree_order,
    const ui root
) {
    const ui qn = static_cast<ui>(candidate_nodes.size());
    std::vector<std::vector<double>> dp(qn);
    for (int order_idx = static_cast<int>(tree_order.size()) - 1; order_idx >= 0; --order_idx) {
        ui u = static_cast<ui>(tree_order[order_idx]);
        dp[u].assign(candidate_nodes[u].size(), 1.0);
        for (ui v = 0; v < qn; ++v) {
            if (tree_parent[v] != static_cast<int>(u)) continue;
            std::unordered_map<ui, double> child_dp_by_node;
            child_dp_by_node.reserve(candidate_nodes[v].size() * 2 + 1);
            for (ui j = 0; j < candidate_nodes[v].size(); ++j) {
                child_dp_by_node[candidate_nodes[v][j]] += dp[v][j];
            }
            for (ui i = 0; i < candidate_nodes[u].size(); ++i) {
                double child_sum = 0.0;
                ui left = candidate_nodes[u][i];
                for (ui right : candidate_adjacency[left]) {
                    if (left == right) continue;
                    auto found = child_dp_by_node.find(right);
                    if (found != child_dp_by_node.end()) {
                        child_sum += found->second;
                    }
                }
                dp[u][i] *= child_sum;
            }
        }
    }
    double total_trees = 0.0;
    if (root < dp.size()) {
        for (double value : dp[root]) total_trees += value;
    }
    return total_trees > 0.0 ? std::log(total_trees) : 0.0;
}

size_t sample_weighted_index(
    const std::vector<double>& weights,
    std::mt19937_64& rng
) {
    double total = 0.0;
    for (double weight : weights) {
        if (std::isfinite(weight) && weight > 0.0) total += weight;
    }
    if (total <= 0.0) return weights.size();
    std::uniform_real_distribution<double> dist(0.0, total);
    double draw = dist(rng);
    double cumulative = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        double weight = weights[i];
        if (!std::isfinite(weight) || weight <= 0.0) continue;
        cumulative += weight;
        if (draw <= cumulative) return i;
    }
    return weights.empty() ? weights.size() : weights.size() - 1;
}

void populate_tree_sample_success_stats(
    QueryStats& stats,
    const std::vector<std::vector<double>>& tree_dp,
    const std::vector<std::unordered_set<ui>>& candidate_adjacency,
    int sample_count,
    unsigned int seed,
    const std::string& sample_strategy
) {
    if (sample_count <= 0 || stats.candidate_nodes.empty()) return;
    const ui qn = static_cast<ui>(stats.candidate_nodes.size());
    const ui root = stats.tree_root >= 0 ? static_cast<ui>(stats.tree_root) : 0;
    if (root >= qn || root >= tree_dp.size() || tree_dp[root].empty()) return;

    std::vector<std::vector<ui>> children(qn);
    for (ui v = 0; v < qn; ++v) {
        int parent = stats.tree_parent[v];
        if (parent >= 0 && static_cast<ui>(parent) < qn) {
            children[static_cast<ui>(parent)].push_back(v);
        }
    }

    std::mt19937_64 rng(static_cast<uint64_t>(seed));
    std::vector<double> root_weights = tree_dp[root];
    double root_weight_sum = 0.0;
    std::vector<size_t> positive_root_indices;
    positive_root_indices.reserve(root_weights.size());
    for (size_t i = 0; i < root_weights.size(); ++i) {
        if (std::isfinite(root_weights[i]) && root_weights[i] > 0.0) {
            root_weight_sum += root_weights[i];
            positive_root_indices.push_back(i);
        }
    }
    if (root_weight_sum <= 0.0 || positive_root_indices.empty()) return;

    struct TransitionDistribution {
        std::vector<double> weights;
        std::vector<ui> candidate_values;
    };
    std::vector<std::unordered_map<ui, size_t>> candidate_indices(qn);
    for (ui u = 0; u < qn; ++u) {
        if (children[u].empty()) continue;
        auto& indices = candidate_indices[u];
        indices.reserve(stats.candidate_nodes[u].size() * 2 + 1);
        for (size_t i = 0; i < stats.candidate_nodes[u].size(); ++i) {
            indices.emplace(stats.candidate_nodes[u][i], i);
        }
    }
    std::vector<std::vector<TransitionDistribution>> transitions_by_child(qn);
    for (ui u = 0; u < qn; ++u) {
        if (children[u].empty()) continue;
        for (ui v : children[u]) {
            std::unordered_map<ui, std::vector<ui>> child_positions;
            child_positions.reserve(stats.candidate_nodes[v].size() * 2 + 1);
            for (ui j = 0; j < stats.candidate_nodes[v].size(); ++j) {
                child_positions[stats.candidate_nodes[v][j]].push_back(j);
            }

            auto& transitions = transitions_by_child[v];
            transitions.resize(stats.candidate_nodes[u].size());
            for (size_t parent_index = 0;
                 parent_index < stats.candidate_nodes[u].size();
                 ++parent_index) {
                const ui parent_candidate =
                    stats.candidate_nodes[u][parent_index];
                if (parent_candidate >= candidate_adjacency.size()) continue;

                std::vector<std::pair<ui, double>> ordered_candidates;
                for (ui neighbor : candidate_adjacency[parent_candidate]) {
                    const auto position_it = child_positions.find(neighbor);
                    if (position_it == child_positions.end()) continue;
                    for (ui child_index : position_it->second) {
                        const double weight =
                            (v < tree_dp.size()
                             && child_index < tree_dp[v].size())
                            ? tree_dp[v][child_index]
                            : 0.0;
                        if (!std::isfinite(weight) || weight <= 0.0) {
                            continue;
                        }
                        ordered_candidates.emplace_back(
                            child_index,
                            weight
                        );
                    }
                }
                std::sort(
                    ordered_candidates.begin(),
                    ordered_candidates.end(),
                    [](const std::pair<ui, double>& left,
                       const std::pair<ui, double>& right) {
                        return left.first < right.first;
                    }
                );

                auto& transition = transitions[parent_index];
                transition.weights.reserve(ordered_candidates.size());
                transition.candidate_values.reserve(
                    ordered_candidates.size()
                );
                for (const auto& item : ordered_candidates) {
                    transition.weights.push_back(item.second);
                    transition.candidate_values.push_back(
                        stats.candidate_nodes[v][item.first]
                    );
                }
            }
        }
    }

    std::vector<double> non_tree_edge_success_counts(stats.query_edge_list.size(), 0.0);
    int non_tree_edge_count = 0;
    double importance_all_edge_success_sum = 0.0;
    double importance_injective_success_sum = 0.0;
    double importance_exact_success_sum = 0.0;
    for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size(); ++edge_idx) {
        if (edge_idx >= stats.tree_edge_mask.size() || stats.tree_edge_mask[edge_idx] <= 0.0) {
            ++non_tree_edge_count;
        }
    }

    for (int trial = 0; trial < sample_count; ++trial) {
        std::vector<ui> assignment(qn, std::numeric_limits<ui>::max());
        size_t root_choice = root_weights.size();
        double importance_weight = 1.0;
        if (sample_strategy == "root_uniform") {
            root_choice = positive_root_indices[static_cast<size_t>(trial) % positive_root_indices.size()];
            const double root_probability = root_weights[root_choice] / root_weight_sum;
            const double proposal_probability = 1.0 / static_cast<double>(positive_root_indices.size());
            importance_weight = root_probability / proposal_probability;
        }
        else {
            root_choice = sample_weighted_index(root_weights, rng);
        }
        if (root_choice >= stats.candidate_nodes[root].size()) continue;
        assignment[root] = stats.candidate_nodes[root][root_choice];

        bool tree_sample_valid = true;
        for (int node_id : stats.tree_order) {
            ui u = static_cast<ui>(node_id);
            if (u >= qn || assignment[u] == std::numeric_limits<ui>::max()) {
                tree_sample_valid = false;
                break;
            }
            for (ui v : children[u]) {
                const auto parent_index =
                    candidate_indices[u].find(assignment[u]);
                if (parent_index == candidate_indices[u].end()
                    || parent_index->second >= transitions_by_child[v].size()) {
                    tree_sample_valid = false;
                    break;
                }
                const auto& transition =
                    transitions_by_child[v][parent_index->second];
                const size_t choice =
                    sample_weighted_index(transition.weights, rng);
                if (choice >= transition.candidate_values.size()) {
                    tree_sample_valid = false;
                    break;
                }
                assignment[v] = transition.candidate_values[choice];
            }
            if (!tree_sample_valid) break;
        }
        if (!tree_sample_valid) continue;

        stats.tree_sample_trials += 1;
        bool all_edges_ok = true;
        int satisfied_non_tree_edges = 0;
        for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size(); ++edge_idx) {
            const auto& edge = stats.query_edge_list[edge_idx];
            ui left = assignment[static_cast<ui>(edge.first)];
            ui right = assignment[static_cast<ui>(edge.second)];
            bool edge_ok = edge_idx < stats.query_edge_candidate_pair_sets.size()
                && stats.query_edge_candidate_pair_sets[edge_idx].find(directed_candidate_pair_key(left, right)) != stats.query_edge_candidate_pair_sets[edge_idx].end();
            if (!edge_ok) all_edges_ok = false;
            if (edge_idx >= stats.tree_edge_mask.size() || stats.tree_edge_mask[edge_idx] <= 0.0) {
                if (edge_ok) {
                    satisfied_non_tree_edges += 1;
                    non_tree_edge_success_counts[edge_idx] += 1.0;
                }
            }
        }

        std::unordered_set<ui> used;
        used.reserve(qn * 2 + 1);
        bool injective_ok = true;
        for (ui value : assignment) {
            if (value == std::numeric_limits<ui>::max() || used.find(value) != used.end()) {
                injective_ok = false;
                break;
            }
            used.insert(value);
        }
        if (all_edges_ok) stats.tree_sample_all_edge_success += 1;
        if (injective_ok) stats.tree_sample_injective_success += 1;
        if (all_edges_ok && injective_ok) stats.tree_sample_exact_success += 1;
        if (all_edges_ok) importance_all_edge_success_sum += importance_weight;
        if (injective_ok) importance_injective_success_sum += importance_weight;
        if (all_edges_ok && injective_ok) importance_exact_success_sum += importance_weight;
    }

    if (stats.tree_sample_trials <= 0) return;
    auto smooth_rate = [&](int success) -> double {
        return (static_cast<double>(success) + 0.5) / (static_cast<double>(stats.tree_sample_trials) + 1.0);
    };
    stats.tree_sample_all_edge_success_rate = smooth_rate(stats.tree_sample_all_edge_success);
    stats.tree_sample_injective_success_rate = smooth_rate(stats.tree_sample_injective_success);
    stats.tree_sample_exact_success_rate = smooth_rate(stats.tree_sample_exact_success);
    stats.tree_sample_log_success_gap = -safe_log(stats.tree_sample_exact_success_rate);
    stats.tree_sample_pred_log_count = stats.log_candidate_tree_count + safe_log(stats.tree_sample_exact_success_rate);
    auto smooth_importance_rate = [&](double weighted_success_sum) -> double {
        double rate = (weighted_success_sum + 0.5) / (static_cast<double>(stats.tree_sample_trials) + 1.0);
        return std::min(1.0, std::max(rate, 1e-12));
    };
    stats.tree_sample_importance_all_edge_success_rate = smooth_importance_rate(importance_all_edge_success_sum);
    stats.tree_sample_importance_injective_success_rate = smooth_importance_rate(importance_injective_success_sum);
    stats.tree_sample_importance_exact_success_rate = smooth_importance_rate(importance_exact_success_sum);
    stats.tree_sample_importance_pred_log_count =
        stats.log_candidate_tree_count + safe_log(stats.tree_sample_importance_exact_success_rate);

    if (non_tree_edge_count > 0) {
        std::vector<double> fractions;
        fractions.reserve(non_tree_edge_count);
        for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size(); ++edge_idx) {
            if (edge_idx < stats.tree_edge_mask.size() && stats.tree_edge_mask[edge_idx] > 0.0) continue;
            fractions.push_back(non_tree_edge_success_counts[edge_idx] / static_cast<double>(stats.tree_sample_trials));
        }
        stats.tree_sample_non_tree_edge_success_frac_mean = mean_or_zero(fractions);
        stats.tree_sample_non_tree_edge_success_frac_min = min_or_zero(fractions);
        stats.tree_sample_non_tree_edge_success_frac_std = std_or_zero(fractions);
    }
}

void populate_probe_fastest2_stats(
    const Graph* query_graph,
    QueryStats& stats,
    long long budget_cap,
    int ub_initial,
    int success_threshold,
    double strata_ratio,
    unsigned int seed
) {
    stats.probe_fastest2_enabled = budget_cap > 0 ? 1 : 0;
    stats.probe_fastest2_tree_success_threshold = success_threshold;
    stats.probe_fastest2_budget_cap = budget_cap;
    stats.probe_fastest2_pred_log_count = stats.tree_sample_pred_log_count;
    if (budget_cap <= 0 || stats.candidate_nodes.empty() || stats.tree_sample_trials <= 0) return;

    const auto sample_start = Clock::now();
    if (stats.tree_sample_exact_success > success_threshold) {
        stats.probe_fastest2_graph_sample_seconds = seconds_since(sample_start);
        return;
    }

    const ui qn = query_graph->getVerticesCount();
    if (qn == 0) {
        stats.probe_fastest2_graph_sample_seconds = seconds_since(sample_start);
        return;
    }
    stats.probe_fastest2_fallback_used = 1;
    stats.probe_fastest2_requested_budget = static_cast<long long>(
        std::ceil(static_cast<double>(ub_initial) * static_cast<double>(qn)
            / std::sqrt(static_cast<double>(stats.tree_sample_exact_success) + 1.0))
    );
    stats.probe_fastest2_budget = std::max<long long>(
        0,
        std::min(stats.probe_fastest2_requested_budget, budget_cap)
    );
    if (stats.probe_fastest2_budget <= 0) {
        stats.probe_fastest2_graph_sample_seconds = seconds_since(sample_start);
        return;
    }

    ui root = 0;
    for (ui u = 1; u < qn; ++u) {
        if (stats.candidate_nodes[u].size() < stats.candidate_nodes[root].size()) root = u;
    }
    stats.probe_fastest2_root = static_cast<int>(root);
    if (stats.candidate_nodes[root].empty()) {
        stats.probe_fastest2_graph_sample_seconds = seconds_since(sample_start);
        return;
    }
    if (qn == 1) {
        stats.probe_fastest2_graph_estimate_count = static_cast<double>(stats.candidate_nodes[root].size());
        stats.probe_fastest2_pred_log_count = safe_log(stats.probe_fastest2_graph_estimate_count);
        stats.probe_fastest2_used_samples = 1;
        stats.probe_fastest2_graph_sample_seconds = seconds_since(sample_start);
        return;
    }

    auto query_edge_key = [](int a, int b) -> unsigned long long {
        int x = std::min(a, b);
        int y = std::max(a, b);
        return (static_cast<unsigned long long>(static_cast<unsigned int>(x)) << 32)
            | static_cast<unsigned int>(y);
    };
    std::unordered_map<unsigned long long, size_t> edge_index_by_key;
    edge_index_by_key.reserve(stats.query_edge_list.size() * 2 + 1);
    for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size(); ++edge_idx) {
        edge_index_by_key[query_edge_key(stats.query_edge_list[edge_idx].first, stats.query_edge_list[edge_idx].second)] = edge_idx;
    }

    std::vector<std::unordered_set<ui>> candidate_sets(qn);
    for (ui u = 0; u < qn; ++u) {
        candidate_sets[u].reserve(stats.candidate_nodes[u].size() * 2 + 1);
        for (ui candidate : stats.candidate_nodes[u]) {
            candidate_sets[u].insert(candidate);
        }
    }

    std::mt19937_64 rng(static_cast<uint64_t>(seed));
    std::vector<ui> assignment(qn, std::numeric_limits<ui>::max());
    std::vector<int> assigned(qn, 0);

    auto candidate_is_used = [&](ui candidate) -> bool {
        for (ui u = 0; u < qn; ++u) {
            if (assigned[u] && assignment[u] == candidate) return true;
        }
        return false;
    };

    auto relation_has_edge = [&](int left_query, ui left_candidate, int right_query, ui right_candidate) -> bool {
        auto found = edge_index_by_key.find(query_edge_key(left_query, right_query));
        if (found == edge_index_by_key.end() || found->second >= stats.query_edge_candidate_pair_sets.size()) {
            return false;
        }
        const auto& relation = stats.query_edge_candidate_pair_sets[found->second];
        return relation.find(directed_candidate_pair_key(left_candidate, right_candidate)) != relation.end();
    };

    auto intersect_sorted = [](const std::vector<ui>& left, const std::vector<ui>& right) -> std::vector<ui> {
        std::vector<ui> out;
        out.reserve(std::min(left.size(), right.size()));
        size_t i = 0;
        size_t j = 0;
        while (i < left.size() && j < right.size()) {
            if (left[i] == right[j]) {
                out.push_back(left[i]);
                ++i;
                ++j;
            }
            else if (left[i] < right[j]) {
                ++i;
            }
            else {
                ++j;
            }
        }
        return out;
    };

    std::function<std::vector<ui>(int)> build_extendable_candidates = [&](int query_vertex) -> std::vector<ui> {
        std::vector<std::vector<ui>> filtered_lists;
        ui neighbor_count = 0;
        const VertexID* neighbors = query_graph->getVertexNeighbors(static_cast<ui>(query_vertex), neighbor_count);
        for (ui ni = 0; ni < neighbor_count; ++ni) {
            int other_query = static_cast<int>(neighbors[ni]);
            if (!assigned[static_cast<ui>(other_query)]) continue;
            ui other_candidate = assignment[static_cast<ui>(other_query)];
            if (other_candidate >= stats.candidate_neighbors.size()) continue;
            std::vector<ui> filtered;
            const auto& neighbor_candidates = stats.candidate_neighbors[other_candidate];
            filtered.reserve(neighbor_candidates.size());
            for (ui candidate : neighbor_candidates) {
                if (candidate_sets[static_cast<ui>(query_vertex)].find(candidate)
                    == candidate_sets[static_cast<ui>(query_vertex)].end()) {
                    continue;
                }
                if (!relation_has_edge(other_query, other_candidate, query_vertex, candidate)) continue;
                filtered.push_back(candidate);
            }
            filtered_lists.push_back(std::move(filtered));
        }
        if (filtered_lists.empty()) {
            return stats.candidate_nodes[static_cast<ui>(query_vertex)];
        }
        std::sort(filtered_lists.begin(), filtered_lists.end(), [](const std::vector<ui>& left, const std::vector<ui>& right) {
            return left.size() < right.size();
        });
        std::vector<ui> local = std::move(filtered_lists[0]);
        for (size_t idx = 1; idx < filtered_lists.size(); ++idx) {
            local = intersect_sorted(local, filtered_lists[idx]);
            if (local.empty()) break;
        }
        return local;
    };

    auto choose_extendable_vertex = [&]() -> int {
        int selected = -1;
        int best_assigned_neighbors = -1;
        size_t best_candidate_count = std::numeric_limits<size_t>::max();
        int best_degree = -1;
        for (ui u = 0; u < qn; ++u) {
            if (assigned[u]) continue;
            ui neighbor_count = 0;
            const VertexID* neighbors = query_graph->getVertexNeighbors(u, neighbor_count);
            int assigned_neighbors = 0;
            for (ui ni = 0; ni < neighbor_count; ++ni) {
                if (assigned[neighbors[ni]]) assigned_neighbors += 1;
            }
            const size_t candidate_count = stats.candidate_nodes[u].size();
            const int degree = static_cast<int>(neighbor_count);
            if (assigned_neighbors > best_assigned_neighbors
                || (assigned_neighbors == best_assigned_neighbors && candidate_count < best_candidate_count)
                || (assigned_neighbors == best_assigned_neighbors && candidate_count == best_candidate_count && degree > best_degree)) {
                selected = static_cast<int>(u);
                best_assigned_neighbors = assigned_neighbors;
                best_candidate_count = candidate_count;
                best_degree = degree;
            }
        }
        return selected;
    };

    const int min_cand = (stats.num_candidates > 10000 || qn > 20) ? 2 : 4;
    std::function<std::pair<double, int>(int, int, double)> stratified =
        [&](int assigned_count, int budget, double weight) -> std::pair<double, int> {
            stats.probe_fastest2_max_depth = std::max(
                stats.probe_fastest2_max_depth,
                assigned_count
            );
            if (budget <= 0) return {0.0, 0};
            int query_vertex = choose_extendable_vertex();
            if (query_vertex < 0) return {weight, 1};

            std::vector<ui> local_candidates = build_extendable_candidates(query_vertex);
            size_t write_pos = 0;
            for (size_t i = 0; i < local_candidates.size(); ++i) {
                if (!candidate_is_used(local_candidates[i])) {
                    local_candidates[write_pos++] = local_candidates[i];
                }
            }
            local_candidates.resize(write_pos);

            if (local_candidates.empty()) {
                stats.probe_fastest2_dead_end_count += 1;
                return {0.0, 1};
            }
            if (assigned_count == static_cast<int>(qn) - 1) {
                const double terminal_fanout =
                    static_cast<double>(local_candidates.size());
                stats.probe_fastest2_terminal_leaf_count += 1;
                stats.probe_fastest2_observed_terminal_match_count +=
                    terminal_fanout;
                stats.probe_fastest2_terminal_fanout_max = std::max(
                    stats.probe_fastest2_terminal_fanout_max,
                    terminal_fanout
                );
                if (stats.probe_fastest2_observed_terminal_match_count
                    >= 1000.0) {
                    stats.probe_fastest2_observed_match_capped = 1;
                }
                return {weight * static_cast<double>(local_candidates.size()), 1};
            }

            const int sample_space_size = static_cast<int>(local_candidates.size());
            int num_strata = static_cast<int>(std::ceil(static_cast<double>(sample_space_size) * strata_ratio));
            num_strata = std::min(std::max(num_strata, min_cand), budget);
            num_strata = std::min(num_strata, sample_space_size);
            if (num_strata <= 0) return {0.0, 0};

            int used = 0;
            int strata_seen = 0;
            double estimate_sum = 0.0;
            while (used < budget && !local_candidates.empty() && strata_seen < num_strata) {
                std::uniform_int_distribution<size_t> dist(0, local_candidates.size() - 1);
                size_t idx = dist(rng);
                ui candidate = local_candidates[idx];
                assigned[static_cast<ui>(query_vertex)] = 1;
                assignment[static_cast<ui>(query_vertex)] = candidate;
                int remaining_strata = std::max(num_strata - strata_seen, 1);
                int next_budget = (budget - used) / remaining_strata;
                if (next_budget <= 0) next_budget = budget - used;
                auto child = stratified(
                    assigned_count + 1,
                    next_budget,
                    weight * static_cast<double>(sample_space_size)
                );
                estimate_sum += child.first;
                used += child.second;
                assignment[static_cast<ui>(query_vertex)] = std::numeric_limits<ui>::max();
                assigned[static_cast<ui>(query_vertex)] = 0;
                local_candidates[idx] = local_candidates.back();
                local_candidates.pop_back();
                strata_seen += 1;
            }
            if (strata_seen <= 0) return {0.0, used};
            return {estimate_sum / static_cast<double>(strata_seen), used};
        };

    std::vector<ui> root_candidates = stats.candidate_nodes[root];
    std::shuffle(root_candidates.begin(), root_candidates.end(), rng);
    const int root_draws = static_cast<int>(
        std::min<long long>(static_cast<long long>(root_candidates.size()), stats.probe_fastest2_budget)
    );
    stats.probe_fastest2_root_draws = root_draws;
    double estimate_sum = 0.0;
    int used = 0;
    std::vector<double> root_estimates;
    root_estimates.reserve(static_cast<size_t>(std::max(root_draws, 0)));
    for (int i = 0; i < root_draws && used < stats.probe_fastest2_budget; ++i) {
        std::fill(assigned.begin(), assigned.end(), 0);
        std::fill(assignment.begin(), assignment.end(), std::numeric_limits<ui>::max());
        assignment[root] = root_candidates[static_cast<size_t>(i)];
        assigned[root] = 1;
        const int remaining_roots = std::max(root_draws - i, 1);
        int next_budget = static_cast<int>((stats.probe_fastest2_budget - used) / remaining_roots);
        if (next_budget <= 0) next_budget = static_cast<int>(stats.probe_fastest2_budget - used);
        auto result = stratified(1, next_budget, static_cast<double>(stats.candidate_nodes[root].size()));
        estimate_sum += result.first;
        used += result.second;
        root_estimates.push_back(result.first);
    }
    stats.probe_fastest2_used_samples = used;
    stats.probe_fastest2_roots_processed =
        static_cast<int>(root_estimates.size());
    for (double estimate : root_estimates) {
        if (estimate > 0.0) stats.probe_fastest2_positive_roots += 1;
    }
    stats.probe_fastest2_budget_exhausted =
        used >= stats.probe_fastest2_budget ? 1 : 0;
    stats.probe_fastest2_root_coverage = root_draws > 0
        ? static_cast<double>(stats.probe_fastest2_roots_processed)
            / static_cast<double>(root_draws)
        : 0.0;
    stats.probe_fastest2_positive_root_frac =
        stats.probe_fastest2_roots_processed > 0
        ? static_cast<double>(stats.probe_fastest2_positive_roots)
            / static_cast<double>(stats.probe_fastest2_roots_processed)
        : 0.0;
    stats.probe_fastest2_root_estimate_max =
        max_or_zero(root_estimates);
    stats.probe_fastest2_root_estimate_std =
        std_or_zero(root_estimates);
    stats.probe_fastest2_graph_estimate_count = root_draws > 0
        ? estimate_sum / static_cast<double>(root_draws)
        : 0.0;
    stats.probe_fastest2_pred_log_count = safe_log(stats.probe_fastest2_graph_estimate_count);
    stats.probe_fastest2_graph_sample_seconds = seconds_since(sample_start);
}

std::vector<int> build_tree_path_between(
    int source,
    int target,
    const std::vector<int>& tree_parent
);

void populate_tree_dp_weighted_closure_stats(
    QueryStats& stats,
    const std::vector<std::unordered_set<ui>>& candidate_adjacency,
    long long per_edge_budget
) {
    if (stats.candidate_nodes.empty() || stats.query_edge_list.empty()) return;
    const ui qn = static_cast<ui>(stats.candidate_nodes.size());
    const double large_count = 1e300;
    auto cap_add = [&](double left, double right) -> double {
        if (left >= large_count || right >= large_count) return large_count;
        double value = left + right;
        return std::isfinite(value) ? std::min(value, large_count) : large_count;
    };
    auto cap_mul = [&](double left, double right) -> double {
        if (left <= 0.0 || right <= 0.0) return 0.0;
        if (left > large_count / right) return large_count;
        double value = left * right;
        return std::isfinite(value) ? std::min(value, large_count) : large_count;
    };

    std::vector<std::vector<ui>> tree_neighbors(qn);
    for (ui v = 0; v < qn; ++v) {
        int parent = v < stats.tree_parent.size() ? stats.tree_parent[v] : -1;
        if (parent >= 0 && static_cast<ui>(parent) < qn) {
            tree_neighbors[v].push_back(static_cast<ui>(parent));
            tree_neighbors[static_cast<ui>(parent)].push_back(v);
        }
    }

    std::vector<std::unordered_map<ui, size_t>> candidate_pos(qn);
    for (ui u = 0; u < qn; ++u) {
        candidate_pos[u].reserve(stats.candidate_nodes[u].size() * 2 + 1);
        for (size_t i = 0; i < stats.candidate_nodes[u].size(); ++i) {
            candidate_pos[u][stats.candidate_nodes[u][i]] = i;
        }
    }

    std::unordered_map<unsigned long long, std::vector<double>> message_cache;
    auto directed_query_edge_key = [](ui from, ui to) -> unsigned long long {
        return (static_cast<unsigned long long>(from) << 32) | static_cast<unsigned long long>(to);
    };

    std::function<std::vector<double>(ui, ui)> compute_message = [&](ui from, ui to) -> std::vector<double> {
        const unsigned long long key = directed_query_edge_key(from, to);
        auto cached = message_cache.find(key);
        if (cached != message_cache.end()) return cached->second;

        std::vector<double> from_weight(stats.candidate_nodes[from].size(), 1.0);
        for (ui neighbor : tree_neighbors[from]) {
            if (neighbor == to) continue;
            std::vector<double> incoming = compute_message(neighbor, from);
            for (size_t i = 0; i < from_weight.size() && i < incoming.size(); ++i) {
                from_weight[i] = cap_mul(from_weight[i], incoming[i]);
            }
        }

        std::vector<double> out(stats.candidate_nodes[to].size(), 0.0);
        const auto& to_pos = candidate_pos[to];
        for (size_t i = 0; i < stats.candidate_nodes[from].size(); ++i) {
            double weight = from_weight[i];
            if (weight <= 0.0) continue;
            ui left = stats.candidate_nodes[from][i];
            if (left >= candidate_adjacency.size()) continue;
            for (ui right : candidate_adjacency[left]) {
                if (left == right) continue;
                auto found = to_pos.find(right);
                if (found == to_pos.end()) continue;
                out[found->second] = cap_add(out[found->second], weight);
            }
        }
        message_cache[key] = out;
        return out;
    };

    auto external_weights_on_path = [&](const std::vector<int>& path) {
        std::vector<std::vector<double>> external;
        external.reserve(path.size());
        for (size_t pos = 0; pos < path.size(); ++pos) {
            ui u = static_cast<ui>(path[pos]);
            std::vector<double> weights(stats.candidate_nodes[u].size(), 1.0);
            int prev = pos > 0 ? path[pos - 1] : -1;
            int next = pos + 1 < path.size() ? path[pos + 1] : -1;
            for (ui neighbor : tree_neighbors[u]) {
                if (static_cast<int>(neighbor) == prev || static_cast<int>(neighbor) == next) continue;
                std::vector<double> incoming = compute_message(neighbor, u);
                for (size_t i = 0; i < weights.size() && i < incoming.size(); ++i) {
                    weights[i] = cap_mul(weights[i], incoming[i]);
                }
            }
            external.push_back(std::move(weights));
        }
        return external;
    };

    auto propagate_layer = [&](
        const std::vector<double>& layer,
        ui from,
        ui to,
        const std::vector<double>& to_external,
        long long& op_count,
        bool& skipped
    ) -> std::vector<double> {
        std::vector<double> next_layer(stats.candidate_nodes[to].size(), 0.0);
        const auto& to_pos = candidate_pos[to];
        for (size_t i = 0; i < layer.size(); ++i) {
            double weight = layer[i];
            if (weight <= 0.0) continue;
            ui left = stats.candidate_nodes[from][i];
            if (left >= candidate_adjacency.size()) continue;
            for (ui right : candidate_adjacency[left]) {
                op_count += 1;
                if (op_count > per_edge_budget) {
                    skipped = true;
                    return next_layer;
                }
                if (left == right) continue;
                auto found = to_pos.find(right);
                if (found == to_pos.end()) continue;
                double combined = cap_mul(weight, to_external[found->second]);
                next_layer[found->second] = cap_add(next_layer[found->second], combined);
            }
        }
        return next_layer;
    };

    std::vector<double> probs;
    std::vector<double> log_gaps;
    for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size(); ++edge_idx) {
        if (edge_idx < stats.tree_edge_mask.size() && stats.tree_edge_mask[edge_idx] > 0.0) continue;
        if (edge_idx >= stats.query_edge_candidate_pair_sets.size()) continue;
        int source = stats.query_edge_list[edge_idx].first;
        int target = stats.query_edge_list[edge_idx].second;
        if (source < 0 || target < 0 || static_cast<ui>(source) >= qn || static_cast<ui>(target) >= qn) continue;
        std::vector<int> path = build_tree_path_between(source, target, stats.tree_parent);
        if (path.size() <= 2) continue;
        auto external = external_weights_on_path(path);

        long long op_count = 0;
        bool skipped = false;
        std::vector<double> denom_layer = external.front();
        for (size_t path_pos = 1; path_pos < path.size(); ++path_pos) {
            denom_layer = propagate_layer(
                denom_layer,
                static_cast<ui>(path[path_pos - 1]),
                static_cast<ui>(path[path_pos]),
                external[path_pos],
                op_count,
                skipped
            );
            if (skipped) break;
        }
        if (skipped) {
            stats.tree_dp_weighted_closure_budget_skipped += 1;
            continue;
        }
        double denominator = 0.0;
        for (double value : denom_layer) denominator = cap_add(denominator, value);
        if (denominator <= 0.0) continue;

        double numerator = 0.0;
        const ui source_u = static_cast<ui>(source);
        const ui target_u = static_cast<ui>(target);
        const auto& closure_relation = stats.query_edge_candidate_pair_sets[edge_idx];
        for (size_t start_idx = 0; start_idx < stats.candidate_nodes[source_u].size(); ++start_idx) {
            double start_weight = external.front()[start_idx];
            if (start_weight <= 0.0) continue;
            std::vector<double> layer(stats.candidate_nodes[source_u].size(), 0.0);
            layer[start_idx] = start_weight;
            for (size_t path_pos = 1; path_pos < path.size(); ++path_pos) {
                layer = propagate_layer(
                    layer,
                    static_cast<ui>(path[path_pos - 1]),
                    static_cast<ui>(path[path_pos]),
                    external[path_pos],
                    op_count,
                    skipped
                );
                if (skipped) break;
            }
            if (skipped) break;
            ui source_candidate = stats.candidate_nodes[source_u][start_idx];
            for (size_t target_idx = 0; target_idx < stats.candidate_nodes[target_u].size(); ++target_idx) {
                double weight = layer[target_idx];
                if (weight <= 0.0) continue;
                ui target_candidate = stats.candidate_nodes[target_u][target_idx];
                if (closure_relation.find(directed_candidate_pair_key(source_candidate, target_candidate)) != closure_relation.end()) {
                    numerator = cap_add(numerator, weight);
                }
            }
        }
        if (skipped) {
            stats.tree_dp_weighted_closure_budget_skipped += 1;
            continue;
        }

        double prob = std::min(1.0, std::max(numerator / denominator, 0.0));
        probs.push_back(prob);
        log_gaps.push_back(-safe_log(prob));
        stats.tree_dp_weighted_closure_edge_count += 1;
    }

    stats.tree_dp_weighted_closure_prob_mean = mean_or_zero(probs);
    stats.tree_dp_weighted_closure_prob_min = min_or_zero(probs);
    stats.tree_dp_weighted_closure_prob_std = std_or_zero(probs);
    stats.tree_dp_weighted_closure_log_gap_sum = 0.0;
    for (double gap : log_gaps) stats.tree_dp_weighted_closure_log_gap_sum += gap;
    stats.tree_dp_weighted_closure_log_gap_mean = mean_or_zero(log_gaps);
    stats.tree_dp_weighted_closure_log_gap_max = max_or_zero(log_gaps);
    stats.tree_dp_weighted_closure_pred_log_count =
        stats.log_candidate_tree_count - stats.tree_dp_weighted_closure_log_gap_sum;
}

void populate_factor_bp_stats(
    QueryStats& stats,
    const std::vector<std::unordered_set<ui>>& candidate_adjacency,
    int iterations,
    long long op_budget
) {
    if (stats.candidate_nodes.empty() || stats.query_edge_list.empty()) return;
    const ui qn = static_cast<ui>(stats.candidate_nodes.size());
    const double eps = 1e-300;
    std::vector<std::unordered_map<ui, size_t>> candidate_pos(qn);
    for (ui u = 0; u < qn; ++u) {
        candidate_pos[u].reserve(stats.candidate_nodes[u].size() * 2 + 1);
        for (size_t i = 0; i < stats.candidate_nodes[u].size(); ++i) {
            candidate_pos[u][stats.candidate_nodes[u][i]] = i;
        }
    }

    std::vector<std::vector<ui>> query_neighbors(qn);
    std::unordered_map<unsigned long long, size_t> directed_slot;
    auto dkey = [](ui from, ui to) -> unsigned long long {
        return (static_cast<unsigned long long>(from) << 32) | static_cast<unsigned long long>(to);
    };
    for (const auto& edge : stats.query_edge_list) {
        if (edge.first < 0 || edge.second < 0) continue;
        ui u = static_cast<ui>(edge.first);
        ui v = static_cast<ui>(edge.second);
        if (u >= qn || v >= qn) continue;
        query_neighbors[u].push_back(v);
        query_neighbors[v].push_back(u);
    }

    std::vector<std::pair<ui, ui>> directed_edges;
    directed_edges.reserve(stats.query_edge_list.size() * 2);
    for (ui u = 0; u < qn; ++u) {
        for (ui v : query_neighbors[u]) {
            directed_slot[dkey(u, v)] = directed_edges.size();
            directed_edges.emplace_back(u, v);
        }
    }

    std::vector<std::vector<double>> messages(directed_edges.size());
    for (size_t slot = 0; slot < directed_edges.size(); ++slot) {
        ui to = directed_edges[slot].second;
        double init = 1.0 / static_cast<double>(std::max<size_t>(stats.candidate_nodes[to].size(), 1));
        messages[slot].assign(stats.candidate_nodes[to].size(), init);
    }

    long long op_count = 0;
    auto incoming_product = [&](ui u, ui excluded_neighbor, const std::vector<std::vector<double>>& current) {
        std::vector<double> weights(stats.candidate_nodes[u].size(), 1.0);
        for (ui neighbor : query_neighbors[u]) {
            if (neighbor == excluded_neighbor) continue;
            auto found_slot = directed_slot.find(dkey(neighbor, u));
            if (found_slot == directed_slot.end()) continue;
            const auto& msg = current[found_slot->second];
            for (size_t i = 0; i < weights.size() && i < msg.size(); ++i) {
                weights[i] *= std::max(msg[i], eps);
            }
        }
        return weights;
    };

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<std::vector<double>> next_messages = messages;
        for (size_t slot = 0; slot < directed_edges.size(); ++slot) {
            ui from = directed_edges[slot].first;
            ui to = directed_edges[slot].second;
            std::vector<double> from_weights = incoming_product(from, to, messages);
            std::vector<double> out(stats.candidate_nodes[to].size(), 0.0);
            const auto& to_pos = candidate_pos[to];
            for (size_t i = 0; i < stats.candidate_nodes[from].size(); ++i) {
                double weight = from_weights[i];
                if (weight <= 0.0) continue;
                ui left = stats.candidate_nodes[from][i];
                if (left >= candidate_adjacency.size()) continue;
                for (ui right : candidate_adjacency[left]) {
                    op_count += 1;
                    if (op_count > op_budget) {
                        stats.factor_bp_budget_skipped = 1;
                        return;
                    }
                    if (left == right) continue;
                    auto found = to_pos.find(right);
                    if (found == to_pos.end()) continue;
                    out[found->second] += weight;
                }
            }
            double sum = 0.0;
            for (double value : out) sum += value;
            if (sum <= 0.0 || !std::isfinite(sum)) {
                double uniform = 1.0 / static_cast<double>(std::max<size_t>(out.size(), 1));
                std::fill(out.begin(), out.end(), uniform);
            }
            else {
                for (double& value : out) value = std::max(value / sum, eps);
            }
            next_messages[slot] = std::move(out);
        }
        messages.swap(next_messages);
        stats.factor_bp_iterations = iter + 1;
    }

    std::vector<std::vector<double>> all_incoming(qn);
    std::vector<double> log_z_node(qn, 0.0);
    for (ui u = 0; u < qn; ++u) {
        all_incoming[u] = incoming_product(u, qn, messages);
        double z = 0.0;
        for (double value : all_incoming[u]) z += value;
        log_z_node[u] = safe_log(z);
    }

    double log_z_edges = 0.0;
    for (const auto& edge : stats.query_edge_list) {
        if (edge.first < 0 || edge.second < 0) continue;
        ui u = static_cast<ui>(edge.first);
        ui v = static_cast<ui>(edge.second);
        if (u >= qn || v >= qn) continue;
        std::vector<double> left_weights = incoming_product(u, v, messages);
        std::vector<double> right_weights = incoming_product(v, u, messages);
        const auto& v_pos = candidate_pos[v];
        double z_edge = 0.0;
        for (size_t i = 0; i < stats.candidate_nodes[u].size(); ++i) {
            double left_weight = left_weights[i];
            if (left_weight <= 0.0) continue;
            ui left = stats.candidate_nodes[u][i];
            if (left >= candidate_adjacency.size()) continue;
            for (ui right : candidate_adjacency[left]) {
                op_count += 1;
                if (op_count > op_budget) {
                    stats.factor_bp_budget_skipped = 1;
                    return;
                }
                auto found = v_pos.find(right);
                if (found == v_pos.end()) continue;
                z_edge += left_weight * right_weights[found->second];
            }
        }
        log_z_edges += safe_log(z_edge);
    }

    double log_z_nodes = 0.0;
    for (ui u = 0; u < qn; ++u) {
        int degree = static_cast<int>(query_neighbors[u].size());
        if (degree > 1) log_z_nodes += static_cast<double>(degree - 1) * log_z_node[u];
    }
    stats.factor_bp_full_edge_log_count = log_z_edges - log_z_nodes;
    stats.factor_bp_tree_to_full_edge_gap =
        stats.log_candidate_tree_count - stats.factor_bp_full_edge_log_count;
}

void populate_factor_bp_injective_stats(
    const Graph* query_graph,
    QueryStats& stats,
    const std::vector<std::unordered_set<ui>>& candidate_adjacency,
    int iterations,
    long long op_budget
) {
    if (stats.candidate_nodes.empty() || stats.query_edge_list.empty()) return;
    const ui qn = static_cast<ui>(stats.candidate_nodes.size());
    const double eps = 1e-300;
    struct FactorEdge {
        ui left;
        ui right;
        int type;  // 0: query edge adjacency, 1: same-label inequality.
    };
    auto edge_key = [](ui a, ui b) -> unsigned long long {
        ui x = std::min(a, b);
        ui y = std::max(a, b);
        return (static_cast<unsigned long long>(x) << 32) | static_cast<unsigned long long>(y);
    };
    std::vector<FactorEdge> factors;
    factors.reserve(stats.query_edge_list.size() + qn);
    std::unordered_set<unsigned long long> query_edge_keys;
    for (const auto& edge : stats.query_edge_list) {
        if (edge.first < 0 || edge.second < 0) continue;
        ui u = static_cast<ui>(edge.first);
        ui v = static_cast<ui>(edge.second);
        if (u >= qn || v >= qn) continue;
        factors.push_back({u, v, 0});
        query_edge_keys.insert(edge_key(u, v));
    }
    for (ui u = 0; u < qn; ++u) {
        for (ui v = u + 1; v < qn; ++v) {
            if (query_graph->getVertexLabel(u) != query_graph->getVertexLabel(v)) continue;
            if (query_edge_keys.find(edge_key(u, v)) != query_edge_keys.end()) continue;
            factors.push_back({u, v, 1});
        }
    }
    if (factors.empty()) return;

    std::vector<std::unordered_map<ui, size_t>> candidate_pos(qn);
    for (ui u = 0; u < qn; ++u) {
        candidate_pos[u].reserve(stats.candidate_nodes[u].size() * 2 + 1);
        for (size_t i = 0; i < stats.candidate_nodes[u].size(); ++i) {
            candidate_pos[u][stats.candidate_nodes[u][i]] = i;
        }
    }

    std::vector<std::vector<size_t>> incident(qn);
    std::vector<std::pair<size_t, int>> directed_refs;
    std::unordered_map<unsigned long long, size_t> directed_slot;
    auto dkey = [](size_t factor_idx, int side) -> unsigned long long {
        return (static_cast<unsigned long long>(factor_idx) << 1) | static_cast<unsigned long long>(side);
    };
    for (size_t idx = 0; idx < factors.size(); ++idx) {
        incident[factors[idx].left].push_back(idx);
        incident[factors[idx].right].push_back(idx);
        directed_slot[dkey(idx, 0)] = directed_refs.size();
        directed_refs.emplace_back(idx, 0);  // left -> right
        directed_slot[dkey(idx, 1)] = directed_refs.size();
        directed_refs.emplace_back(idx, 1);  // right -> left
    }

    auto sender = [&](size_t factor_idx, int side) -> ui {
        return side == 0 ? factors[factor_idx].left : factors[factor_idx].right;
    };
    auto receiver = [&](size_t factor_idx, int side) -> ui {
        return side == 0 ? factors[factor_idx].right : factors[factor_idx].left;
    };
    auto incoming_slot_to = [&](size_t factor_idx, ui variable) -> size_t {
        const auto& factor = factors[factor_idx];
        int side = variable == factor.left ? 1 : 0;
        return directed_slot[dkey(factor_idx, side)];
    };

    std::vector<std::vector<double>> messages(directed_refs.size());
    for (size_t slot = 0; slot < directed_refs.size(); ++slot) {
        ui to = receiver(directed_refs[slot].first, directed_refs[slot].second);
        double init = 1.0 / static_cast<double>(std::max<size_t>(stats.candidate_nodes[to].size(), 1));
        messages[slot].assign(stats.candidate_nodes[to].size(), init);
    }

    long long op_count = 0;
    auto incoming_product = [&](ui u, size_t excluded_factor, const std::vector<std::vector<double>>& current) {
        std::vector<double> weights(stats.candidate_nodes[u].size(), 1.0);
        for (size_t factor_idx : incident[u]) {
            if (factor_idx == excluded_factor) continue;
            size_t slot = incoming_slot_to(factor_idx, u);
            const auto& msg = current[slot];
            for (size_t i = 0; i < weights.size() && i < msg.size(); ++i) {
                weights[i] *= std::max(msg[i], eps);
            }
        }
        return weights;
    };

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<std::vector<double>> next_messages = messages;
        for (size_t slot = 0; slot < directed_refs.size(); ++slot) {
            size_t factor_idx = directed_refs[slot].first;
            int side = directed_refs[slot].second;
            const auto& factor = factors[factor_idx];
            ui from = sender(factor_idx, side);
            ui to = receiver(factor_idx, side);
            std::vector<double> from_weights = incoming_product(from, factor_idx, messages);
            std::vector<double> out(stats.candidate_nodes[to].size(), 0.0);
            if (factor.type == 1) {
                double total = 0.0;
                for (double value : from_weights) total += value;
                const auto& from_pos = candidate_pos[from];
                for (size_t j = 0; j < stats.candidate_nodes[to].size(); ++j) {
                    ui candidate = stats.candidate_nodes[to][j];
                    double subtract = 0.0;
                    auto found = from_pos.find(candidate);
                    if (found != from_pos.end() && found->second < from_weights.size()) {
                        subtract = from_weights[found->second];
                    }
                    out[j] = std::max(total - subtract, 0.0);
                    op_count += 1;
                    if (op_count > op_budget) {
                        stats.factor_bp_injective_budget_skipped = 1;
                        return;
                    }
                }
            }
            else {
                const auto& to_pos = candidate_pos[to];
                for (size_t i = 0; i < stats.candidate_nodes[from].size(); ++i) {
                    double weight = from_weights[i];
                    if (weight <= 0.0) continue;
                    ui left = stats.candidate_nodes[from][i];
                    if (left >= candidate_adjacency.size()) continue;
                    for (ui right : candidate_adjacency[left]) {
                        op_count += 1;
                        if (op_count > op_budget) {
                            stats.factor_bp_injective_budget_skipped = 1;
                            return;
                        }
                        if (left == right) continue;
                        auto found = to_pos.find(right);
                        if (found == to_pos.end()) continue;
                        out[found->second] += weight;
                    }
                }
            }
            double sum = 0.0;
            for (double value : out) sum += value;
            if (sum <= 0.0 || !std::isfinite(sum)) {
                double uniform = 1.0 / static_cast<double>(std::max<size_t>(out.size(), 1));
                std::fill(out.begin(), out.end(), uniform);
            }
            else {
                for (double& value : out) value = std::max(value / sum, eps);
            }
            next_messages[slot] = std::move(out);
        }
        messages.swap(next_messages);
        stats.factor_bp_injective_iterations = iter + 1;
    }

    std::vector<double> log_z_node(qn, 0.0);
    for (ui u = 0; u < qn; ++u) {
        std::vector<double> incoming = incoming_product(u, std::numeric_limits<size_t>::max(), messages);
        double z = 0.0;
        for (double value : incoming) z += value;
        log_z_node[u] = safe_log(z);
    }

    double log_z_factors = 0.0;
    for (size_t factor_idx = 0; factor_idx < factors.size(); ++factor_idx) {
        const auto& factor = factors[factor_idx];
        ui u = factor.left;
        ui v = factor.right;
        std::vector<double> left_weights = incoming_product(u, factor_idx, messages);
        std::vector<double> right_weights = incoming_product(v, factor_idx, messages);
        double z_factor = 0.0;
        if (factor.type == 1) {
            double right_total = 0.0;
            for (double value : right_weights) right_total += value;
            const auto& right_pos = candidate_pos[v];
            for (size_t i = 0; i < stats.candidate_nodes[u].size(); ++i) {
                ui candidate = stats.candidate_nodes[u][i];
                double subtract = 0.0;
                auto found = right_pos.find(candidate);
                if (found != right_pos.end() && found->second < right_weights.size()) {
                    subtract = right_weights[found->second];
                }
                z_factor += left_weights[i] * std::max(right_total - subtract, 0.0);
                op_count += 1;
                if (op_count > op_budget) {
                    stats.factor_bp_injective_budget_skipped = 1;
                    return;
                }
            }
        }
        else {
            const auto& v_pos = candidate_pos[v];
            for (size_t i = 0; i < stats.candidate_nodes[u].size(); ++i) {
                double left_weight = left_weights[i];
                if (left_weight <= 0.0) continue;
                ui left = stats.candidate_nodes[u][i];
                if (left >= candidate_adjacency.size()) continue;
                for (ui right : candidate_adjacency[left]) {
                    op_count += 1;
                    if (op_count > op_budget) {
                        stats.factor_bp_injective_budget_skipped = 1;
                        return;
                    }
                    auto found = v_pos.find(right);
                    if (found == v_pos.end()) continue;
                    z_factor += left_weight * right_weights[found->second];
                }
            }
        }
        log_z_factors += safe_log(z_factor);
    }

    double log_z_nodes = 0.0;
    for (ui u = 0; u < qn; ++u) {
        int degree = static_cast<int>(incident[u].size());
        if (degree > 1) log_z_nodes += static_cast<double>(degree - 1) * log_z_node[u];
    }
    stats.factor_bp_injective_log_count = log_z_factors - log_z_nodes;
    stats.factor_bp_tree_to_injective_gap =
        stats.log_candidate_tree_count - stats.factor_bp_injective_log_count;
}

std::vector<int> build_tree_path_between(
    int source,
    int target,
    const std::vector<int>& tree_parent
) {
    std::unordered_map<int, int> source_pos;
    std::vector<int> source_ancestors;
    int current = source;
    while (current >= 0) {
        source_pos[current] = static_cast<int>(source_ancestors.size());
        source_ancestors.push_back(current);
        current = tree_parent[static_cast<size_t>(current)];
    }

    std::vector<int> target_ancestors;
    current = target;
    int lca = -1;
    while (current >= 0) {
        target_ancestors.push_back(current);
        if (source_pos.find(current) != source_pos.end()) {
            lca = current;
            break;
        }
        current = tree_parent[static_cast<size_t>(current)];
    }
    if (lca < 0) return {};

    std::vector<int> path;
    const int source_lca_pos = source_pos[lca];
    for (int i = 0; i <= source_lca_pos; ++i) {
        path.push_back(source_ancestors[static_cast<size_t>(i)]);
    }
    for (int i = static_cast<int>(target_ancestors.size()) - 2; i >= 0; --i) {
        path.push_back(target_ancestors[static_cast<size_t>(i)]);
    }
    return path;
}

void populate_cycle_path_consistency_stats(QueryStats& stats, long long per_edge_budget) {
    if (stats.candidate_nodes.empty() || stats.candidate_neighbors.empty()) return;
    const ui qn = static_cast<ui>(stats.candidate_nodes.size());
    std::vector<std::unordered_set<ui>> candidate_sets(qn);
    for (ui u = 0; u < qn; ++u) {
        candidate_sets[u].reserve(stats.candidate_nodes[u].size() * 2 + 1);
        for (ui candidate : stats.candidate_nodes[u]) {
            candidate_sets[u].insert(candidate);
        }
    }

    std::unordered_map<unsigned long long, size_t> edge_index_by_key;
    auto query_edge_key = [](int a, int b) -> unsigned long long {
        int x = std::min(a, b);
        int y = std::max(a, b);
        return (static_cast<unsigned long long>(static_cast<unsigned int>(x)) << 32)
            | static_cast<unsigned int>(y);
    };
    edge_index_by_key.reserve(stats.query_edge_list.size() * 2 + 1);
    for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size(); ++edge_idx) {
        edge_index_by_key[query_edge_key(stats.query_edge_list[edge_idx].first, stats.query_edge_list[edge_idx].second)] = edge_idx;
    }

    using RelationAdjacency =
        std::unordered_map<ui, std::vector<ui>>;
    std::unordered_map<size_t, RelationAdjacency>
        relation_adjacency_cache;
    relation_adjacency_cache.reserve(
        stats.query_edge_candidate_pair_sets.size() * 2 + 1
    );
    auto relation_adjacency =
        [&](size_t relation_index) -> const RelationAdjacency& {
        auto found = relation_adjacency_cache.find(relation_index);
        if (found != relation_adjacency_cache.end()) {
            return found->second;
        }
        RelationAdjacency adjacency;
        const auto& relation =
            stats.query_edge_candidate_pair_sets[relation_index];
        adjacency.reserve(relation.size() / 2 + 1);
        for (unsigned long long pair_key : relation) {
            const ui left = static_cast<ui>(pair_key >> 32);
            const ui right = static_cast<ui>(
                pair_key & 0xffffffffULL
            );
            adjacency[left].push_back(right);
        }
        for (auto& item : adjacency) {
            std::sort(item.second.begin(), item.second.end());
        }
        return relation_adjacency_cache.emplace(
            relation_index,
            std::move(adjacency)
        ).first->second;
    };

    std::vector<double> support_fracs;
    std::vector<double> log_gaps;
    std::vector<double> local_edge_logs;
    std::vector<double> supported_edge_logs;
    std::vector<double> tree_pair_support_fracs;
    std::vector<double> tree_pair_log_gaps;
    std::vector<double> tree_pair_logs;

    for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size(); ++edge_idx) {
        if (edge_idx < stats.tree_edge_mask.size() && stats.tree_edge_mask[edge_idx] > 0.0) {
            continue;
        }
        int source = stats.query_edge_list[edge_idx].first;
        int target = stats.query_edge_list[edge_idx].second;
        std::vector<int> path = build_tree_path_between(source, target, stats.tree_parent);
        if (path.size() <= 2) continue;
        if (edge_idx >= stats.query_edge_candidate_pair_sets.size()) continue;

        long long op_count = 0;
        bool skipped = false;
        double supported_edges = 0.0;
        double tree_path_pairs = 0.0;
        const auto& closure_relation = stats.query_edge_candidate_pair_sets[edge_idx];
        const double local_edges = edge_idx < stats.query_edge_candidate_counts.size()
            ? stats.query_edge_candidate_counts[edge_idx]
            : static_cast<double>(closure_relation.size()) / 2.0;

        for (ui start_candidate : stats.candidate_nodes[static_cast<ui>(source)]) {
            std::unordered_set<ui> current_layer;
            current_layer.insert(start_candidate);
            for (size_t path_pos = 1; path_pos < path.size(); ++path_pos) {
                const int prev_query = path[path_pos - 1];
                const ui next_query = static_cast<ui>(path[path_pos]);
                auto relation_index = edge_index_by_key.find(query_edge_key(prev_query, static_cast<int>(next_query)));
                if (relation_index == edge_index_by_key.end()
                    || relation_index->second >= stats.query_edge_candidate_pair_sets.size()) {
                    current_layer.clear();
                    break;
                }
                const auto& path_adjacency =
                    relation_adjacency(relation_index->second);
                const auto& next_candidates = candidate_sets[next_query];
                std::unordered_set<ui> next_layer;
                for (ui current_candidate : current_layer) {
                    if (current_candidate >= stats.candidate_neighbors.size()) continue;
                    const auto& global_neighbors =
                        stats.candidate_neighbors[current_candidate];
                    const long long candidate_operations =
                        static_cast<long long>(global_neighbors.size());
                    if (candidate_operations
                        > per_edge_budget - op_count) {
                        skipped = true;
                        break;
                    }
                    op_count += candidate_operations;
                    const auto relation_neighbors =
                        path_adjacency.find(current_candidate);
                    if (relation_neighbors == path_adjacency.end()) {
                        continue;
                    }
                    for (ui neighbor : relation_neighbors->second) {
                        if (next_candidates.find(neighbor)
                            != next_candidates.end()) {
                            next_layer.insert(neighbor);
                        }
                    }
                }
                if (skipped) break;
                current_layer.swap(next_layer);
                if (current_layer.empty()) break;
            }
            if (skipped) break;
            if (current_layer.empty()) continue;
            tree_path_pairs += static_cast<double>(current_layer.size());
            for (ui target_candidate : current_layer) {
                if (closure_relation.find(directed_candidate_pair_key(start_candidate, target_candidate)) != closure_relation.end()) {
                    supported_edges += 1.0;
                }
            }
        }

        if (skipped) {
            stats.cycle_path_budget_skipped += 1;
            continue;
        }
        stats.cycle_path_edge_count += 1;
        const double denom = std::max(local_edges, 1.0);
        const double support_frac = supported_edges / denom;
        support_fracs.push_back(support_frac);
        local_edge_logs.push_back(safe_log1p(local_edges));
        supported_edge_logs.push_back(safe_log1p(supported_edges));
        log_gaps.push_back(std::max(0.0, safe_log1p(local_edges) - safe_log1p(supported_edges)));
        const double tree_pair_denom = std::max(tree_path_pairs, 1.0);
        tree_pair_support_fracs.push_back(
            supported_edges / tree_pair_denom
        );
        tree_pair_logs.push_back(safe_log1p(tree_path_pairs));
        tree_pair_log_gaps.push_back(
            safe_log1p(tree_path_pairs) - safe_log1p(supported_edges)
        );
    }

    stats.cycle_path_support_frac_mean = mean_or_zero(support_fracs);
    stats.cycle_path_support_frac_min = min_or_zero(support_fracs);
    stats.cycle_path_support_frac_std = std_or_zero(support_fracs);
    stats.cycle_path_log_gap_mean = mean_or_zero(log_gaps);
    stats.cycle_path_log_gap_max = max_or_zero(log_gaps);
    stats.cycle_path_local_edge_log_mean = mean_or_zero(local_edge_logs);
    stats.cycle_path_supported_edge_log_mean = mean_or_zero(supported_edge_logs);
    stats.cycle_path_tree_pair_support_frac_mean =
        mean_or_zero(tree_pair_support_fracs);
    stats.cycle_path_tree_pair_support_frac_min =
        min_or_zero(tree_pair_support_fracs);
    stats.cycle_path_tree_pair_support_frac_std =
        std_or_zero(tree_pair_support_fracs);
    stats.cycle_path_tree_pair_log_gap_mean =
        mean_or_zero(tree_pair_log_gaps);
    stats.cycle_path_tree_pair_log_gap_max =
        max_or_zero(tree_pair_log_gaps);
    stats.cycle_path_tree_pair_log_mean =
        mean_or_zero(tree_pair_logs);
}

double log_falling_permutation(ui universe_size, int picks);

void populate_label_injectivity_stats(const Graph* query_graph, QueryStats& stats) {
    const ui qn = query_graph->getVerticesCount();
    std::unordered_map<LabelID, std::vector<ui>> vertices_by_label;
    vertices_by_label.reserve(qn * 2 + 1);
    for (ui u = 0; u < qn; ++u) {
        vertices_by_label[query_graph->getVertexLabel(u)].push_back(u);
    }

    std::vector<double> gaps;
    std::vector<double> overlap_fracs;
    std::vector<double> union_slacks;
    double expected_pair_collisions = 0.0;
    for (const auto& item : vertices_by_label) {
        const auto& vertices = item.second;
        if (vertices.size() < 2) continue;
        std::unordered_set<ui> union_candidates;
        double product_log = 0.0;
        for (ui vertex : vertices) {
            const auto& candidates = stats.candidate_nodes[vertex];
            product_log += safe_log(static_cast<double>(std::max<size_t>(candidates.size(), 1)));
            for (ui candidate : candidates) {
                union_candidates.insert(candidate);
            }
        }
        const double distinct_bound = log_falling_permutation(
            static_cast<ui>(union_candidates.size()),
            static_cast<int>(vertices.size())
        );
        const double gap = std::max(0.0, product_log - distinct_bound);
        gaps.push_back(gap);
        union_slacks.push_back(
            static_cast<double>(union_candidates.size())
            - static_cast<double>(vertices.size())
        );
        stats.label_injectivity_product_log_sum += product_log;
        stats.label_injectivity_distinct_bound_log_sum += distinct_bound;

        for (size_t left_idx = 0; left_idx < vertices.size(); ++left_idx) {
            const auto& left = stats.candidate_nodes[vertices[left_idx]];
            for (size_t right_idx = left_idx + 1; right_idx < vertices.size(); ++right_idx) {
                const auto& right = stats.candidate_nodes[vertices[right_idx]];
                std::vector<ui> left_sorted = left;
                std::vector<ui> right_sorted = right;
                std::sort(left_sorted.begin(), left_sorted.end());
                std::sort(right_sorted.begin(), right_sorted.end());
                const size_t intersection = sorted_intersection_count(left_sorted, right_sorted);
                const double denom = static_cast<double>(
                    std::max<size_t>(std::min(left_sorted.size(), right_sorted.size()), 1)
                );
                overlap_fracs.push_back(static_cast<double>(intersection) / denom);
                const double collision_denom = std::max(
                    static_cast<double>(left_sorted.size())
                        * static_cast<double>(right_sorted.size()),
                    1.0
                );
                expected_pair_collisions +=
                    static_cast<double>(intersection) / collision_denom;
            }
        }
    }

    stats.label_injectivity_group_count = static_cast<int>(gaps.size());
    stats.label_injectivity_gap_sum = std::accumulate(gaps.begin(), gaps.end(), 0.0);
    stats.label_injectivity_gap_mean = mean_or_zero(gaps);
    stats.label_injectivity_gap_max = max_or_zero(gaps);
    stats.label_injectivity_overlap_frac_mean = mean_or_zero(overlap_fracs);
    stats.label_injectivity_overlap_frac_max = max_or_zero(overlap_fracs);
    stats.label_injectivity_min_union_slack = min_or_zero(union_slacks);
    stats.label_injectivity_pair_collision_log1p =
        safe_log1p(expected_pair_collisions);
}

void populate_final11d_label_injectivity_stats(
    const Graph* query_graph,
    QueryStats& stats
) {
    const ui qn = query_graph->getVerticesCount();
    std::unordered_map<LabelID, std::vector<ui>> vertices_by_label;
    vertices_by_label.reserve(qn * 2 + 1);
    for (ui u = 0; u < qn; ++u) {
        vertices_by_label[query_graph->getVertexLabel(u)].push_back(u);
    }
    for (const auto& item : vertices_by_label) {
        const auto& vertices = item.second;
        if (vertices.size() < 2) continue;
        double product_log = 0.0;
        for (ui vertex : vertices) {
            product_log += safe_log(
                static_cast<double>(
                    std::max<size_t>(
                        stats.candidate_nodes[vertex].size(),
                        1
                    )
                )
            );
        }
        stats.label_injectivity_product_log_sum += product_log;
        for (size_t left_idx = 0; left_idx < vertices.size(); ++left_idx) {
            std::vector<ui> left =
                stats.candidate_nodes[vertices[left_idx]];
            std::sort(left.begin(), left.end());
            for (
                size_t right_idx = left_idx + 1;
                right_idx < vertices.size();
                ++right_idx
            ) {
                std::vector<ui> right =
                    stats.candidate_nodes[vertices[right_idx]];
                std::sort(right.begin(), right.end());
                const size_t intersection =
                    sorted_intersection_count(left, right);
                const double denom = std::max(
                    static_cast<double>(left.size())
                        * static_cast<double>(right.size()),
                    1.0
                );
                stats.label_injectivity_pair_collision_log1p +=
                    static_cast<double>(intersection) / denom;
            }
        }
    }
    stats.label_injectivity_pair_collision_log1p =
        safe_log1p(stats.label_injectivity_pair_collision_log1p);
}

void populate_triangle_joint_stats(const Graph* query_graph, QueryStats& stats, long long per_triangle_budget) {
    const ui qn = query_graph->getVerticesCount();
    if (qn < 3 || stats.query_edge_candidate_pair_sets.empty()) return;

    auto query_edge_key = [](int a, int b) -> unsigned long long {
        int x = std::min(a, b);
        int y = std::max(a, b);
        return (static_cast<unsigned long long>(static_cast<unsigned int>(x)) << 32)
            | static_cast<unsigned int>(y);
    };
    std::unordered_map<unsigned long long, size_t> edge_index_by_key;
    edge_index_by_key.reserve(stats.query_edge_list.size() * 2 + 1);
    for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size(); ++edge_idx) {
        edge_index_by_key[query_edge_key(stats.query_edge_list[edge_idx].first, stats.query_edge_list[edge_idx].second)] = edge_idx;
    }

    std::vector<std::unordered_set<ui>> candidate_sets(qn);
    for (ui u = 0; u < qn; ++u) {
        candidate_sets[u].reserve(stats.candidate_nodes[u].size() * 2 + 1);
        for (ui candidate : stats.candidate_nodes[u]) {
            candidate_sets[u].insert(candidate);
        }
    }

    std::vector<double> exact_logs;
    std::vector<double> gaps;
    int zero_exact = 0;

    for (ui a = 0; a < qn; ++a) {
        for (ui b = a + 1; b < qn; ++b) {
            if (!query_graph->checkEdgeExistence(a, b)) continue;
            for (ui c = b + 1; c < qn; ++c) {
                if (!query_graph->checkEdgeExistence(a, c) || !query_graph->checkEdgeExistence(b, c)) continue;
                auto ab_it = edge_index_by_key.find(query_edge_key(static_cast<int>(a), static_cast<int>(b)));
                auto ac_it = edge_index_by_key.find(query_edge_key(static_cast<int>(a), static_cast<int>(c)));
                auto bc_it = edge_index_by_key.find(query_edge_key(static_cast<int>(b), static_cast<int>(c)));
                if (ab_it == edge_index_by_key.end() || ac_it == edge_index_by_key.end() || bc_it == edge_index_by_key.end()) {
                    continue;
                }
                const auto& ab = stats.query_edge_candidate_pair_sets[ab_it->second];
                const auto& ac = stats.query_edge_candidate_pair_sets[ac_it->second];
                const auto& bc = stats.query_edge_candidate_pair_sets[bc_it->second];

                long long op_count = 0;
                bool skipped = false;
                bool capped = false;
                double exact_count = 0.0;
                const double max_exact_count = 10000000.0;

                for (ui da : stats.candidate_nodes[a]) {
                    if (da >= stats.candidate_neighbors.size()) continue;
                    std::vector<ui> b_neighbors;
                    std::vector<ui> c_neighbors;
                    for (ui neighbor : stats.candidate_neighbors[da]) {
                        op_count += 1;
                        if (op_count > per_triangle_budget) {
                            skipped = true;
                            break;
                        }
                        if (candidate_sets[b].find(neighbor) != candidate_sets[b].end()
                            && ab.find(directed_candidate_pair_key(da, neighbor)) != ab.end()) {
                            b_neighbors.push_back(neighbor);
                        }
                        if (candidate_sets[c].find(neighbor) != candidate_sets[c].end()
                            && ac.find(directed_candidate_pair_key(da, neighbor)) != ac.end()) {
                            c_neighbors.push_back(neighbor);
                        }
                    }
                    if (skipped) break;
                    for (ui db : b_neighbors) {
                        if (db == da) continue;
                        for (ui dc : c_neighbors) {
                            op_count += 1;
                            if (op_count > per_triangle_budget) {
                                skipped = true;
                                break;
                            }
                            if (dc == da || dc == db) continue;
                            if (bc.find(directed_candidate_pair_key(db, dc)) == bc.end()) continue;
                            exact_count += 1.0;
                            if (exact_count >= max_exact_count) {
                                capped = true;
                                break;
                            }
                        }
                        if (skipped || capped) break;
                    }
                    if (skipped || capped) break;
                }

                if (skipped) {
                    stats.triangle_joint_budget_skipped += 1;
                    continue;
                }
                if (capped) {
                    stats.triangle_joint_capped += 1;
                }
                stats.triangle_joint_probe_count += 1;
                if (exact_count <= 0.0) {
                    zero_exact += 1;
                }
                const double exact_log = safe_log1p(exact_count);
                const double domain_log =
                    safe_log(static_cast<double>(std::max<size_t>(stats.candidate_nodes[a].size(), 1)))
                    + safe_log(static_cast<double>(std::max<size_t>(stats.candidate_nodes[b].size(), 1)))
                    + safe_log(static_cast<double>(std::max<size_t>(stats.candidate_nodes[c].size(), 1)));
                exact_logs.push_back(exact_log);
                gaps.push_back(std::max(0.0, domain_log - exact_log));
            }
        }
    }

    stats.triangle_joint_exact_log_mean = mean_or_zero(exact_logs);
    stats.triangle_joint_exact_log_min = min_or_zero(exact_logs);
    stats.triangle_joint_domain_to_exact_gap_mean = mean_or_zero(gaps);
    stats.triangle_joint_domain_to_exact_gap_max = max_or_zero(gaps);
    stats.triangle_joint_zero_exact_frac = stats.triangle_joint_probe_count > 0
        ? static_cast<double>(zero_exact) / static_cast<double>(stats.triangle_joint_probe_count)
        : 0.0;
}

struct Motif4Spec {
    std::vector<int> vertices;
    std::vector<std::pair<int, int>> edges;
    double domain_log = 0.0;
};

double count_exact_motif_matches(
    const QueryStats& stats,
    const std::vector<int>& vertices,
    const std::vector<std::pair<int, int>>& edges,
    const std::unordered_map<unsigned long long, size_t>& edge_index_by_key,
    long long per_motif_budget,
    double max_exact_count,
    bool& skipped,
    bool& capped,
    std::vector<unsigned char>* used_workspace = nullptr
) {
    skipped = false;
    capped = false;
    if (vertices.empty()) return 0.0;
    auto query_edge_key = [](int a, int b) -> unsigned long long {
        int x = std::min(a, b);
        int y = std::max(a, b);
        return (static_cast<unsigned long long>(static_cast<unsigned int>(x)) << 32)
            | static_cast<unsigned int>(y);
    };

    struct AssignedConstraint {
        size_t other_local_index;
        size_t relation_index;

        AssignedConstraint(size_t other, size_t relation)
            : other_local_index(other), relation_index(relation) {}
    };
    std::vector<int> local_index(stats.candidate_nodes.size(), -1);
    for (size_t i = 0; i < vertices.size(); ++i) {
        const int vertex = vertices[i];
        if (vertex >= 0
            && static_cast<size_t>(vertex) < local_index.size()) {
            local_index[static_cast<size_t>(vertex)] =
                static_cast<int>(i);
        }
    }
    std::vector<int> internal_degree(vertices.size(), 0);
    std::vector<std::vector<AssignedConstraint>> constraints(
        vertices.size()
    );
    for (const auto& edge : edges) {
        if (edge.first < 0 || edge.second < 0
            || static_cast<size_t>(edge.first) >= local_index.size()
            || static_cast<size_t>(edge.second) >= local_index.size()) {
            continue;
        }
        const int left_local = local_index[static_cast<size_t>(edge.first)];
        const int right_local =
            local_index[static_cast<size_t>(edge.second)];
        if (left_local < 0 || right_local < 0) continue;

        const auto edge_it = edge_index_by_key.find(
            query_edge_key(edge.first, edge.second)
        );
        const size_t relation_index =
            edge_it == edge_index_by_key.end()
            ? std::numeric_limits<size_t>::max()
            : edge_it->second;
        internal_degree[static_cast<size_t>(left_local)] += 1;
        internal_degree[static_cast<size_t>(right_local)] += 1;
        constraints[static_cast<size_t>(left_local)].push_back(
            AssignedConstraint(
                static_cast<size_t>(right_local),
                relation_index
            )
        );
        constraints[static_cast<size_t>(right_local)].push_back(
            AssignedConstraint(
                static_cast<size_t>(left_local),
                relation_index
            )
        );
    }

    std::vector<size_t> order(vertices.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        const size_t left_size =
            stats.candidate_nodes[static_cast<ui>(vertices[left])].size();
        const size_t right_size =
            stats.candidate_nodes[static_cast<ui>(vertices[right])].size();
        if (left_size != right_size) return left_size < right_size;
        return internal_degree[left] > internal_degree[right];
    });

    std::vector<ui> assignment(
        vertices.size(),
        std::numeric_limits<ui>::max()
    );
    std::vector<unsigned char> local_used;
    if (used_workspace == nullptr) {
        local_used.assign(stats.num_candidates, 0);
        used_workspace = &local_used;
    }
    auto& used = *used_workspace;
    long long op_count = 0;
    double exact_count = 0.0;

    std::function<void(size_t)> dfs = [&](size_t depth) {
        if (skipped || capped) return;
        if (depth >= order.size()) {
            exact_count += 1.0;
            if (exact_count >= max_exact_count) capped = true;
            return;
        }
        const size_t query_local_index = order[depth];
        const int query_vertex = vertices[query_local_index];
        for (ui candidate :
             stats.candidate_nodes[static_cast<ui>(query_vertex)]) {
            op_count += 1;
            if (op_count > per_motif_budget) {
                skipped = true;
                return;
            }
            if (candidate >= used.size() || used[candidate]) continue;
            bool ok = true;
            for (const auto& constraint :
                 constraints[query_local_index]) {
                const ui assigned =
                    assignment[constraint.other_local_index];
                if (assigned == std::numeric_limits<ui>::max()) continue;
                if (constraint.relation_index
                        >= stats.query_edge_candidate_pair_sets.size()) {
                    ok = false;
                    break;
                }
                const auto& relation =
                    stats.query_edge_candidate_pair_sets[
                        constraint.relation_index
                    ];
                if (relation.find(directed_candidate_pair_key(
                        candidate,
                        assigned
                    )) == relation.end()) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            assignment[query_local_index] = candidate;
            used[candidate] = 1;
            dfs(depth + 1);
            used[candidate] = 0;
            assignment[query_local_index] = std::numeric_limits<ui>::max();
            if (skipped || capped) return;
        }
    };
    dfs(0);
    return exact_count;
}

void populate_dense_clique_stats(
    const Graph* query_graph,
    QueryStats& stats,
    int max_vertices,
    long long exact_budget
) {
    const ui qn = query_graph->getVerticesCount();
    if (qn < 2 || max_vertices < 2
        || stats.query_edge_candidate_pair_sets.empty()) {
        return;
    }
    const int capped_max_vertices = std::min(
        max_vertices,
        static_cast<int>(qn)
    );
    std::vector<int> best;
    double best_domain_log = -1.0;
    std::vector<int> current;
    std::function<void(int, double)> enumerate =
        [&](int start, double domain_log) {
            if (current.size() >= 2
                && (
                    current.size() > best.size()
                    || (
                        current.size() == best.size()
                        && domain_log > best_domain_log
                    )
                )) {
                best = current;
                best_domain_log = domain_log;
            }
            if (static_cast<int>(current.size()) >= capped_max_vertices) {
                return;
            }
            for (int vertex = start; vertex < static_cast<int>(qn); ++vertex) {
                bool clique_ok = true;
                for (int selected : current) {
                    if (!query_graph->checkEdgeExistence(
                        static_cast<ui>(vertex),
                        static_cast<ui>(selected)
                    )) {
                        clique_ok = false;
                        break;
                    }
                }
                if (!clique_ok) continue;
                current.push_back(vertex);
                const double candidate_log = safe_log(
                    static_cast<double>(
                        std::max<size_t>(
                            stats.candidate_nodes[
                                static_cast<ui>(vertex)
                            ].size(),
                            1
                        )
                    )
                );
                enumerate(vertex + 1, domain_log + candidate_log);
                current.pop_back();
            }
    };
    enumerate(0, 0.0);
    stats.dense_clique_vertex_count = static_cast<int>(best.size());
    stats.dense_clique_query_edge_count = static_cast<int>(
        best.size() * (best.size() - 1) / 2
    );
    if (best.size() < 4) return;
    if (best_domain_log > safe_log(10000000.0)) {
        stats.dense_clique_budget_exhausted = 1;
        stats.dense_clique_domain_log = best_domain_log;
        return;
    }

    auto query_edge_key = [](int a, int b) -> unsigned long long {
        int x = std::min(a, b);
        int y = std::max(a, b);
        return (
            static_cast<unsigned long long>(
                static_cast<unsigned int>(x)
            ) << 32
        ) | static_cast<unsigned int>(y);
    };
    std::unordered_map<unsigned long long, size_t> edge_index_by_key;
    edge_index_by_key.reserve(stats.query_edge_list.size() * 2 + 1);
    for (size_t edge_idx = 0;
         edge_idx < stats.query_edge_list.size();
         ++edge_idx) {
        edge_index_by_key[
            query_edge_key(
                stats.query_edge_list[edge_idx].first,
                stats.query_edge_list[edge_idx].second
            )
        ] = edge_idx;
    }
    std::vector<std::pair<int, int>> edges;
    for (size_t left = 0; left < best.size(); ++left) {
        for (size_t right = left + 1; right < best.size(); ++right) {
            edges.emplace_back(best[left], best[right]);
        }
    }

    bool skipped = false;
    bool capped = false;
    const double exact_count = count_exact_motif_matches(
        stats,
        best,
        edges,
        edge_index_by_key,
        exact_budget,
        10000000.0,
        skipped,
        capped
    );
    long double tightest_tree_count =
        std::numeric_limits<long double>::infinity();
    for (int root : best) {
        long double root_total = 0.0L;
        for (ui root_candidate :
             stats.candidate_nodes[static_cast<ui>(root)]) {
            long double branch_count = 1.0L;
            for (int child : best) {
                if (child == root) continue;
                auto edge_idx = edge_index_by_key.find(
                    query_edge_key(root, child)
                );
                if (edge_idx == edge_index_by_key.end()
                    || edge_idx->second
                        >= stats.query_edge_candidate_pair_sets.size()) {
                    branch_count = 0.0L;
                    break;
                }
                const auto& relation =
                    stats.query_edge_candidate_pair_sets[edge_idx->second];
                long double child_support = 0.0L;
                for (ui child_candidate :
                     stats.candidate_nodes[static_cast<ui>(child)]) {
                    if (relation.find(directed_candidate_pair_key(
                            root_candidate,
                            child_candidate
                        )) != relation.end()) {
                        child_support += 1.0L;
                    }
                }
                branch_count *= child_support;
                if (branch_count <= 0.0L) break;
            }
            root_total += branch_count;
        }
        tightest_tree_count = std::min(
            tightest_tree_count,
            root_total
        );
    }
    if (!std::isfinite(tightest_tree_count)) {
        tightest_tree_count = 0.0L;
    }
    stats.dense_clique_budget_exhausted = skipped ? 1 : 0;
    stats.dense_clique_count_capped = capped ? 1 : 0;
    stats.dense_clique_domain_log = std::max(best_domain_log, 0.0);
    stats.dense_clique_exact_log_count = safe_log1p(exact_count);
    stats.dense_clique_domain_to_exact_gap = std::max(
        0.0,
        stats.dense_clique_domain_log
            - stats.dense_clique_exact_log_count
    );
    stats.dense_clique_tree_log_count = static_cast<double>(
        std::log1p(std::max(tightest_tree_count, 0.0L))
    );
    if (!skipped && !capped) {
        stats.dense_clique_tree_to_exact_gap = std::max(
            0.0,
            stats.dense_clique_tree_log_count
                - stats.dense_clique_exact_log_count
        );
        stats.dense_clique_corrected_tree_log = std::max(
            0.0,
            stats.log_candidate_tree_count
                - stats.dense_clique_tree_to_exact_gap
        );
    }
}

void populate_motif4_joint_stats(const Graph* query_graph, QueryStats& stats, int max_motifs, long long per_motif_budget) {
    const ui qn = query_graph->getVerticesCount();
    if (qn < 4 || stats.query_edge_candidate_pair_sets.empty()) return;

    auto query_edge_key = [](int a, int b) -> unsigned long long {
        int x = std::min(a, b);
        int y = std::max(a, b);
        return (static_cast<unsigned long long>(static_cast<unsigned int>(x)) << 32)
            | static_cast<unsigned int>(y);
    };
    std::unordered_map<unsigned long long, size_t> edge_index_by_key;
    edge_index_by_key.reserve(stats.query_edge_list.size() * 2 + 1);
    for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size(); ++edge_idx) {
        edge_index_by_key[query_edge_key(stats.query_edge_list[edge_idx].first, stats.query_edge_list[edge_idx].second)] = edge_idx;
    }

    std::vector<Motif4Spec> motifs;
    for (ui a = 0; a < qn; ++a) {
        for (ui b = a + 1; b < qn; ++b) {
            for (ui c = b + 1; c < qn; ++c) {
                for (ui d = c + 1; d < qn; ++d) {
                    std::vector<int> vertices = {
                        static_cast<int>(a),
                        static_cast<int>(b),
                        static_cast<int>(c),
                        static_cast<int>(d),
                    };
                    std::vector<std::pair<int, int>> edges;
                    for (size_t i = 0; i < vertices.size(); ++i) {
                        for (size_t j = i + 1; j < vertices.size(); ++j) {
                            if (query_graph->checkEdgeExistence(static_cast<ui>(vertices[i]), static_cast<ui>(vertices[j]))) {
                                edges.emplace_back(vertices[i], vertices[j]);
                            }
                        }
                    }
                    if (edges.size() < 4) continue;
                    Motif4Spec spec;
                    spec.vertices = vertices;
                    spec.edges = edges;
                    for (int vertex : vertices) {
                        spec.domain_log += safe_log(
                            static_cast<double>(std::max<size_t>(stats.candidate_nodes[static_cast<ui>(vertex)].size(), 1))
                        );
                    }
                    motifs.push_back(std::move(spec));
                }
            }
        }
    }
    stats.motif4_joint_candidate_count = static_cast<int>(motifs.size());
    std::sort(motifs.begin(), motifs.end(), [](const Motif4Spec& left, const Motif4Spec& right) {
        if (left.edges.size() != right.edges.size()) return left.edges.size() > right.edges.size();
        return left.domain_log > right.domain_log;
    });
    if (static_cast<int>(motifs.size()) > max_motifs) {
        motifs.resize(static_cast<size_t>(max_motifs));
    }

    std::vector<double> edge_counts;
    std::vector<double> exact_logs;
    std::vector<double> gaps;
    int zero_exact = 0;
    std::vector<unsigned char> used_workspace(stats.num_candidates, 0);
    for (const auto& motif : motifs) {
        bool skipped = false;
        bool capped = false;
        double exact_count = count_exact_motif_matches(
            stats,
            motif.vertices,
            motif.edges,
            edge_index_by_key,
            per_motif_budget,
            10000000.0,
            skipped,
            capped,
            &used_workspace
        );
        if (skipped) {
            stats.motif4_joint_budget_skipped += 1;
            continue;
        }
        if (capped) {
            stats.motif4_joint_capped += 1;
        }
        stats.motif4_joint_probe_count += 1;
        if (exact_count <= 0.0) {
            zero_exact += 1;
        }
        double exact_log = safe_log1p(exact_count);
        edge_counts.push_back(static_cast<double>(motif.edges.size()));
        exact_logs.push_back(exact_log);
        gaps.push_back(std::max(0.0, motif.domain_log - exact_log));
    }
    stats.motif4_joint_edge_count_mean = mean_or_zero(edge_counts);
    stats.motif4_joint_exact_log_mean = mean_or_zero(exact_logs);
    stats.motif4_joint_exact_log_min = min_or_zero(exact_logs);
    stats.motif4_joint_domain_to_exact_gap_mean = mean_or_zero(gaps);
    stats.motif4_joint_domain_to_exact_gap_max = max_or_zero(gaps);
    stats.motif4_joint_zero_exact_frac = stats.motif4_joint_probe_count > 0
        ? static_cast<double>(zero_exact) / static_cast<double>(stats.motif4_joint_probe_count)
        : 0.0;
}

void populate_full_match_probe(const Graph* query_graph, QueryStats& stats, long long search_budget, double count_cap) {
    const ui qn = query_graph->getVerticesCount();
    if (qn == 0 || stats.query_edge_candidate_pair_sets.empty()) return;
    auto query_edge_key = [](int a, int b) -> unsigned long long {
        int x = std::min(a, b);
        int y = std::max(a, b);
        return (static_cast<unsigned long long>(static_cast<unsigned int>(x)) << 32)
            | static_cast<unsigned int>(y);
    };
    std::unordered_map<unsigned long long, size_t> edge_index_by_key;
    edge_index_by_key.reserve(stats.query_edge_list.size() * 2 + 1);
    for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size(); ++edge_idx) {
        edge_index_by_key[query_edge_key(stats.query_edge_list[edge_idx].first, stats.query_edge_list[edge_idx].second)] = edge_idx;
    }

    std::vector<int> degrees(qn, 0);
    for (ui u = 0; u < qn; ++u) {
        ui neighbor_count = 0;
        query_graph->getVertexNeighbors(u, neighbor_count);
        degrees[u] = static_cast<int>(neighbor_count);
    }
    std::vector<int> assigned(qn, 0);
    std::vector<ui> assignment(qn, 0);
    std::vector<unsigned char> used(stats.num_candidates, 0);
    long long search_nodes = 0;
    bool exhausted = false;
    bool capped = false;
    double match_count = 0.0;
    int max_depth = 0;

    std::function<void(size_t)> dfs = [&](size_t depth) {
        if (exhausted || capped) return;
        max_depth = std::max(max_depth, static_cast<int>(depth));
        if (depth >= qn) {
            match_count += 1.0;
            if (match_count >= count_cap) {
                capped = true;
            }
            return;
        }
        int query_vertex = -1;
        int best_assigned_neighbor_count = -1;
        size_t best_candidate_count = std::numeric_limits<size_t>::max();
        int best_degree = -1;
        for (ui u = 0; u < qn; ++u) {
            if (assigned[u]) continue;
            ui neighbor_count = 0;
            const VertexID* neighbors = query_graph->getVertexNeighbors(u, neighbor_count);
            int assigned_neighbor_count = 0;
            for (ui ni = 0; ni < neighbor_count; ++ni) {
                if (assigned[neighbors[ni]]) assigned_neighbor_count += 1;
            }
            const size_t candidate_count = stats.candidate_nodes[u].size();
            const int degree = degrees[u];
            if (assigned_neighbor_count > best_assigned_neighbor_count
                || (assigned_neighbor_count == best_assigned_neighbor_count && candidate_count < best_candidate_count)
                || (assigned_neighbor_count == best_assigned_neighbor_count && candidate_count == best_candidate_count && degree > best_degree)) {
                query_vertex = static_cast<int>(u);
                best_assigned_neighbor_count = assigned_neighbor_count;
                best_candidate_count = candidate_count;
                best_degree = degree;
            }
        }
        if (query_vertex < 0) return;
        for (ui candidate : stats.candidate_nodes[static_cast<ui>(query_vertex)]) {
            search_nodes += 1;
            if (search_nodes > search_budget) {
                exhausted = true;
                return;
            }
            if (candidate >= used.size() || used[candidate]) continue;
            bool ok = true;
            ui neighbor_count = 0;
            const VertexID* neighbors = query_graph->getVertexNeighbors(static_cast<ui>(query_vertex), neighbor_count);
            for (ui ni = 0; ni < neighbor_count; ++ni) {
                int other_query = static_cast<int>(neighbors[ni]);
                if (!assigned[static_cast<ui>(other_query)]) continue;
                auto edge_idx = edge_index_by_key.find(query_edge_key(query_vertex, other_query));
                if (edge_idx == edge_index_by_key.end()
                    || edge_idx->second >= stats.query_edge_candidate_pair_sets.size()) {
                    ok = false;
                    break;
                }
                const auto& relation = stats.query_edge_candidate_pair_sets[edge_idx->second];
                if (relation.find(directed_candidate_pair_key(candidate, assignment[static_cast<ui>(other_query)])) == relation.end()) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            assigned[static_cast<ui>(query_vertex)] = 1;
            assignment[static_cast<ui>(query_vertex)] = candidate;
            used[candidate] = 1;
            dfs(depth + 1);
            used[candidate] = 0;
            assigned[static_cast<ui>(query_vertex)] = 0;
            if (capped || exhausted) return;
        }
    };
    dfs(0);
    stats.full_match_probe_found = match_count > 0.0 ? 1 : 0;
    stats.full_match_probe_budget_exhausted = exhausted ? 1 : 0;
    stats.full_match_probe_count_capped = capped ? 1 : 0;
    stats.full_match_probe_max_depth = max_depth;
    stats.full_match_probe_search_nodes = static_cast<double>(search_nodes);
    stats.full_match_probe_count = match_count;
    stats.full_match_probe_log_count = safe_log1p(match_count);
}

ui select_root_by_candidate_count(const ui qn, ui* candidates_count, bool use_min) {
    ui best = 0;
    for (ui u = 1; u < qn; ++u) {
        if (use_min) {
            if (candidates_count[u] < candidates_count[best]) best = u;
        } else {
            if (candidates_count[u] > candidates_count[best]) best = u;
        }
    }
    return best;
}

ui select_root_by_query_degree(const Graph* query_graph, const ui qn, bool use_min) {
    ui best = 0;
    ui best_degree = 0;
    query_graph->getVertexNeighbors(0, best_degree);
    for (ui u = 1; u < qn; ++u) {
        ui degree = 0;
        query_graph->getVertexNeighbors(u, degree);
        if (use_min) {
            if (degree < best_degree) {
                best = u;
                best_degree = degree;
            }
        } else {
            if (degree > best_degree) {
                best = u;
                best_degree = degree;
            }
        }
    }
    return best;
}

std::vector<int> build_bfs_tree_parent(
    const Graph* query_graph,
    const ui qn,
    const ui root,
    std::vector<int>& tree_order
) {
    std::vector<int> parent(qn, -1);
    std::vector<int> visited(qn, 0);
    std::queue<ui> queue;
    tree_order.clear();
    visited[root] = 1;
    queue.push(root);
    while (!queue.empty()) {
        ui u = queue.front();
        queue.pop();
        tree_order.push_back(static_cast<int>(u));
        ui neighbor_count = 0;
        const VertexID* neighbors = query_graph->getVertexNeighbors(u, neighbor_count);
        for (ui i = 0; i < neighbor_count; ++i) {
            ui v = neighbors[i];
            if (visited[v]) continue;
            visited[v] = 1;
            parent[v] = static_cast<int>(u);
            queue.push(v);
        }
    }
    return parent;
}

std::vector<int> build_weighted_spanning_tree_parent(
    const ui qn,
    const std::vector<std::pair<int, int>>& edge_list,
    const std::vector<double>& edge_scores,
    const ui root,
    std::vector<int>& tree_order
) {
    std::vector<int> dsu_parent(qn);
    for (ui u = 0; u < qn; ++u) dsu_parent[u] = static_cast<int>(u);
    std::function<int(int)> find_root = [&](int x) {
        while (dsu_parent[x] != x) {
            dsu_parent[x] = dsu_parent[dsu_parent[x]];
            x = dsu_parent[x];
        }
        return x;
    };
    std::vector<size_t> ids(edge_list.size());
    for (size_t i = 0; i < edge_list.size(); ++i) ids[i] = i;
    std::sort(ids.begin(), ids.end(), [&](size_t left, size_t right) {
        return edge_scores[left] > edge_scores[right];
    });
    std::vector<std::vector<ui>> tree_adjacency(qn);
    for (size_t id : ids) {
        int u = edge_list[id].first;
        int v = edge_list[id].second;
        int ru = find_root(u);
        int rv = find_root(v);
        if (ru == rv) continue;
        dsu_parent[ru] = rv;
        tree_adjacency[static_cast<ui>(u)].push_back(static_cast<ui>(v));
        tree_adjacency[static_cast<ui>(v)].push_back(static_cast<ui>(u));
    }
    std::vector<int> parent;
    tree_order = orient_tree_order(qn, root, tree_adjacency, parent);
    return parent;
}

struct RepeatedLabelGroupProbe {
    int label = -1;
    int vertex_n = 0;
    int internal_edge_n = 0;
    int cycle_rank = 0;
    int tree_edge_n = 0;
    int nontree_edge_n = 0;
    double log_candidate_product = 0.0;
    double log_exact_internal_match_count = 0.0;
    double product_to_exact_log_gap = 0.0;
    int exact_computed = 0;
    int exact_capped = 0;
    double min_log_density = 0.0;
    double mean_log_density = 0.0;
    double std_log_density = 0.0;
    double min_log_count = 0.0;
    double mean_log_count = 0.0;
    double std_log_count = 0.0;
    double closure_complexity = 0.0;
    double complexity_weak_density = 0.0;
    double complexity_small_count = 0.0;
    double candidate_union_log_size = 0.0;
    double candidate_pair_overlap_frac_mean = 0.0;
    double candidate_pair_overlap_frac_max = 0.0;
    double candidate_pair_jaccard_mean = 0.0;
    double candidate_pair_jaccard_max = 0.0;
    double injectivity_log_gap = 0.0;
    double vertex_all_edge_support_frac_min = 0.0;
    double vertex_all_edge_support_frac_mean = 0.0;
    double vertex_min_edge_support_log_min = 0.0;
    double vertex_min_edge_support_log_mean = 0.0;
    double vertex_min_edge_support_log_std = 0.0;
    int support_pattern_skipped_by_budget = 0;
};

struct MlpFeatures {
    std::vector<std::vector<double>> query_node_features;
    std::vector<std::vector<double>> data_node_features;
    std::vector<double> global_features;
    std::vector<int> tree_order_ids;
    double aux_log_tree_count = 0.0;
};

bool local_edge_exists(const QueryStats& stats, ui left, ui right) {
    if (left >= stats.candidate_neighbors.size()) return false;
    const auto& neighbors = stats.candidate_neighbors[left];
    return std::binary_search(neighbors.begin(), neighbors.end(), right);
}

void write_core_outside_stats(std::ostream& out, const QueryStats& stats) {
    out << ",\"core_outside_disabled\":" << stats.core_outside_disabled
        << ",\"core_outside_policy\":\"" << stats.core_outside_policy << "\""
        << ",\"core_outside_vertices\":";
    write_int_array(out, stats.core_outside_vertices);
    out << ",\"core_outside_vertex_count\":" << stats.core_outside_vertex_count
        << ",\"core_outside_truncated\":" << stats.core_outside_truncated
        << ",\"core_outside_edge_count\":" << stats.core_outside_edge_count
        << ",\"core_outside_frontier_edge_count\":" << stats.core_outside_frontier_edge_count
        << ",\"core_outside_uncovered_non_tree_edge_count\":" << stats.core_outside_uncovered_non_tree_edge_count
        << ",\"core_exact_count\":" << stats.core_exact_count
        << ",\"core_exact_log_count\":" << stats.core_exact_log_count
        << ",\"core_exact_count_capped\":" << stats.core_exact_count_capped
        << ",\"core_exact_budget_exhausted\":" << stats.core_exact_budget_exhausted
        << ",\"core_exact_search_nodes\":" << stats.core_exact_search_nodes
        << ",\"core_outside_estimate_count\":" << stats.core_outside_estimate_count
        << ",\"core_outside_estimate_log_count\":" << stats.core_outside_estimate_log_count
        << ",\"core_outside_estimate_capped\":" << stats.core_outside_estimate_capped;
}

double core_outside_tree_extension_count(
    const Graph* query_graph,
    const QueryStats& stats,
    const std::vector<char>& in_core,
    const std::vector<ui>& core_assignment,
    const std::unordered_set<ui>& used_core_nodes,
    long long& op_count,
    long long op_budget,
    bool& exhausted
) {
    const ui qn = query_graph->getVerticesCount();
    if (qn == 0) return 0.0;
    const double large_count = 1e300;
    std::vector<std::vector<double>> dp(qn);
    std::vector<int> order = stats.tree_order;
    if (order.size() < qn) {
        std::vector<char> seen(qn, 0);
        for (int u : order) {
            if (u >= 0 && static_cast<ui>(u) < qn) seen[static_cast<ui>(u)] = 1;
        }
        for (ui u = 0; u < qn; ++u) {
            if (!seen[u]) order.push_back(static_cast<int>(u));
        }
    }

    auto cap_mul = [&](double left, double right) -> double {
        if (left <= 0.0 || right <= 0.0) return 0.0;
        if (left > large_count / right) return large_count;
        double value = left * right;
        return std::isfinite(value) ? std::min(value, large_count) : large_count;
    };

    for (int order_idx = static_cast<int>(order.size()) - 1; order_idx >= 0; --order_idx) {
        ui u = static_cast<ui>(order[order_idx]);
        if (u >= qn || in_core[u]) continue;
        dp[u].assign(stats.candidate_nodes[u].size(), 0.0);
        ui neighbor_count = 0;
        const VertexID* neighbors = query_graph->getVertexNeighbors(u, neighbor_count);
        for (ui i = 0; i < stats.candidate_nodes[u].size(); ++i) {
            op_count += 1;
            if (op_count > op_budget) {
                exhausted = true;
                return 0.0;
            }
            ui candidate = stats.candidate_nodes[u][i];
            if (used_core_nodes.find(candidate) != used_core_nodes.end()) continue;
            bool ok = true;
            for (ui ni = 0; ni < neighbor_count; ++ni) {
                ui v = neighbors[ni];
                if (v >= qn || !in_core[v]) continue;
                ui assigned_core = core_assignment[v];
                if (candidate == assigned_core || !local_edge_exists(stats, candidate, assigned_core)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            double value = 1.0;
            for (ui v = 0; v < qn; ++v) {
                if (v >= stats.tree_parent.size() || stats.tree_parent[v] != static_cast<int>(u) || in_core[v]) continue;
                double child_sum = 0.0;
                for (ui j = 0; j < stats.candidate_nodes[v].size(); ++j) {
                    op_count += 1;
                    if (op_count > op_budget) {
                        exhausted = true;
                        return 0.0;
                    }
                    ui child_candidate = stats.candidate_nodes[v][j];
                    if (child_candidate == candidate) continue;
                    if (!local_edge_exists(stats, candidate, child_candidate)) continue;
                    child_sum += dp[v][j];
                    if (child_sum > large_count) {
                        child_sum = large_count;
                        break;
                    }
                }
                value = cap_mul(value, child_sum);
                if (value <= 0.0) break;
            }
            dp[u][i] = value;
        }
    }

    double total = 1.0;
    bool has_outside_root = false;
    for (ui u = 0; u < qn; ++u) {
        if (in_core[u]) continue;
        const bool parent_is_core = u < stats.tree_parent.size()
            && stats.tree_parent[u] >= 0
            && in_core[static_cast<ui>(stats.tree_parent[u])];
        const bool parent_missing = u >= stats.tree_parent.size() || stats.tree_parent[u] < 0;
        if (!parent_is_core && !parent_missing) continue;
        has_outside_root = true;
        double root_sum = 0.0;
        for (double value : dp[u]) {
            root_sum += value;
            if (root_sum > large_count) {
                root_sum = large_count;
                break;
            }
        }
        total = cap_mul(total, root_sum);
        if (total <= 0.0) break;
    }
    return has_outside_root ? total : 1.0;
}

void populate_core_outside_probe(
    const Graph* query_graph,
    QueryStats& stats,
    long long search_budget,
    double count_cap,
    int max_core_vertices,
    const std::string& core_policy
) {
    const ui qn = query_graph->getVerticesCount();
    if (qn == 0) return;
    const double large_count = 1e300;
    stats.core_outside_policy = core_policy + "_top" + std::to_string(max_core_vertices);

    std::unordered_map<LabelID, int> label_counts;
    label_counts.reserve(qn * 2 + 1);
    std::vector<int> query_degrees(qn, 0);
    for (ui u = 0; u < qn; ++u) {
        label_counts[query_graph->getVertexLabel(u)] += 1;
        ui degree = 0;
        query_graph->getVertexNeighbors(u, degree);
        query_degrees[u] = static_cast<int>(degree);
    }

    std::vector<double> scores(qn, 0.0);
    std::vector<char> selected(qn, 0);
    std::vector<size_t> non_tree_edge_indices;
    non_tree_edge_indices.reserve(stats.query_edge_list.size());
    for (ui u = 0; u < qn; ++u) {
        if (label_counts[query_graph->getVertexLabel(u)] > 1) {
            scores[u] += 10.0;
            if (core_policy != "nontree_vertex_cover") selected[u] = 1;
        }
        scores[u] += static_cast<double>(query_degrees[u]) * 0.1;
        scores[u] -= 0.001 * static_cast<double>(stats.candidate_nodes[u].size());
    }
    for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size(); ++edge_idx) {
        if (edge_idx < stats.tree_edge_mask.size() && stats.tree_edge_mask[edge_idx] > 0.0) continue;
        non_tree_edge_indices.push_back(edge_idx);
        int left = stats.query_edge_list[edge_idx].first;
        int right = stats.query_edge_list[edge_idx].second;
        if (core_policy != "nontree_vertex_cover") {
            if (left >= 0 && static_cast<ui>(left) < qn) {
                scores[static_cast<ui>(left)] += 100.0;
                selected[static_cast<ui>(left)] = 1;
            }
            if (right >= 0 && static_cast<ui>(right) < qn) {
                scores[static_cast<ui>(right)] += 100.0;
                selected[static_cast<ui>(right)] = 1;
            }
        }
    }

    std::vector<int> candidates_for_core;
    if (core_policy == "nontree_vertex_cover") {
        std::vector<char> edge_covered(stats.query_edge_list.size(), 0);
        auto better_vertex = [&](ui left, ui right, int left_cover, int right_cover) {
            if (left_cover != right_cover) return left_cover > right_cover;
            bool left_repeated = label_counts[query_graph->getVertexLabel(left)] > 1;
            bool right_repeated = label_counts[query_graph->getVertexLabel(right)] > 1;
            if (left_repeated != right_repeated) return left_repeated;
            if (query_degrees[left] != query_degrees[right]) return query_degrees[left] > query_degrees[right];
            if (stats.candidate_nodes[left].size() != stats.candidate_nodes[right].size()) {
                return stats.candidate_nodes[left].size() < stats.candidate_nodes[right].size();
            }
            return left < right;
        };
        while (static_cast<int>(candidates_for_core.size()) < max_core_vertices) {
            int best_vertex = -1;
            int best_cover = 0;
            for (ui u = 0; u < qn; ++u) {
                if (selected[u]) continue;
                int cover = 0;
                for (size_t edge_idx : non_tree_edge_indices) {
                    if (edge_covered[edge_idx]) continue;
                    int left = stats.query_edge_list[edge_idx].first;
                    int right = stats.query_edge_list[edge_idx].second;
                    if (left == static_cast<int>(u) || right == static_cast<int>(u)) cover += 1;
                }
                if (cover <= 0) continue;
                if (best_vertex < 0 || better_vertex(u, static_cast<ui>(best_vertex), cover, best_cover)) {
                    best_vertex = static_cast<int>(u);
                    best_cover = cover;
                }
            }
            if (best_vertex < 0 || best_cover <= 0) break;
            selected[static_cast<ui>(best_vertex)] = 1;
            candidates_for_core.push_back(best_vertex);
            for (size_t edge_idx : non_tree_edge_indices) {
                int left = stats.query_edge_list[edge_idx].first;
                int right = stats.query_edge_list[edge_idx].second;
                if (left == best_vertex || right == best_vertex) edge_covered[edge_idx] = 1;
            }
        }
        std::vector<int> repeated_vertices;
        for (ui u = 0; u < qn; ++u) {
            if (selected[u]) continue;
            if (label_counts[query_graph->getVertexLabel(u)] <= 1) continue;
            repeated_vertices.push_back(static_cast<int>(u));
        }
        std::sort(repeated_vertices.begin(), repeated_vertices.end(), [&](int left, int right) {
            ui l = static_cast<ui>(left);
            ui r = static_cast<ui>(right);
            if (query_degrees[l] != query_degrees[r]) return query_degrees[l] > query_degrees[r];
            if (stats.candidate_nodes[l].size() != stats.candidate_nodes[r].size()) {
                return stats.candidate_nodes[l].size() < stats.candidate_nodes[r].size();
            }
            return left < right;
        });
        int selected_plus_repeated = static_cast<int>(candidates_for_core.size() + repeated_vertices.size());
        for (int u : repeated_vertices) {
            if (static_cast<int>(candidates_for_core.size()) >= max_core_vertices) break;
            candidates_for_core.push_back(u);
            selected[static_cast<ui>(u)] = 1;
        }
        int uncovered_non_tree = 0;
        for (size_t edge_idx : non_tree_edge_indices) {
            int left = stats.query_edge_list[edge_idx].first;
            int right = stats.query_edge_list[edge_idx].second;
            bool left_selected = left >= 0 && static_cast<ui>(left) < qn && selected[static_cast<ui>(left)];
            bool right_selected = right >= 0 && static_cast<ui>(right) < qn && selected[static_cast<ui>(right)];
            if (!left_selected && !right_selected) uncovered_non_tree += 1;
        }
        if (uncovered_non_tree > 0 || selected_plus_repeated > max_core_vertices) {
            stats.core_outside_truncated = 1;
        }
    }
    else {
        for (ui u = 0; u < qn; ++u) {
            if (selected[u]) candidates_for_core.push_back(static_cast<int>(u));
        }
    }
    if (candidates_for_core.empty()) {
        stats.core_outside_vertices.clear();
        stats.core_outside_vertex_count = 0;
        stats.core_exact_count = 1.0;
        stats.core_exact_log_count = safe_log1p(1.0);
        stats.core_outside_estimate_count = std::exp(std::min(stats.log_candidate_tree_count, std::log(large_count)));
        stats.core_outside_estimate_log_count = stats.log_candidate_tree_count;
        return;
    }
    if (core_policy != "nontree_vertex_cover") {
        std::sort(candidates_for_core.begin(), candidates_for_core.end(), [&](int left, int right) {
            if (scores[static_cast<ui>(left)] != scores[static_cast<ui>(right)]) {
                return scores[static_cast<ui>(left)] > scores[static_cast<ui>(right)];
            }
            if (stats.candidate_nodes[static_cast<ui>(left)].size() != stats.candidate_nodes[static_cast<ui>(right)].size()) {
                return stats.candidate_nodes[static_cast<ui>(left)].size() < stats.candidate_nodes[static_cast<ui>(right)].size();
            }
            return left < right;
        });
        if (static_cast<int>(candidates_for_core.size()) > max_core_vertices) {
            stats.core_outside_truncated = 1;
            candidates_for_core.resize(static_cast<size_t>(max_core_vertices));
        }
    }
    std::sort(candidates_for_core.begin(), candidates_for_core.end());
    stats.core_outside_vertices = candidates_for_core;
    stats.core_outside_vertex_count = static_cast<int>(stats.core_outside_vertices.size());
    std::vector<char> in_core(qn, 0);
    for (int u : stats.core_outside_vertices) {
        if (u >= 0 && static_cast<ui>(u) < qn) in_core[static_cast<ui>(u)] = 1;
    }
    for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size(); ++edge_idx) {
        int left = stats.query_edge_list[edge_idx].first;
        int right = stats.query_edge_list[edge_idx].second;
        bool left_core = left >= 0 && static_cast<ui>(left) < qn && in_core[static_cast<ui>(left)];
        bool right_core = right >= 0 && static_cast<ui>(right) < qn && in_core[static_cast<ui>(right)];
        if (left_core && right_core) stats.core_outside_edge_count += 1;
        if (left_core != right_core) stats.core_outside_frontier_edge_count += 1;
        if (!left_core && !right_core && edge_idx < stats.tree_edge_mask.size() && stats.tree_edge_mask[edge_idx] <= 0.0) {
            stats.core_outside_uncovered_non_tree_edge_count += 1;
        }
    }

    std::vector<char> assigned(qn, 0);
    std::vector<ui> assignment(qn, std::numeric_limits<ui>::max());
    std::unordered_set<ui> used;
    long long search_nodes = 0;
    bool exhausted = false;
    bool core_count_capped = false;
    bool outside_capped = false;
    double core_count = 0.0;
    double outside_estimate = 0.0;

    std::function<void(size_t)> dfs = [&](size_t depth) {
        if (exhausted || outside_capped) return;
        if (depth >= stats.core_outside_vertices.size()) {
            core_count += 1.0;
            if (core_count >= count_cap) {
                core_count = count_cap;
                core_count_capped = true;
                outside_estimate = std::max(outside_estimate, count_cap);
                stats.core_outside_estimate_capped = 1;
                outside_capped = true;
                return;
            }
            double outside = core_outside_tree_extension_count(
                query_graph,
                stats,
                in_core,
                assignment,
                used,
                search_nodes,
                search_budget,
                exhausted
            );
            if (exhausted) return;
            outside_estimate += outside;
            if (!std::isfinite(outside_estimate) || outside_estimate >= count_cap) {
                outside_estimate = std::min(count_cap, large_count);
                stats.core_outside_estimate_capped = 1;
                outside_capped = true;
            }
            return;
        }

        int query_vertex = -1;
        int best_assigned_neighbor_count = -1;
        size_t best_candidate_count = std::numeric_limits<size_t>::max();
        int best_degree = -1;
        for (int u : stats.core_outside_vertices) {
            if (u < 0 || static_cast<ui>(u) >= qn || assigned[static_cast<ui>(u)]) continue;
            ui neighbor_count = 0;
            const VertexID* neighbors = query_graph->getVertexNeighbors(static_cast<ui>(u), neighbor_count);
            int assigned_neighbor_count = 0;
            for (ui ni = 0; ni < neighbor_count; ++ni) {
                ui v = neighbors[ni];
                if (v < qn && in_core[v] && assigned[v]) assigned_neighbor_count += 1;
            }
            size_t candidate_count = stats.candidate_nodes[static_cast<ui>(u)].size();
            int degree = query_degrees[static_cast<ui>(u)];
            if (assigned_neighbor_count > best_assigned_neighbor_count
                || (assigned_neighbor_count == best_assigned_neighbor_count && candidate_count < best_candidate_count)
                || (assigned_neighbor_count == best_assigned_neighbor_count && candidate_count == best_candidate_count && degree > best_degree)) {
                query_vertex = u;
                best_assigned_neighbor_count = assigned_neighbor_count;
                best_candidate_count = candidate_count;
                best_degree = degree;
            }
        }
        if (query_vertex < 0) return;
        ui qv = static_cast<ui>(query_vertex);
        ui neighbor_count = 0;
        const VertexID* neighbors = query_graph->getVertexNeighbors(qv, neighbor_count);
        auto outside_neighbors_still_feasible = [&](ui assigned_core_vertex) {
            ui assigned_neighbor_count = 0;
            const VertexID* assigned_neighbors = query_graph->getVertexNeighbors(assigned_core_vertex, assigned_neighbor_count);
            for (ui ni = 0; ni < assigned_neighbor_count; ++ni) {
                ui outside_vertex = assigned_neighbors[ni];
                if (outside_vertex >= qn || in_core[outside_vertex]) continue;
                ui outside_neighbor_count = 0;
                const VertexID* outside_neighbors = query_graph->getVertexNeighbors(outside_vertex, outside_neighbor_count);
                bool has_candidate = false;
                for (ui candidate : stats.candidate_nodes[outside_vertex]) {
                    search_nodes += 1;
                    if (search_nodes > search_budget) {
                        exhausted = true;
                        return false;
                    }
                    if (used.find(candidate) != used.end()) continue;
                    bool ok = true;
                    for (ui oi = 0; oi < outside_neighbor_count; ++oi) {
                        ui core_neighbor = outside_neighbors[oi];
                        if (core_neighbor >= qn || !in_core[core_neighbor] || !assigned[core_neighbor]) continue;
                        if (candidate == assignment[core_neighbor] || !local_edge_exists(stats, candidate, assignment[core_neighbor])) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        has_candidate = true;
                        break;
                    }
                }
                if (!has_candidate) return false;
            }
            return true;
        };
        for (ui candidate : stats.candidate_nodes[qv]) {
            search_nodes += 1;
            if (search_nodes > search_budget) {
                exhausted = true;
                return;
            }
            if (used.find(candidate) != used.end()) continue;
            bool ok = true;
            for (ui ni = 0; ni < neighbor_count; ++ni) {
                ui other = neighbors[ni];
                if (other >= qn || !in_core[other] || !assigned[other]) continue;
                if (!local_edge_exists(stats, candidate, assignment[other])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            assigned[qv] = 1;
            assignment[qv] = candidate;
            used.insert(candidate);
            if (outside_neighbors_still_feasible(qv)) {
                dfs(depth + 1);
            }
            used.erase(candidate);
            assignment[qv] = std::numeric_limits<ui>::max();
            assigned[qv] = 0;
            if (exhausted || outside_capped) return;
        }
    };
    dfs(0);
    if (exhausted) {
        stats.core_outside_truncated = 1;
        stats.core_outside_estimate_capped = 1;
    }

    stats.core_exact_count = core_count;
    stats.core_exact_log_count = safe_log1p(core_count);
    stats.core_exact_count_capped = core_count_capped ? 1 : 0;
    stats.core_exact_budget_exhausted = exhausted ? 1 : 0;
    stats.core_exact_search_nodes = static_cast<double>(search_nodes);
    stats.core_outside_estimate_count = outside_estimate;
    stats.core_outside_estimate_log_count = outside_estimate > 0.0 ? std::log(outside_estimate) : 0.0;
}

int component_count_for_group(
    const std::vector<int>& group_vertices,
    const std::vector<std::pair<int, int>>& internal_edges
) {
    if (group_vertices.empty()) return 0;
    std::unordered_map<int, std::vector<int>> adjacency;
    adjacency.reserve(group_vertices.size() * 2 + 1);
    for (int vertex : group_vertices) {
        adjacency[vertex] = {};
    }
    for (const auto& edge : internal_edges) {
        adjacency[edge.first].push_back(edge.second);
        adjacency[edge.second].push_back(edge.first);
    }
    std::unordered_set<int> visited;
    int component_count = 0;
    for (int vertex : group_vertices) {
        if (visited.find(vertex) != visited.end()) continue;
        component_count += 1;
        std::queue<int> queue;
        queue.push(vertex);
        visited.insert(vertex);
        while (!queue.empty()) {
            int current = queue.front();
            queue.pop();
            for (int neighbor : adjacency[current]) {
                if (visited.find(neighbor) != visited.end()) continue;
                visited.insert(neighbor);
                queue.push(neighbor);
            }
        }
    }
    return component_count;
}

double log_falling_permutation(ui universe_size, int picks) {
    if (picks <= 0) return 0.0;
    if (universe_size < static_cast<ui>(picks)) return 0.0;
    double out = 0.0;
    for (int i = 0; i < picks; ++i) {
        out += safe_log(static_cast<double>(universe_size - static_cast<ui>(i)));
    }
    return out;
}

double count_exact_internal_group_matches(
    const QueryStats& stats,
    const std::vector<int>& group_vertices,
    const std::vector<std::pair<int, int>>& internal_edges,
    double max_count,
    bool& capped
) {
    capped = false;
    if (group_vertices.empty()) return 0.0;
    std::unordered_map<int, int> internal_degree;
    std::unordered_map<int, std::vector<std::pair<int, int>>> edges_by_query;
    internal_degree.reserve(group_vertices.size() * 2 + 1);
    edges_by_query.reserve(group_vertices.size() * 2 + 1);
    for (int vertex : group_vertices) {
        internal_degree[vertex] = 0;
        edges_by_query[vertex] = {};
    }
    for (const auto& edge : internal_edges) {
        internal_degree[edge.first] += 1;
        internal_degree[edge.second] += 1;
        edges_by_query[edge.first].push_back(edge);
        edges_by_query[edge.second].push_back(edge);
    }

    std::vector<int> order = group_vertices;
    std::sort(order.begin(), order.end(), [&](int left, int right) {
        const size_t left_size = stats.candidate_nodes[static_cast<ui>(left)].size();
        const size_t right_size = stats.candidate_nodes[static_cast<ui>(right)].size();
        if (left_size != right_size) return left_size < right_size;
        return internal_degree[left] > internal_degree[right];
    });

    std::unordered_map<int, ui> assigned;
    std::unordered_set<ui> used_data_nodes;
    assigned.reserve(group_vertices.size() * 2 + 1);
    used_data_nodes.reserve(group_vertices.size() * 4 + 1);
    double count = 0.0;

    std::function<void(size_t)> dfs = [&](size_t depth) {
        if (capped) return;
        if (depth >= order.size()) {
            count += 1.0;
            if (count >= max_count) {
                capped = true;
            }
            return;
        }
        int query_vertex = order[depth];
        const auto& candidates = stats.candidate_nodes[static_cast<ui>(query_vertex)];
        for (ui data_vertex : candidates) {
            if (used_data_nodes.find(data_vertex) != used_data_nodes.end()) continue;
            bool ok = true;
            for (const auto& edge : edges_by_query[query_vertex]) {
                int other_query = edge.first == query_vertex ? edge.second : edge.first;
                auto found = assigned.find(other_query);
                if (found == assigned.end()) continue;
                if (!local_edge_exists(stats, data_vertex, found->second)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            assigned[query_vertex] = data_vertex;
            used_data_nodes.insert(data_vertex);
            dfs(depth + 1);
            used_data_nodes.erase(data_vertex);
            assigned.erase(query_vertex);
            if (capped) return;
        }
    };
    dfs(0);
    return count;
}

std::vector<RepeatedLabelGroupProbe> build_repeated_label_group_probes(
    const Graph* query_graph,
    const QueryStats& stats
) {
    const ui qn = query_graph->getVerticesCount();
    std::unordered_map<LabelID, std::vector<int>> vertices_by_label;
    vertices_by_label.reserve(qn * 2 + 1);
    for (ui u = 0; u < qn; ++u) {
        vertices_by_label[query_graph->getVertexLabel(u)].push_back(static_cast<int>(u));
    }

    std::unordered_map<unsigned long long, size_t> edge_index_by_key;
    auto edge_key = [](int a, int b) -> unsigned long long {
        int x = std::min(a, b);
        int y = std::max(a, b);
        return (static_cast<unsigned long long>(static_cast<unsigned int>(x)) << 32)
            | static_cast<unsigned int>(y);
    };
    edge_index_by_key.reserve(stats.query_edge_list.size() * 2 + 1);
    for (size_t idx = 0; idx < stats.query_edge_list.size(); ++idx) {
        edge_index_by_key[edge_key(stats.query_edge_list[idx].first, stats.query_edge_list[idx].second)] = idx;
    }

    std::vector<RepeatedLabelGroupProbe> probes;
    for (const auto& item : vertices_by_label) {
        const auto& vertices = item.second;
        if (vertices.size() < 2) continue;
        std::unordered_set<int> vertex_set(vertices.begin(), vertices.end());
        std::vector<std::pair<int, int>> internal_edges;
        std::vector<double> log_densities;
        std::vector<double> log_counts;
        int tree_edge_n = 0;
        int nontree_edge_n = 0;
        for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size(); ++edge_idx) {
            int left = stats.query_edge_list[edge_idx].first;
            int right = stats.query_edge_list[edge_idx].second;
            if (vertex_set.find(left) == vertex_set.end() || vertex_set.find(right) == vertex_set.end()) continue;
            internal_edges.emplace_back(left, right);
            double density = edge_idx < stats.query_edge_densities.size() ? stats.query_edge_densities[edge_idx] : 0.0;
            double edge_count = edge_idx < stats.query_edge_candidate_counts.size() ? stats.query_edge_candidate_counts[edge_idx] : 0.0;
            log_densities.push_back(safe_log(density));
            log_counts.push_back(safe_log1p(edge_count));
            if (edge_idx < stats.tree_edge_mask.size() && stats.tree_edge_mask[edge_idx] > 0.0) {
                tree_edge_n += 1;
            }
            else {
                nontree_edge_n += 1;
            }
        }
        if (internal_edges.empty()) continue;
        RepeatedLabelGroupProbe probe;
        probe.label = static_cast<int>(item.first);
        probe.vertex_n = static_cast<int>(vertices.size());
        probe.internal_edge_n = static_cast<int>(internal_edges.size());
        const int component_n = component_count_for_group(vertices, internal_edges);
        probe.cycle_rank = std::max<int>(0, probe.internal_edge_n - probe.vertex_n + component_n);
        probe.tree_edge_n = tree_edge_n;
        probe.nontree_edge_n = nontree_edge_n;
        double log_product = 0.0;
        double product_cap_check = 1.0;
        bool product_too_large = false;
        for (int vertex : vertices) {
            const double size = static_cast<double>(std::max<size_t>(stats.candidate_nodes[static_cast<ui>(vertex)].size(), 1));
            log_product += safe_log(size);
            if (!product_too_large) {
                product_cap_check *= size;
                if (product_cap_check > 1000000.0) {
                    product_too_large = true;
                }
            }
        }
        probe.log_candidate_product = log_product;

        std::unordered_map<int, std::unordered_set<ui>> group_candidate_sets;
        group_candidate_sets.reserve(vertices.size() * 2 + 1);
        std::unordered_set<ui> union_candidates;
        for (int vertex : vertices) {
            auto& target = group_candidate_sets[vertex];
            target.reserve(stats.candidate_nodes[static_cast<ui>(vertex)].size() * 2 + 1);
            for (ui candidate : stats.candidate_nodes[static_cast<ui>(vertex)]) {
                target.insert(candidate);
                union_candidates.insert(candidate);
            }
        }
        probe.candidate_union_log_size = safe_log1p(static_cast<double>(union_candidates.size()));
        const double log_distinct_upper = log_falling_permutation(static_cast<ui>(union_candidates.size()), probe.vertex_n);
        probe.injectivity_log_gap = std::max(0.0, probe.log_candidate_product - log_distinct_upper);

        std::vector<double> overlap_fracs;
        std::vector<double> jaccards;
        for (size_t left_idx = 0; left_idx < vertices.size(); ++left_idx) {
            const auto& left_set = group_candidate_sets[vertices[left_idx]];
            for (size_t right_idx = left_idx + 1; right_idx < vertices.size(); ++right_idx) {
                const auto& right_set = group_candidate_sets[vertices[right_idx]];
                const auto* small = &left_set;
                const auto* large = &right_set;
                if (small->size() > large->size()) std::swap(small, large);
                ui intersection = 0;
                for (ui candidate : *small) {
                    if (large->find(candidate) != large->end()) intersection += 1;
                }
                const double min_size = static_cast<double>(std::max<size_t>(std::min(left_set.size(), right_set.size()), 1));
                const double union_size = static_cast<double>(std::max<size_t>(left_set.size() + right_set.size() - intersection, 1));
                overlap_fracs.push_back(static_cast<double>(intersection) / min_size);
                jaccards.push_back(static_cast<double>(intersection) / union_size);
            }
        }
        probe.candidate_pair_overlap_frac_mean = mean_or_zero(overlap_fracs);
        probe.candidate_pair_overlap_frac_max = max_or_zero(overlap_fracs);
        probe.candidate_pair_jaccard_mean = mean_or_zero(jaccards);
        probe.candidate_pair_jaccard_max = max_or_zero(jaccards);

        std::unordered_map<int, std::vector<int>> internal_neighbors;
        internal_neighbors.reserve(vertices.size() * 2 + 1);
        for (int vertex : vertices) {
            internal_neighbors[vertex] = {};
        }
        for (const auto& edge : internal_edges) {
            internal_neighbors[edge.first].push_back(edge.second);
            internal_neighbors[edge.second].push_back(edge.first);
        }
        long long support_budget = 0;
        for (int vertex : vertices) {
            support_budget += static_cast<long long>(stats.candidate_nodes[static_cast<ui>(vertex)].size())
                * static_cast<long long>(std::max<size_t>(internal_neighbors[vertex].size(), 1));
        }
        std::vector<double> all_edge_support_fracs;
        std::vector<double> min_support_logs;
        if (support_budget <= 5000000LL) {
            for (int vertex : vertices) {
                const auto& incident = internal_neighbors[vertex];
                if (incident.empty()) continue;
                const auto& candidates_for_vertex = stats.candidate_nodes[static_cast<ui>(vertex)];
                ui all_supported = 0;
                for (ui candidate : candidates_for_vertex) {
                    bool candidate_supports_all_edges = true;
                    double min_support = std::numeric_limits<double>::infinity();
                    for (int neighbor_vertex : incident) {
                        const auto& neighbor_set = group_candidate_sets[neighbor_vertex];
                        ui support_count = 0;
                        if (candidate < stats.candidate_neighbors.size()) {
                            for (ui neighbor_candidate : stats.candidate_neighbors[candidate]) {
                                if (neighbor_set.find(neighbor_candidate) != neighbor_set.end()) {
                                    support_count += 1;
                                }
                            }
                        }
                        if (support_count == 0) {
                            candidate_supports_all_edges = false;
                        }
                        min_support = std::min(min_support, static_cast<double>(support_count));
                    }
                    if (candidate_supports_all_edges) all_supported += 1;
                    if (std::isfinite(min_support)) {
                        min_support_logs.push_back(safe_log1p(min_support));
                    }
                }
                const double denom = static_cast<double>(std::max<size_t>(candidates_for_vertex.size(), 1));
                all_edge_support_fracs.push_back(static_cast<double>(all_supported) / denom);
            }
        }
        else {
            probe.support_pattern_skipped_by_budget = 1;
        }
        probe.vertex_all_edge_support_frac_min = min_or_zero(all_edge_support_fracs);
        probe.vertex_all_edge_support_frac_mean = mean_or_zero(all_edge_support_fracs);
        probe.vertex_min_edge_support_log_min = min_or_zero(min_support_logs);
        probe.vertex_min_edge_support_log_mean = mean_or_zero(min_support_logs);
        probe.vertex_min_edge_support_log_std = std_or_zero(min_support_logs);

        probe.min_log_density = min_or_zero(log_densities);
        probe.mean_log_density = mean_or_zero(log_densities);
        probe.std_log_density = std_or_zero(log_densities);
        probe.min_log_count = min_or_zero(log_counts);
        probe.mean_log_count = mean_or_zero(log_counts);
        probe.std_log_count = std_or_zero(log_counts);
        probe.closure_complexity = safe_log1p(static_cast<double>(probe.vertex_n))
            * static_cast<double>(std::max<int>(probe.internal_edge_n, 1))
            * static_cast<double>(1 + probe.cycle_rank);
        probe.complexity_weak_density = probe.closure_complexity * std::max(0.0, -probe.min_log_density);
        probe.complexity_small_count = probe.closure_complexity * std::max(0.0, probe.log_candidate_product - probe.min_log_count);
        if (probe.vertex_n <= 5 && !product_too_large) {
            bool capped = false;
            const double exact_count = count_exact_internal_group_matches(stats, vertices, internal_edges, 10000000.0, capped);
            probe.exact_computed = 1;
            probe.exact_capped = capped ? 1 : 0;
            probe.log_exact_internal_match_count = safe_log1p(exact_count);
            probe.product_to_exact_log_gap = std::max(0.0, probe.log_candidate_product - probe.log_exact_internal_match_count);
        }
        probes.push_back(probe);
    }
    std::sort(probes.begin(), probes.end(), [](const RepeatedLabelGroupProbe& left, const RepeatedLabelGroupProbe& right) {
        if (left.complexity_weak_density != right.complexity_weak_density) {
            return left.complexity_weak_density > right.complexity_weak_density;
        }
        if (left.internal_edge_n != right.internal_edge_n) return left.internal_edge_n > right.internal_edge_n;
        return left.vertex_n > right.vertex_n;
    });
    if (probes.size() > 8) probes.resize(8);
    return probes;
}

void write_repeated_label_group_probes(
    std::ostream& out,
    const std::vector<RepeatedLabelGroupProbe>& probes
) {
    out << '[';
    for (size_t i = 0; i < probes.size(); ++i) {
        const auto& p = probes[i];
        if (i > 0) out << ',';
        out << '{'
            << "\"label\":" << p.label
            << ",\"vertex_n\":" << p.vertex_n
            << ",\"internal_edge_n\":" << p.internal_edge_n
            << ",\"cycle_rank\":" << p.cycle_rank
            << ",\"tree_edge_n\":" << p.tree_edge_n
            << ",\"nontree_edge_n\":" << p.nontree_edge_n
            << ",\"log_candidate_product\":" << p.log_candidate_product
            << ",\"log_exact_internal_match_count\":" << p.log_exact_internal_match_count
            << ",\"product_to_exact_log_gap\":" << p.product_to_exact_log_gap
            << ",\"exact_computed\":" << p.exact_computed
            << ",\"exact_capped\":" << p.exact_capped
            << ",\"min_log_density\":" << p.min_log_density
            << ",\"mean_log_density\":" << p.mean_log_density
            << ",\"std_log_density\":" << p.std_log_density
            << ",\"min_log_count\":" << p.min_log_count
            << ",\"mean_log_count\":" << p.mean_log_count
            << ",\"std_log_count\":" << p.std_log_count
            << ",\"closure_complexity\":" << p.closure_complexity
            << ",\"complexity_weak_density\":" << p.complexity_weak_density
            << ",\"complexity_small_count\":" << p.complexity_small_count
            << ",\"candidate_union_log_size\":" << p.candidate_union_log_size
            << ",\"candidate_pair_overlap_frac_mean\":" << p.candidate_pair_overlap_frac_mean
            << ",\"candidate_pair_overlap_frac_max\":" << p.candidate_pair_overlap_frac_max
            << ",\"candidate_pair_jaccard_mean\":" << p.candidate_pair_jaccard_mean
            << ",\"candidate_pair_jaccard_max\":" << p.candidate_pair_jaccard_max
            << ",\"injectivity_log_gap\":" << p.injectivity_log_gap
            << ",\"vertex_all_edge_support_frac_min\":" << p.vertex_all_edge_support_frac_min
            << ",\"vertex_all_edge_support_frac_mean\":" << p.vertex_all_edge_support_frac_mean
            << ",\"vertex_min_edge_support_log_min\":" << p.vertex_min_edge_support_log_min
            << ",\"vertex_min_edge_support_log_mean\":" << p.vertex_min_edge_support_log_mean
            << ",\"vertex_min_edge_support_log_std\":" << p.vertex_min_edge_support_log_std
            << ",\"support_pattern_skipped_by_budget\":" << p.support_pattern_skipped_by_budget
            << '}';
    }
    out << ']';
}

void write_tree_prior_variants(
    std::ostream& out,
    const std::vector<std::pair<std::string, double>>& variants
) {
    out << '{';
    for (size_t i = 0; i < variants.size(); ++i) {
        if (i > 0) out << ',';
        out << '"' << json_escape(variants[i].first) << "\":" << variants[i].second;
    }
    out << '}';
}

void write_mlp_binary_record(
    std::ostream& out,
    const std::string& query_path,
    const QueryStats& stats,
    const MlpFeatures& features,
    const std::vector<uint32_t>& cs_sizes,
    double elapsed_seconds
) {
    const uint32_t magic = 0x4e545343;  // NTSC
    const uint32_t version = 1;
    const uint32_t q_rows = static_cast<uint32_t>(features.query_node_features.size());
    const uint32_t q_cols = q_rows == 0 ? 0 : static_cast<uint32_t>(features.query_node_features.front().size());
    const uint32_t d_rows = static_cast<uint32_t>(features.data_node_features.size());
    const uint32_t d_cols = d_rows == 0 ? 0 : static_cast<uint32_t>(features.data_node_features.front().size());
    const uint32_t g_cols = static_cast<uint32_t>(features.global_features.size());
    const uint32_t tree_len = static_cast<uint32_t>(features.tree_order_ids.size());
    const uint32_t cs_len = static_cast<uint32_t>(cs_sizes.size());
    write_binary_value(out, magic);
    write_binary_value(out, version);
    write_binary_string(out, query_path);
    write_binary_value(out, elapsed_seconds);
    write_binary_value(out, features.aux_log_tree_count);
    write_binary_value(out, stats.log_candidate_tree_count);
    write_binary_value(out, stats.tree_density_log_mean);
    write_binary_value(out, stats.non_tree_density_log_mean);
    write_binary_value(out, stats.density_log_gap);
    write_binary_value(out, stats.candidate_edge_count);
    write_binary_value(out, q_rows);
    write_binary_value(out, q_cols);
    write_binary_value(out, d_rows);
    write_binary_value(out, d_cols);
    write_binary_value(out, g_cols);
    write_binary_value(out, tree_len);
    write_binary_value(out, cs_len);
    for (uint32_t value : cs_sizes) {
        write_binary_value(out, value);
    }
    write_binary_nested_float_array(out, features.query_node_features);
    write_binary_nested_float_array(out, features.data_node_features);
    write_binary_float_array(out, features.global_features);
    for (int value : features.tree_order_ids) {
        int32_t converted = static_cast<int32_t>(value);
        write_binary_value(out, converted);
    }
}

MlpFeatures build_mlp_features(
    const Graph* data_graph,
    const Graph* query_graph,
    const QueryStats& stats,
    const std::vector<std::vector<ui>>* components = nullptr
) {
    MlpFeatures features;
    const ui qn = query_graph->getVerticesCount();
    const ui query_edge_count = query_graph->getEdgesCount();

    double max_cs = 1.0;
    double sum_cs = 0.0;
    std::vector<double> cs_values;
    std::vector<double> cs_logs;
    cs_values.reserve(qn);
    cs_logs.reserve(qn);
    double log_product_cs = 0.0;
    for (ui u = 0; u < qn; ++u) {
        double cs = static_cast<double>(stats.candidate_nodes[u].size());
        max_cs = std::max(max_cs, cs);
        sum_cs += cs;
        log_product_cs += safe_log(std::max(cs, 1.0));
        cs_values.push_back(cs);
        cs_logs.push_back(safe_log1p(cs));
    }
    double mean_cs = qn > 0 ? sum_cs / static_cast<double>(qn) : 0.0;
    double query_cycle_rank = std::max<double>(
        static_cast<double>(query_edge_count) - static_cast<double>(qn) + 1.0,
        0.0
    );
    double max_query_degree = 1.0;
    for (ui u = 0; u < qn; ++u) {
        max_query_degree = std::max(max_query_degree, static_cast<double>(query_graph->getVertexDegree(u)));
    }
    std::unordered_map<LabelID, std::vector<ui>> query_label_vertices;
    for (ui u = 0; u < qn; ++u) {
        query_label_vertices[query_graph->getVertexLabel(u)].push_back(u);
    }
    std::vector<double> query_label_count_values;
    query_label_count_values.reserve(query_label_vertices.size());
    double query_label_max_frequency = 0.0;
    double query_label_duplicate_count = 0.0;
    double query_label_collision_pairs = 0.0;
    double label_injection_bound_log = 0.0;
    std::vector<double> same_label_candidate_jaccards;
    double same_label_candidate_jaccard_max = 0.0;
    for (const auto& item : query_label_vertices) {
        const auto& vertices = item.second;
        const double count = static_cast<double>(vertices.size());
        query_label_count_values.push_back(count);
        query_label_max_frequency = std::max(query_label_max_frequency, count);
        query_label_duplicate_count += std::max(count - 1.0, 0.0);
        query_label_collision_pairs += count * (count - 1.0) * 0.5;

        std::unordered_set<ui> label_candidate_union;
        size_t union_reserve = 0;
        for (ui u : vertices) {
            union_reserve += stats.candidate_nodes[u].size();
        }
        label_candidate_union.reserve(union_reserve * 2 + 1);
        for (ui u : vertices) {
            for (ui node : stats.candidate_nodes[u]) {
                label_candidate_union.insert(node);
            }
        }
        label_injection_bound_log += log_injection_bound(
            static_cast<double>(label_candidate_union.size()),
            count
        );

        if (vertices.size() > 1) {
            for (size_t i = 0; i < vertices.size(); ++i) {
                for (size_t j = i + 1; j < vertices.size(); ++j) {
                    const auto& left = stats.candidate_nodes[vertices[i]];
                    const auto& right = stats.candidate_nodes[vertices[j]];
                    const double intersection = static_cast<double>(sorted_intersection_count(left, right));
                    const double union_size = static_cast<double>(left.size() + right.size()) - intersection;
                    const double jaccard = union_size > 0.0 ? intersection / union_size : 0.0;
                    same_label_candidate_jaccards.push_back(jaccard);
                    same_label_candidate_jaccard_max = std::max(same_label_candidate_jaccard_max, jaccard);
                }
            }
        }
    }
    const double query_vertex_count = std::max<double>(static_cast<double>(qn), 1.0);
    const double query_label_unique_fraction = static_cast<double>(query_label_vertices.size()) / query_vertex_count;
    const double query_label_entropy = normalized_entropy(query_label_count_values);
    const double query_label_max_frequency_fraction = query_label_max_frequency / query_vertex_count;
    const double query_label_collision_fraction = query_label_collision_pairs
        / std::max<double>(query_vertex_count * (query_vertex_count - 1.0) * 0.5, 1.0);
    const double same_label_candidate_jaccard_mean = mean_or_zero(same_label_candidate_jaccards);
    const double product_to_label_injection_log_gap = std::max(log_product_cs - label_injection_bound_log, 0.0);
    double max_depth = 1.0;
    for (int depth : stats.tree_depth_by_node) {
        max_depth = std::max(max_depth, static_cast<double>(depth));
    }
    double max_data_degree = 1.0;
    for (double degree : stats.candidate_data_degrees) {
        max_data_degree = std::max(max_data_degree, degree);
    }

    features.query_node_features.reserve(qn);
    for (ui u = 0; u < qn; ++u) {
        double cs = static_cast<double>(stats.candidate_nodes[u].size());
        double degree = static_cast<double>(query_graph->getVertexDegree(u));
        double tree_depth = u < stats.tree_depth_by_node.size() ? static_cast<double>(stats.tree_depth_by_node[u]) : 0.0;
        double leaf_mask = u < stats.tree_leaf_mask.size() ? stats.tree_leaf_mask[u] : 0.0;
        double parent_log_density = u < stats.tree_parent_log_density.size() ? stats.tree_parent_log_density[u] : 0.0;
        features.query_node_features.push_back({
            safe_log(cs),
            cs / max_cs,
            degree / max_query_degree,
            static_cast<int>(u) == stats.tree_root ? 1.0 : 0.0,
            leaf_mask,
            tree_depth / max_depth,
            parent_log_density,
        });
    }

    std::vector<double> membership_count(stats.num_candidates, 0.0);
    std::vector<double> root_overlap(stats.num_candidates, 0.0);
    std::vector<double> tree_support(stats.num_candidates, 0.0);
    std::vector<ui> component_id(stats.num_candidates, 0);
    std::vector<double> component_sizes(1, static_cast<double>(stats.num_candidates));
    double component_count = 1.0;
    double max_component_size = static_cast<double>(stats.num_candidates);
    double largest_component_fraction = 1.0;
    if (components != nullptr && !components->empty()) {
        component_count = static_cast<double>(components->size());
        component_sizes.assign(components->size(), 0.0);
        max_component_size = 0.0;
        for (ui cid = 0; cid < components->size(); ++cid) {
            component_sizes[cid] = static_cast<double>((*components)[cid].size());
            max_component_size = std::max(max_component_size, component_sizes[cid]);
            for (ui node : (*components)[cid]) {
                if (node < component_id.size()) component_id[node] = cid;
            }
        }
        largest_component_fraction = max_component_size / std::max<double>(static_cast<double>(stats.num_candidates), 1.0);
    }
    std::vector<std::vector<double>> query_component_candidate_counts;
    if (components != nullptr && !components->empty()) {
        query_component_candidate_counts.assign(qn, std::vector<double>(components->size(), 0.0));
        for (ui u = 0; u < qn; ++u) {
            for (ui node : stats.candidate_nodes[u]) {
                if (node < component_id.size()) {
                    query_component_candidate_counts[u][component_id[node]] += 1.0;
                }
            }
        }
    }
    for (ui u = 0; u < qn; ++u) {
        const auto& nodes = stats.candidate_nodes[u];
        double cluster_size = std::max<double>(static_cast<double>(nodes.size()), 1.0);
        for (ui node : nodes) {
            membership_count[node] += 1.0;
            if (static_cast<int>(u) == stats.tree_root) {
                double local_cluster_size = cluster_size;
                if (components != nullptr && !components->empty() && node < component_id.size()) {
                    local_cluster_size = std::max<double>(query_component_candidate_counts[u][component_id[node]], 1.0);
                }
                root_overlap[node] += 1.0 / local_cluster_size;
            }
        }
    }
    for (size_t edge_idx = 0; edge_idx < stats.query_edge_list.size() && edge_idx < stats.tree_edge_mask.size(); ++edge_idx) {
        if (stats.tree_edge_mask[edge_idx] <= 0.0) continue;
        int left_query = stats.query_edge_list[edge_idx].first;
        int right_query = stats.query_edge_list[edge_idx].second;
        const auto& left_nodes = stats.candidate_nodes[left_query];
        const auto& right_nodes = stats.candidate_nodes[right_query];
        double left_weight = 1.0 / std::max<double>(static_cast<double>(left_nodes.size()), 1.0);
        double right_weight = 1.0 / std::max<double>(static_cast<double>(right_nodes.size()), 1.0);
        for (ui node : left_nodes) {
            double weight = left_weight;
            if (components != nullptr && !components->empty() && node < component_id.size()) {
                weight = 1.0 / std::max<double>(query_component_candidate_counts[left_query][component_id[node]], 1.0);
            }
            tree_support[node] += weight;
        }
        for (ui node : right_nodes) {
            double weight = right_weight;
            if (components != nullptr && !components->empty() && node < component_id.size()) {
                weight = 1.0 / std::max<double>(query_component_candidate_counts[right_query][component_id[node]], 1.0);
            }
            tree_support[node] += weight;
        }
    }
    double max_membership = 1.0;
    double max_candidate_degree = 1.0;
    double max_tree_support = 1.0;
    std::vector<double> component_max_membership(component_sizes.size(), 1.0);
    std::vector<double> component_max_candidate_degree(component_sizes.size(), 1.0);
    std::vector<double> component_max_data_degree(component_sizes.size(), 1.0);
    std::vector<double> component_max_tree_support(component_sizes.size(), 1.0);
    for (ui i = 0; i < stats.num_candidates; ++i) {
        max_membership = std::max(max_membership, membership_count[i]);
        ui cid = i < component_id.size() ? component_id[i] : 0;
        if (cid < component_max_membership.size()) {
            component_max_membership[cid] = std::max(component_max_membership[cid], membership_count[i]);
        }
        if (i < stats.candidate_node_degrees.size()) {
            max_candidate_degree = std::max(max_candidate_degree, stats.candidate_node_degrees[i]);
            if (cid < component_max_candidate_degree.size()) {
                component_max_candidate_degree[cid] = std::max(component_max_candidate_degree[cid], stats.candidate_node_degrees[i]);
            }
        }
        if (i < stats.candidate_data_degrees.size() && cid < component_max_data_degree.size()) {
            component_max_data_degree[cid] = std::max(component_max_data_degree[cid], stats.candidate_data_degrees[i]);
        }
        max_tree_support = std::max(max_tree_support, tree_support[i]);
        if (cid < component_max_tree_support.size()) {
            component_max_tree_support[cid] = std::max(component_max_tree_support[cid], tree_support[i]);
        }
    }
    features.data_node_features.reserve(stats.num_candidates);
    for (ui i = 0; i < stats.num_candidates; ++i) {
        ui cid = i < component_id.size() ? component_id[i] : 0;
        double membership_denom = max_membership;
        double candidate_degree_denom = max_candidate_degree;
        double data_degree_denom = max_data_degree;
        double tree_support_denom = max_tree_support;
        if (components != nullptr && !components->empty() && cid < component_sizes.size()) {
            membership_denom = component_max_membership[cid];
            candidate_degree_denom = component_max_candidate_degree[cid];
            data_degree_denom = component_max_data_degree[cid];
            tree_support_denom = component_max_tree_support[cid];
        }
        double candidate_degree = i < stats.candidate_node_degrees.size() ? stats.candidate_node_degrees[i] : 0.0;
        double data_degree = i < stats.candidate_data_degrees.size() ? stats.candidate_data_degrees[i] : candidate_degree;
        features.data_node_features.push_back({
            safe_log(membership_count[i]),
            membership_count[i] / std::max<double>(membership_denom, 1.0),
            safe_log(candidate_degree),
            candidate_degree / std::max<double>(candidate_degree_denom, 1.0),
            data_degree / std::max<double>(data_degree_denom, 1.0),
            root_overlap[i],
            tree_support[i] / std::max<double>(tree_support_denom, 1.0),
        });
    }

    double query_density = 0.0;
    if (qn > 1) {
        query_density = (2.0 * static_cast<double>(query_edge_count))
            / std::max<double>(static_cast<double>(qn) * static_cast<double>(qn - 1), 1.0);
    }
    features.global_features = {
        safe_log(static_cast<double>(qn)),
        safe_log(static_cast<double>(std::max<ui>(query_edge_count, 1))),
        query_density,
        safe_log(max_query_degree),
        safe_log(sum_cs),
        safe_log(mean_cs),
        safe_log(max_cs),
        safe_log(max_data_degree),
        stats.log_candidate_tree_count,
        stats.tree_density_log_mean,
        stats.non_tree_density_log_mean,
        stats.density_log_gap,
        stats.tree_density_log_min,
        stats.tree_density_log_std,
        stats.non_tree_density_log_min,
        stats.non_tree_density_log_std,
        stats.tree_edge_log_count_mean,
        stats.tree_edge_log_count_min,
        stats.tree_edge_log_count_std,
        stats.non_tree_edge_log_count_mean,
        stats.non_tree_edge_log_count_min,
        stats.non_tree_edge_log_count_std,
        stats.candidate_degree_log_mean,
        stats.candidate_degree_log_std,
        stats.candidate_degree_log_max,
        stats.candidate_degree_l2_log,
        std_or_zero(cs_logs),
        normalized_entropy(cs_values),
        gini_coefficient(cs_values),
        query_cycle_rank,
        query_label_unique_fraction,
        query_label_entropy,
        query_label_max_frequency_fraction,
        query_label_collision_fraction,
        safe_log1p(query_label_duplicate_count),
        same_label_candidate_jaccard_mean,
        same_label_candidate_jaccard_max,
        label_injection_bound_log,
        product_to_label_injection_log_gap,
    };
    if (components != nullptr) {
        features.global_features.push_back(safe_log(component_count));
        features.global_features.push_back(safe_log(std::max<double>(max_component_size, 1.0)));
        features.global_features.push_back(largest_component_fraction);
    }
    features.tree_order_ids = stats.tree_order;
    features.aux_log_tree_count = stats.log_candidate_tree_count;
    return features;
}

MlpFeatures build_component_view_mlp_features(
    const Graph* data_graph,
    const Graph* query_graph,
    const QueryStats& stats,
    const std::vector<ui>& component
) {
    MlpFeatures features;
    const ui qn = query_graph->getVerticesCount();
    const ui query_edge_count = query_graph->getEdgesCount();

    std::vector<char> in_component(stats.num_candidates, 0);
    for (ui node : component) {
        if (node < in_component.size()) in_component[node] = 1;
    }

    std::vector<std::vector<ui>> view_candidate_nodes(qn);
    double max_cs = 1.0;
    double sum_cs = 0.0;
    for (ui u = 0; u < qn; ++u) {
        for (ui node : stats.candidate_nodes[u]) {
            if (node < in_component.size() && in_component[node]) {
                view_candidate_nodes[u].push_back(node);
            }
        }
        if (view_candidate_nodes[u].empty()) {
            features.aux_log_tree_count = 0.0;
            return features;
        }
        double cs = static_cast<double>(view_candidate_nodes[u].size());
        max_cs = std::max(max_cs, cs);
        sum_cs += cs;
    }
    double mean_cs = qn > 0 ? sum_cs / static_cast<double>(qn) : 0.0;

    double max_query_degree = 1.0;
    for (ui u = 0; u < qn; ++u) {
        max_query_degree = std::max(max_query_degree, static_cast<double>(query_graph->getVertexDegree(u)));
    }
    double max_depth = 1.0;
    for (int depth : stats.tree_depth_by_node) {
        max_depth = std::max(max_depth, static_cast<double>(depth));
    }

    features.query_node_features.reserve(qn);
    std::unordered_map<unsigned long long, double> view_edge_density_map;
    std::vector<std::pair<int, int>> query_edges;
    std::vector<double> query_edge_densities;
    auto edge_key = [](int a, int b) -> unsigned long long {
        int x = std::min(a, b);
        int y = std::max(a, b);
        return (static_cast<unsigned long long>(static_cast<unsigned int>(x)) << 32)
            | static_cast<unsigned int>(y);
    };
    std::unordered_set<unsigned long long> tree_edge_keys;
    for (size_t original_idx = 0; original_idx < stats.query_edge_list.size() && original_idx < stats.tree_edge_mask.size(); ++original_idx) {
        if (stats.tree_edge_mask[original_idx] > 0.0) {
            tree_edge_keys.insert(edge_key(stats.query_edge_list[original_idx].first, stats.query_edge_list[original_idx].second));
        }
    }
    auto build_node_set = [](const std::vector<ui>& nodes) {
        std::unordered_set<ui> result;
        result.reserve(nodes.size() * 2 + 1);
        for (ui node : nodes) result.insert(node);
        return result;
    };
    std::vector<std::unordered_set<ui>> view_candidate_sets;
    view_candidate_sets.reserve(qn);
    for (ui u = 0; u < qn; ++u) {
        view_candidate_sets.push_back(build_node_set(view_candidate_nodes[u]));
    }

    for (ui u = 0; u < qn; ++u) {
        ui neighbor_count = 0;
        const VertexID* neighbors = query_graph->getVertexNeighbors(u, neighbor_count);
        for (ui ni = 0; ni < neighbor_count; ++ni) {
            ui v = neighbors[ni];
            if (u > v) continue;
            double edge_count = 0.0;
            for (ui left : view_candidate_nodes[u]) {
                if (left >= stats.candidate_neighbors.size()) continue;
                for (ui right : stats.candidate_neighbors[left]) {
                    if (view_candidate_sets[v].find(right) != view_candidate_sets[v].end()) {
                        edge_count += 1.0;
                    }
                }
            }
            double denom = static_cast<double>(std::max<size_t>(view_candidate_nodes[u].size(), 1))
                * static_cast<double>(std::max<size_t>(view_candidate_nodes[v].size(), 1));
            double density = edge_count / denom;
            query_edges.emplace_back(static_cast<int>(u), static_cast<int>(v));
            query_edge_densities.push_back(density);
            view_edge_density_map[edge_key(static_cast<int>(u), static_cast<int>(v))] = density;
        }
    }

    for (ui u = 0; u < qn; ++u) {
        double cs = static_cast<double>(view_candidate_nodes[u].size());
        double degree = static_cast<double>(query_graph->getVertexDegree(u));
        double tree_depth = u < stats.tree_depth_by_node.size() ? static_cast<double>(stats.tree_depth_by_node[u]) : 0.0;
        double leaf_mask = u < stats.tree_leaf_mask.size() ? stats.tree_leaf_mask[u] : 0.0;
        double parent_log_density = 0.0;
        if (u < stats.tree_parent.size() && stats.tree_parent[u] >= 0) {
            parent_log_density = safe_log(view_edge_density_map[edge_key(static_cast<int>(u), stats.tree_parent[u])]);
        }
        features.query_node_features.push_back({
            safe_log(cs),
            cs / max_cs,
            degree / max_query_degree,
            static_cast<int>(u) == stats.tree_root ? 1.0 : 0.0,
            leaf_mask,
            tree_depth / max_depth,
            parent_log_density,
        });
    }

    std::vector<double> membership_count(stats.num_candidates, 0.0);
    std::vector<double> root_overlap(stats.num_candidates, 0.0);
    std::vector<double> tree_support(stats.num_candidates, 0.0);
    for (ui u = 0; u < qn; ++u) {
        double cluster_size = std::max<double>(static_cast<double>(view_candidate_nodes[u].size()), 1.0);
        for (ui node : view_candidate_nodes[u]) {
            membership_count[node] += 1.0;
            if (static_cast<int>(u) == stats.tree_root) {
                root_overlap[node] += 1.0 / cluster_size;
            }
        }
    }
    for (size_t edge_idx = 0; edge_idx < query_edges.size(); ++edge_idx) {
        int left_query = query_edges[edge_idx].first;
        int right_query = query_edges[edge_idx].second;
        if (tree_edge_keys.find(edge_key(left_query, right_query)) == tree_edge_keys.end()) continue;
        double left_weight = 1.0 / std::max<double>(static_cast<double>(view_candidate_nodes[left_query].size()), 1.0);
        double right_weight = 1.0 / std::max<double>(static_cast<double>(view_candidate_nodes[right_query].size()), 1.0);
        for (ui node : view_candidate_nodes[left_query]) tree_support[node] += left_weight;
        for (ui node : view_candidate_nodes[right_query]) tree_support[node] += right_weight;
    }

    double max_membership = 1.0;
    double max_candidate_degree = 1.0;
    double max_data_degree = 1.0;
    double max_tree_support = 1.0;
    std::vector<double> view_candidate_degree(stats.num_candidates, 0.0);
    for (ui node : component) {
        if (node >= stats.candidate_neighbors.size()) continue;
        double degree = 0.0;
        for (ui neighbor : stats.candidate_neighbors[node]) {
            if (neighbor < in_component.size() && in_component[neighbor]) degree += 1.0;
        }
        view_candidate_degree[node] = degree;
        max_membership = std::max(max_membership, membership_count[node]);
        max_candidate_degree = std::max(max_candidate_degree, degree);
        if (node < stats.candidate_data_degrees.size()) {
            max_data_degree = std::max(max_data_degree, stats.candidate_data_degrees[node]);
        }
        max_tree_support = std::max(max_tree_support, tree_support[node]);
    }

    features.data_node_features.reserve(component.size());
    for (ui node : component) {
        double candidate_degree = node < view_candidate_degree.size() ? view_candidate_degree[node] : 0.0;
        double data_degree = node < stats.candidate_data_degrees.size() ? stats.candidate_data_degrees[node] : candidate_degree;
        features.data_node_features.push_back({
            safe_log(membership_count[node]),
            membership_count[node] / max_membership,
            safe_log(candidate_degree),
            candidate_degree / max_candidate_degree,
            data_degree / max_data_degree,
            root_overlap[node],
            tree_support[node] / max_tree_support,
        });
    }

    std::vector<std::vector<double>> dp(qn);
    for (int order_idx = static_cast<int>(stats.tree_order.size()) - 1; order_idx >= 0; --order_idx) {
        ui u = static_cast<ui>(stats.tree_order[order_idx]);
        dp[u].assign(view_candidate_nodes[u].size(), 1.0);
        for (ui v = 0; v < qn; ++v) {
            if (stats.tree_parent[v] != static_cast<int>(u)) continue;
            std::unordered_map<ui, double> child_dp_by_node;
            child_dp_by_node.reserve(view_candidate_nodes[v].size() * 2 + 1);
            for (ui j = 0; j < view_candidate_nodes[v].size(); ++j) {
                child_dp_by_node[view_candidate_nodes[v][j]] += dp[v][j];
            }
            for (ui i = 0; i < view_candidate_nodes[u].size(); ++i) {
                double child_sum = 0.0;
                ui left = view_candidate_nodes[u][i];
                if (left < stats.candidate_neighbors.size()) {
                    for (ui right : stats.candidate_neighbors[left]) {
                        auto found = child_dp_by_node.find(right);
                        if (found != child_dp_by_node.end()) child_sum += found->second;
                    }
                }
                dp[u][i] *= child_sum;
            }
        }
    }
    double total_trees = 0.0;
    if (stats.tree_root >= 0 && static_cast<ui>(stats.tree_root) < dp.size()) {
        for (double value : dp[static_cast<ui>(stats.tree_root)]) total_trees += value;
    }
    double log_candidate_tree_count = total_trees > 0.0 ? std::log(total_trees) : 0.0;

    std::vector<double> tree_logs;
    std::vector<double> non_tree_logs;
    for (size_t i = 0; i < query_edges.size(); ++i) {
        bool is_tree_edge = tree_edge_keys.find(edge_key(query_edges[i].first, query_edges[i].second)) != tree_edge_keys.end();
        double log_density = safe_log(query_edge_densities[i]);
        if (is_tree_edge) tree_logs.push_back(log_density);
        else non_tree_logs.push_back(log_density);
    }
    auto mean = [](const std::vector<double>& values) -> double {
        if (values.empty()) return 0.0;
        double sum = 0.0;
        for (double value : values) sum += value;
        return sum / static_cast<double>(values.size());
    };
    double tree_density_log_mean = mean(tree_logs);
    double non_tree_density_log_mean = non_tree_logs.empty() ? tree_density_log_mean : mean(non_tree_logs);
    double density_log_gap = non_tree_density_log_mean - tree_density_log_mean;

    double query_density = 0.0;
    if (qn > 1) {
        query_density = (2.0 * static_cast<double>(query_edge_count))
            / std::max<double>(static_cast<double>(qn) * static_cast<double>(qn - 1), 1.0);
    }
    features.global_features = {
        safe_log(static_cast<double>(qn)),
        safe_log(static_cast<double>(std::max<ui>(query_edge_count, 1))),
        query_density,
        safe_log(max_query_degree),
        safe_log(sum_cs),
        safe_log(mean_cs),
        safe_log(max_cs),
        safe_log(max_data_degree),
        log_candidate_tree_count,
        tree_density_log_mean,
        non_tree_density_log_mean,
        density_log_gap,
    };
    features.tree_order_ids = stats.tree_order;
    features.aux_log_tree_count = log_candidate_tree_count;
    return features;
}

struct Final11dFeatures {
    double candidate_tree_log_count = 0.0;
    double candidate_domain_size_dispersion_mean = 0.0;
    double sum_support_concentration = 0.0;
    double candidate_relation_cycle_path_log_gap_mean = 0.0;
    double candidate_tree_path_non_tree_log_gap_mean = 0.0;
    double probe_edge_feasible_sample_neglog_survival = 0.0;
    double label_injectivity_product_log_sum = 0.0;
    double repeated_label_pair_collision_log1p = 0.0;
    double probe_injective_sample_neglog_survival = 0.0;
    double probe_full_match_sample_neglog_survival = 0.0;
    double probe_graph_estimate_tree_log_gap = 0.0;
    double probe_graph_estimate_signed_used_sample_fraction = 0.0;
    double positive_sample_fraction_weighted_graph_gap_ratio = 0.0;

    double supported_graph_anchor_context_log1p = 0.0;
    double support_concentration = 0.0;
    double leaf_frac = 0.0;
    double bridge_frac = 0.0;
    double repeat_internal_density_max = 0.0;
    double repeat_neighbor_label_similarity_max = 0.0;
    double wl1_collision_excess_frac = 0.0;
    double repeated_internal_group_count_log1p = 0.0;
    double motif_hierarchy_gap = 0.0;
    double context_alias = 0.0;
    double distributed_closure_support = 0.0;
    double fallback_anchor_log = 0.0;
};

double final11d_support_concentration(const QueryStats& stats) {
    if (stats.num_candidates == 0) return 0.0;
    std::vector<double> membership_count(stats.num_candidates, 0.0);
    std::vector<double> tree_support(stats.num_candidates, 0.0);
    for (const auto& nodes : stats.candidate_nodes) {
        for (ui node : nodes) {
            if (node < membership_count.size()) {
                membership_count[node] += 1.0;
            }
        }
    }
    for (size_t edge_idx = 0;
         edge_idx < stats.query_edge_list.size()
            && edge_idx < stats.tree_edge_mask.size();
         ++edge_idx) {
        if (stats.tree_edge_mask[edge_idx] <= 0.0) continue;
        const int left_query = stats.query_edge_list[edge_idx].first;
        const int right_query = stats.query_edge_list[edge_idx].second;
        if (left_query < 0 || right_query < 0
            || static_cast<size_t>(left_query) >= stats.candidate_nodes.size()
            || static_cast<size_t>(right_query) >= stats.candidate_nodes.size()) {
            continue;
        }
        const auto& left_nodes =
            stats.candidate_nodes[static_cast<size_t>(left_query)];
        const auto& right_nodes =
            stats.candidate_nodes[static_cast<size_t>(right_query)];
        const double left_weight =
            1.0 / std::max<double>(left_nodes.size(), 1.0);
        const double right_weight =
            1.0 / std::max<double>(right_nodes.size(), 1.0);
        for (ui node : left_nodes) {
            if (node < tree_support.size()) {
                tree_support[node] += left_weight;
            }
        }
        for (ui node : right_nodes) {
            if (node < tree_support.size()) {
                tree_support[node] += right_weight;
            }
        }
    }

    double max_tree_support = 1.0;
    for (double value : tree_support) {
        max_tree_support = std::max(max_tree_support, value);
    }
    std::vector<double> membership_logs;
    std::vector<double> candidate_degree_logs;
    std::vector<double> tree_support_fractions;
    membership_logs.reserve(stats.num_candidates);
    candidate_degree_logs.reserve(stats.num_candidates);
    tree_support_fractions.reserve(stats.num_candidates);
    for (ui node = 0; node < stats.num_candidates; ++node) {
        membership_logs.push_back(safe_log(membership_count[node]));
        const double degree =
            node < stats.candidate_node_degrees.size()
            ? stats.candidate_node_degrees[node]
            : 0.0;
        candidate_degree_logs.push_back(safe_log(degree));
        tree_support_fractions.push_back(
            tree_support[node] / std::max(max_tree_support, 1.0)
        );
    }
    const double membership_concentration =
        max_or_zero(membership_logs) - mean_or_zero(membership_logs);
    const double degree_concentration =
        max_or_zero(candidate_degree_logs)
        - mean_or_zero(candidate_degree_logs);
    const double tree_support_concentration =
        max_or_zero(tree_support_fractions)
        - mean_or_zero(tree_support_fractions);
    return (
        membership_concentration
        + degree_concentration
        + tree_support_concentration
    ) / 3.0;
}

Final11dFeatures build_final11d_features(
    const Graph* query_graph,
    const QueryStats& stats,
    long long configured_probe_budget
) {
    Final11dFeatures features;
    const ui qn = query_graph->getVerticesCount();
    const ui query_edge_count = query_graph->getEdgesCount();

    std::vector<double> cs_values;
    cs_values.reserve(stats.candidate_nodes.size());
    for (const auto& nodes : stats.candidate_nodes) {
        cs_values.push_back(static_cast<double>(nodes.size()));
    }
    const double cs_mean = mean_or_zero(cs_values);
    const double cs_std = std_or_zero(cs_values);
    const double cs_cv = cs_std / std::max(cs_mean, 1.0);

    std::vector<int> discovery(qn, -1);
    std::vector<int> low(qn, -1);
    int discovery_clock = 0;
    int bridge_count = 0;
    std::function<void(ui, int)> bridge_dfs = [&](ui u, int parent) {
        discovery[u] = discovery_clock;
        low[u] = discovery_clock;
        discovery_clock += 1;
        ui neighbor_count = 0;
        const VertexID* neighbors =
            query_graph->getVertexNeighbors(u, neighbor_count);
        for (ui index = 0; index < neighbor_count; ++index) {
            const ui v = neighbors[index];
            if (static_cast<int>(v) == parent) continue;
            if (discovery[v] < 0) {
                bridge_dfs(v, static_cast<int>(u));
                low[u] = std::min(low[u], low[v]);
                if (low[v] > discovery[u]) {
                    bridge_count += 1;
                }
            }
            else {
                low[u] = std::min(low[u], discovery[v]);
            }
        }
    };
    for (ui u = 0; u < qn; ++u) {
        if (discovery[u] < 0) bridge_dfs(u, -1);
    }

    std::unordered_map<LabelID, std::vector<ui>> vertices_by_label;
    vertices_by_label.reserve(qn * 2 + 1);
    std::vector<std::unordered_map<LabelID, int>> neighbor_label_counts(qn);
    int leaf_count = 0;
    for (ui u = 0; u < qn; ++u) {
        vertices_by_label[query_graph->getVertexLabel(u)].push_back(u);
        ui neighbor_count = 0;
        const VertexID* neighbors =
            query_graph->getVertexNeighbors(u, neighbor_count);
        if (neighbor_count == 1) leaf_count += 1;
        auto& signature = neighbor_label_counts[u];
        signature.reserve(neighbor_count * 2 + 1);
        for (ui index = 0; index < neighbor_count; ++index) {
            signature[query_graph->getVertexLabel(neighbors[index])] += 1;
        }
    }

    auto signature_similarity = [](
        const std::unordered_map<LabelID, int>& left,
        const std::unordered_map<LabelID, int>& right
    ) -> double {
        double intersection = 0.0;
        double union_sum = 0.0;
        for (const auto& item : left) {
            const auto found = right.find(item.first);
            const int right_value =
                found == right.end() ? 0 : found->second;
            intersection += std::min(item.second, right_value);
            union_sum += std::max(item.second, right_value);
        }
        for (const auto& item : right) {
            if (left.find(item.first) == left.end()) {
                union_sum += item.second;
            }
        }
        return union_sum > 0.0 ? intersection / union_sum : 0.0;
    };

    double repeat_internal_density_max = 0.0;
    double repeat_neighbor_similarity_max = 0.0;
    int repeated_internal_group_count = 0;
    for (const auto& item : vertices_by_label) {
        const auto& vertices = item.second;
        if (vertices.size() < 2) continue;
        int internal_edges = 0;
        for (size_t left = 0; left < vertices.size(); ++left) {
            for (size_t right = left + 1; right < vertices.size(); ++right) {
                if (query_graph->checkEdgeExistence(
                        vertices[left],
                        vertices[right])) {
                    internal_edges += 1;
                }
                repeat_neighbor_similarity_max = std::max(
                    repeat_neighbor_similarity_max,
                    signature_similarity(
                        neighbor_label_counts[vertices[left]],
                        neighbor_label_counts[vertices[right]]
                    )
                );
            }
        }
        const double possible =
            static_cast<double>(vertices.size())
            * static_cast<double>(vertices.size() - 1) / 2.0;
        repeat_internal_density_max = std::max(
            repeat_internal_density_max,
            possible > 0.0
            ? static_cast<double>(internal_edges) / possible
            : 0.0
        );
        if (internal_edges > 0) repeated_internal_group_count += 1;
    }
    repeated_internal_group_count =
        std::min(repeated_internal_group_count, 8);

    std::unordered_map<std::string, int> wl1_signature_counts;
    wl1_signature_counts.reserve(qn * 2 + 1);
    for (ui u = 0; u < qn; ++u) {
        std::vector<std::pair<LabelID, int>> entries(
            neighbor_label_counts[u].begin(),
            neighbor_label_counts[u].end()
        );
        std::sort(entries.begin(), entries.end());
        std::ostringstream signature;
        signature << query_graph->getVertexLabel(u) << '|';
        for (const auto& entry : entries) {
            signature << entry.first << ':' << entry.second << ',';
        }
        wl1_signature_counts[signature.str()] += 1;
    }
    int wl1_collision_excess = 0;
    for (const auto& item : wl1_signature_counts) {
        wl1_collision_excess += std::max(item.second - 1, 0);
    }

    features.candidate_tree_log_count =
        stats.log_candidate_tree_count;
    std::vector<double> cs_logs;
    cs_logs.reserve(cs_values.size());
    for (double value : cs_values) {
        cs_logs.push_back(safe_log(value));
    }
    features.candidate_domain_size_dispersion_mean =
        0.5 * (cs_cv + std_or_zero(cs_logs));
    features.probe_edge_feasible_sample_neglog_survival =
        -safe_log(stats.tree_sample_all_edge_success_rate);
    features.probe_injective_sample_neglog_survival =
        -safe_log(stats.tree_sample_injective_success_rate);
    features.probe_full_match_sample_neglog_survival =
        -safe_log(stats.tree_sample_exact_success_rate);
    features.candidate_relation_cycle_path_log_gap_mean =
        stats.cycle_path_log_gap_mean;
    features.candidate_tree_path_non_tree_log_gap_mean =
        stats.cycle_path_tree_pair_log_gap_mean;
    features.label_injectivity_product_log_sum =
        stats.label_injectivity_product_log_sum;
    features.repeated_label_pair_collision_log1p =
        stats.label_injectivity_pair_collision_log1p;

    const bool graph_positive =
        stats.probe_fastest2_graph_estimate_count > 0.0;
    features.probe_graph_estimate_tree_log_gap =
        graph_positive
        ? stats.log_candidate_tree_count
            - safe_log(stats.probe_fastest2_graph_estimate_count)
        : 0.0;
    const double budget_denom = std::max<double>(
        static_cast<double>(configured_probe_budget),
        1.0
    );
    const double used_fraction = std::min(
        1.0,
        std::max(
            0.0,
            static_cast<double>(stats.probe_fastest2_used_samples)
                / budget_denom
        )
    );
    features.probe_graph_estimate_signed_used_sample_fraction =
        (graph_positive ? 1.0 : -1.0) * used_fraction;
    features.positive_sample_fraction_weighted_graph_gap_ratio =
        std::max(
            features.probe_graph_estimate_signed_used_sample_fraction,
            0.0
        )
        * features.probe_graph_estimate_tree_log_gap
        / std::max(features.candidate_tree_log_count, 1.0);

    features.support_concentration =
        final11d_support_concentration(stats);
    features.sum_support_concentration =
        features.support_concentration;
    features.leaf_frac =
        static_cast<double>(leaf_count) / std::max<double>(qn, 1.0);
    features.bridge_frac =
        static_cast<double>(bridge_count)
        / std::max<double>(query_edge_count, 1.0);
    features.repeat_internal_density_max =
        repeat_internal_density_max;
    features.repeat_neighbor_label_similarity_max =
        repeat_neighbor_similarity_max;
    features.wl1_collision_excess_frac =
        static_cast<double>(wl1_collision_excess)
        / std::max<double>(qn, 1.0);
    features.repeated_internal_group_count_log1p =
        std::log1p(static_cast<double>(repeated_internal_group_count));
    features.motif_hierarchy_gap =
        stats.triangle_joint_domain_to_exact_gap_mean
        + stats.motif4_joint_domain_to_exact_gap_mean;
    const double symmetry_periphery =
        features.repeat_neighbor_label_similarity_max
        * (features.leaf_frac + features.bridge_frac)
        + features.wl1_collision_excess_frac;
    const double repeat_group_sparse_density =
        features.repeated_internal_group_count_log1p
        * (1.0 - features.repeat_internal_density_max);
    features.context_alias =
        symmetry_periphery * features.motif_hierarchy_gap
        + repeat_group_sparse_density;
    const double non_tree_floor =
        features.context_alias * stats.non_tree_edge_log_count_min;
    features.distributed_closure_support =
        non_tree_floor / (0.25 + features.support_concentration);
    features.fallback_anchor_log =
        features.candidate_tree_log_count
        - features.probe_graph_estimate_tree_log_gap;
    features.supported_graph_anchor_context_log1p = std::log1p(
        std::max(
            features.fallback_anchor_log
            + features.context_alias
                * features.distributed_closure_support,
            0.0
        )
    );
    return features;
}

Final11dFeatures build_clean3224_final11d_features(
    const QueryStats& stats,
    long long configured_probe_budget
) {
    Final11dFeatures features;
    std::vector<double> candidate_domain_sizes;
    candidate_domain_sizes.reserve(stats.candidate_nodes.size());
    for (const auto& nodes : stats.candidate_nodes) {
        candidate_domain_sizes.push_back(
            static_cast<double>(nodes.size())
        );
    }
    const double domain_mean =
        mean_or_zero(candidate_domain_sizes);
    const double domain_cv =
        std_or_zero(candidate_domain_sizes)
        / std::max(domain_mean, 1.0);
    std::vector<double> domain_logs;
    domain_logs.reserve(candidate_domain_sizes.size());
    for (double value : candidate_domain_sizes) {
        domain_logs.push_back(safe_log(value));
    }

    features.candidate_tree_log_count =
        stats.log_candidate_tree_count;
    features.candidate_domain_size_dispersion_mean =
        0.5 * (domain_cv + std_or_zero(domain_logs));
    features.sum_support_concentration =
        final11d_support_concentration(stats);
    features.candidate_relation_cycle_path_log_gap_mean =
        stats.cycle_path_log_gap_mean;
    features.candidate_tree_path_non_tree_log_gap_mean =
        stats.cycle_path_tree_pair_log_gap_mean;
    features.probe_edge_feasible_sample_neglog_survival =
        -safe_log(stats.tree_sample_all_edge_success_rate);
    features.label_injectivity_product_log_sum =
        stats.label_injectivity_product_log_sum;
    features.repeated_label_pair_collision_log1p =
        stats.label_injectivity_pair_collision_log1p;
    features.probe_injective_sample_neglog_survival =
        -safe_log(stats.tree_sample_injective_success_rate);
    features.probe_full_match_sample_neglog_survival =
        -safe_log(stats.tree_sample_exact_success_rate);

    const bool graph_positive =
        stats.probe_fastest2_graph_estimate_count > 0.0;
    features.probe_graph_estimate_tree_log_gap =
        graph_positive
        ? stats.log_candidate_tree_count
            - safe_log(stats.probe_fastest2_graph_estimate_count)
        : 0.0;
    const double used_fraction = std::min(
        1.0,
        std::max(
            0.0,
            static_cast<double>(stats.probe_fastest2_used_samples)
                / std::max<double>(
                    static_cast<double>(configured_probe_budget),
                    1.0
                )
        )
    );
    features.probe_graph_estimate_signed_used_sample_fraction =
        (graph_positive ? 1.0 : -1.0) * used_fraction;
    features.positive_sample_fraction_weighted_graph_gap_ratio =
        std::max(
            features.probe_graph_estimate_signed_used_sample_fraction,
            0.0
        )
        * features.probe_graph_estimate_tree_log_gap
        / std::max(features.candidate_tree_log_count, 1.0);
    return features;
}

QueryStats build_query_stats(
    const Graph* data_graph,
    const Graph* query_graph,
    ui** candidates,
    ui* candidates_count,
    const FilterVertices::CandidateNeighborMaps* candidate_neighbors_from_filter = nullptr,
    long long full_match_probe_budget = 500000LL,
    double full_match_probe_count_cap = 1000.0,
    int tree_sample_count = 0,
    unsigned int tree_sample_seed = 42,
    const std::string& tree_sample_strategy = "weighted",
    const std::string& candidate_tree_strategy = "mwst_low_density_edges",
    int core_max_vertices = 10,
    const std::string& core_policy = "cycle_repeat",
    bool enable_tree_dp_weighted_closure = true,
    bool enable_factor_bp = true,
    bool enable_factor_bp_injective = true,
    bool enable_core_outside = true,
    bool minimal_probe_features = false,
    bool stage5_probe_only = false,
    long long probe_fastest2_cap = 0,
    int probe_fastest2_ub_initial = 100000,
    int probe_fastest2_success_threshold = 10,
    double probe_fastest2_strata_ratio = 0.5,
    unsigned int probe_fastest2_seed = 43,
    bool enable_dense_clique_feature = false,
    long long cycle_path_budget = 5000000LL,
    bool final11d_features_only = false
) {
    QueryStats stats;
    stats.candidate_tree_strategy = candidate_tree_strategy;
    const ui qn = query_graph->getVerticesCount();
    stats.tree_parent.assign(qn, -1);
    stats.tree_depth_by_node.assign(qn, 0);
    stats.tree_child_count_by_node.assign(qn, 0);
    stats.tree_leaf_mask.assign(qn, 0.0);
    stats.tree_parent_log_density.assign(qn, 0.0);

    auto section_start = Clock::now();
    std::unordered_map<VertexID, ui> reindex;
    std::vector<VertexID> original_by_reindex;
    for (ui u = 0; u < qn; ++u) {
        for (ui i = 0; i < candidates_count[u]; ++i) {
            VertexID v = candidates[u][i];
            if (reindex.find(v) == reindex.end()) {
                ui idx = static_cast<ui>(original_by_reindex.size());
                reindex[v] = idx;
                original_by_reindex.push_back(v);
            }
        }
    }
    std::sort(original_by_reindex.begin(), original_by_reindex.end());
    reindex.clear();
    for (ui idx = 0; idx < original_by_reindex.size(); ++idx) {
        reindex[original_by_reindex[idx]] = idx;
    }
    stats.num_candidates = static_cast<ui>(original_by_reindex.size());
    stats.candidate_nodes.resize(qn);
    std::vector<std::vector<ui>> candidate_local_by_input_order(qn);
    std::vector<std::unordered_map<VertexID, ui>> candidate_original_pos(qn);
    for (ui u = 0; u < qn; ++u) {
        stats.candidate_nodes[u].reserve(candidates_count[u]);
        candidate_local_by_input_order[u].reserve(candidates_count[u]);
        candidate_original_pos[u].reserve(candidates_count[u] * 2 + 1);
        for (ui i = 0; i < candidates_count[u]; ++i) {
            VertexID original_id = candidates[u][i];
            ui local_id = reindex[original_id];
            stats.candidate_nodes[u].push_back(local_id);
            candidate_local_by_input_order[u].push_back(local_id);
            candidate_original_pos[u][original_id] = i;
        }
        std::sort(stats.candidate_nodes[u].begin(), stats.candidate_nodes[u].end());
    }
    stats.timing_reindex_seconds = seconds_since(section_start);

    section_start = Clock::now();
    std::vector<std::unordered_set<ui>> candidate_adjacency(stats.num_candidates);
    std::unordered_set<unsigned long long> tree_edge_keys;
    std::unordered_map<unsigned long long, double> edge_density_map;
    auto edge_key = [](int a, int b) -> unsigned long long {
        int x = std::min(a, b);
        int y = std::max(a, b);
        return (static_cast<unsigned long long>(static_cast<unsigned int>(x)) << 32)
            | static_cast<unsigned int>(y);
    };

    for (ui u = 0; u < qn; ++u) {
        ui neighbor_count = 0;
        const VertexID* neighbors = query_graph->getVertexNeighbors(u, neighbor_count);
        for (ui ni = 0; ni < neighbor_count; ++ni) {
            ui v = neighbors[ni];
            if (u > v) continue;
            double edge_count = 0.0;
            std::unordered_set<unsigned long long> edge_candidate_pairs;
            std::vector<double> left_support;
            std::vector<double> right_support;
            if (!final11d_features_only) {
                left_support.assign(candidates_count[u], 0.0);
                right_support.assign(candidates_count[v], 0.0);
            }
            const auto& right_pos = candidate_original_pos[v];
            if (candidate_neighbors_from_filter != nullptr) {
                const auto& neighbor_map = (*candidate_neighbors_from_filter)[u];
                for (ui i = 0; i < candidates_count[u]; ++i) {
                    VertexID du = candidates[u][i];
                    auto found_neighbors = neighbor_map.find(du);
                    if (found_neighbors == neighbor_map.end()) continue;
                    ui ru = candidate_local_by_input_order[u][i];
                    for (VertexID dv : found_neighbors->second) {
                        if (du == dv) continue;
                        auto right_found = right_pos.find(dv);
                        if (right_found == right_pos.end()) continue;
                        ui rv = candidate_local_by_input_order[v][right_found->second];
                        candidate_adjacency[ru].insert(rv);
                        candidate_adjacency[rv].insert(ru);
                        edge_candidate_pairs.insert(directed_candidate_pair_key(ru, rv));
                        edge_candidate_pairs.insert(directed_candidate_pair_key(rv, ru));
                        edge_count += 1.0;
                        if (!final11d_features_only) {
                            left_support[i] += 1.0;
                            right_support[right_found->second] += 1.0;
                        }
                    }
                }
            }
            else {
                for (ui i = 0; i < candidates_count[u]; ++i) {
                    VertexID du = candidates[u][i];
                    ui ru = candidate_local_by_input_order[u][i];
                    ui data_neighbor_count = 0;
                    const VertexID* data_neighbors = data_graph->getVertexNeighbors(du, data_neighbor_count);
                    for (ui j = 0; j < data_neighbor_count; ++j) {
                        VertexID dv = data_neighbors[j];
                        if (du == dv) continue;
                        auto right_found = right_pos.find(dv);
                        if (right_found == right_pos.end()) continue;
                        ui rv = candidate_local_by_input_order[v][right_found->second];
                        candidate_adjacency[ru].insert(rv);
                        candidate_adjacency[rv].insert(ru);
                        edge_candidate_pairs.insert(directed_candidate_pair_key(ru, rv));
                        edge_candidate_pairs.insert(directed_candidate_pair_key(rv, ru));
                        edge_count += 1.0;
                        if (!final11d_features_only) {
                            left_support[i] += 1.0;
                            right_support[right_found->second] += 1.0;
                        }
                    }
                }
            }
            stats.query_edge_list.emplace_back(static_cast<int>(u), static_cast<int>(v));
            stats.query_edge_candidate_pair_sets.push_back(std::move(edge_candidate_pairs));
            stats.query_edge_candidate_counts.push_back(edge_count);
            const double denom =
                static_cast<double>(
                    std::max<ui>(candidates_count[u], 1)
                )
                * static_cast<double>(
                    std::max<ui>(candidates_count[v], 1)
                );
            const double density = edge_count / denom;
            stats.query_edge_densities.push_back(density);
            if (!final11d_features_only) {
                auto append_endpoint_support_stats = [](
                    const std::vector<double>& counts,
                    std::vector<double>& nonzero_fracs,
                    std::vector<double>& log_mins,
                    std::vector<double>& log_means,
                    std::vector<double>& log_stds,
                    std::vector<double>& log_maxs
                ) {
                    std::vector<double> logs;
                    logs.reserve(counts.size());
                    double nonzero = 0.0;
                    for (double count : counts) {
                        if (count > 0.0) nonzero += 1.0;
                        logs.push_back(safe_log1p(count));
                    }
                    nonzero_fracs.push_back(
                        counts.empty()
                        ? 0.0
                        : nonzero / static_cast<double>(counts.size())
                    );
                    log_mins.push_back(min_or_zero(logs));
                    log_means.push_back(mean_or_zero(logs));
                    log_stds.push_back(std_or_zero(logs));
                    log_maxs.push_back(max_or_zero(logs));
                };
                append_endpoint_support_stats(
                    left_support,
                    stats.query_edge_left_support_nonzero_frac,
                    stats.query_edge_left_log_support_min,
                    stats.query_edge_left_log_support_mean,
                    stats.query_edge_left_log_support_std,
                    stats.query_edge_left_log_support_max
                );
                append_endpoint_support_stats(
                    right_support,
                    stats.query_edge_right_support_nonzero_frac,
                    stats.query_edge_right_log_support_min,
                    stats.query_edge_right_log_support_mean,
                    stats.query_edge_right_log_support_std,
                    stats.query_edge_right_log_support_max
                );
                edge_density_map[
                    edge_key(static_cast<int>(u), static_cast<int>(v))
                ] = density;
            }
        }
    }
    stats.timing_candidate_adjacency_seconds = seconds_since(section_start);

    auto compute_components = [&]() {
        std::vector<char> visited(stats.num_candidates, 0);
        std::vector<std::vector<ui>> components;
        for (ui start = 0; start < stats.num_candidates; ++start) {
            if (visited[start]) continue;
            components.emplace_back();
            std::queue<ui> component_queue;
            visited[start] = 1;
            component_queue.push(start);
            while (!component_queue.empty()) {
                ui current = component_queue.front();
                component_queue.pop();
                components.back().push_back(current);
                for (ui neighbor : candidate_adjacency[current]) {
                    if (visited[neighbor]) continue;
                    visited[neighbor] = 1;
                    component_queue.push(neighbor);
                }
            }
        }
        return components;
    };

    std::vector<std::vector<ui>> components;
    if (!final11d_features_only) {
        components = compute_components();
    }
    stats.components = components;
    stats.component_count_before_prune = static_cast<int>(components.size());
    stats.component_count_after_prune = stats.component_count_before_prune;

    section_start = Clock::now();
    stats.candidate_node_degrees.reserve(candidate_adjacency.size());
    if (!final11d_features_only) {
        stats.candidate_data_degrees.reserve(
            candidate_adjacency.size()
        );
    }
    double directed_edges = 0.0;
    std::vector<double> candidate_degree_logs;
    double candidate_degree_l2_sum = 0.0;
    for (ui local_id = 0; local_id < candidate_adjacency.size(); ++local_id) {
        const auto& neighbors = candidate_adjacency[local_id];
        double candidate_degree = static_cast<double>(neighbors.size());
        stats.candidate_node_degrees.push_back(candidate_degree);
        if (!final11d_features_only) {
            candidate_degree_logs.push_back(
                safe_log1p(candidate_degree)
            );
            candidate_degree_l2_sum +=
                candidate_degree * candidate_degree;
        }
        stats.candidate_neighbors.emplace_back(neighbors.begin(), neighbors.end());
        std::sort(stats.candidate_neighbors.back().begin(), stats.candidate_neighbors.back().end());
        if (!final11d_features_only) {
            stats.candidate_data_degrees.push_back(
                static_cast<double>(data_graph->getVertexDegree(original_by_reindex[local_id]))
            );
        }
        directed_edges += static_cast<double>(neighbors.size());
    }
    stats.candidate_edge_count = directed_edges;
    stats.num_candidate_edges = static_cast<ui>(directed_edges / 2.0);
    if (!final11d_features_only) {
        stats.candidate_degree_log_mean =
            mean_or_zero(candidate_degree_logs);
        stats.candidate_degree_log_std =
            std_or_zero(candidate_degree_logs);
        stats.candidate_degree_log_max =
            max_or_zero(candidate_degree_logs);
        stats.candidate_degree_l2_log =
            safe_log1p(std::sqrt(candidate_degree_l2_sum));
    }
    stats.timing_degree_stats_seconds = seconds_since(section_start);

    section_start = Clock::now();
    ui best_root = select_root_by_candidate_count(
        qn,
        candidates_count,
        false
    );
    if (candidate_tree_strategy == "bfs_mincand_root") {
        best_root = select_root_by_candidate_count(
            qn,
            candidates_count,
            true
        );
        stats.tree_parent = build_bfs_tree_parent(
            query_graph,
            qn,
            best_root,
            stats.tree_order
        );
    }
    else if (candidate_tree_strategy == "bfs_max_query_degree_root") {
        best_root = select_root_by_query_degree(
            query_graph,
            qn,
            false
        );
        stats.tree_parent = build_bfs_tree_parent(
            query_graph,
            qn,
            best_root,
            stats.tree_order
        );
    }
    else if (candidate_tree_strategy == "bfs_min_query_degree_root") {
        best_root = select_root_by_query_degree(
            query_graph,
            qn,
            true
        );
        stats.tree_parent = build_bfs_tree_parent(
            query_graph,
            qn,
            best_root,
            stats.tree_order
        );
    }
    else if (
        candidate_tree_strategy == "mwst_low_density_edges"
        || candidate_tree_strategy == "mwst_low_count_edges"
        || candidate_tree_strategy == "mwst_high_density_edges"
    ) {
        std::vector<double> edge_scores;
        edge_scores.reserve(stats.query_edge_list.size());
        for (size_t i = 0; i < stats.query_edge_list.size(); ++i) {
            const double density =
                i < stats.query_edge_densities.size()
                ? stats.query_edge_densities[i]
                : 0.0;
            const double count =
                i < stats.query_edge_candidate_counts.size()
                ? stats.query_edge_candidate_counts[i]
                : 0.0;
            if (candidate_tree_strategy == "mwst_low_density_edges") {
                edge_scores.push_back(-safe_log(density));
            }
            else if (candidate_tree_strategy == "mwst_low_count_edges") {
                edge_scores.push_back(-safe_log1p(count));
            }
            else {
                edge_scores.push_back(safe_log(density));
            }
        }
        stats.tree_parent = build_weighted_spanning_tree_parent(
            qn,
            stats.query_edge_list,
            edge_scores,
            best_root,
            stats.tree_order
        );
    }
    else if (candidate_tree_strategy == "current_bfs_maxcand") {
        stats.tree_parent = build_bfs_tree_parent(
            query_graph,
            qn,
            best_root,
            stats.tree_order
        );
    }
    else {
        throw std::runtime_error(
            "Unknown candidate tree strategy: "
            + candidate_tree_strategy
        );
    }
    stats.tree_root = static_cast<int>(best_root);
    if (stats.tree_order.size() != qn) {
        throw std::runtime_error(
            "Candidate tree does not cover all query vertices"
        );
    }
    for (int node : stats.tree_order) {
        const int parent = stats.tree_parent[static_cast<ui>(node)];
        if (parent < 0) continue;
        stats.tree_depth_by_node[static_cast<ui>(node)] =
            stats.tree_depth_by_node[static_cast<ui>(parent)] + 1;
        stats.tree_child_count_by_node[static_cast<ui>(parent)] += 1;
        tree_edge_keys.insert(edge_key(node, parent));
    }
    for (ui u = 0; u < qn; ++u) {
        if (stats.tree_child_count_by_node[u] == 0) stats.tree_leaf_mask[u] = 1.0;
        if (!final11d_features_only && stats.tree_parent[u] >= 0) {
            stats.tree_parent_log_density[u] = safe_log(edge_density_map[edge_key(static_cast<int>(u), stats.tree_parent[u])]);
        }
    }
    for (const auto& e : stats.query_edge_list) {
        stats.tree_edge_mask.push_back(tree_edge_keys.count(edge_key(e.first, e.second)) ? 1.0 : 0.0);
    }
    stats.timing_tree_setup_seconds = seconds_since(section_start);

    section_start = Clock::now();
    std::vector<std::vector<double>> dp(qn);
    for (int order_idx = static_cast<int>(stats.tree_order.size()) - 1; order_idx >= 0; --order_idx) {
        ui u = static_cast<ui>(stats.tree_order[order_idx]);
        dp[u].assign(stats.candidate_nodes[u].size(), 1.0);
        for (ui v = 0; v < qn; ++v) {
            if (stats.tree_parent[v] != static_cast<int>(u)) continue;
            std::unordered_map<ui, double> child_dp_by_node;
            child_dp_by_node.reserve(stats.candidate_nodes[v].size() * 2 + 1);
            for (ui j = 0; j < stats.candidate_nodes[v].size(); ++j) {
                child_dp_by_node[stats.candidate_nodes[v][j]] += dp[v][j];
            }
            for (ui i = 0; i < stats.candidate_nodes[u].size(); ++i) {
                double child_sum = 0.0;
                ui left = stats.candidate_nodes[u][i];
                for (ui right : candidate_adjacency[left]) {
                    if (left == right) continue;
                    auto found = child_dp_by_node.find(right);
                    if (found != child_dp_by_node.end()) {
                        child_sum += found->second;
                    }
                }
                dp[u][i] *= child_sum;
            }
        }
    }
    double total_trees = 0.0;
    for (double value : dp[best_root]) total_trees += value;
    stats.log_candidate_tree_count = total_trees > 0.0 ? std::log(total_trees) : 0.0;
    if (!final11d_features_only
        && !dp[best_root].empty()
        && total_trees > 0.0) {
        std::vector<double> root_weights = dp[best_root];
        int positive_count = 0;
        for (double value : root_weights) {
            if (std::isfinite(value) && value > 0.0) positive_count += 1;
        }
        stats.tree_root_positive_candidate_count = positive_count;
        stats.tree_root_positive_candidate_frac =
            static_cast<double>(positive_count) / static_cast<double>(std::max<size_t>(root_weights.size(), 1));
        stats.tree_root_contrib_entropy = normalized_entropy(root_weights);
        stats.tree_root_contrib_gini = gini_coefficient(root_weights);
        std::sort(root_weights.begin(), root_weights.end(), std::greater<double>());
        auto top_frac = [&](size_t k) -> double {
            double subtotal = 0.0;
            for (size_t i = 0; i < std::min(k, root_weights.size()); ++i) {
                subtotal += std::max(root_weights[i], 0.0);
            }
            return subtotal / total_trees;
        };
        stats.tree_root_contrib_top1_frac = top_frac(1);
        stats.tree_root_contrib_top5_frac = top_frac(5);
        stats.tree_root_contrib_top10_frac = top_frac(10);
    }
    stats.timing_tree_dp_seconds = seconds_since(section_start);

    section_start = Clock::now();
    if (!minimal_probe_features && !stage5_probe_only
        && !final11d_features_only && enable_tree_dp_weighted_closure) {
        populate_tree_dp_weighted_closure_stats(
            stats,
            candidate_adjacency,
            10000000LL
        );
    }
    else {
        stats.tree_dp_weighted_closure_disabled = 1;
    }
    if (!minimal_probe_features && !stage5_probe_only
        && !final11d_features_only && enable_factor_bp) {
        populate_factor_bp_stats(
            stats,
            candidate_adjacency,
            10,
            50000000LL
        );
    }
    else {
        stats.factor_bp_disabled = 1;
    }
    if (!minimal_probe_features && !stage5_probe_only
        && !final11d_features_only && enable_factor_bp_injective) {
        populate_factor_bp_injective_stats(
            query_graph,
            stats,
            candidate_adjacency,
            10,
            50000000LL
        );
    }
    else {
        stats.factor_bp_injective_disabled = 1;
    }
    stats.timing_tree_weighted_closure_seconds = seconds_since(section_start);

    if (!minimal_probe_features) {
        section_start = Clock::now();
        populate_tree_sample_success_stats(
            stats,
            dp,
            candidate_adjacency,
            tree_sample_count,
            tree_sample_seed,
            tree_sample_strategy
        );
        stats.timing_tree_sampling_seconds = seconds_since(section_start);

        section_start = Clock::now();
        populate_probe_fastest2_stats(
            query_graph,
            stats,
            probe_fastest2_cap,
            probe_fastest2_ub_initial,
            probe_fastest2_success_threshold,
            probe_fastest2_strata_ratio,
            probe_fastest2_seed
        );
        stats.timing_probe_fastest2_seconds = seconds_since(section_start);
    }

    if (enable_dense_clique_feature) {
        section_start = Clock::now();
        populate_dense_clique_stats(
            query_graph,
            stats,
            5,
            100000LL
        );
        stats.timing_dense_clique_seconds =
            seconds_since(section_start);
    }

    if (stage5_probe_only) {
        return stats;
    }

    if (!minimal_probe_features && !final11d_features_only) {
        stats.tree_prior_variants.clear();
        stats.tree_prior_variants.push_back({"current_bfs_maxcand", stats.log_candidate_tree_count});
        auto add_bfs_variant = [&](const std::string& name, ui root) {
            std::vector<int> variant_order;
            std::vector<int> variant_parent = build_bfs_tree_parent(query_graph, qn, root, variant_order);
            double variant_log_count = compute_tree_log_count(
                stats.candidate_nodes,
                candidate_adjacency,
                variant_parent,
                variant_order,
                root
            );
            stats.tree_prior_variants.push_back({name, variant_log_count});
        };
        add_bfs_variant("bfs_mincand_root", select_root_by_candidate_count(qn, candidates_count, true));
        add_bfs_variant("bfs_max_query_degree_root", select_root_by_query_degree(query_graph, qn, false));
        add_bfs_variant("bfs_min_query_degree_root", select_root_by_query_degree(query_graph, qn, true));
        auto add_weighted_tree_variant = [&](const std::string& name, const std::vector<double>& scores) {
            std::vector<int> variant_order;
            std::vector<int> variant_parent = build_weighted_spanning_tree_parent(
                qn,
                stats.query_edge_list,
                scores,
                best_root,
                variant_order
            );
            double variant_log_count = compute_tree_log_count(
                stats.candidate_nodes,
                candidate_adjacency,
                variant_parent,
                variant_order,
                best_root
            );
            stats.tree_prior_variants.push_back({name, variant_log_count});
        };
        std::vector<double> low_density_scores;
        std::vector<double> low_count_scores;
        std::vector<double> high_density_scores;
        low_density_scores.reserve(stats.query_edge_list.size());
        low_count_scores.reserve(stats.query_edge_list.size());
        high_density_scores.reserve(stats.query_edge_list.size());
        for (size_t i = 0; i < stats.query_edge_list.size(); ++i) {
            double density = i < stats.query_edge_densities.size() ? stats.query_edge_densities[i] : 0.0;
            double count = i < stats.query_edge_candidate_counts.size() ? stats.query_edge_candidate_counts[i] : 0.0;
            low_density_scores.push_back(-safe_log(density));
            low_count_scores.push_back(-safe_log1p(count));
            high_density_scores.push_back(safe_log(density));
        }
        add_weighted_tree_variant("mwst_low_density_edges", low_density_scores);
        add_weighted_tree_variant("mwst_low_count_edges", low_count_scores);
        add_weighted_tree_variant("mwst_high_density_edges", high_density_scores);
    }
    else {
        stats.tree_prior_variants.clear();
        stats.tree_prior_variants.push_back({"current_bfs_maxcand", stats.log_candidate_tree_count});
    }

    section_start = Clock::now();
    if (!final11d_features_only) {
        std::vector<double> tree_logs;
        std::vector<double> non_tree_logs;
        std::vector<double> tree_edge_log_counts;
        std::vector<double> non_tree_edge_log_counts;
        stats.label_pair_log_density_hash_sum.assign(kLabelPairSketchBuckets, 0.0);
        stats.label_pair_log_count_hash_sum.assign(kLabelPairSketchBuckets, 0.0);
        stats.label_pair_tree_log_density_hash_sum.assign(kLabelPairSketchBuckets, 0.0);
        stats.label_pair_non_tree_log_density_hash_sum.assign(kLabelPairSketchBuckets, 0.0);
        stats.label_pair_edge_hash_count.assign(kLabelPairSketchBuckets, 0.0);
        for (size_t i = 0; i < stats.query_edge_list.size(); ++i) {
            const auto& edge = stats.query_edge_list[i];
            double log_density = safe_log(stats.query_edge_densities[i]);
            double log_count = i < stats.query_edge_candidate_counts.size()
                ? safe_log1p(stats.query_edge_candidate_counts[i])
                : 0.0;
            size_t label_bucket = label_pair_bucket(
                query_graph->getVertexLabel(static_cast<ui>(edge.first)),
                query_graph->getVertexLabel(static_cast<ui>(edge.second))
            );
            stats.label_pair_log_density_hash_sum[label_bucket] += log_density;
            stats.label_pair_log_count_hash_sum[label_bucket] += log_count;
            stats.label_pair_edge_hash_count[label_bucket] += 1.0;
            if (stats.tree_edge_mask[i] > 0.0) {
                tree_logs.push_back(log_density);
                tree_edge_log_counts.push_back(log_count);
                stats.label_pair_tree_log_density_hash_sum[label_bucket] += log_density;
            }
            else {
                non_tree_logs.push_back(log_density);
                non_tree_edge_log_counts.push_back(log_count);
                stats.label_pair_non_tree_log_density_hash_sum[label_bucket] += log_density;
            }
        }
        for (double value : tree_logs) stats.tree_density_log_sum += value;
        stats.tree_density_log_mean = mean_or_zero(tree_logs);
        stats.tree_density_log_min = min_or_zero(tree_logs);
        stats.tree_density_log_max = max_or_zero(tree_logs);
        stats.tree_density_log_std = std_or_zero(tree_logs);
        stats.non_tree_density_log_mean = non_tree_logs.empty() ? stats.tree_density_log_mean : mean_or_zero(non_tree_logs);
        stats.non_tree_density_log_min = non_tree_logs.empty() ? stats.tree_density_log_min : min_or_zero(non_tree_logs);
        stats.non_tree_density_log_std = non_tree_logs.empty() ? 0.0 : std_or_zero(non_tree_logs);
        stats.density_log_gap = stats.non_tree_density_log_mean - stats.tree_density_log_mean;
        stats.tree_edge_log_count_mean = mean_or_zero(tree_edge_log_counts);
        stats.tree_edge_log_count_min = min_or_zero(tree_edge_log_counts);
        stats.tree_edge_log_count_std = std_or_zero(tree_edge_log_counts);
        stats.non_tree_edge_log_count_mean = non_tree_edge_log_counts.empty()
            ? stats.tree_edge_log_count_mean
            : mean_or_zero(non_tree_edge_log_counts);
        stats.non_tree_edge_log_count_min = non_tree_edge_log_counts.empty()
            ? stats.tree_edge_log_count_min
            : min_or_zero(non_tree_edge_log_counts);
        stats.non_tree_edge_log_count_std = non_tree_edge_log_counts.empty()
            ? 0.0
            : std_or_zero(non_tree_edge_log_counts);
    }
    stats.timing_density_summary_seconds = seconds_since(section_start);
    if (!minimal_probe_features) {
        section_start = Clock::now();
        populate_cycle_path_consistency_stats(
            stats,
            cycle_path_budget
        );
        stats.timing_cycle_path_seconds = seconds_since(section_start);

        section_start = Clock::now();
        if (final11d_features_only) {
            populate_final11d_label_injectivity_stats(
                query_graph,
                stats
            );
        }
        else {
            populate_label_injectivity_stats(query_graph, stats);
        }
        stats.timing_label_injectivity_seconds =
            seconds_since(section_start);

        if (!final11d_features_only) {
            section_start = Clock::now();
            populate_triangle_joint_stats(
                query_graph,
                stats,
                5000000LL
            );
            stats.timing_triangle_joint_seconds =
                seconds_since(section_start);

            section_start = Clock::now();
            populate_motif4_joint_stats(
                query_graph,
                stats,
                16,
                2000000LL
            );
            stats.timing_motif4_joint_seconds =
                seconds_since(section_start);
        }
    }
    if (!final11d_features_only) {
        populate_full_match_probe(
            query_graph,
            stats,
            full_match_probe_budget,
            full_match_probe_count_cap
        );
    }
    if (!minimal_probe_features && !final11d_features_only
        && enable_core_outside) {
        populate_core_outside_probe(query_graph, stats, full_match_probe_budget, full_match_probe_count_cap, core_max_vertices, core_policy);
    }
    else {
        stats.core_outside_disabled = 1;
        stats.core_outside_policy = "disabled";
    }
    return stats;
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_graph_path;
    std::string query_list_path;
    std::string out_jsonl_path;
    std::string out_binary_path;
    std::string mode = "pruned";
    std::string payload_mode = "legacy";
    bool emit_timing = false;
    long long full_match_probe_budget = 500000LL;
    double full_match_probe_count_cap = 1000.0;
    int tree_sample_count = 0;
    unsigned int tree_sample_seed = 42;
    std::string tree_sample_strategy = "weighted";
    std::string candidate_tree_strategy = "mwst_low_density_edges";
    int core_max_vertices = 10;
    std::string core_policy = "cycle_repeat";
    bool enable_tree_dp_weighted_closure = true;
    bool enable_factor_bp = true;
    bool enable_factor_bp_injective = true;
    bool enable_core_outside = true;
    bool minimal_probe_features = false;
    bool stage5_probe_only = false;
    long long probe_fastest2_cap = 0;
    int probe_fastest2_ub_initial = 100000;
    int probe_fastest2_success_threshold = 10;
    double probe_fastest2_strata_ratio = 0.5;
    unsigned int probe_fastest2_seed = 43;
    bool enable_dense_clique_feature = false;
    long long cycle_path_budget = 5000000LL;
    bool enable_dualsim_prefilter = false;
    ui gql_refinement_rounds = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-d") {
            data_graph_path = require_arg(argc, argv, i);
        }
        else if (arg == "--query_list") {
            query_list_path = require_arg(argc, argv, i);
        }
        else if (arg == "--out_jsonl") {
            out_jsonl_path = require_arg(argc, argv, i);
        }
        else if (arg == "--out_binary") {
            out_binary_path = require_arg(argc, argv, i);
        }
        else if (arg == "--mode") {
            mode = require_arg(argc, argv, i);
        }
        else if (arg == "--payload") {
            payload_mode = require_arg(argc, argv, i);
        }
        else if (arg == "--emit_timing") {
            emit_timing = true;
        }
        else if (arg == "--full_probe_budget") {
            full_match_probe_budget = std::stoll(require_arg(argc, argv, i));
        }
        else if (arg == "--full_probe_count_cap") {
            full_match_probe_count_cap = std::stod(require_arg(argc, argv, i));
        }
        else if (arg == "--tree_sample_count") {
            tree_sample_count = std::stoi(require_arg(argc, argv, i));
        }
        else if (arg == "--tree_sample_seed") {
            tree_sample_seed = static_cast<unsigned int>(std::stoul(require_arg(argc, argv, i)));
        }
        else if (arg == "--tree_sample_strategy") {
            tree_sample_strategy = require_arg(argc, argv, i);
        }
        else if (arg == "--candidate_tree_strategy") {
            candidate_tree_strategy = require_arg(argc, argv, i);
        }
        else if (arg == "--core_max_vertices") {
            core_max_vertices = std::stoi(require_arg(argc, argv, i));
        }
        else if (arg == "--core_policy") {
            core_policy = require_arg(argc, argv, i);
        }
        else if (arg == "--disable_tree_dp_weighted_closure") {
            enable_tree_dp_weighted_closure = false;
        }
        else if (arg == "--disable_factor_bp") {
            enable_factor_bp = false;
        }
        else if (arg == "--disable_factor_bp_injective") {
            enable_factor_bp_injective = false;
        }
        else if (arg == "--disable_core_outside") {
            enable_core_outside = false;
        }
        else if (arg == "--minimal_probe_features") {
            minimal_probe_features = true;
            enable_tree_dp_weighted_closure = false;
            enable_factor_bp = false;
            enable_factor_bp_injective = false;
            enable_core_outside = false;
        }
        else if (arg == "--stage5_probe_only") {
            stage5_probe_only = true;
            enable_tree_dp_weighted_closure = false;
            enable_factor_bp = false;
            enable_factor_bp_injective = false;
            enable_core_outside = false;
        }
        else if (arg == "--probe_fastest2_cap") {
            probe_fastest2_cap = std::stoll(require_arg(argc, argv, i));
        }
        else if (arg == "--probe_fastest2_ub_initial") {
            probe_fastest2_ub_initial = std::stoi(require_arg(argc, argv, i));
        }
        else if (arg == "--probe_fastest2_success_threshold") {
            probe_fastest2_success_threshold = std::stoi(require_arg(argc, argv, i));
        }
        else if (arg == "--probe_fastest2_strata_ratio") {
            probe_fastest2_strata_ratio = std::stod(require_arg(argc, argv, i));
        }
        else if (arg == "--probe_fastest2_seed") {
            probe_fastest2_seed = static_cast<unsigned int>(std::stoul(require_arg(argc, argv, i)));
        }
        else if (arg == "--enable_dense_clique_feature") {
            enable_dense_clique_feature = true;
        }
        else if (arg == "--cycle_path_budget") {
            cycle_path_budget = std::stoll(
                require_arg(argc, argv, i)
            );
        }
        else if (arg == "--enable_dualsim_prefilter") {
            enable_dualsim_prefilter = true;
        }
        else if (arg == "--gql_refinement_rounds") {
            gql_refinement_rounds = static_cast<ui>(std::stoul(require_arg(argc, argv, i)));
        }
    }

    if (data_graph_path.empty() || query_list_path.empty() || (out_jsonl_path.empty() && out_binary_path.empty())) {
        std::cerr << "Usage: GQLBatchExport -d <data_graph> --query_list <queries.txt> [--out_jsonl <out.jsonl|->] [--out_binary <out.bin>] [--mode basic|pruned] [--payload legacy|mlp_features|probe_features|stage5_probe|final11d_features|component_mlp_features|component_views_mlp_features] [--emit_timing] [--full_probe_budget N] [--full_probe_count_cap N] [--tree_sample_count N] [--tree_sample_seed N] [--tree_sample_strategy weighted|root_uniform] [--candidate_tree_strategy current_bfs_maxcand|bfs_mincand_root|bfs_max_query_degree_root|bfs_min_query_degree_root|mwst_low_density_edges|mwst_low_count_edges|mwst_high_density_edges] [--core_policy cycle_repeat|nontree_vertex_cover] [--core_max_vertices N] [--probe_fastest2_cap N] [--probe_fastest2_ub_initial N] [--probe_fastest2_success_threshold N] [--probe_fastest2_strata_ratio R] [--probe_fastest2_seed N] [--cycle_path_budget N] [--enable_dense_clique_feature] [--disable_tree_dp_weighted_closure] [--disable_factor_bp] [--disable_factor_bp_injective] [--disable_core_outside] [--minimal_probe_features] [--stage5_probe_only] [--enable_dualsim_prefilter] [--gql_refinement_rounds N]\n";
        return 1;
    }
    bool output_component_view_mlp_features = payload_mode == "component_views_mlp_features";
    bool output_probe_features = payload_mode == "probe_features";
    bool output_stage5_probe = payload_mode == "stage5_probe";
    bool output_final11d_features =
        payload_mode == "final11d_features";
    bool output_mlp_features = payload_mode == "mlp_features" || payload_mode == "component_mlp_features" || output_component_view_mlp_features;
    bool output_component_mlp_features = payload_mode == "component_mlp_features";
    if (output_stage5_probe) {
        stage5_probe_only = true;
        enable_tree_dp_weighted_closure = false;
        enable_factor_bp = false;
        enable_factor_bp_injective = false;
        enable_core_outside = false;
    }
    if (output_final11d_features) {
        enable_tree_dp_weighted_closure = false;
        enable_factor_bp = false;
        enable_factor_bp_injective = false;
        enable_core_outside = false;
        enable_dense_clique_feature = false;
    }

    std::ifstream query_list_stream(query_list_path);
    if (!query_list_stream.good()) {
        std::cerr << "Failed to open query list: " << query_list_path << "\n";
        return 1;
    }
    std::ofstream out_file;
    std::ostream* out_ptr = &std::cout;
    if (out_jsonl_path.empty()) {
        out_ptr = nullptr;
    }
    else if (out_jsonl_path != "-") {
        out_file.open(out_jsonl_path);
        if (!out_file.good()) {
            std::cerr << "Failed to open output jsonl: " << out_jsonl_path << "\n";
            return 1;
        }
        out_ptr = &out_file;
    }
    std::ofstream binary_file;
    if (!out_binary_path.empty()) {
        binary_file.open(out_binary_path, std::ios::binary);
        if (!binary_file.good()) {
            std::cerr << "Failed to open output binary: " << out_binary_path << "\n";
            return 1;
        }
    }

    auto load_start = std::chrono::high_resolution_clock::now();
    Graph* data_graph = new Graph(true);
    data_graph->loadGraphFromFile(data_graph_path);
    auto load_end = std::chrono::high_resolution_clock::now();
    std::cerr << "GQL DataGraph Load Seconds: "
              << std::chrono::duration<double>(load_end - load_start).count()
              << "\n";

    std::string query_path;
    while (std::getline(query_list_stream, query_path)) {
        while (!query_path.empty() && (query_path.back() == '\r' || query_path.back() == '\n' || query_path.back() == ' ' || query_path.back() == '\t')) {
            query_path.pop_back();
        }
        if (query_path.empty()) {
            continue;
        }
        auto query_start = std::chrono::high_resolution_clock::now();
        try {
            auto query_load_start = Clock::now();
            Graph* query_graph = new Graph(true);
            query_graph->loadGraphFromFile(query_path);
            double query_load_seconds = seconds_since(query_load_start);
            ui** candidates = NULL;
            ui* candidates_count = NULL;
            FilterVertices::CandidateNeighborMaps candidate_neighbors_from_filter;
            std::streambuf* original_cout = std::cout.rdbuf();
            std::ostringstream discarded_stdout;
            std::cout.rdbuf(discarded_stdout.rdbuf());
            auto gql_start = Clock::now();
            bool gql_ok = FilterVertices::GQLFilterWithCandidateNeighbors(
                data_graph,
                query_graph,
                candidates,
                candidates_count,
                &candidate_neighbors_from_filter,
                enable_dualsim_prefilter,
                gql_refinement_rounds
            );
            double gql_filter_seconds = seconds_since(gql_start);
            std::cout.rdbuf(original_cout);
            if (!gql_ok) {
                throw std::runtime_error("GQLFilter failed");
            }
            auto stats_start = Clock::now();
            QueryStats stats = build_query_stats(
                data_graph,
                query_graph,
                candidates,
                candidates_count,
                &candidate_neighbors_from_filter,
                full_match_probe_budget,
                full_match_probe_count_cap,
                tree_sample_count,
                tree_sample_seed,
                tree_sample_strategy,
                candidate_tree_strategy,
                core_max_vertices,
                core_policy,
                enable_tree_dp_weighted_closure,
                enable_factor_bp,
                enable_factor_bp_injective,
                enable_core_outside,
                minimal_probe_features,
                stage5_probe_only,
                probe_fastest2_cap,
                probe_fastest2_ub_initial,
                probe_fastest2_success_threshold,
                probe_fastest2_strata_ratio,
                probe_fastest2_seed,
                enable_dense_clique_feature,
                cycle_path_budget,
                output_final11d_features
            );
            double build_stats_seconds = seconds_since(stats_start);
            std::vector<RepeatedLabelGroupProbe> repeated_label_group_probes;
            if (!stage5_probe_only && !output_final11d_features) {
                repeated_label_group_probes = build_repeated_label_group_probes(
                    query_graph,
                    stats
                );
            }
            Final11dFeatures final11d_features;
            double final11d_feature_seconds = 0.0;
            if (output_final11d_features) {
                auto final11d_start = Clock::now();
                final11d_features = build_clean3224_final11d_features(
                    stats,
                    probe_fastest2_cap
                );
                final11d_feature_seconds =
                    seconds_since(final11d_start);
            }
            MlpFeatures mlp_features;
            std::vector<std::pair<int, MlpFeatures>> component_view_features;
            double mlp_feature_seconds = 0.0;
            if (output_component_view_mlp_features) {
                auto mlp_start = Clock::now();
                for (size_t component_idx = 0; component_idx < stats.components.size(); ++component_idx) {
                    MlpFeatures view_features = build_component_view_mlp_features(
                        data_graph,
                        query_graph,
                        stats,
                        stats.components[component_idx]
                    );
                    if (view_features.query_node_features.empty() || view_features.data_node_features.empty()) {
                        continue;
                    }
                    if (view_features.aux_log_tree_count <= 0.0) {
                        continue;
                    }
                    component_view_features.emplace_back(static_cast<int>(component_idx), std::move(view_features));
                }
                mlp_feature_seconds = seconds_since(mlp_start);
            }
            else if (output_mlp_features) {
                auto mlp_start = Clock::now();
                mlp_features = build_mlp_features(
                    data_graph,
                    query_graph,
                    stats,
                    output_component_mlp_features ? &stats.components : nullptr
                );
                mlp_feature_seconds = seconds_since(mlp_start);
            }

            ui query_vertices = query_graph->getVerticesCount();
            ui sum_candidates = 0;
            std::vector<uint32_t> cs_sizes;
            cs_sizes.reserve(query_vertices);
            if (out_ptr != nullptr) {
                (*out_ptr) << '{'
                    << "\"query_path\":\"" << json_escape(query_path) << "\","
                    << "\"cs_size\":[";
            }
            for (ui u = 0; u < query_vertices; ++u) {
                uint32_t cs_size = static_cast<uint32_t>(stats.candidate_nodes[u].size());
                cs_sizes.push_back(cs_size);
                if (out_ptr != nullptr) {
                    if (u > 0) (*out_ptr) << ',';
                    (*out_ptr) << cs_size;
                }
                sum_candidates += cs_size;
            }
            auto query_end = std::chrono::high_resolution_clock::now();
            double elapsed_seconds = std::chrono::duration<double>(query_end - query_start).count();
            if (output_component_view_mlp_features && binary_file.good()) {
                for (const auto& item : component_view_features) {
                    std::string view_query_path = query_path + "#component=" + std::to_string(item.first);
                    std::vector<uint32_t> view_cs_sizes;
                    view_cs_sizes.reserve(item.second.query_node_features.size());
                    for (const auto& row : item.second.query_node_features) {
                        view_cs_sizes.push_back(row.empty() ? 0 : static_cast<uint32_t>(std::round(std::expm1(row[0]))));
                    }
                    write_mlp_binary_record(binary_file, view_query_path, stats, item.second, view_cs_sizes, elapsed_seconds);
                }
            }
            else if (output_mlp_features && binary_file.good()) {
                write_mlp_binary_record(binary_file, query_path, stats, mlp_features, cs_sizes, elapsed_seconds);
            }
            if (out_ptr == nullptr) {
                free_candidates(query_graph, candidates, candidates_count);
                delete query_graph;
                release_query_heap();
                continue;
            }
            std::ostream& out = *out_ptr;
            if (output_component_view_mlp_features) {
                for (const auto& item : component_view_features) {
                    out << '{'
                        << "\"query_path\":\"" << json_escape(query_path + "#component=" + std::to_string(item.first)) << "\","
                        << "\"parent_query_path\":\"" << json_escape(query_path) << "\","
                        << "\"component_index\":" << item.first << ','
                        << "\"component_count\":" << stats.components.size() << ','
                        << "\"elapsed_seconds\":" << elapsed_seconds
                        << ",\"mlp_features\":{"
                        << "\"query_node_features\":";
                    write_nested_double_array(out, item.second.query_node_features);
                    out << ",\"data_node_features\":";
                    write_nested_double_array(out, item.second.data_node_features);
                    out << ",\"global_features\":";
                    write_double_array(out, item.second.global_features);
                    out << ",\"tree_order_ids\":";
                    write_int_array(out, item.second.tree_order_ids);
                    out << ",\"aux_log_tree_count\":" << item.second.aux_log_tree_count
                        << "},\"static_stats\":{"
                        << "\"component_index\":" << item.first
                        << ",\"component_count\":" << stats.components.size()
                        << ",\"export_elapsed_seconds\":" << elapsed_seconds
                        << "}}\n";
                }
                free_candidates(query_graph, candidates, candidates_count);
                delete query_graph;
                out.flush();
                release_query_heap();
                continue;
            }
            if (output_final11d_features) {
                const std::streamsize previous_precision =
                    out.precision();
                out << std::setprecision(17);
                out << "],"
                    << "\"sum_candidates\":" << sum_candidates << ','
                    << "\"num_candidates\":" << stats.num_candidates << ','
                    << "\"num_candidate_edges\":"
                    << stats.num_candidate_edges << ','
                    << "\"elapsed_seconds\":" << elapsed_seconds
                    << ",\"features\":{"
                    << "\"candidate_tree_log_count\":"
                    << final11d_features.candidate_tree_log_count
                    << ",\"candidate_domain_size_dispersion_mean\":"
                    << final11d_features.candidate_domain_size_dispersion_mean
                    << ",\"sum_support_concentration\":"
                    << final11d_features.sum_support_concentration
                    << ",\"candidate_relation_cycle_path_log_gap_mean\":"
                    << final11d_features.candidate_relation_cycle_path_log_gap_mean
                    << ",\"candidate_tree_path_non_tree_log_gap_mean\":"
                    << final11d_features.candidate_tree_path_non_tree_log_gap_mean
                    << ",\"probe_edge_feasible_sample_neglog_survival\":"
                    << final11d_features.probe_edge_feasible_sample_neglog_survival
                    << ",\"label_injectivity_product_log_sum\":"
                    << final11d_features.label_injectivity_product_log_sum
                    << ",\"repeated_label_pair_collision_log1p\":"
                    << final11d_features.repeated_label_pair_collision_log1p
                    << ",\"probe_injective_sample_neglog_survival\":"
                    << final11d_features.probe_injective_sample_neglog_survival
                    << ",\"probe_full_match_sample_neglog_survival\":"
                    << final11d_features.probe_full_match_sample_neglog_survival
                    << ",\"probe_graph_estimate_tree_log_gap\":"
                    << final11d_features.probe_graph_estimate_tree_log_gap
                    << ",\"probe_graph_estimate_signed_used_sample_fraction\":"
                    << final11d_features.probe_graph_estimate_signed_used_sample_fraction
                    << ",\"positive_sample_fraction_weighted_graph_gap_ratio\":"
                    << final11d_features.positive_sample_fraction_weighted_graph_gap_ratio
                    << "},\"probe_state\":{"
                    << "\"tree_sample_trials\":"
                    << stats.tree_sample_trials
                    << ",\"tree_sample_all_edge_success\":"
                    << stats.tree_sample_all_edge_success
                    << ",\"tree_sample_injective_success\":"
                    << stats.tree_sample_injective_success
                    << ",\"tree_sample_exact_success\":"
                    << stats.tree_sample_exact_success
                    << ",\"graph_probe_used_samples\":"
                    << stats.probe_fastest2_used_samples
                    << ",\"graph_probe_estimate_count\":"
                    << stats.probe_fastest2_graph_estimate_count
                    << "}";
                if (emit_timing) {
                    out << ",\"timing\":{"
                        << "\"query_load_seconds\":"
                        << query_load_seconds
                        << ",\"gql_filter_seconds\":"
                        << gql_filter_seconds
                        << ",\"build_stats_seconds\":"
                        << build_stats_seconds
                        << ",\"reindex_seconds\":"
                        << stats.timing_reindex_seconds
                        << ",\"candidate_adjacency_seconds\":"
                        << stats.timing_candidate_adjacency_seconds
                        << ",\"degree_stats_seconds\":"
                        << stats.timing_degree_stats_seconds
                        << ",\"tree_setup_seconds\":"
                        << stats.timing_tree_setup_seconds
                        << ",\"tree_dp_seconds\":"
                        << stats.timing_tree_dp_seconds
                        << ",\"tree_sampling_seconds\":"
                        << stats.timing_tree_sampling_seconds
                        << ",\"graph_probe_seconds\":"
                        << stats.timing_probe_fastest2_seconds
                        << ",\"density_summary_seconds\":"
                        << stats.timing_density_summary_seconds
                        << ",\"cycle_path_seconds\":"
                        << stats.timing_cycle_path_seconds
                        << ",\"label_injectivity_seconds\":"
                        << stats.timing_label_injectivity_seconds
                        << ",\"triangle_joint_seconds\":"
                        << stats.timing_triangle_joint_seconds
                        << ",\"motif4_joint_seconds\":"
                        << stats.timing_motif4_joint_seconds
                        << ",\"final11d_assembly_seconds\":"
                        << final11d_feature_seconds
                        << ",\"elapsed_before_json_seconds\":"
                        << std::chrono::duration<double>(
                            query_end - query_start
                        ).count()
                        << "}";
                }
                out << "}\n";
                out.precision(previous_precision);
                free_candidates(query_graph, candidates, candidates_count);
                delete query_graph;
                out.flush();
                release_query_heap();
                continue;
            }
            if (output_stage5_probe) {
                out << "],"
                    << "\"elapsed_seconds\":" << elapsed_seconds
                    << ",\"static_stats\":{";
                write_stage5_probe_stats(out, stats);
                out << "}";
                if (emit_timing) {
                    out << ",\"timing\":{"
                        << "\"query_load_seconds\":" << query_load_seconds
                        << ",\"gql_filter_seconds\":" << gql_filter_seconds
                        << ",\"build_stats_seconds\":" << build_stats_seconds
                        << ",\"reindex_seconds\":" << stats.timing_reindex_seconds
                        << ",\"candidate_adjacency_seconds\":" << stats.timing_candidate_adjacency_seconds
                        << ",\"tree_setup_seconds\":" << stats.timing_tree_setup_seconds
                        << ",\"tree_dp_seconds\":" << stats.timing_tree_dp_seconds
                        << ",\"tree_sampling_seconds\":" << stats.timing_tree_sampling_seconds
                        << ",\"probe_fastest2_seconds\":" << stats.timing_probe_fastest2_seconds
                        << ",\"dense_clique_seconds\":" << stats.timing_dense_clique_seconds
                        << ",\"elapsed_before_json_seconds\":" << std::chrono::duration<double>(query_end - query_start).count()
                        << "}";
                }
                out << "}\n";
                free_candidates(query_graph, candidates, candidates_count);
                delete query_graph;
                out.flush();
                release_query_heap();
                continue;
            }
            if (output_probe_features) {
                out << "],"
                    << "\"sum_candidates\":" << sum_candidates << ','
                    << "\"num_candidates\":" << stats.num_candidates << ','
                    << "\"num_candidate_edges\":" << stats.num_candidate_edges << ','
                    << "\"elapsed_seconds\":" << elapsed_seconds
                    << ",\"static_stats\":{"
                    << "\"log_candidate_tree_count\":" << stats.log_candidate_tree_count
                    << ",\"tree_density_log_mean\":" << stats.tree_density_log_mean
                    << ",\"tree_density_log_min\":" << stats.tree_density_log_min
                    << ",\"tree_density_log_std\":" << stats.tree_density_log_std
                    << ",\"non_tree_density_log_mean\":" << stats.non_tree_density_log_mean
                    << ",\"non_tree_density_log_min\":" << stats.non_tree_density_log_min
                    << ",\"non_tree_density_log_std\":" << stats.non_tree_density_log_std
                    << ",\"density_log_gap\":" << stats.density_log_gap
                    << ",\"tree_edge_log_count_mean\":" << stats.tree_edge_log_count_mean
                    << ",\"tree_edge_log_count_min\":" << stats.tree_edge_log_count_min
                    << ",\"tree_edge_log_count_std\":" << stats.tree_edge_log_count_std
                    << ",\"non_tree_edge_log_count_mean\":" << stats.non_tree_edge_log_count_mean
                    << ",\"non_tree_edge_log_count_min\":" << stats.non_tree_edge_log_count_min
                    << ",\"non_tree_edge_log_count_std\":" << stats.non_tree_edge_log_count_std
                    << ",\"label_pair_log_density_hash_sum\":";
                write_double_array(out, stats.label_pair_log_density_hash_sum);
                out << ",\"label_pair_log_count_hash_sum\":";
                write_double_array(out, stats.label_pair_log_count_hash_sum);
                out << ",\"label_pair_tree_log_density_hash_sum\":";
                write_double_array(out, stats.label_pair_tree_log_density_hash_sum);
                out << ",\"label_pair_non_tree_log_density_hash_sum\":";
                write_double_array(out, stats.label_pair_non_tree_log_density_hash_sum);
                out << ",\"label_pair_edge_hash_count\":";
                write_double_array(out, stats.label_pair_edge_hash_count);
                out
                    << ",\"candidate_degree_log_mean\":" << stats.candidate_degree_log_mean
                    << ",\"candidate_degree_log_std\":" << stats.candidate_degree_log_std
                    << ",\"candidate_degree_log_max\":" << stats.candidate_degree_log_max
                    << ",\"candidate_degree_l2_log\":" << stats.candidate_degree_l2_log
                    << ",\"candidate_edge_count\":" << stats.candidate_edge_count
                    << ",\"component_count_before_prune\":" << stats.component_count_before_prune
                    << ",\"component_count_after_prune\":" << stats.component_count_after_prune
                    << ",\"pruned_component_count\":" << stats.pruned_component_count
                    << ",\"pruned_candidate_node_count\":" << stats.pruned_candidate_node_count
                    << ",\"edge_fixpoint_iterations\":" << stats.edge_fixpoint_iterations
                    << ",\"edge_fixpoint_pruned_candidate_count\":" << stats.edge_fixpoint_pruned_candidate_count
                    << ",\"triangle_edges_checked\":" << stats.triangle_edges_checked
                    << ",\"triangle_candidate_edges_removed\":" << stats.triangle_candidate_edges_removed
                    << ",\"triangle_filter_skipped_by_budget\":" << stats.triangle_filter_skipped_by_budget
                    << ",\"cycle_path_edge_count\":" << stats.cycle_path_edge_count
                    << ",\"cycle_path_budget_skipped\":" << stats.cycle_path_budget_skipped
                    << ",\"cycle_path_support_frac_mean\":" << stats.cycle_path_support_frac_mean
                    << ",\"cycle_path_support_frac_min\":" << stats.cycle_path_support_frac_min
                    << ",\"cycle_path_support_frac_std\":" << stats.cycle_path_support_frac_std
                    << ",\"cycle_path_log_gap_mean\":" << stats.cycle_path_log_gap_mean
                    << ",\"cycle_path_log_gap_max\":" << stats.cycle_path_log_gap_max
                    << ",\"cycle_path_local_edge_log_mean\":" << stats.cycle_path_local_edge_log_mean
                    << ",\"cycle_path_supported_edge_log_mean\":" << stats.cycle_path_supported_edge_log_mean
                    << ",\"cycle_path_tree_pair_support_frac_mean\":" << stats.cycle_path_tree_pair_support_frac_mean
                    << ",\"cycle_path_tree_pair_support_frac_min\":" << stats.cycle_path_tree_pair_support_frac_min
                    << ",\"cycle_path_tree_pair_support_frac_std\":" << stats.cycle_path_tree_pair_support_frac_std
                    << ",\"cycle_path_tree_pair_log_gap_mean\":" << stats.cycle_path_tree_pair_log_gap_mean
                    << ",\"cycle_path_tree_pair_log_gap_max\":" << stats.cycle_path_tree_pair_log_gap_max
                    << ",\"cycle_path_tree_pair_log_mean\":" << stats.cycle_path_tree_pair_log_mean
                    << ",\"label_injectivity_group_count\":" << stats.label_injectivity_group_count
                    << ",\"label_injectivity_gap_sum\":" << stats.label_injectivity_gap_sum
                    << ",\"label_injectivity_gap_mean\":" << stats.label_injectivity_gap_mean
                    << ",\"label_injectivity_gap_max\":" << stats.label_injectivity_gap_max
                    << ",\"label_injectivity_overlap_frac_mean\":" << stats.label_injectivity_overlap_frac_mean
                    << ",\"label_injectivity_overlap_frac_max\":" << stats.label_injectivity_overlap_frac_max
                    << ",\"label_injectivity_min_union_slack\":" << stats.label_injectivity_min_union_slack
                    << ",\"label_injectivity_product_log_sum\":" << stats.label_injectivity_product_log_sum
                    << ",\"label_injectivity_distinct_bound_log_sum\":" << stats.label_injectivity_distinct_bound_log_sum
                    << ",\"label_injectivity_pair_collision_log1p\":" << stats.label_injectivity_pair_collision_log1p
                    << ",\"triangle_joint_probe_count\":" << stats.triangle_joint_probe_count
                    << ",\"triangle_joint_budget_skipped\":" << stats.triangle_joint_budget_skipped
                    << ",\"triangle_joint_capped\":" << stats.triangle_joint_capped
                    << ",\"triangle_joint_exact_log_mean\":" << stats.triangle_joint_exact_log_mean
                    << ",\"triangle_joint_exact_log_min\":" << stats.triangle_joint_exact_log_min
                    << ",\"triangle_joint_domain_to_exact_gap_mean\":" << stats.triangle_joint_domain_to_exact_gap_mean
                    << ",\"triangle_joint_domain_to_exact_gap_max\":" << stats.triangle_joint_domain_to_exact_gap_max
                    << ",\"triangle_joint_zero_exact_frac\":" << stats.triangle_joint_zero_exact_frac
                    << ",\"motif4_joint_candidate_count\":" << stats.motif4_joint_candidate_count
                    << ",\"motif4_joint_probe_count\":" << stats.motif4_joint_probe_count
                    << ",\"motif4_joint_budget_skipped\":" << stats.motif4_joint_budget_skipped
                    << ",\"motif4_joint_capped\":" << stats.motif4_joint_capped
                    << ",\"motif4_joint_edge_count_mean\":" << stats.motif4_joint_edge_count_mean
                    << ",\"motif4_joint_exact_log_mean\":" << stats.motif4_joint_exact_log_mean
                    << ",\"motif4_joint_exact_log_min\":" << stats.motif4_joint_exact_log_min
                    << ",\"motif4_joint_domain_to_exact_gap_mean\":" << stats.motif4_joint_domain_to_exact_gap_mean
                    << ",\"motif4_joint_domain_to_exact_gap_max\":" << stats.motif4_joint_domain_to_exact_gap_max
                    << ",\"motif4_joint_zero_exact_frac\":" << stats.motif4_joint_zero_exact_frac
                    << ",\"full_match_probe_found\":" << stats.full_match_probe_found
                    << ",\"full_match_probe_budget_exhausted\":" << stats.full_match_probe_budget_exhausted
                    << ",\"full_match_probe_count_capped\":" << stats.full_match_probe_count_capped
                    << ",\"full_match_probe_max_depth\":" << stats.full_match_probe_max_depth
                    << ",\"full_match_probe_search_nodes\":" << stats.full_match_probe_search_nodes
                    << ",\"full_match_probe_count\":" << stats.full_match_probe_count
                    << ",\"full_match_probe_log_count\":" << stats.full_match_probe_log_count;
                write_core_outside_stats(out, stats);
                write_tree_sample_stats(out, stats);
                write_probe_fastest2_stats(out, stats);
                write_tree_dp_weighted_closure_stats(out, stats);
                write_tree_root_contribution_stats(out, stats);
                write_factor_bp_stats(out, stats);
                out << ",\"tree_prior_variants\":";
                write_tree_prior_variants(out, stats.tree_prior_variants);
                out << ",\"repeated_label_group_probe\":";
                write_repeated_label_group_probes(out, repeated_label_group_probes);
                out << "}";
                if (emit_timing) {
                    out << ",\"timing\":{"
                        << "\"query_load_seconds\":" << query_load_seconds
                        << ",\"gql_filter_seconds\":" << gql_filter_seconds
                        << ",\"build_stats_seconds\":" << build_stats_seconds
                        << ",\"mlp_feature_seconds\":" << mlp_feature_seconds
                        << ",\"reindex_seconds\":" << stats.timing_reindex_seconds
                        << ",\"candidate_adjacency_seconds\":" << stats.timing_candidate_adjacency_seconds
                        << ",\"component_prune_seconds\":" << stats.timing_component_prune_seconds
                        << ",\"edge_fixpoint_seconds\":" << stats.timing_edge_fixpoint_seconds
                        << ",\"triangle_prune_seconds\":" << stats.timing_triangle_prune_seconds
                        << ",\"degree_stats_seconds\":" << stats.timing_degree_stats_seconds
                        << ",\"tree_setup_seconds\":" << stats.timing_tree_setup_seconds
                        << ",\"tree_dp_seconds\":" << stats.timing_tree_dp_seconds
                        << ",\"tree_weighted_closure_seconds\":" << stats.timing_tree_weighted_closure_seconds
                        << ",\"tree_sampling_seconds\":" << stats.timing_tree_sampling_seconds
                        << ",\"probe_fastest2_seconds\":" << stats.timing_probe_fastest2_seconds
                        << ",\"density_summary_seconds\":" << stats.timing_density_summary_seconds
                        << ",\"elapsed_before_json_seconds\":" << std::chrono::duration<double>(query_end - query_start).count()
                        << "}";
                }
                out << "}\n";
                free_candidates(query_graph, candidates, candidates_count);
                delete query_graph;
                out.flush();
                release_query_heap();
                continue;
            }
            if (output_mlp_features) {
                out << "],"
                    << "\"candidate_nodes\":";
                write_nested_ui_array(out, stats.candidate_nodes);
                out << ','
                    << "\"sum_candidates\":" << sum_candidates << ','
                    << "\"num_candidates\":" << stats.num_candidates << ','
                    << "\"num_candidate_edges\":" << stats.num_candidate_edges << ','
                    << "\"elapsed_seconds\":" << elapsed_seconds
                    << ",\"mlp_features\":{"
                    << "\"query_node_features\":";
                write_nested_double_array(out, mlp_features.query_node_features);
                out << ",\"data_node_features\":";
                write_nested_double_array(out, mlp_features.data_node_features);
                out << ",\"global_features\":";
                write_double_array(out, mlp_features.global_features);
                out << ",\"tree_order_ids\":";
                write_int_array(out, mlp_features.tree_order_ids);
                out << ",\"aux_log_tree_count\":" << mlp_features.aux_log_tree_count
                    << "},\"static_stats\":{"
                    << "\"query_edge_list\":";
                write_pair_array(out, stats.query_edge_list);
                out << ",\"query_edge_candidate_counts\":";
                write_double_array(out, stats.query_edge_candidate_counts);
                out << ",\"query_edge_densities\":";
                write_double_array(out, stats.query_edge_densities);
                out << ",\"query_edge_left_support_nonzero_frac\":";
                write_double_array(out, stats.query_edge_left_support_nonzero_frac);
                out << ",\"query_edge_right_support_nonzero_frac\":";
                write_double_array(out, stats.query_edge_right_support_nonzero_frac);
                out << ",\"query_edge_left_log_support_min\":";
                write_double_array(out, stats.query_edge_left_log_support_min);
                out << ",\"query_edge_left_log_support_mean\":";
                write_double_array(out, stats.query_edge_left_log_support_mean);
                out << ",\"query_edge_left_log_support_std\":";
                write_double_array(out, stats.query_edge_left_log_support_std);
                out << ",\"query_edge_left_log_support_max\":";
                write_double_array(out, stats.query_edge_left_log_support_max);
                out << ",\"query_edge_right_log_support_min\":";
                write_double_array(out, stats.query_edge_right_log_support_min);
                out << ",\"query_edge_right_log_support_mean\":";
                write_double_array(out, stats.query_edge_right_log_support_mean);
                out << ",\"query_edge_right_log_support_std\":";
                write_double_array(out, stats.query_edge_right_log_support_std);
                out << ",\"query_edge_right_log_support_max\":";
                write_double_array(out, stats.query_edge_right_log_support_max);
                out << ",\"tree_edge_mask\":";
                write_double_array(out, stats.tree_edge_mask);
                out << ",\"log_candidate_tree_count\":" << stats.log_candidate_tree_count
                    << ",\"tree_density_log_mean\":" << stats.tree_density_log_mean
                    << ",\"tree_density_log_min\":" << stats.tree_density_log_min
                    << ",\"tree_density_log_std\":" << stats.tree_density_log_std
                    << ",\"non_tree_density_log_mean\":" << stats.non_tree_density_log_mean
                    << ",\"non_tree_density_log_min\":" << stats.non_tree_density_log_min
                    << ",\"non_tree_density_log_std\":" << stats.non_tree_density_log_std
                    << ",\"density_log_gap\":" << stats.density_log_gap
                    << ",\"tree_edge_log_count_mean\":" << stats.tree_edge_log_count_mean
                    << ",\"tree_edge_log_count_min\":" << stats.tree_edge_log_count_min
                    << ",\"tree_edge_log_count_std\":" << stats.tree_edge_log_count_std
                    << ",\"non_tree_edge_log_count_mean\":" << stats.non_tree_edge_log_count_mean
                    << ",\"non_tree_edge_log_count_min\":" << stats.non_tree_edge_log_count_min
                    << ",\"non_tree_edge_log_count_std\":" << stats.non_tree_edge_log_count_std
                    << ",\"candidate_degree_log_mean\":" << stats.candidate_degree_log_mean
                    << ",\"candidate_degree_log_std\":" << stats.candidate_degree_log_std
                    << ",\"candidate_degree_log_max\":" << stats.candidate_degree_log_max
                    << ",\"candidate_degree_l2_log\":" << stats.candidate_degree_l2_log
                    << ",\"candidate_edge_count\":" << stats.candidate_edge_count
                    << ",\"component_count_before_prune\":" << stats.component_count_before_prune
                    << ",\"component_count_after_prune\":" << stats.component_count_after_prune
                    << ",\"pruned_component_count\":" << stats.pruned_component_count
                    << ",\"pruned_candidate_node_count\":" << stats.pruned_candidate_node_count
                    << ",\"edge_fixpoint_iterations\":" << stats.edge_fixpoint_iterations
                    << ",\"edge_fixpoint_pruned_candidate_count\":" << stats.edge_fixpoint_pruned_candidate_count
                    << ",\"triangle_edges_checked\":" << stats.triangle_edges_checked
                    << ",\"triangle_candidate_edges_removed\":" << stats.triangle_candidate_edges_removed
                    << ",\"triangle_filter_skipped_by_budget\":" << stats.triangle_filter_skipped_by_budget
                    << ",\"cycle_path_edge_count\":" << stats.cycle_path_edge_count
                    << ",\"cycle_path_budget_skipped\":" << stats.cycle_path_budget_skipped
                    << ",\"cycle_path_support_frac_mean\":" << stats.cycle_path_support_frac_mean
                    << ",\"cycle_path_support_frac_min\":" << stats.cycle_path_support_frac_min
                    << ",\"cycle_path_support_frac_std\":" << stats.cycle_path_support_frac_std
                    << ",\"cycle_path_log_gap_mean\":" << stats.cycle_path_log_gap_mean
                    << ",\"cycle_path_log_gap_max\":" << stats.cycle_path_log_gap_max
                    << ",\"cycle_path_local_edge_log_mean\":" << stats.cycle_path_local_edge_log_mean
                    << ",\"cycle_path_supported_edge_log_mean\":" << stats.cycle_path_supported_edge_log_mean
                    << ",\"cycle_path_tree_pair_support_frac_mean\":" << stats.cycle_path_tree_pair_support_frac_mean
                    << ",\"cycle_path_tree_pair_support_frac_min\":" << stats.cycle_path_tree_pair_support_frac_min
                    << ",\"cycle_path_tree_pair_support_frac_std\":" << stats.cycle_path_tree_pair_support_frac_std
                    << ",\"cycle_path_tree_pair_log_gap_mean\":" << stats.cycle_path_tree_pair_log_gap_mean
                    << ",\"cycle_path_tree_pair_log_gap_max\":" << stats.cycle_path_tree_pair_log_gap_max
                    << ",\"cycle_path_tree_pair_log_mean\":" << stats.cycle_path_tree_pair_log_mean
                    << ",\"label_injectivity_group_count\":" << stats.label_injectivity_group_count
                    << ",\"label_injectivity_gap_sum\":" << stats.label_injectivity_gap_sum
                    << ",\"label_injectivity_gap_mean\":" << stats.label_injectivity_gap_mean
                    << ",\"label_injectivity_gap_max\":" << stats.label_injectivity_gap_max
                    << ",\"label_injectivity_overlap_frac_mean\":" << stats.label_injectivity_overlap_frac_mean
                    << ",\"label_injectivity_overlap_frac_max\":" << stats.label_injectivity_overlap_frac_max
                    << ",\"label_injectivity_min_union_slack\":" << stats.label_injectivity_min_union_slack
                    << ",\"label_injectivity_product_log_sum\":" << stats.label_injectivity_product_log_sum
                    << ",\"label_injectivity_distinct_bound_log_sum\":" << stats.label_injectivity_distinct_bound_log_sum
                    << ",\"label_injectivity_pair_collision_log1p\":" << stats.label_injectivity_pair_collision_log1p
                    << ",\"triangle_joint_probe_count\":" << stats.triangle_joint_probe_count
                    << ",\"triangle_joint_budget_skipped\":" << stats.triangle_joint_budget_skipped
                    << ",\"triangle_joint_capped\":" << stats.triangle_joint_capped
                    << ",\"triangle_joint_exact_log_mean\":" << stats.triangle_joint_exact_log_mean
                    << ",\"triangle_joint_exact_log_min\":" << stats.triangle_joint_exact_log_min
                    << ",\"triangle_joint_domain_to_exact_gap_mean\":" << stats.triangle_joint_domain_to_exact_gap_mean
                    << ",\"triangle_joint_domain_to_exact_gap_max\":" << stats.triangle_joint_domain_to_exact_gap_max
                    << ",\"triangle_joint_zero_exact_frac\":" << stats.triangle_joint_zero_exact_frac
                    << ",\"motif4_joint_candidate_count\":" << stats.motif4_joint_candidate_count
                    << ",\"motif4_joint_probe_count\":" << stats.motif4_joint_probe_count
                    << ",\"motif4_joint_budget_skipped\":" << stats.motif4_joint_budget_skipped
                    << ",\"motif4_joint_capped\":" << stats.motif4_joint_capped
                    << ",\"motif4_joint_edge_count_mean\":" << stats.motif4_joint_edge_count_mean
                    << ",\"motif4_joint_exact_log_mean\":" << stats.motif4_joint_exact_log_mean
                    << ",\"motif4_joint_exact_log_min\":" << stats.motif4_joint_exact_log_min
                    << ",\"motif4_joint_domain_to_exact_gap_mean\":" << stats.motif4_joint_domain_to_exact_gap_mean
                    << ",\"motif4_joint_domain_to_exact_gap_max\":" << stats.motif4_joint_domain_to_exact_gap_max
                    << ",\"motif4_joint_zero_exact_frac\":" << stats.motif4_joint_zero_exact_frac
                    << ",\"full_match_probe_found\":" << stats.full_match_probe_found
                    << ",\"full_match_probe_budget_exhausted\":" << stats.full_match_probe_budget_exhausted
                    << ",\"full_match_probe_count_capped\":" << stats.full_match_probe_count_capped
                    << ",\"full_match_probe_max_depth\":" << stats.full_match_probe_max_depth
                    << ",\"full_match_probe_search_nodes\":" << stats.full_match_probe_search_nodes
                    << ",\"full_match_probe_count\":" << stats.full_match_probe_count
                    << ",\"full_match_probe_log_count\":" << stats.full_match_probe_log_count;
                write_core_outside_stats(out, stats);
                write_tree_sample_stats(out, stats);
                write_probe_fastest2_stats(out, stats);
                write_tree_dp_weighted_closure_stats(out, stats);
                write_tree_root_contribution_stats(out, stats);
                write_factor_bp_stats(out, stats);
                out << ",\"repeated_label_group_probe\":";
                write_repeated_label_group_probes(out, repeated_label_group_probes);
                out << "}";
                if (emit_timing) {
                    out << ",\"timing\":{"
                        << "\"query_load_seconds\":" << query_load_seconds
                        << ",\"gql_filter_seconds\":" << gql_filter_seconds
                        << ",\"build_stats_seconds\":" << build_stats_seconds
                        << ",\"mlp_feature_seconds\":" << mlp_feature_seconds
                        << ",\"reindex_seconds\":" << stats.timing_reindex_seconds
                        << ",\"candidate_adjacency_seconds\":" << stats.timing_candidate_adjacency_seconds
                        << ",\"component_prune_seconds\":" << stats.timing_component_prune_seconds
                        << ",\"edge_fixpoint_seconds\":" << stats.timing_edge_fixpoint_seconds
                        << ",\"triangle_prune_seconds\":" << stats.timing_triangle_prune_seconds
                        << ",\"degree_stats_seconds\":" << stats.timing_degree_stats_seconds
                        << ",\"tree_setup_seconds\":" << stats.timing_tree_setup_seconds
                        << ",\"tree_dp_seconds\":" << stats.timing_tree_dp_seconds
                        << ",\"tree_weighted_closure_seconds\":" << stats.timing_tree_weighted_closure_seconds
                        << ",\"tree_sampling_seconds\":" << stats.timing_tree_sampling_seconds
                        << ",\"probe_fastest2_seconds\":" << stats.timing_probe_fastest2_seconds
                        << ",\"density_summary_seconds\":" << stats.timing_density_summary_seconds
                        << ",\"elapsed_before_json_seconds\":" << std::chrono::duration<double>(query_end - query_start).count()
                        << "}";
                }
                out << "}\n";
                free_candidates(query_graph, candidates, candidates_count);
                delete query_graph;
                out.flush();
                release_query_heap();
                continue;
            }
            out << "],"
                << "\"candidate_nodes\":";
            write_nested_ui_array(out, stats.candidate_nodes);
            out << ','
                << "\"sum_candidates\":" << sum_candidates << ','
                << "\"num_candidates\":" << stats.num_candidates << ','
                << "\"num_candidate_edges\":" << stats.num_candidate_edges << ','
                << "\"elapsed_seconds\":" << elapsed_seconds
                << ",\"static_stats\":{"
                << "\"tree_root\":" << stats.tree_root << ','
                << "\"tree_parent\":";
            write_int_array(out, stats.tree_parent);
            out << ",\"tree_order\":";
            write_int_array(out, stats.tree_order);
            out << ",\"tree_depth_by_node\":";
            write_int_array(out, stats.tree_depth_by_node);
            out << ",\"tree_child_count_by_node\":";
            write_int_array(out, stats.tree_child_count_by_node);
            out << ",\"tree_leaf_mask\":";
            write_double_array(out, stats.tree_leaf_mask);
            out << ",\"tree_parent_log_density\":";
            write_double_array(out, stats.tree_parent_log_density);
            out << ",\"query_edge_list\":";
            write_pair_array(out, stats.query_edge_list);
            out << ",\"query_edge_candidate_counts\":";
            write_double_array(out, stats.query_edge_candidate_counts);
            out << ",\"query_edge_densities\":";
            write_double_array(out, stats.query_edge_densities);
            out << ",\"query_edge_left_support_nonzero_frac\":";
            write_double_array(out, stats.query_edge_left_support_nonzero_frac);
            out << ",\"query_edge_right_support_nonzero_frac\":";
            write_double_array(out, stats.query_edge_right_support_nonzero_frac);
            out << ",\"query_edge_left_log_support_min\":";
            write_double_array(out, stats.query_edge_left_log_support_min);
            out << ",\"query_edge_left_log_support_mean\":";
            write_double_array(out, stats.query_edge_left_log_support_mean);
            out << ",\"query_edge_left_log_support_std\":";
            write_double_array(out, stats.query_edge_left_log_support_std);
            out << ",\"query_edge_left_log_support_max\":";
            write_double_array(out, stats.query_edge_left_log_support_max);
            out << ",\"query_edge_right_log_support_min\":";
            write_double_array(out, stats.query_edge_right_log_support_min);
            out << ",\"query_edge_right_log_support_mean\":";
            write_double_array(out, stats.query_edge_right_log_support_mean);
            out << ",\"query_edge_right_log_support_std\":";
            write_double_array(out, stats.query_edge_right_log_support_std);
            out << ",\"query_edge_right_log_support_max\":";
            write_double_array(out, stats.query_edge_right_log_support_max);
            out << ",\"tree_edge_mask\":";
            write_double_array(out, stats.tree_edge_mask);
            out << ",\"candidate_node_degrees\":";
            write_double_array(out, stats.candidate_node_degrees);
            out << ",\"candidate_data_degrees\":";
            write_double_array(out, stats.candidate_data_degrees);
            out << ",\"log_candidate_tree_count\":" << stats.log_candidate_tree_count
                << ",\"tree_density_log_sum\":" << stats.tree_density_log_sum
                << ",\"tree_density_log_mean\":" << stats.tree_density_log_mean
                << ",\"tree_density_log_min\":" << stats.tree_density_log_min
                << ",\"tree_density_log_max\":" << stats.tree_density_log_max
                << ",\"tree_density_log_std\":" << stats.tree_density_log_std
                << ",\"non_tree_density_log_mean\":" << stats.non_tree_density_log_mean
                << ",\"non_tree_density_log_min\":" << stats.non_tree_density_log_min
                << ",\"non_tree_density_log_std\":" << stats.non_tree_density_log_std
                << ",\"density_log_gap\":" << stats.density_log_gap
                << ",\"tree_edge_log_count_mean\":" << stats.tree_edge_log_count_mean
                << ",\"tree_edge_log_count_min\":" << stats.tree_edge_log_count_min
                << ",\"tree_edge_log_count_std\":" << stats.tree_edge_log_count_std
                << ",\"non_tree_edge_log_count_mean\":" << stats.non_tree_edge_log_count_mean
                << ",\"non_tree_edge_log_count_min\":" << stats.non_tree_edge_log_count_min
                << ",\"non_tree_edge_log_count_std\":" << stats.non_tree_edge_log_count_std
                << ",\"candidate_degree_log_mean\":" << stats.candidate_degree_log_mean
                << ",\"candidate_degree_log_std\":" << stats.candidate_degree_log_std
                << ",\"candidate_degree_log_max\":" << stats.candidate_degree_log_max
                << ",\"candidate_degree_l2_log\":" << stats.candidate_degree_l2_log
                << ",\"candidate_edge_count\":" << stats.candidate_edge_count
                << ",\"component_count_before_prune\":" << stats.component_count_before_prune
                << ",\"component_count_after_prune\":" << stats.component_count_after_prune
                << ",\"pruned_component_count\":" << stats.pruned_component_count
                << ",\"pruned_candidate_node_count\":" << stats.pruned_candidate_node_count
                << ",\"edge_fixpoint_iterations\":" << stats.edge_fixpoint_iterations
                << ",\"edge_fixpoint_pruned_candidate_count\":" << stats.edge_fixpoint_pruned_candidate_count
                << ",\"triangle_edges_checked\":" << stats.triangle_edges_checked
                << ",\"triangle_candidate_edges_removed\":" << stats.triangle_candidate_edges_removed
                << ",\"triangle_filter_skipped_by_budget\":" << stats.triangle_filter_skipped_by_budget
                << ",\"cycle_path_edge_count\":" << stats.cycle_path_edge_count
                << ",\"cycle_path_budget_skipped\":" << stats.cycle_path_budget_skipped
                << ",\"cycle_path_support_frac_mean\":" << stats.cycle_path_support_frac_mean
                << ",\"cycle_path_support_frac_min\":" << stats.cycle_path_support_frac_min
                << ",\"cycle_path_support_frac_std\":" << stats.cycle_path_support_frac_std
                << ",\"cycle_path_log_gap_mean\":" << stats.cycle_path_log_gap_mean
                << ",\"cycle_path_log_gap_max\":" << stats.cycle_path_log_gap_max
                << ",\"cycle_path_local_edge_log_mean\":" << stats.cycle_path_local_edge_log_mean
                << ",\"cycle_path_supported_edge_log_mean\":" << stats.cycle_path_supported_edge_log_mean
                << ",\"cycle_path_tree_pair_support_frac_mean\":" << stats.cycle_path_tree_pair_support_frac_mean
                << ",\"cycle_path_tree_pair_support_frac_min\":" << stats.cycle_path_tree_pair_support_frac_min
                << ",\"cycle_path_tree_pair_support_frac_std\":" << stats.cycle_path_tree_pair_support_frac_std
                << ",\"cycle_path_tree_pair_log_gap_mean\":" << stats.cycle_path_tree_pair_log_gap_mean
                << ",\"cycle_path_tree_pair_log_gap_max\":" << stats.cycle_path_tree_pair_log_gap_max
                << ",\"cycle_path_tree_pair_log_mean\":" << stats.cycle_path_tree_pair_log_mean
                << ",\"label_injectivity_group_count\":" << stats.label_injectivity_group_count
                << ",\"label_injectivity_gap_sum\":" << stats.label_injectivity_gap_sum
                << ",\"label_injectivity_gap_mean\":" << stats.label_injectivity_gap_mean
                << ",\"label_injectivity_gap_max\":" << stats.label_injectivity_gap_max
                << ",\"label_injectivity_overlap_frac_mean\":" << stats.label_injectivity_overlap_frac_mean
                << ",\"label_injectivity_overlap_frac_max\":" << stats.label_injectivity_overlap_frac_max
                << ",\"label_injectivity_min_union_slack\":" << stats.label_injectivity_min_union_slack
                << ",\"label_injectivity_product_log_sum\":" << stats.label_injectivity_product_log_sum
                << ",\"label_injectivity_distinct_bound_log_sum\":" << stats.label_injectivity_distinct_bound_log_sum
                << ",\"label_injectivity_pair_collision_log1p\":" << stats.label_injectivity_pair_collision_log1p
                << ",\"triangle_joint_probe_count\":" << stats.triangle_joint_probe_count
                << ",\"triangle_joint_budget_skipped\":" << stats.triangle_joint_budget_skipped
                << ",\"triangle_joint_capped\":" << stats.triangle_joint_capped
                << ",\"triangle_joint_exact_log_mean\":" << stats.triangle_joint_exact_log_mean
                << ",\"triangle_joint_exact_log_min\":" << stats.triangle_joint_exact_log_min
                << ",\"triangle_joint_domain_to_exact_gap_mean\":" << stats.triangle_joint_domain_to_exact_gap_mean
                << ",\"triangle_joint_domain_to_exact_gap_max\":" << stats.triangle_joint_domain_to_exact_gap_max
                << ",\"triangle_joint_zero_exact_frac\":" << stats.triangle_joint_zero_exact_frac
                << ",\"motif4_joint_candidate_count\":" << stats.motif4_joint_candidate_count
                << ",\"motif4_joint_probe_count\":" << stats.motif4_joint_probe_count
                << ",\"motif4_joint_budget_skipped\":" << stats.motif4_joint_budget_skipped
                << ",\"motif4_joint_capped\":" << stats.motif4_joint_capped
                << ",\"motif4_joint_edge_count_mean\":" << stats.motif4_joint_edge_count_mean
                << ",\"motif4_joint_exact_log_mean\":" << stats.motif4_joint_exact_log_mean
                << ",\"motif4_joint_exact_log_min\":" << stats.motif4_joint_exact_log_min
                << ",\"motif4_joint_domain_to_exact_gap_mean\":" << stats.motif4_joint_domain_to_exact_gap_mean
                << ",\"motif4_joint_domain_to_exact_gap_max\":" << stats.motif4_joint_domain_to_exact_gap_max
                << ",\"motif4_joint_zero_exact_frac\":" << stats.motif4_joint_zero_exact_frac
                << ",\"full_match_probe_found\":" << stats.full_match_probe_found
                << ",\"full_match_probe_budget_exhausted\":" << stats.full_match_probe_budget_exhausted
                << ",\"full_match_probe_count_capped\":" << stats.full_match_probe_count_capped
                << ",\"full_match_probe_max_depth\":" << stats.full_match_probe_max_depth
                << ",\"full_match_probe_search_nodes\":" << stats.full_match_probe_search_nodes
                << ",\"full_match_probe_count\":" << stats.full_match_probe_count
                << ",\"full_match_probe_log_count\":" << stats.full_match_probe_log_count;
            write_core_outside_stats(out, stats);
            write_tree_sample_stats(out, stats);
            write_probe_fastest2_stats(out, stats);
            write_tree_dp_weighted_closure_stats(out, stats);
            write_tree_root_contribution_stats(out, stats);
            write_factor_bp_stats(out, stats);
            out << ",\"repeated_label_group_probe\":";
            write_repeated_label_group_probes(out, repeated_label_group_probes);
            out << "}";
            if (emit_timing) {
                out << ",\"timing\":{"
                    << "\"query_load_seconds\":" << query_load_seconds
                    << ",\"gql_filter_seconds\":" << gql_filter_seconds
                    << ",\"build_stats_seconds\":" << build_stats_seconds
                    << ",\"mlp_feature_seconds\":" << mlp_feature_seconds
                    << ",\"reindex_seconds\":" << stats.timing_reindex_seconds
                    << ",\"candidate_adjacency_seconds\":" << stats.timing_candidate_adjacency_seconds
                    << ",\"component_prune_seconds\":" << stats.timing_component_prune_seconds
                    << ",\"edge_fixpoint_seconds\":" << stats.timing_edge_fixpoint_seconds
                    << ",\"triangle_prune_seconds\":" << stats.timing_triangle_prune_seconds
                    << ",\"degree_stats_seconds\":" << stats.timing_degree_stats_seconds
                    << ",\"tree_setup_seconds\":" << stats.timing_tree_setup_seconds
                    << ",\"tree_dp_seconds\":" << stats.timing_tree_dp_seconds
                    << ",\"tree_weighted_closure_seconds\":" << stats.timing_tree_weighted_closure_seconds
                    << ",\"tree_sampling_seconds\":" << stats.timing_tree_sampling_seconds
                    << ",\"probe_fastest2_seconds\":" << stats.timing_probe_fastest2_seconds
                    << ",\"density_summary_seconds\":" << stats.timing_density_summary_seconds
                    << ",\"elapsed_before_json_seconds\":" << std::chrono::duration<double>(query_end - query_start).count()
                    << "}";
            }
            out << "}\n";

            free_candidates(query_graph, candidates, candidates_count);
            delete query_graph;
            release_query_heap();
        }
        catch (const std::exception& exc) {
            if (out_ptr != nullptr) {
                (*out_ptr) << '{'
                    << "\"query_path\":\"" << json_escape(query_path) << "\","
                    << "\"error\":\"" << json_escape(exc.what()) << "\""
                    << "}\n";
            }
        }
        if (out_ptr != nullptr) {
            out_ptr->flush();
        }
        if (binary_file.good()) {
            binary_file.flush();
        }
    }

    delete data_graph;
    return 0;
}
