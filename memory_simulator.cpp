#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr unsigned int PAGE_SHIFT = 12;
constexpr std::size_t TLB_CAPACITY = 16;
constexpr std::size_t DEFAULT_DRAM_CAPACITY = 8;

struct TraceEntry {
    std::uint64_t timestamp;
    std::uint64_t address;
    bool is_write;
};

struct PageMetadata {
    std::size_t access_count = 0;
    std::uint64_t last_access_timestamp = 0;
    std::uint64_t total_reuse_distance_us = 0;
    std::size_t reuse_samples = 0;
    bool is_write = false;
    bool is_in_dram = false;
};

struct Prediction {
    int tier = 0;
    std::size_t access_frequency = 0;
    std::uint64_t reuse_distance = 0;
    double dram_score = 0.0;
};

struct TimelinePoint {
    double progress = 0.0;
    double fault_rate = 0.0;
    double emat = 0.0;
};

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

bool parse_trace(const std::string& path, std::vector<TraceEntry>& entries) {
    std::ifstream input(path);
    if (!input) {
        std::cerr << "Cannot open trace file: " << path << '\n';
        return false;
    }

    std::string line;
    std::size_t line_number = 0;
    std::uint64_t previous_timestamp = 0;
    bool has_previous_timestamp = false;
    while (std::getline(input, line)) {
        ++line_number;
        if (trim(line).empty()) {
            continue;
        }

        std::stringstream row(line);
        std::string time_text;
        std::string operation;
        std::string address_text;
        if (!std::getline(row, time_text, ',') || !std::getline(row, operation, ',') ||
            !std::getline(row, address_text)) {
            std::cerr << "Malformed trace row " << line_number << '\n';
            return false;
        }

        try {
            const std::uint64_t timestamp = std::stoull(trim(time_text));
            if (has_previous_timestamp && timestamp < previous_timestamp) {
                std::cerr << "Trace timestamps must be nondecreasing (row " << line_number << ")\n";
                return false;
            }
            previous_timestamp = timestamp;
            has_previous_timestamp = true;
            const std::string address_value = trim(address_text);
            std::size_t consumed = 0;
            const std::uint64_t address = std::stoull(address_value, &consumed, 16);
            if (consumed != address_value.size()) {
                std::cerr << "Invalid address on trace row " << line_number << '\n';
                return false;
            }
            const std::string normalized_operation = trim(operation);
            entries.push_back({timestamp, address,
                               normalized_operation == "W" || normalized_operation == "w"});
        } catch (const std::exception&) {
            std::cerr << "Invalid value on trace row " << line_number << '\n';
            return false;
        }
    }

    if (entries.empty()) {
        std::cerr << "Trace file contains no memory references.\n";
        return false;
    }
    return true;
}

bool load_predictions(const std::string& path,
                      std::unordered_map<std::uint64_t, Prediction>& predictions) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }

    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        std::stringstream row(line);
        std::string page_text;
        std::string frequency_text;
        std::string write_text;
        std::string reuse_text;
        std::string tier_text;
        std::string score_text;
        if (!std::getline(row, page_text, ',') || !std::getline(row, frequency_text, ',') ||
            !std::getline(row, write_text, ',') || !std::getline(row, reuse_text, ',') ||
            !std::getline(row, tier_text, ',') || !std::getline(row, score_text)) {
            continue;
        }
        try {
            predictions[std::stoull(trim(page_text))] =
                {std::stoi(trim(tier_text)),
                 static_cast<std::size_t>(std::stoull(trim(frequency_text))),
                 std::stoull(trim(reuse_text)),
                 std::stod(trim(score_text))};
        } catch (const std::exception&) {
            continue;
        }
    }
    return !predictions.empty();
}

void touch_frame(std::uint64_t page,
                 std::list<std::uint64_t>& frames,
                 std::unordered_map<std::uint64_t, std::list<std::uint64_t>::iterator>& positions) {
    const auto found = positions.find(page);
    if (found != positions.end()) {
        frames.erase(found->second);
    }
    frames.push_front(page);
    positions[page] = frames.begin();
}

void write_metrics(const std::string& policy,
                   double emat,
                   double fault_rate,
                   double tlb_hit_rate,
                   std::size_t total_pages,
                   std::size_t dram_capacity) {
    std::ofstream output("run_metrics_" + policy + ".csv");
    output << "emat,fault_rate,tlb_hit_rate,total_pages,dram_capacity\n";
    output << std::fixed << std::setprecision(6) << emat << ',' << fault_rate << ','
           << tlb_hit_rate << ',' << total_pages << ',' << dram_capacity << '\n';
}

void write_timeline(const std::string& policy, const std::vector<TimelinePoint>& timeline) {
    std::ofstream output("run_timeline_" + policy + ".csv");
    output << "progress_percent,fault_rate,emat\n";
    output << std::fixed << std::setprecision(6);
    for (const TimelinePoint& point : timeline) {
        output << point.progress << ',' << point.fault_rate << ',' << point.emat << '\n';
    }
}

void write_page_map(const std::string& policy,
                    const std::unordered_map<std::uint64_t, PageMetadata>& pages) {
    std::vector<std::uint64_t> page_ids;
    page_ids.reserve(pages.size());
    for (const auto& item : pages) {
        page_ids.push_back(item.first);
    }
    std::sort(page_ids.begin(), page_ids.end());

    std::ofstream output("page_map_" + policy + ".csv");
    output << "page_id,tier,access_frequency,is_write\n";
    for (const std::uint64_t page_id : page_ids) {
        const PageMetadata& page = pages.at(page_id);
        output << page_id << ',' << (page.is_in_dram ? "DRAM" : "NVM") << ','
               << page.access_count << ',' << (page.is_write ? 1 : 0) << '\n';
    }
}

void write_training_data(const std::unordered_map<std::uint64_t, PageMetadata>& pages,
                         std::size_t dram_capacity) {
    std::vector<std::pair<std::uint64_t, long double>> ranked_pages;
    for (const auto& item : pages) {
        const PageMetadata& page = item.second;
        const std::uint64_t reuse_distance = page.reuse_samples == 0
                                                 ? static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
                                                 : page.total_reuse_distance_us / page.reuse_samples;
        if (page.access_count > 30 && reuse_distance < 2000) {
            const long double score = static_cast<long double>(page.access_count) /
                                      (static_cast<long double>(reuse_distance) + 1.0L);
            ranked_pages.emplace_back(item.first, score);
        }
    }
    std::sort(ranked_pages.begin(), ranked_pages.end(), [](const auto& left, const auto& right) {
        if (left.second != right.second) {
            return left.second > right.second;
        }
        return left.first < right.first;
    });

    std::unordered_set<std::uint64_t> preferred_pages;
    const std::size_t preferred_count = std::min(dram_capacity, ranked_pages.size());
    for (std::size_t index = 0; index < preferred_count; ++index) {
        preferred_pages.insert(ranked_pages[index].first);
    }

    std::ofstream output("telemetry_ml_dataset.csv");
    output << "page_id,access_frequency,is_write,reuse_distance,optimal_tier\n";
    for (const auto& item : pages) {
        const PageMetadata& page = item.second;
        const std::uint64_t reuse_distance = page.reuse_samples == 0
                                                 ? static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
                                                 : page.total_reuse_distance_us / page.reuse_samples;
        const int optimal_tier = preferred_pages.count(item.first) ? 1 : 0;
        output << item.first << ',' << item.second.access_count << ','
               << (page.is_write ? 1 : 0) << ',' << reuse_distance << ',' << optimal_tier << '\n';
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    const double tlb_walk_latency = argc > 1 ? std::atof(argv[1]) : 15.0;
    const double dram_latency = argc > 2 ? std::atof(argv[2]) : 50.0;
    const double nvm_latency = argc > 3 ? std::atof(argv[3]) : 200.0;
    const std::string requested_policy = argc > 4 ? argv[4] : "LRU";
    const std::size_t dram_capacity = argc > 5
                                          ? std::max<std::size_t>(1, std::strtoull(argv[5], nullptr, 10))
                                          : DEFAULT_DRAM_CAPACITY;
    const std::string prediction_path = argc > 6 ? argv[6] : "page_predictions.csv";

    std::vector<TraceEntry> trace;
    if (!parse_trace("memory_trace.txt", trace)) {
        return 1;
    }

    std::unordered_map<std::uint64_t, Prediction> predictions;
    const std::string output_policy = requested_policy == "AI" ? "AI" : "LRU";
    std::string policy = output_policy;
    bool ai_fallback = false;
    std::unordered_map<std::uint64_t, double> ai_scores;
    if (policy == "AI") {
        bool predictions_valid = load_predictions(prediction_path, predictions);
        if (predictions_valid) {
            for (const TraceEntry& entry : trace) {
                const auto prediction = predictions.find(entry.address >> PAGE_SHIFT);
                if (prediction == predictions.end() || prediction->second.tier < 0 || prediction->second.tier > 1) {
                    predictions_valid = false;
                    break;
                }
            }
        }
        if (!predictions_valid) {
            std::cerr << "AI predictions unavailable or incomplete; falling back to LRU.\n";
            policy = "LRU";
            ai_fallback = true;
        } else {
            for (const auto& prediction : predictions) {
                ai_scores[prediction.first] = prediction.second.dram_score;
            }
        }
    }

    std::list<std::uint64_t> tlb;
    std::list<std::uint64_t> dram_frames;
    std::unordered_map<std::uint64_t, std::list<std::uint64_t>::iterator> frame_positions;
    std::unordered_map<std::uint64_t, PageMetadata> page_table;
    std::size_t tlb_hits = 0;
    std::size_t page_faults = 0;
    double accumulated_latency = 0.0;
    double interval_latency = 0.0;
    std::size_t interval_faults = 0;
    std::size_t interval_references = 0;
    std::vector<TimelinePoint> timeline;
    const std::size_t interval_size = std::max<std::size_t>(1, (trace.size() + 99) / 100);

    for (std::size_t index = 0; index < trace.size(); ++index) {
        const TraceEntry& entry = trace[index];
        const std::uint64_t page_id = entry.address >> PAGE_SHIFT;
        double access_latency = 0.0;

        const auto tlb_page = std::find(tlb.begin(), tlb.end(), page_id);
        if (tlb_page != tlb.end()) {
            ++tlb_hits;
            tlb.erase(tlb_page);
            tlb.push_front(page_id);
            access_latency += 1.0;
        } else {
            access_latency += 1.0 + tlb_walk_latency;
            if (tlb.size() == TLB_CAPACITY) {
                tlb.pop_back();
            }
            tlb.push_front(page_id);
        }

        PageMetadata& page = page_table[page_id];
        if (page.access_count > 0) {
            page.total_reuse_distance_us += entry.timestamp - page.last_access_timestamp;
            ++page.reuse_samples;
        }
        ++page.access_count;
        page.last_access_timestamp = entry.timestamp;
        page.is_write = entry.is_write;

        if (policy == "AI") {
            if (page.is_in_dram) {
                access_latency += dram_latency;
                touch_frame(page_id, dram_frames, frame_positions);
            } else {
                ++page_faults;
                ++interval_faults;
                access_latency += nvm_latency;

                if (dram_frames.size() == dram_capacity) {
                    auto victim_position = std::prev(dram_frames.end());
                    double lowest_score = ai_scores[*victim_position];
                    for (auto position = dram_frames.rbegin(); position != dram_frames.rend(); ++position) {
                        const double candidate_score = ai_scores[*position];
                        if (candidate_score < lowest_score) {
                            lowest_score = candidate_score;
                            victim_position = std::prev(position.base());
                        }
                    }
                    const std::uint64_t victim = *victim_position;
                    dram_frames.erase(victim_position);
                    frame_positions.erase(victim);
                    page_table[victim].is_in_dram = false;
                }
                touch_frame(page_id, dram_frames, frame_positions);
                page.is_in_dram = true;
            }
        } else if (page.is_in_dram) {
            access_latency += dram_latency;
            touch_frame(page_id, dram_frames, frame_positions);
        } else {
            ++page_faults;
            ++interval_faults;
            access_latency += nvm_latency;

            if (dram_frames.size() == dram_capacity) {
                const std::uint64_t victim = dram_frames.back();
                dram_frames.pop_back();
                frame_positions.erase(victim);
                page_table[victim].is_in_dram = false;
            }
            touch_frame(page_id, dram_frames, frame_positions);
            page.is_in_dram = true;
        }

        accumulated_latency += access_latency;
        interval_latency += access_latency;
        ++interval_references;

        if (interval_references == interval_size || index + 1 == trace.size()) {
            timeline.push_back({
                100.0 * static_cast<double>(index + 1) / static_cast<double>(trace.size()),
                100.0 * static_cast<double>(interval_faults) / static_cast<double>(interval_references),
                interval_latency / static_cast<double>(interval_references),
            });
            interval_latency = 0.0;
            interval_faults = 0;
            interval_references = 0;
        }
    }

    const double total_references = static_cast<double>(trace.size());
    const double emat = accumulated_latency / total_references;
    const double fault_rate = 100.0 * static_cast<double>(page_faults) / total_references;
    const double tlb_hit_rate = 100.0 * static_cast<double>(tlb_hits) / total_references;

    write_metrics(output_policy, emat, fault_rate, tlb_hit_rate, page_table.size(), dram_capacity);
    write_timeline(output_policy, timeline);
    write_page_map(output_policy, page_table);
    if (policy == "LRU") {
        write_training_data(page_table, dram_capacity);
    }

    std::cout << "Policy: " << output_policy << (ai_fallback ? " (LRU fallback)" : "")
              << "\nReferences: " << trace.size()
              << "\nUnique pages: " << page_table.size() << "\nEMAT: " << emat
              << " ns\nPage fault rate: " << fault_rate << "%\n";
    return 0;
}