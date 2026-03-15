#include "dr_api.h"
#include "drmemtrace/analyzer.h"
#include "drmemtrace/analysis_tool.h"
#include "drmemtrace/memref.h"
#include "drmemtrace/memtrace_stream.h"
#include "drmemtrace/trace_entry.h"

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using dynamorio::drmemtrace::analysis_tool_t;
using dynamorio::drmemtrace::analyzer_t;
using dynamorio::drmemtrace::is_any_instr_type;
using dynamorio::drmemtrace::memref_t;
using dynamorio::drmemtrace::memref_tid_t;
using dynamorio::drmemtrace::memtrace_stream_t;
using dynamorio::drmemtrace::_memref_data_t;
using dynamorio::drmemtrace::_memref_instr_t;
using dynamorio::drmemtrace::addr_t;
using dynamorio::drmemtrace::TRACE_TYPE_INSTR_CONDITIONAL_JUMP;
using dynamorio::drmemtrace::TRACE_TYPE_INSTR_DIRECT_CALL;
using dynamorio::drmemtrace::TRACE_TYPE_INSTR_DIRECT_JUMP;
using dynamorio::drmemtrace::TRACE_TYPE_INSTR_INDIRECT_CALL;
using dynamorio::drmemtrace::TRACE_TYPE_INSTR_INDIRECT_JUMP;
using dynamorio::drmemtrace::TRACE_TYPE_INSTR_RETURN;
using dynamorio::drmemtrace::TRACE_TYPE_INSTR_TAKEN_JUMP;
using dynamorio::drmemtrace::TRACE_TYPE_INSTR_UNTAKEN_JUMP;
using dynamorio::drmemtrace::TRACE_TYPE_READ;
using dynamorio::drmemtrace::TRACE_TYPE_THREAD_EXIT;
using dynamorio::drmemtrace::TRACE_TYPE_WRITE;
using dynamorio::drmemtrace::trace_type_t;
using dynamorio::drmemtrace::type_is_instr_conditional_branch;
using dynamorio::drmemtrace::type_is_instr_direct_branch;
using dynamorio::drmemtrace::type_is_prefetch;

struct Options {
    std::string trace_dir;
    std::string config_path;
    std::string output_json;
    uint64_t max_instructions = 0;
    int worker_count = 0;
};

struct NativeConfig {
    int rob_entries = 128;
    int window_size = 400;

    int fetch_width = 4;
    int decode_width = 4;
    int rename_width = 4;
    int commit_width = 4;
    int alu_issue_width = 3;
    int fp_issue_width = 2;
    int ls_issue_width = 2;

    int load_store_pipes = 2;
    int load_only_pipes = 10;

    int cache_line_size = 64;
    int l1_size_bytes = 65536;
    int l1_associativity = 4;
    int l1_hit_latency = 1;
    int l2_size_bytes = 524288;
    int l2_associativity = 8;
    int l2_hit_latency = 10;
    int l3_size_bytes = 32000000;
    int l3_associativity = 16;
    int l3_hit_latency = 20;
    int memory_latency = 60;

    int icache_max_fills = 16;
    int icache_fill_latency = 40;
    int icache_size_bytes = 65536;
    int icache_line_size = 64;
    int icache_fetch_width = 4;

    int fetch_buffer_entries = 64;

    double simple_misprediction_rate = 0.05;
    int simple_seed = 1;

    int tage_num_tables = 8;
    int tage_table_size = 2048;
    int tage_tag_bits = 10;
    int tage_ghr_bits = 200;
    int tage_base_size = 4096;
    int tage_counter_bits = 3;
    int tage_usefulness_bits = 2;
    int tage_seed = 1;
};

enum class IssueGroup : uint8_t {
    ALU = 0,
    FP = 1,
    LS = 2,
};

enum class BranchKind : uint8_t {
    NONE = 0,
    CONDITIONAL = 1,
    DIRECT_UNCONDITIONAL = 2,
    INDIRECT = 3,
    OTHER = 4,
};

enum class InstructionKind : uint8_t {
    NON_MEM = 0,
    LOAD = 1,
    STORE = 2,
};

enum class HitLevel : uint8_t {
    L1 = 0,
    L2 = 1,
    L3 = 2,
    MEM = 3,
};

struct DecodedInstruction {
    std::string mnemonic;
    uint64_t uses_mask = 0;
    uint64_t defs_mask = 0;
    IssueGroup issue_group = IssueGroup::ALU;
    BranchKind branch_kind = BranchKind::NONE;
    bool has_direct_target = false;
    addr_t direct_target = 0;
    bool branch_taken_known = false;
    bool branch_taken = false;
};

struct MemOp {
    bool is_write = false;
    addr_t addr = 0;
    uint32_t size = 0;

    MemOp() = default;
    MemOp(bool write_value, addr_t addr_value, uint32_t size_value)
        : is_write(write_value)
        , addr(addr_value)
        , size(size_value)
    {
    }
};

struct CurrentInstruction {
    uint64_t instruction_ordinal = 0;
    memref_tid_t tid = 0;
    addr_t pc = 0;
    DecodedInstruction decoded;
    std::vector<MemOp> memops;
};

struct BranchStats {
    uint64_t total = 0;
    uint64_t misp = 0;
};

struct ThroughputSeries {
    std::vector<double> rob_thr_chunks;
    std::vector<double> static_fetch_width;
    std::vector<double> static_decode_width;
    std::vector<double> static_rename_width;
    std::vector<double> static_commit_width;
    std::vector<double> static_alu_issue_width;
    std::vector<double> static_fp_issue_width;
    std::vector<double> static_ls_issue_width;
    std::vector<double> dyn_pipes_thr_lower;
    std::vector<double> dyn_pipes_thr_upper;
    std::vector<double> dyn_icache_fills_thr;
    std::vector<double> dyn_fb_decode_thr;
    std::vector<double> br_type_direct_unconditional;
    std::vector<double> br_type_direct_conditional;
    std::vector<double> br_type_indirect;
};

struct ThreadResults {
    int64_t tid = -1;
    int shard_index = -1;
    int64_t input_id = -1;
    std::string stream_name;
    uint64_t instruction_count = 0;
    uint64_t window_size = 0;
    double rob_avg_ipc = 0.0;
    BranchStats simple_stats;
    BranchStats tage_stats;
    ThroughputSeries series;
};

struct NativeResults {
    int64_t target_tid = -1;
    int target_shard_index = -1;
    int64_t target_input_id = -1;
    std::string target_stream_name;
    uint64_t instruction_count = 0;
    uint64_t window_size = 0;
    double rob_avg_ipc = 0.0;
    BranchStats simple_stats;
    BranchStats tage_stats;
    ThroughputSeries series;
    std::vector<ThreadResults> thread_results;
    std::unordered_map<std::string, double> timings;
};

class StopAnalysis : public std::exception {
public:
    const char *
    what() const noexcept override
    {
        return "stop analysis";
    }
};

std::string
trim_copy(const std::string &value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
        return "";
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string
to_lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool
is_directory(const std::string &path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string
resolve_trace_input_path(const std::string &path)
{
    const std::string nested_trace_dir = path + "/trace";
    if (is_directory(nested_trace_dir))
        return nested_trace_dir;
    return path;
}

void
set_config_value(NativeConfig *config, const std::string &key, const std::string &value)
{
    auto to_int = [&]() { return std::stoi(value); };
    auto to_double = [&]() { return std::stod(value); };

    if (key == "rob.entries")
        config->rob_entries = to_int();
    else if (key == "rob.window_size")
        config->window_size = to_int();
    else if (key == "pipeline.fetch_width")
        config->fetch_width = to_int();
    else if (key == "pipeline.decode_width")
        config->decode_width = to_int();
    else if (key == "pipeline.rename_width")
        config->rename_width = to_int();
    else if (key == "pipeline.commit_width")
        config->commit_width = to_int();
    else if (key == "pipeline.issue_widths.alu")
        config->alu_issue_width = to_int();
    else if (key == "pipeline.issue_widths.fp")
        config->fp_issue_width = to_int();
    else if (key == "pipeline.issue_widths.ls")
        config->ls_issue_width = to_int();
    else if (key == "load_store_pipes.load_store_pipes")
        config->load_store_pipes = to_int();
    else if (key == "load_store_pipes.load_only_pipes")
        config->load_only_pipes = to_int();
    else if (key == "cache_hierarchy.line_size")
        config->cache_line_size = to_int();
    else if (key == "cache_hierarchy.l1.size_bytes")
        config->l1_size_bytes = to_int();
    else if (key == "cache_hierarchy.l1.associativity")
        config->l1_associativity = to_int();
    else if (key == "cache_hierarchy.l1.hit_latency")
        config->l1_hit_latency = to_int();
    else if (key == "cache_hierarchy.l2.size_bytes")
        config->l2_size_bytes = to_int();
    else if (key == "cache_hierarchy.l2.associativity")
        config->l2_associativity = to_int();
    else if (key == "cache_hierarchy.l2.hit_latency")
        config->l2_hit_latency = to_int();
    else if (key == "cache_hierarchy.l3.size_bytes")
        config->l3_size_bytes = to_int();
    else if (key == "cache_hierarchy.l3.associativity")
        config->l3_associativity = to_int();
    else if (key == "cache_hierarchy.l3.hit_latency")
        config->l3_hit_latency = to_int();
    else if (key == "cache_hierarchy.memory.latency")
        config->memory_latency = to_int();
    else if (key == "icache.max_fills")
        config->icache_max_fills = to_int();
    else if (key == "icache.fill_latency")
        config->icache_fill_latency = to_int();
    else if (key == "icache.size_bytes")
        config->icache_size_bytes = to_int();
    else if (key == "icache.line_size")
        config->icache_line_size = to_int();
    else if (key == "icache.fetch_width")
        config->icache_fetch_width = to_int();
    else if (key == "fetch_buffer.entries")
        config->fetch_buffer_entries = to_int();
    else if (key == "branch_prediction.simple.misprediction_rate")
        config->simple_misprediction_rate = to_double();
    else if (key == "branch_prediction.simple.seed")
        config->simple_seed = to_int();
    else if (key == "branch_prediction.tage.num_tables")
        config->tage_num_tables = to_int();
    else if (key == "branch_prediction.tage.table_size")
        config->tage_table_size = to_int();
    else if (key == "branch_prediction.tage.tag_bits")
        config->tage_tag_bits = to_int();
    else if (key == "branch_prediction.tage.ghr_bits")
        config->tage_ghr_bits = to_int();
    else if (key == "branch_prediction.tage.base_size")
        config->tage_base_size = to_int();
    else if (key == "branch_prediction.tage.counter_bits")
        config->tage_counter_bits = to_int();
    else if (key == "branch_prediction.tage.usefulness_bits")
        config->tage_usefulness_bits = to_int();
    else if (key == "branch_prediction.tage.seed")
        config->tage_seed = to_int();
}

bool
load_native_config(const std::string &path, NativeConfig *config, std::string *error)
{
    std::ifstream input(path);
    if (!input) {
        *error = "Failed to open native config: " + path;
        return false;
    }

    std::string line;
    while (std::getline(input, line)) {
        line = trim_copy(line);
        if (line.empty() || line[0] == '#')
            continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        const auto key = trim_copy(line.substr(0, eq));
        const auto value = trim_copy(line.substr(eq + 1));
        if (key.empty() || value.empty())
            continue;
        try {
            set_config_value(config, key, value);
        } catch (const std::exception &ex) {
            *error = "Invalid native config entry for key '" + key + "': " + ex.what();
            return false;
        }
    }
    return true;
}

std::string
normalize_mnemonic(std::string mnemonic)
{
    mnemonic = to_lower_copy(std::move(mnemonic));
    if (mnemonic == "bcond")
        return "b.cond";
    return mnemonic;
}

std::string
fallback_mnemonic(trace_type_t type)
{
    switch (type) {
    case TRACE_TYPE_INSTR_TAKEN_JUMP:
    case TRACE_TYPE_INSTR_UNTAKEN_JUMP:
    case TRACE_TYPE_INSTR_CONDITIONAL_JUMP: return "b.cond";
    case TRACE_TYPE_INSTR_DIRECT_JUMP: return "b";
    case TRACE_TYPE_INSTR_DIRECT_CALL: return "bl";
    case TRACE_TYPE_INSTR_INDIRECT_JUMP: return "br";
    case TRACE_TYPE_INSTR_INDIRECT_CALL: return "blr";
    case TRACE_TYPE_INSTR_RETURN: return "ret";
    default: return "undecoded";
    }
}

reg_id_t
canon_reg(reg_id_t reg)
{
    if (reg >= DR_REG_W0 && reg <= DR_REG_W30)
        return static_cast<reg_id_t>(DR_REG_X0 + (reg - DR_REG_W0));
    if (reg == DR_REG_WZR)
        return DR_REG_XZR;
    return reg;
}

int
tracked_reg_index(reg_id_t reg)
{
    reg = canon_reg(reg);
    if (reg >= DR_REG_X0 && reg <= DR_REG_X30)
        return static_cast<int>(reg - DR_REG_X0);
    if (reg == DR_REG_SP)
        return 31;
    if (reg == DR_REG_XZR)
        return 32;
    return -1;
}

IssueGroup
classify_issue_group(const std::string &mnemonic)
{
    const std::string lower = to_lower_copy(mnemonic);
    if (lower.rfind("fc", 0) == 0 || lower.rfind("fm", 0) == 0 ||
        lower.rfind("fs", 0) == 0 || lower.rfind("fd", 0) == 0 ||
        lower.rfind("f", 0) == 0)
        return IssueGroup::FP;
    return IssueGroup::ALU;
}

BranchKind
classify_branch_kind(trace_type_t type, const std::string &mnemonic)
{
    if (type == TRACE_TYPE_INSTR_TAKEN_JUMP || type == TRACE_TYPE_INSTR_UNTAKEN_JUMP ||
        type == TRACE_TYPE_INSTR_CONDITIONAL_JUMP)
        return BranchKind::CONDITIONAL;
    if (type == TRACE_TYPE_INSTR_DIRECT_JUMP || type == TRACE_TYPE_INSTR_DIRECT_CALL)
        return BranchKind::DIRECT_UNCONDITIONAL;
    if (type == TRACE_TYPE_INSTR_INDIRECT_JUMP || type == TRACE_TYPE_INSTR_INDIRECT_CALL ||
        type == TRACE_TYPE_INSTR_RETURN)
        return BranchKind::INDIRECT;

    const std::string lower = to_lower_copy(mnemonic);
    if (lower.rfind("b.", 0) == 0 || lower == "cbz" || lower == "cbnz" || lower == "tbz" ||
        lower == "tbnz")
        return BranchKind::CONDITIONAL;
    if (lower == "b" || lower == "bl")
        return BranchKind::DIRECT_UNCONDITIONAL;
    if (lower == "br" || lower == "blr" || lower == "ret")
        return BranchKind::INDIRECT;
    return BranchKind::NONE;
}

const char *
distribution_branch_name(BranchKind kind)
{
    switch (kind) {
    case BranchKind::CONDITIONAL: return "Direct Conditional Branch";
    case BranchKind::DIRECT_UNCONDITIONAL: return "Direct Unconditional Branch";
    case BranchKind::INDIRECT: return "Indirect Branch";
    default: return "Other Branch";
    }
}

const char *
predictor_branch_name(BranchKind kind)
{
    switch (kind) {
    case BranchKind::CONDITIONAL: return "Conditional Branch";
    case BranchKind::DIRECT_UNCONDITIONAL: return "Unconditional Branch";
    case BranchKind::INDIRECT: return "Indirect Branch";
    default: return "Other Branch";
    }
}

std::vector<uint64_t>
cache_lines_covered(addr_t addr, uint32_t size, int line_size)
{
    std::vector<uint64_t> lines;
    const uint64_t start = static_cast<uint64_t>(addr) / static_cast<uint64_t>(line_size);
    const uint64_t end =
        (static_cast<uint64_t>(addr) + static_cast<uint64_t>(size) - 1) / static_cast<uint64_t>(line_size);
    lines.reserve(static_cast<size_t>(end - start + 1));
    for (uint64_t line = start; line <= end; ++line)
        lines.push_back(line);
    return lines;
}

int
trailing_zero_index(uint64_t value)
{
    return static_cast<int>(__builtin_ctzll(value));
}

struct CacheAccess {
    int latency = 0;
    HitLevel hit_level = HitLevel::MEM;

    CacheAccess() = default;
    CacheAccess(int latency_value, HitLevel hit_level_value)
        : latency(latency_value)
        , hit_level(hit_level_value)
    {
    }
};

class MemoryLevel {
public:
    explicit MemoryLevel(int latency)
        : latency_(latency)
    {
    }

    CacheAccess
    access(addr_t, bool) const
    {
        return { latency_, HitLevel::MEM };
    }

private:
    int latency_;
};

class CacheLevel {
public:
    CacheLevel(std::string name, int size_bytes, int assoc, int line_size, int hit_latency,
               std::shared_ptr<void> lower, HitLevel self_level, HitLevel lower_level)
        : name_(std::move(name))
        , size_bytes_(std::max(size_bytes, line_size * assoc))
        , assoc_(std::max(assoc, 1))
        , line_size_(std::max(line_size, 1))
        , hit_latency_(std::max(hit_latency, 1))
        , lower_(std::move(lower))
        , self_level_(self_level)
        , lower_level_(lower_level)
    {
        num_sets_ = std::max(1, size_bytes_ / (line_size_ * assoc_));
        sets_.resize(static_cast<size_t>(num_sets_));
    }

    CacheAccess
    access(addr_t addr, bool is_write)
    {
        const auto line = static_cast<uint64_t>(addr) / static_cast<uint64_t>(line_size_);
        const auto set_idx = static_cast<size_t>(line % static_cast<uint64_t>(num_sets_));
        const auto tag = line / static_cast<uint64_t>(num_sets_);
        auto &set = sets_[set_idx];

        for (size_t i = 0; i < set.size(); ++i) {
            if (set[i].first == tag) {
                auto entry = set[i];
                set.erase(set.begin() + static_cast<long>(i));
                set.push_back({ tag, entry.second || is_write });
                return { hit_latency_, self_level_ };
            }
        }

        CacheAccess lower_access;
        if (self_level_ == HitLevel::L1) {
            auto *lower = static_cast<CacheLevel *>(lower_.get());
            lower_access = lower->access(addr, is_write);
        } else if (self_level_ == HitLevel::L2) {
            auto *lower = static_cast<CacheLevel *>(lower_.get());
            lower_access = lower->access(addr, is_write);
        } else {
            auto *lower = static_cast<MemoryLevel *>(lower_.get());
            lower_access = lower->access(addr, is_write);
        }

        if (set.size() >= static_cast<size_t>(assoc_))
            set.erase(set.begin());
        set.push_back({ tag, is_write });
        return { hit_latency_ + lower_access.latency, lower_access.hit_level };
    }

private:
    std::string name_;
    int size_bytes_;
    int assoc_;
    int line_size_;
    int hit_latency_;
    std::shared_ptr<void> lower_;
    HitLevel self_level_;
    HitLevel lower_level_;
    int num_sets_ = 1;
    std::vector<std::vector<std::pair<uint64_t, bool>>> sets_;
};

class ICacheModel {
public:
    explicit ICacheModel(const NativeConfig &config)
        : max_fills_(std::max(config.icache_max_fills, 1))
        , fill_latency_(std::max(config.icache_fill_latency, 1))
        , line_size_(std::max(config.icache_line_size, 1))
        , fetch_width_(std::max(config.icache_fetch_width, 1))
        , capacity_lines_(std::max(1, config.icache_size_bytes / std::max(config.icache_line_size, 1)))
    {
    }

    double
    on_instruction(uint64_t pc)
    {
        retire_completed(cur_time_);
        const uint64_t line = pc / static_cast<uint64_t>(line_size_);

        if (fetched_this_cycle_ >= fetch_width_) {
            cur_time_ += 1.0;
            fetched_this_cycle_ = 0;
            retire_completed(cur_time_);
        }

        auto it = cache_entries_.find(line);
        if (it != cache_entries_.end()) {
            touch(it);
            ++fetched_this_cycle_;
            return cur_time_;
        }

        auto inflight = inflight_.find(line);
        if (inflight != inflight_.end()) {
            ++fetched_this_cycle_;
            return inflight->second;
        }

        while (fills_.size() >= static_cast<size_t>(max_fills_)) {
            cur_time_ = fills_.top().first;
            fetched_this_cycle_ = 0;
            retire_completed(cur_time_);
        }

        const double completion = cur_time_ + static_cast<double>(fill_latency_);
        fills_.push({ completion, line });
        inflight_[line] = completion;
        ++fetched_this_cycle_;
        return completion;
    }

private:
    using LruList = std::list<uint64_t>;

    void
    touch(std::unordered_map<uint64_t, LruList::iterator>::iterator it)
    {
        lru_.splice(lru_.end(), lru_, it->second);
        it->second = std::prev(lru_.end());
    }

    void
    insert(uint64_t line)
    {
        auto existing = cache_entries_.find(line);
        if (existing != cache_entries_.end()) {
            touch(existing);
            return;
        }
        if (cache_entries_.size() >= static_cast<size_t>(capacity_lines_)) {
            const auto evicted = lru_.front();
            lru_.pop_front();
            cache_entries_.erase(evicted);
        }
        lru_.push_back(line);
        cache_entries_[line] = std::prev(lru_.end());
    }

    void
    retire_completed(double upto_time)
    {
        while (!fills_.empty() && fills_.top().first <= upto_time) {
            const std::pair<double, uint64_t> top = fills_.top();
            fills_.pop();
            inflight_.erase(top.second);
            insert(top.second);
        }
    }

    struct FillCompare {
        bool operator()(const std::pair<double, uint64_t> &lhs,
                        const std::pair<double, uint64_t> &rhs) const
        {
            return lhs.first > rhs.first;
        }
    };

    int max_fills_;
    int fill_latency_;
    int line_size_;
    int fetch_width_;
    int capacity_lines_;
    double cur_time_ = 0.0;
    int fetched_this_cycle_ = 0;
    std::priority_queue<std::pair<double, uint64_t>,
                        std::vector<std::pair<double, uint64_t>>,
                        FillCompare>
        fills_;
    std::unordered_map<uint64_t, double> inflight_;
    LruList lru_;
    std::unordered_map<uint64_t, LruList::iterator> cache_entries_;
};

class FetchBufferModel {
public:
    explicit FetchBufferModel(const NativeConfig &config)
        : fb_entries_(std::max(config.fetch_buffer_entries, 1))
        , decode_width_(std::max(config.decode_width, 1))
    {
    }

    double
    on_instruction(double ready_time)
    {
        if (ready_time > time_) {
            while (time_ + 1.0 <= ready_time && occupancy_ > 0) {
                do_one_cycle_decode(time_);
                time_ += 1.0;
            }
            if (time_ < ready_time)
                time_ = ready_time;
        }

        while (occupancy_ >= fb_entries_) {
            do_one_cycle_decode(time_);
            time_ += 1.0;
        }

        ++occupancy_;
        while (occupancy_ > 0 && decoded_pending_ < 1) {
            do_one_cycle_decode(time_);
            if (last_decoded_time_ == time_)
                break;
            time_ += 1.0;
        }
        if (decoded_pending_ == 0) {
            do_one_cycle_decode(time_);
            time_ += 1.0;
        }
        return last_decoded_time_;
    }

private:
    void
    do_one_cycle_decode(double cycle)
    {
        const int to_decode = std::min(decode_width_, occupancy_);
        if (to_decode <= 0) {
            decoded_pending_ = 0;
            return;
        }
        occupancy_ -= to_decode;
        decoded_pending_ = to_decode;
        last_decoded_time_ = cycle;
    }

    int fb_entries_;
    int decode_width_;
    int occupancy_ = 0;
    int decoded_pending_ = 0;
    double time_ = 0.0;
    double last_decoded_time_ = 0.0;
};

class SimplePredictor {
public:
    explicit SimplePredictor(const NativeConfig &config)
        : p_(config.simple_misprediction_rate)
        , rng_(config.simple_seed)
        , dist_(0.0, 1.0)
    {
    }

    bool
    update_and_count(uint64_t pc, bool actual_taken)
    {
        (void)pc;
        (void)actual_taken;
        return dist_(rng_) < p_;
    }

private:
    double p_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;
};

struct TageEntry {
    int ctr = 0;
    uint64_t tag = 0;
    int u = 0;
};

class TAGEPredictor {
public:
    explicit TAGEPredictor(const NativeConfig &config)
        : num_tables_(config.tage_num_tables)
        , table_size_(config.tage_table_size)
        , tag_bits_(config.tage_tag_bits)
        , ghr_bits_(config.tage_ghr_bits)
        , base_size_(config.tage_base_size)
        , ctr_bits_(config.tage_counter_bits)
        , u_bits_(config.tage_usefulness_bits)
        , rng_(config.tage_seed)
        , chance_(0.0, 1.0)
    {
        hist_lengths_.reserve(num_tables_);
        int length = 2;
        for (int i = 0; i < num_tables_; ++i) {
            hist_lengths_.push_back(std::min(length, ghr_bits_));
            length = static_cast<int>(length * 1.4);
        }
        base_table_.assign(static_cast<size_t>(base_size_), 0);
        tables_.resize(static_cast<size_t>(num_tables_));
        for (auto &table : tables_)
            table.assign(static_cast<size_t>(table_size_), TageEntry{});
    }

    bool
    update_and_count(uint64_t pc, bool actual_taken)
    {
        auto prediction = predict(pc);
        const bool is_misp = prediction.pred != actual_taken;
        update(pc, actual_taken, prediction);
        return is_misp;
    }

private:
    struct Prediction {
        bool pred = false;
        int provider = -1;
        bool alt_pred = false;
    };

    int
    saturate(int value, int bits) const
    {
        const int max_value = (1 << bits) - 1;
        if (value < 0)
            return 0;
        if (value > max_value)
            return max_value;
        return value;
    }

    uint64_t
    get_history_bits(int length) const
    {
        if (length >= 63)
            return ghr_;
        return ghr_ & ((uint64_t{1} << length) - 1);
    }

    uint64_t
    fold(uint64_t hist, int length, int out_bits) const
    {
        uint64_t result = 0;
        const uint64_t mask = (uint64_t{1} << out_bits) - 1;
        for (int i = 0; i < length; i += out_bits)
            result ^= (hist >> i) & mask;
        return result & mask;
    }

    std::pair<int, uint64_t>
    idx_tag(uint64_t pc, int table_index) const
    {
        const int history_len = hist_lengths_[static_cast<size_t>(table_index)];
        const uint64_t hist_bits = get_history_bits(history_len);
        const int idx_bits = std::max(1, static_cast<int>(std::log2(table_size_)));
        const uint64_t folded_idx = fold(hist_bits, history_len, idx_bits);
        const int idx =
            static_cast<int>((folded_idx ^ (pc & ((uint64_t{1} << idx_bits) - 1))) % static_cast<uint64_t>(table_size_));
        const uint64_t folded_tag = fold(hist_bits, history_len, tag_bits_);
        const uint64_t tag =
            folded_tag ^ ((pc >> idx_bits) & ((uint64_t{1} << tag_bits_) - 1));
        return { idx, tag };
    }

    bool
    base_pred(uint64_t pc) const
    {
        const int idx = static_cast<int>(pc % static_cast<uint64_t>(base_size_));
        return base_table_[static_cast<size_t>(idx)] >= (1 << (ctr_bits_ - 1));
    }

    bool
    ctr_pred(int ctr) const
    {
        return ctr >= (1 << (ctr_bits_ - 1));
    }

    Prediction
    predict(uint64_t pc) const
    {
        Prediction prediction;
        prediction.pred = base_pred(pc);
        prediction.alt_pred = prediction.pred;

        for (int i = num_tables_ - 1; i >= 0; --i) {
            const std::pair<int, uint64_t> location = idx_tag(pc, i);
            const TageEntry &entry =
                tables_[static_cast<size_t>(i)][static_cast<size_t>(location.first)];
            if (entry.tag != location.second)
                continue;
            if (prediction.provider < 0) {
                prediction.provider = i;
                prediction.pred = ctr_pred(entry.ctr);
            } else {
                prediction.alt_pred = ctr_pred(entry.ctr);
                break;
            }
        }
        return prediction;
    }

    void
    update(uint64_t pc, bool actual_taken, const Prediction &prediction)
    {
        const int base_idx = static_cast<int>(pc % static_cast<uint64_t>(base_size_));
        base_table_[static_cast<size_t>(base_idx)] =
            saturate(base_table_[static_cast<size_t>(base_idx)] + (actual_taken ? 1 : -1), ctr_bits_);

        if (prediction.provider >= 0) {
            const std::pair<int, uint64_t> location = idx_tag(pc, prediction.provider);
            TageEntry &entry =
                tables_[static_cast<size_t>(prediction.provider)][static_cast<size_t>(location.first)];
            entry.ctr = saturate(entry.ctr + (actual_taken ? 1 : -1), ctr_bits_);
            if (prediction.pred != prediction.alt_pred) {
                entry.u = saturate(entry.u + ((prediction.pred == actual_taken) ? 1 : -1), u_bits_);
            }
        }

        if (prediction.pred != actual_taken) {
            for (int i = num_tables_ - 1; i >= 0; --i) {
                if (prediction.provider >= 0 && i <= prediction.provider)
                    continue;
                const std::pair<int, uint64_t> location = idx_tag(pc, i);
                TageEntry &entry =
                    tables_[static_cast<size_t>(i)][static_cast<size_t>(location.first)];
                if (entry.u == 0 || chance_(rng_) < 0.1) {
                    entry.tag = location.second;
                    entry.ctr = (1 << (ctr_bits_ - 1)) + (actual_taken ? 1 : -1);
                    entry.u = 0;
                    break;
                }
            }
        }

        if (ghr_bits_ >= 63)
            ghr_ = (ghr_ << 1) | (actual_taken ? 1ULL : 0ULL);
        else
            ghr_ = ((ghr_ << 1) | (actual_taken ? 1ULL : 0ULL)) & ((uint64_t{1} << ghr_bits_) - 1);
    }

    int num_tables_;
    int table_size_;
    int tag_bits_;
    int ghr_bits_;
    int base_size_;
    int ctr_bits_;
    int u_bits_;
    std::vector<int> hist_lengths_;
    std::vector<int> base_table_;
    std::vector<std::vector<TageEntry>> tables_;
    uint64_t ghr_ = 0;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> chance_;
};

class ThreadAnalyzer {
public:
    ThreadAnalyzer(memref_tid_t tid, const NativeConfig &config, int shard_index = -1,
                   int64_t input_id = -1, std::string stream_name = "")
        : tid_(tid)
        , shard_index_(shard_index)
        , input_id_(input_id)
        , stream_name_(std::move(stream_name))
        , config_(config)
        , memory_(std::make_shared<MemoryLevel>(config.memory_latency))
        , l3_("L3", config.l3_size_bytes, config.l3_associativity, config.cache_line_size,
              config.l3_hit_latency, memory_, HitLevel::L3, HitLevel::MEM)
        , l2_("L2", config.l2_size_bytes, config.l2_associativity, config.cache_line_size,
              config.l2_hit_latency, std::shared_ptr<void>(&l3_, [](void *) {}), HitLevel::L2, HitLevel::L3)
        , l1_("L1", config.l1_size_bytes, config.l1_associativity, config.cache_line_size,
              config.l1_hit_latency, std::shared_ptr<void>(&l2_, [](void *) {}), HitLevel::L1, HitLevel::L2)
        , icache_(config)
        , fetch_buffer_(config)
        , simple_bp_(config)
        , tage_bp_(config)
        , commit_ring_(static_cast<size_t>(std::max(config.rob_entries + 1, 2)), 0)
    {
    }

    void
    update_metadata(memref_tid_t tid, int shard_index, int64_t input_id, const std::string &stream_name)
    {
        tid_ = tid;
        if (shard_index >= 0)
            shard_index_ = shard_index;
        if (input_id >= 0)
            input_id_ = input_id;
        if (!stream_name.empty())
            stream_name_ = stream_name;
    }

    void
    process_instruction(const CurrentInstruction &instruction)
    {
        ++instruction_count_;
        const uint64_t index = instruction_count_;
        const double icache_ready = icache_.on_instruction(static_cast<uint64_t>(instruction.pc));
        const double decode_time = fetch_buffer_.on_instruction(icache_ready);

        uint64_t dep_max = dependency_max_from_regs(instruction.decoded.uses_mask);
        InstructionKind kind = InstructionKind::NON_MEM;
        IssueGroup issue_group = instruction.decoded.issue_group;
        uint64_t load_address = 0;
        int load_latency = 0;
        std::vector<uint64_t> written_lines;

        if (instruction.decoded.branch_kind == BranchKind::DIRECT_UNCONDITIONAL)
            ++window_direct_unconditional_;
        else if (instruction.decoded.branch_kind == BranchKind::CONDITIONAL)
            ++window_direct_conditional_;
        else if (instruction.decoded.branch_kind == BranchKind::INDIRECT)
            ++window_indirect_;

        if (instruction.decoded.branch_kind == BranchKind::CONDITIONAL &&
            instruction.decoded.branch_taken_known) {
            ++simple_total_.total;
            ++tage_total_.total;
            if (simple_bp_.update_and_count(static_cast<uint64_t>(instruction.pc),
                                            instruction.decoded.branch_taken))
                ++simple_total_.misp;
            if (tage_bp_.update_and_count(static_cast<uint64_t>(instruction.pc),
                                          instruction.decoded.branch_taken))
                ++tage_total_.misp;
        }

        for (const auto &memop : instruction.memops) {
            if (!memop.is_write) {
                auto access = l1_.access(memop.addr, false);
                const auto covered = cache_lines_covered(memop.addr, memop.size, config_.cache_line_size);
                for (auto line : covered) {
                    const auto found = last_store_finish_by_line_.find(line);
                    if (found != last_store_finish_by_line_.end())
                        dep_max = std::max(dep_max, found->second);
                }
                if (kind != InstructionKind::LOAD) {
                    kind = InstructionKind::LOAD;
                    issue_group = IssueGroup::LS;
                    load_address = static_cast<uint64_t>(memop.addr);
                    load_latency = latency_from_hit(access.hit_level);
                }
            } else {
                (void)l1_.access(memop.addr, true);
                kind = InstructionKind::STORE;
                issue_group = IssueGroup::LS;
                const auto covered = cache_lines_covered(memop.addr, memop.size, config_.cache_line_size);
                written_lines.insert(written_lines.end(), covered.begin(), covered.end());
            }
        }

        ++window_instruction_count_;
        increment_issue_group(issue_group);
        if (kind == InstructionKind::LOAD)
            ++window_loads_;
        else if (kind == InstructionKind::STORE)
            ++window_stores_;

        const uint64_t c_i_rob =
            (index > static_cast<uint64_t>(config_.rob_entries))
                ? commit_ring_[(index - static_cast<uint64_t>(config_.rob_entries)) % commit_ring_.size()]
                : 0;
        const uint64_t arrival = std::max(prev_arrival_, c_i_rob);
        const uint64_t start = std::max(arrival, dep_max);
        const uint64_t finish =
            (kind == InstructionKind::LOAD)
                ? load_resp_cycle(start, load_address, load_latency)
                : (start + 1);
        const uint64_t commit = std::max(prev_commit_, finish);
        prev_arrival_ = arrival;
        prev_commit_ = commit;
        commit_ring_[index % commit_ring_.size()] = commit;

        update_defs(instruction.decoded.defs_mask, finish);
        for (auto line : written_lines)
            last_store_finish_by_line_[line] = finish;

        if (index % static_cast<uint64_t>(config_.window_size) == 0)
            finalize_window(commit, icache_ready, decode_time);
    }

    ThreadResults
    finish() const
    {
        ThreadResults results;
        results.tid = static_cast<int64_t>(tid_);
        results.shard_index = shard_index_;
        results.input_id = input_id_;
        results.stream_name = stream_name_;
        results.instruction_count = instruction_count_;
        results.window_size = static_cast<uint64_t>(config_.window_size);
        results.rob_avg_ipc =
            (prev_commit_ > 0) ? static_cast<double>(instruction_count_) / static_cast<double>(prev_commit_) : 0.0;
        results.simple_stats = simple_total_;
        results.tage_stats = tage_total_;
        results.series = series_;
        return results;
    }

    uint64_t
    instruction_count() const
    {
        return instruction_count_;
    }

private:
    void
    increment_issue_group(IssueGroup group)
    {
        switch (group) {
        case IssueGroup::ALU: ++window_alu_; break;
        case IssueGroup::FP: ++window_fp_; break;
        case IssueGroup::LS: ++window_ls_; break;
        }
    }

    int
    latency_from_hit(HitLevel hit_level) const
    {
        switch (hit_level) {
        case HitLevel::L1: return 4;
        case HitLevel::L2: return 12;
        case HitLevel::L3: return 35;
        case HitLevel::MEM: return 200;
        }
        return 200;
    }

    uint64_t
    dependency_max_from_regs(uint64_t uses_mask) const
    {
        uint64_t dep_max = 0;
        while (uses_mask != 0) {
            const uint64_t lsb = uses_mask & (~uses_mask + 1);
            const int index = trailing_zero_index(uses_mask);
            dep_max = std::max(dep_max, last_def_finish_[static_cast<size_t>(index)]);
            uses_mask ^= lsb;
        }
        return dep_max;
    }

    void
    update_defs(uint64_t defs_mask, uint64_t finish)
    {
        while (defs_mask != 0) {
            const uint64_t lsb = defs_mask & (~defs_mask + 1);
            const int index = trailing_zero_index(defs_mask);
            last_def_finish_[static_cast<size_t>(index)] = finish;
            defs_mask ^= lsb;
        }
    }

    uint64_t
    load_resp_cycle(uint64_t req_cycle, uint64_t addr, int latency)
    {
        const uint64_t line = addr / static_cast<uint64_t>(config_.cache_line_size);
        auto &last_req = last_load_req_cycle_[line];
        auto &last_rsp = last_load_rsp_cycle_[line];
        if (req_cycle < last_req)
            req_cycle = last_req;
        last_req = req_cycle;
        const uint64_t rsp = std::max(req_cycle + static_cast<uint64_t>(latency), last_rsp);
        last_rsp = rsp;
        return rsp;
    }

    void
    finalize_window(uint64_t commit, double icache_ready, double decode_time)
    {
        const int k = config_.window_size;
        const double rob_delta = static_cast<double>(commit - prev_window_commit_);
        series_.rob_thr_chunks.push_back(rob_delta > 0.0 ? static_cast<double>(k) / rob_delta
                                                         : std::numeric_limits<double>::infinity());
        prev_window_commit_ = commit;

        series_.static_fetch_width.push_back(static_cast<double>(config_.fetch_width));
        series_.static_decode_width.push_back(static_cast<double>(config_.decode_width));
        series_.static_rename_width.push_back(static_cast<double>(config_.rename_width));
        series_.static_commit_width.push_back(static_cast<double>(config_.commit_width));
        series_.static_alu_issue_width.push_back(issue_group_thr(window_alu_, config_.alu_issue_width));
        series_.static_fp_issue_width.push_back(issue_group_thr(window_fp_, config_.fp_issue_width));
        series_.static_ls_issue_width.push_back(issue_group_thr(window_ls_, config_.ls_issue_width));

        const double total_pipes = static_cast<double>(std::max(config_.load_store_pipes + config_.load_only_pipes, 1));
        const double lower_time =
            (window_loads_ > 0 ? static_cast<double>(window_loads_) / total_pipes : 0.0) +
            (window_stores_ > 0 ? static_cast<double>(window_stores_) /
                                      static_cast<double>(std::max(config_.load_store_pipes, 1))
                                : 0.0);
        series_.dyn_pipes_thr_lower.push_back(lower_time > 0.0 ? static_cast<double>(k) / lower_time
                                                               : std::numeric_limits<double>::infinity());

        const int t_store =
            (window_stores_ + std::max(config_.load_store_pipes, 1) - 1) /
            std::max(config_.load_store_pipes, 1);
        const int issued_loads = t_store * std::max(config_.load_only_pipes, 0);
        const int remaining_loads = std::max(0, window_loads_ - issued_loads);
        const int t_remaining =
            (remaining_loads + std::max(config_.load_store_pipes + config_.load_only_pipes, 1) - 1) /
            std::max(config_.load_store_pipes + config_.load_only_pipes, 1);
        const int min_time = t_store + t_remaining;
        series_.dyn_pipes_thr_upper.push_back(min_time > 0 ? static_cast<double>(k) / static_cast<double>(min_time)
                                                           : std::numeric_limits<double>::infinity());

        const double icache_delta = icache_ready - prev_window_icache_ready_;
        series_.dyn_icache_fills_thr.push_back(icache_delta > 0.0 ? static_cast<double>(k) / icache_delta
                                                                  : std::numeric_limits<double>::infinity());
        prev_window_icache_ready_ = icache_ready;

        const double decode_delta = decode_time - prev_window_decode_time_;
        series_.dyn_fb_decode_thr.push_back(decode_delta > 0.0 ? static_cast<double>(k) / decode_delta
                                                               : std::numeric_limits<double>::infinity());
        prev_window_decode_time_ = decode_time;

        series_.br_type_direct_unconditional.push_back(static_cast<double>(window_direct_unconditional_));
        series_.br_type_direct_conditional.push_back(static_cast<double>(window_direct_conditional_));
        series_.br_type_indirect.push_back(static_cast<double>(window_indirect_));

        window_instruction_count_ = 0;
        window_alu_ = 0;
        window_fp_ = 0;
        window_ls_ = 0;
        window_loads_ = 0;
        window_stores_ = 0;
        window_direct_unconditional_ = 0;
        window_direct_conditional_ = 0;
        window_indirect_ = 0;
    }

    double
    issue_group_thr(int count, int width) const
    {
        if (count <= 0)
            return std::numeric_limits<double>::infinity();
        const int time = (count + std::max(width, 1) - 1) / std::max(width, 1);
        return static_cast<double>(config_.window_size) / static_cast<double>(time);
    }

    memref_tid_t tid_;
    int shard_index_ = -1;
    int64_t input_id_ = -1;
    std::string stream_name_;
    NativeConfig config_;
    std::shared_ptr<MemoryLevel> memory_;
    CacheLevel l3_;
    CacheLevel l2_;
    CacheLevel l1_;
    ICacheModel icache_;
    FetchBufferModel fetch_buffer_;
    SimplePredictor simple_bp_;
    TAGEPredictor tage_bp_;

    uint64_t instruction_count_ = 0;
    std::array<uint64_t, 64> last_def_finish_{};
    std::unordered_map<uint64_t, uint64_t> last_store_finish_by_line_;
    std::unordered_map<uint64_t, uint64_t> last_load_req_cycle_;
    std::unordered_map<uint64_t, uint64_t> last_load_rsp_cycle_;
    std::vector<uint64_t> commit_ring_;
    uint64_t prev_arrival_ = 0;
    uint64_t prev_commit_ = 0;
    uint64_t prev_window_commit_ = 0;
    double prev_window_icache_ready_ = 0.0;
    double prev_window_decode_time_ = 0.0;

    int window_instruction_count_ = 0;
    int window_alu_ = 0;
    int window_fp_ = 0;
    int window_ls_ = 0;
    int window_loads_ = 0;
    int window_stores_ = 0;
    int window_direct_unconditional_ = 0;
    int window_direct_conditional_ = 0;
    int window_indirect_ = 0;

    BranchStats simple_total_;
    BranchStats tage_total_;
    ThroughputSeries series_;
};

class concorde_analyzer_tool_t : public analysis_tool_t {
public:
    explicit concorde_analyzer_tool_t(const Options &options, const NativeConfig &config)
        : options_(options)
        , config_(config)
    {
    }

    bool
    parallel_shard_supported() override
    {
        return options_.max_instructions == 0;
    }

    std::string
    initialize_stream(memtrace_stream_t *serial_stream) override
    {
        if (serial_stream == nullptr) {
            if (parallel_shard_supported())
                return "";
            return "concorde_native_analyzer requires a serial memtrace stream";
        }
        stream_ = serial_stream;
        return "";
    }

    bool
    process_memref(const memref_t &entry) override
    {
        if (stop_processing_)
            return true;

        const trace_type_t type = entry.data.type;
        if (is_any_instr_type(type))
            return handle_instruction(entry.instr);
        if (type == TRACE_TYPE_READ || type == TRACE_TYPE_WRITE)
            return handle_data_ref(entry.data);
        if (type == TRACE_TYPE_THREAD_EXIT) {
            if (current_ && current_->tid == entry.exit.tid)
                flush_current(false, 0);
            return true;
        }
        if (type_is_prefetch(type))
            return true;
        return true;
    }

    bool
    print_results() override
    {
        if (stream_ != nullptr)
            flush_current(false, 0);
        return true;
    }

    void *
    parallel_shard_init_stream(int shard_index, void *worker_data, memtrace_stream_t *shard_stream) override
    {
        (void)worker_data;
        return new ParallelShardState(shard_index, shard_stream, config_);
    }

    bool
    parallel_shard_memref(void *shard_data, const memref_t &entry) override
    {
        auto *state = reinterpret_cast<ParallelShardState *>(shard_data);
        try {
            return process_parallel_memref(state, entry);
        } catch (const std::exception &ex) {
            state->error = ex.what();
            return false;
        }
    }

    bool
    parallel_shard_exit(void *shard_data) override
    {
        std::unique_ptr<ParallelShardState> state(reinterpret_cast<ParallelShardState *>(shard_data));
        if (state == nullptr)
            return true;
        try {
            flush_parallel_current(state.get(), false, 0);
        } catch (const std::exception &ex) {
            state->error = ex.what();
            return false;
        }
        if (state->analyzer) {
            std::lock_guard<std::mutex> lock(results_mutex_);
            parallel_thread_results_.push_back(state->analyzer->finish());
        }
        return true;
    }

    std::string
    parallel_shard_error(void *shard_data) override
    {
        auto *state = reinterpret_cast<ParallelShardState *>(shard_data);
        return state != nullptr ? state->error : "";
    }

    NativeResults
    results() const
    {
        NativeResults output;
        std::vector<ThreadResults> all_results;
        all_results.reserve(analyzers_.size() + parallel_thread_results_.size());
        for (const auto &item : analyzers_) {
            all_results.push_back(item.second->finish());
        }
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            all_results.insert(all_results.end(), parallel_thread_results_.begin(), parallel_thread_results_.end());
        }
        std::sort(all_results.begin(), all_results.end(),
                  [](const ThreadResults &lhs, const ThreadResults &rhs) {
                      if (lhs.instruction_count != rhs.instruction_count)
                          return lhs.instruction_count > rhs.instruction_count;
                      return lhs.tid < rhs.tid;
                  });
        output.thread_results = all_results;
        if (!all_results.empty()) {
            const auto &best = all_results.front();
            output.target_tid = best.tid;
            output.target_shard_index = best.shard_index;
            output.target_input_id = best.input_id;
            output.target_stream_name = best.stream_name;
            output.instruction_count = best.instruction_count;
            output.window_size = best.window_size;
            output.rob_avg_ipc = best.rob_avg_ipc;
            output.simple_stats = best.simple_stats;
            output.tage_stats = best.tage_stats;
            output.series = best.series;
        }
        return output;
    }

private:
    struct ParallelShardState {
        ParallelShardState(int shard_index_value, memtrace_stream_t *shard_stream,
                           const NativeConfig &config)
            : stream(shard_stream)
            , shard_index(shard_index_value)
        {
            if (stream != nullptr) {
                const auto stream_tid = stream->get_tid();
                if (stream_tid >= 0)
                    tid = static_cast<memref_tid_t>(stream_tid);
                input_id = stream->get_input_id();
                stream_name = stream->get_stream_name();
            }
            analyzer.reset(new ThreadAnalyzer(tid, config, shard_index, input_id, stream_name));
        }

        memtrace_stream_t *stream = nullptr;
        int shard_index = -1;
        memref_tid_t tid = 0;
        int64_t input_id = -1;
        std::string stream_name;
        std::unordered_map<addr_t, DecodedInstruction> decoded_cache;
        std::unique_ptr<CurrentInstruction> current;
        std::unique_ptr<ThreadAnalyzer> analyzer;
        std::string error;
    };

    DecodedInstruction
    decode_instruction(std::unordered_map<addr_t, DecodedInstruction> &decoded_cache,
                       const _memref_instr_t &instr_memref) const
    {
        const addr_t pc = instr_memref.addr;
        const bool refresh =
            instr_memref.encoding_is_new || decoded_cache.find(pc) == decoded_cache.end();
        if (!refresh)
            return decoded_cache.at(pc);

        DecodedInstruction decoded;
        instr_t instr;
        instr_init(GLOBAL_DCONTEXT, &instr);
        const auto *decode_end = decode_from_copy(
            GLOBAL_DCONTEXT,
            const_cast<byte *>(reinterpret_cast<const byte *>(instr_memref.encoding)),
            reinterpret_cast<byte *>(pc),
            &instr);

        if (decode_end != nullptr && instr_get_opcode(&instr) != OP_INVALID) {
            const char *opcode_name = decode_opcode_name(instr_get_opcode(&instr));
            decoded.mnemonic =
                opcode_name != nullptr ? normalize_mnemonic(opcode_name) : fallback_mnemonic(instr_memref.type);
            decoded.issue_group = classify_issue_group(decoded.mnemonic);
            decoded.branch_kind = classify_branch_kind(instr_memref.type, decoded.mnemonic);

            const int src_count = instr_num_srcs(&instr);
            for (int i = 0; i < src_count; ++i) {
                const opnd_t src = instr_get_src(&instr, i);
                const int reg_count = opnd_num_regs_used(src);
                for (int reg_idx = 0; reg_idx < reg_count; ++reg_idx) {
                    const int tracked = tracked_reg_index(opnd_get_reg_used(src, reg_idx));
                    if (tracked >= 0)
                        decoded.uses_mask |= (uint64_t{1} << tracked);
                }
            }

            const int dst_count = instr_num_dsts(&instr);
            for (int i = 0; i < dst_count; ++i) {
                const opnd_t dst = instr_get_dst(&instr, i);
                if (opnd_is_memory_reference(dst)) {
                    const int reg_count = opnd_num_regs_used(dst);
                    for (int reg_idx = 0; reg_idx < reg_count; ++reg_idx) {
                        const int tracked = tracked_reg_index(opnd_get_reg_used(dst, reg_idx));
                        if (tracked >= 0)
                            decoded.uses_mask |= (uint64_t{1} << tracked);
                    }
                    continue;
                }
                if (opnd_is_reg(dst)) {
                    const int tracked = tracked_reg_index(opnd_get_reg(dst));
                    if (tracked >= 0)
                        decoded.defs_mask |= (uint64_t{1} << tracked);
                }
            }

            if (type_is_instr_conditional_branch(instr_memref.type) ||
                type_is_instr_direct_branch(instr_memref.type)) {
                const opnd_t target = instr_get_target(&instr);
                if (opnd_is_pc(target)) {
                    decoded.has_direct_target = true;
                    decoded.direct_target = reinterpret_cast<addr_t>(opnd_get_pc(target));
                }
            }
        } else {
            decoded.mnemonic = fallback_mnemonic(instr_memref.type);
            decoded.issue_group = classify_issue_group(decoded.mnemonic);
            decoded.branch_kind = classify_branch_kind(instr_memref.type, decoded.mnemonic);
        }

        if (instr_memref.type == TRACE_TYPE_INSTR_TAKEN_JUMP) {
            decoded.branch_taken_known = true;
            decoded.branch_taken = true;
        } else if (instr_memref.type == TRACE_TYPE_INSTR_UNTAKEN_JUMP) {
            decoded.branch_taken_known = true;
            decoded.branch_taken = false;
        }

        instr_free(GLOBAL_DCONTEXT, &instr);
        decoded_cache[pc] = decoded;
        return decoded;
    }

    void
    flush_current(bool has_next_pc, addr_t next_pc)
    {
        if (!current_)
            return;

        auto &current = *current_;
        if (!current.decoded.branch_taken_known && has_next_pc && current.decoded.has_direct_target) {
            current.decoded.branch_taken_known = true;
            current.decoded.branch_taken = (current.decoded.direct_target == next_pc);
        }

        auto &analyzer = get_or_create_analyzer(current.tid);
        analyzer.process_instruction(current);
        ++emitted_instruction_count_;
        current_.reset();

        if (options_.max_instructions > 0 && emitted_instruction_count_ >= options_.max_instructions) {
            stop_processing_ = true;
            throw StopAnalysis();
        }
    }

    void
    flush_parallel_current(ParallelShardState *state, bool has_next_pc, addr_t next_pc)
    {
        if (state == nullptr || !state->current)
            return;

        auto &current = *state->current;
        if (!current.decoded.branch_taken_known && has_next_pc && current.decoded.has_direct_target) {
            current.decoded.branch_taken_known = true;
            current.decoded.branch_taken = (current.decoded.direct_target == next_pc);
        }

        state->analyzer->update_metadata(current.tid, state->shard_index, state->input_id, state->stream_name);
        state->analyzer->process_instruction(current);
        state->current.reset();
    }

    bool
    handle_instruction(const _memref_instr_t &instr_memref)
    {
        flush_current(true, instr_memref.addr);
        const auto instruction_ordinal = stream_->get_instruction_ordinal();
        const auto decoded = decode_instruction(decoded_cache_, instr_memref);
        current_.reset(new CurrentInstruction());
        current_->instruction_ordinal = instruction_ordinal;
        current_->tid = instr_memref.tid;
        current_->pc = instr_memref.addr;
        current_->decoded = decoded;
        return true;
    }

    bool
    handle_parallel_instruction(ParallelShardState *state, const _memref_instr_t &instr_memref)
    {
        flush_parallel_current(state, true, instr_memref.addr);
        const auto instruction_ordinal =
            state->stream != nullptr ? state->stream->get_instruction_ordinal() : 0;
        const auto decoded = decode_instruction(state->decoded_cache, instr_memref);
        state->current.reset(new CurrentInstruction());
        state->current->instruction_ordinal = instruction_ordinal;
        state->current->tid = instr_memref.tid;
        state->current->pc = instr_memref.addr;
        state->current->decoded = decoded;
        state->tid = instr_memref.tid;
        if (state->analyzer)
            state->analyzer->update_metadata(state->tid, state->shard_index, state->input_id, state->stream_name);
        return true;
    }

    bool
    handle_data_ref(const _memref_data_t &data_memref)
    {
        if (!current_)
            return true;
        current_->memops.push_back(
            MemOp{ data_memref.type == TRACE_TYPE_WRITE, data_memref.addr, static_cast<uint32_t>(data_memref.size) });
        return true;
    }

    bool
    handle_parallel_data_ref(ParallelShardState *state, const _memref_data_t &data_memref)
    {
        if (state == nullptr || !state->current)
            return true;
        state->current->memops.push_back(
            MemOp{ data_memref.type == TRACE_TYPE_WRITE, data_memref.addr, static_cast<uint32_t>(data_memref.size) });
        return true;
    }

    bool
    process_parallel_memref(ParallelShardState *state, const memref_t &entry)
    {
        if (state == nullptr)
            return false;
        const trace_type_t type = entry.data.type;
        if (is_any_instr_type(type))
            return handle_parallel_instruction(state, entry.instr);
        if (type == TRACE_TYPE_READ || type == TRACE_TYPE_WRITE)
            return handle_parallel_data_ref(state, entry.data);
        if (type == TRACE_TYPE_THREAD_EXIT) {
            if (state->current && state->current->tid == entry.exit.tid)
                flush_parallel_current(state, false, 0);
            return true;
        }
        if (type_is_prefetch(type))
            return true;
        return true;
    }

    ThreadAnalyzer &
    get_or_create_analyzer(memref_tid_t tid)
    {
        auto found = analyzers_.find(tid);
        if (found != analyzers_.end()) {
            if (stream_ != nullptr) {
                found->second->update_metadata(
                    tid, stream_->get_shard_index(), stream_->get_input_id(), stream_->get_stream_name());
            }
            return *found->second;
        }
        auto inserted = analyzers_.emplace(
            tid,
            std::unique_ptr<ThreadAnalyzer>(new ThreadAnalyzer(
                tid, config_, stream_ != nullptr ? stream_->get_shard_index() : -1,
                stream_ != nullptr ? stream_->get_input_id() : -1,
                stream_ != nullptr ? stream_->get_stream_name() : "")));
        return *inserted.first->second;
    }

    Options options_;
    NativeConfig config_;
    memtrace_stream_t *stream_ = nullptr;
    std::unordered_map<addr_t, DecodedInstruction> decoded_cache_;
    std::unique_ptr<CurrentInstruction> current_;
    std::unordered_map<memref_tid_t, std::unique_ptr<ThreadAnalyzer>> analyzers_;
    mutable std::mutex results_mutex_;
    std::vector<ThreadResults> parallel_thread_results_;
    uint64_t emitted_instruction_count_ = 0;
    bool stop_processing_ = false;
};

void
write_json_string(std::ostream &out, const std::string &value)
{
    out << '"';
    for (char c : value) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << c; break;
        }
    }
    out << '"';
}

void
write_json_number(std::ostream &out, double value)
{
    if (std::isnan(value))
        out << "NaN";
    else if (std::isinf(value))
        out << (value > 0 ? "Infinity" : "-Infinity");
    else
        out << std::setprecision(12) << value;
}

void
write_json_vector(std::ostream &out, const std::vector<double> &values)
{
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0)
            out << ',';
        write_json_number(out, values[i]);
    }
    out << ']';
}

void
write_throughput_series_json(std::ostream &out, const ThroughputSeries &series, int indent)
{
    const std::string pad(static_cast<size_t>(indent), ' ');
    const std::string child_pad(static_cast<size_t>(indent + 2), ' ');
    out << "{\n";
    out << child_pad << "\"BR.TYPE.Direct Unconditional Branch\": ";
    write_json_vector(out, series.br_type_direct_unconditional);
    out << ",\n";
    out << child_pad << "\"BR.TYPE.Direct Conditional Branch\": ";
    write_json_vector(out, series.br_type_direct_conditional);
    out << ",\n";
    out << child_pad << "\"BR.TYPE.Indirect Branch\": ";
    write_json_vector(out, series.br_type_indirect);
    out << ",\n";
    out << child_pad << "\"ROB.thr_chunks\": ";
    write_json_vector(out, series.rob_thr_chunks);
    out << ",\n";
    out << child_pad << "\"STATIC.fetch_width\": ";
    write_json_vector(out, series.static_fetch_width);
    out << ",\n";
    out << child_pad << "\"STATIC.decode_width\": ";
    write_json_vector(out, series.static_decode_width);
    out << ",\n";
    out << child_pad << "\"STATIC.rename_width\": ";
    write_json_vector(out, series.static_rename_width);
    out << ",\n";
    out << child_pad << "\"STATIC.commit_width\": ";
    write_json_vector(out, series.static_commit_width);
    out << ",\n";
    out << child_pad << "\"STATIC.alu_issue_width\": ";
    write_json_vector(out, series.static_alu_issue_width);
    out << ",\n";
    out << child_pad << "\"STATIC.fp_issue_width\": ";
    write_json_vector(out, series.static_fp_issue_width);
    out << ",\n";
    out << child_pad << "\"STATIC.ls_issue_width\": ";
    write_json_vector(out, series.static_ls_issue_width);
    out << ",\n";
    out << child_pad << "\"DYN.pipes_thr_lower\": ";
    write_json_vector(out, series.dyn_pipes_thr_lower);
    out << ",\n";
    out << child_pad << "\"DYN.pipes_thr_upper\": ";
    write_json_vector(out, series.dyn_pipes_thr_upper);
    out << ",\n";
    out << child_pad << "\"DYN.icache_fills_thr\": ";
    write_json_vector(out, series.dyn_icache_fills_thr);
    out << ",\n";
    out << child_pad << "\"DYN.fb_decode_thr\": ";
    write_json_vector(out, series.dyn_fb_decode_thr);
    out << '\n' << pad << '}';
}

void
write_branch_prediction_json(std::ostream &out, const BranchStats &simple_stats,
                             const BranchStats &tage_stats, const ThroughputSeries &series,
                             int indent)
{
    const std::string pad(static_cast<size_t>(indent), ' ');
    const std::string child_pad(static_cast<size_t>(indent + 2), ' ');
    out << "{\n";
    out << child_pad << "\"simple\": {\n";
    out << child_pad << "  \"total\": " << simple_stats.total << ",\n";
    out << child_pad << "  \"misp\": " << simple_stats.misp << ",\n";
    out << child_pad << "  \"misp_rate\": ";
    write_json_number(out, simple_stats.total > 0
                               ? static_cast<double>(simple_stats.misp) /
                                     static_cast<double>(simple_stats.total)
                               : 0.0);
    out << ",\n";
    out << child_pad << "  \"by_type\": {\n";
    out << child_pad << "    \"Conditional Branch\": {\"total\": " << simple_stats.total
        << ", \"misp\": " << simple_stats.misp << ", \"misp_rate\": ";
    write_json_number(out, simple_stats.total > 0
                               ? static_cast<double>(simple_stats.misp) /
                                     static_cast<double>(simple_stats.total)
                               : 0.0);
    out << "}\n";
    out << child_pad << "  }\n";
    out << child_pad << "},\n";
    out << child_pad << "\"tage\": {\n";
    out << child_pad << "  \"total\": " << tage_stats.total << ",\n";
    out << child_pad << "  \"misp\": " << tage_stats.misp << ",\n";
    out << child_pad << "  \"misp_rate\": ";
    write_json_number(out, tage_stats.total > 0
                               ? static_cast<double>(tage_stats.misp) /
                                     static_cast<double>(tage_stats.total)
                               : 0.0);
    out << ",\n";
    out << child_pad << "  \"by_type\": {\n";
    out << child_pad << "    \"Conditional Branch\": {\"total\": " << tage_stats.total
        << ", \"misp\": " << tage_stats.misp << ", \"misp_rate\": ";
    write_json_number(out, tage_stats.total > 0
                               ? static_cast<double>(tage_stats.misp) /
                                     static_cast<double>(tage_stats.total)
                               : 0.0);
    out << "}\n";
    out << child_pad << "  }\n";
    out << child_pad << "},\n";
    out << child_pad << "\"branch_type_distribution\": {\n";
    out << child_pad << "  \"Direct Unconditional Branch\": ";
    write_json_vector(out, series.br_type_direct_unconditional);
    out << ",\n";
    out << child_pad << "  \"Direct Conditional Branch\": ";
    write_json_vector(out, series.br_type_direct_conditional);
    out << ",\n";
    out << child_pad << "  \"Indirect Branch\": ";
    write_json_vector(out, series.br_type_indirect);
    out << "\n" << child_pad << "}\n";
    out << pad << '}';
}

void
write_thread_result_json(std::ostream &out, const ThreadResults &result, int indent)
{
    const std::string pad(static_cast<size_t>(indent), ' ');
    out << "{\n";
    out << pad << "  \"tid\": " << result.tid << ",\n";
    out << pad << "  \"shard_index\": " << result.shard_index << ",\n";
    out << pad << "  \"input_id\": " << result.input_id << ",\n";
    out << pad << "  \"stream_name\": ";
    write_json_string(out, result.stream_name);
    out << ",\n";
    out << pad << "  \"instruction_count\": " << result.instruction_count << ",\n";
    out << pad << "  \"window_size\": " << result.window_size << ",\n";
    out << pad << "  \"rob_avg_ipc\": ";
    write_json_number(out, result.rob_avg_ipc);
    out << ",\n";
    out << pad << "  \"throughput_series\": ";
    write_throughput_series_json(out, result.series, indent + 2);
    out << ",\n";
    out << pad << "  \"branch_prediction\": ";
    write_branch_prediction_json(out, result.simple_stats, result.tage_stats, result.series, indent + 2);
    out << '\n' << pad << '}';
}

void
write_results_json(const NativeResults &results, const std::unordered_map<std::string, double> &timings,
                   const std::string &path)
{
    std::ofstream out(path);
    out << "{\n";
    out << "  \"target_tid\": " << results.target_tid << ",\n";
    out << "  \"target_shard_index\": " << results.target_shard_index << ",\n";
    out << "  \"target_input_id\": " << results.target_input_id << ",\n";
    out << "  \"target_stream_name\": ";
    write_json_string(out, results.target_stream_name);
    out << ",\n";
    out << "  \"instruction_count\": " << results.instruction_count << ",\n";
    out << "  \"window_size\": " << results.window_size << ",\n";
    out << "  \"rob_avg_ipc\": ";
    write_json_number(out, results.rob_avg_ipc);
    out << ",\n";
    out << "  \"thread_count\": " << results.thread_results.size() << ",\n";
    out << "  \"throughput_series\": ";
    write_throughput_series_json(out, results.series, 2);
    out << ",\n";
    out << "  \"branch_prediction\": ";
    write_branch_prediction_json(out, results.simple_stats, results.tage_stats, results.series, 2);
    out << ",\n";
    out << "  \"thread_results\": [\n";
    for (size_t i = 0; i < results.thread_results.size(); ++i) {
        out << "    ";
        write_thread_result_json(out, results.thread_results[i], 4);
        if (i + 1 < results.thread_results.size())
            out << ',';
        out << '\n';
    }
    out << "  ],\n";
    out << "  \"timings\": {\n";
    bool first = true;
    for (const auto &item : timings) {
        if (!first)
            out << ",\n";
        out << "    ";
        write_json_string(out, item.first);
        out << ": ";
        write_json_number(out, item.second);
        first = false;
    }
    out << "\n  }\n";
    out << "}\n";
}

void
print_usage(const char *argv0)
{
    std::cerr << "Usage: " << argv0
              << " --trace-dir <drmemtrace.dir|op_dir> --config <native_cfg> --output-json <path>"
              << " [--max-instructions N] [--worker-count N]\n";
}

bool
parse_args(int argc, char **argv, Options *options, bool *showed_help)
{
    *showed_help = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--trace-dir" && i + 1 < argc)
            options->trace_dir = argv[++i];
        else if (arg == "--config" && i + 1 < argc)
            options->config_path = argv[++i];
        else if (arg == "--output-json" && i + 1 < argc)
            options->output_json = argv[++i];
        else if (arg == "--max-instructions" && i + 1 < argc)
            options->max_instructions = std::strtoull(argv[++i], nullptr, 10);
        else if (arg == "--worker-count" && i + 1 < argc)
            options->worker_count = std::max(0, std::atoi(argv[++i]));
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            *showed_help = true;
            return false;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }

    if (options->trace_dir.empty() || options->config_path.empty() || options->output_json.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

} // namespace

int
main(int argc, char **argv)
{
    Options options;
    bool showed_help = false;
    if (!parse_args(argc, argv, &options, &showed_help))
        return showed_help ? 0 : 1;

    NativeConfig config;
    std::string config_error;
    if (!load_native_config(options.config_path, &config, &config_error)) {
        std::cerr << config_error << "\n";
        return 1;
    }

    const auto wall_start = std::chrono::steady_clock::now();

    if (dr_standalone_init() == nullptr) {
        std::cerr << "Failed to initialize DynamoRIO standalone context\n";
        return 1;
    }

    concorde_analyzer_tool_t tool(options, config);
    analysis_tool_t *tools[] = { &tool };

    const auto analyze_start = std::chrono::steady_clock::now();
    analyzer_t analyzer(resolve_trace_input_path(options.trace_dir), tools, 1, options.worker_count);
    if (!analyzer) {
        std::cerr << analyzer.get_error_string() << "\n";
        dr_standalone_exit();
        return 1;
    }
    bool stopped_early = false;
    try {
        if (!analyzer.run()) {
            std::cerr << analyzer.get_error_string() << "\n";
            dr_standalone_exit();
            return 1;
        }
    } catch (const StopAnalysis &) {
        stopped_early = true;
    }
    tool.print_results();
    const auto analyze_end = std::chrono::steady_clock::now();

    auto results = tool.results();
    const auto wall_end = std::chrono::steady_clock::now();
    const double native_total =
        std::chrono::duration<double>(analyze_end - analyze_start).count();
    const double wall_total =
        std::chrono::duration<double>(wall_end - wall_start).count();

    std::unordered_map<std::string, double> timings;
    timings["trace_parse_total"] = native_total;
    timings["native_total"] = native_total;
    timings["wall_total"] = wall_total;

    write_results_json(results, timings, options.output_json);
    std::cerr << "[native-concorde] instruction_count=" << results.instruction_count
              << " native_total=" << native_total << "s wall=" << wall_total << "s";
    if (stopped_early)
        std::cerr << " (stopped early)";
    std::cerr << "\n";
    dr_standalone_exit();
    return 0;
}
