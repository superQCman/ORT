#include "dr_api.h"
#include "drmemtrace/analyzer.h"
#include "drmemtrace/analysis_tool.h"
#include "drmemtrace/memref.h"
#include "drmemtrace/memtrace_stream.h"
#include "drmemtrace/trace_entry.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>
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
using dynamorio::drmemtrace::TRACE_TYPE_INSTR_DIRECT_CALL;
using dynamorio::drmemtrace::TRACE_TYPE_INSTR_DIRECT_JUMP;
using dynamorio::drmemtrace::TRACE_TYPE_INSTR_CONDITIONAL_JUMP;
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
    std::string output_path = "-";
    std::string output_format = "compact-text";
    uint64_t max_instructions = 0;
};

enum class issue_group_t : uint8_t {
    ALU = 0,
    FP = 1,
    LS = 2,
};

enum class branch_kind_t : uint8_t {
    NONE = 0,
    CONDITIONAL = 1,
    DIRECT_UNCONDITIONAL = 2,
    INDIRECT = 3,
    OTHER = 4,
};

struct DecodedInstruction {
    std::string mnemonic;
    std::string uses_csv;
    std::string defs_csv;
    uint64_t uses_mask = 0;
    uint64_t defs_mask = 0;
    issue_group_t issue_group = issue_group_t::ALU;
    branch_kind_t branch_kind = branch_kind_t::NONE;
    bool has_direct_target = false;
    addr_t direct_target = 0;
    bool branch_taken_known = false;
    bool branch_taken = false;
};

struct MemOp {
    char kind = 'R';
    addr_t addr = 0;
    uint32_t size = 0;
};

#pragma pack(push, 1)
struct compact_instr_record_t {
    uint64_t instruction_ordinal = 0;
    uint32_t tid = 0;
    uint8_t issue_group = 0;
    uint8_t branch_kind = 0;
    uint8_t branch_taken = 0;
    uint8_t memop_count = 0;
    uint64_t pc = 0;
    uint64_t uses_mask = 0;
    uint64_t defs_mask = 0;
};

struct compact_memop_record_t {
    uint8_t kind = 0;
    uint64_t addr = 0;
    uint32_t size = 0;
};
#pragma pack(pop)

struct CurrentInstruction {
    uint64_t instruction_ordinal = 0;
    memref_tid_t tid = 0;
    addr_t pc = 0;
    DecodedInstruction decoded;
    std::vector<MemOp> memops;
};

std::string
to_lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
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
hex_string(addr_t value)
{
    std::ostringstream oss;
    oss << std::hex << std::nouppercase << static_cast<uint64_t>(value);
    return oss.str();
}

std::string
bytes_to_hex(const unsigned char *bytes, size_t size)
{
    std::ostringstream oss;
    oss << std::hex << std::nouppercase << std::setfill('0');
    for (size_t i = 0; i < size; ++i)
        oss << std::setw(2) << static_cast<unsigned int>(bytes[i]);
    return oss.str();
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

bool
is_tracked_reg(reg_id_t reg)
{
    reg = canon_reg(reg);
    return (reg >= DR_REG_X0 && reg <= DR_REG_X30) || reg == DR_REG_SP ||
        reg == DR_REG_XZR;
}

std::string
format_reg(reg_id_t reg)
{
    reg = canon_reg(reg);
    if (!is_tracked_reg(reg))
        return "";
    const char *name = get_register_name(reg);
    if (name == nullptr)
        return "";
    std::string text = to_lower_copy(name);
    if (text == "wzr")
        text = "xzr";
    else if (text.size() >= 2 && text[0] == 'w' &&
             std::all_of(text.begin() + 1, text.end(),
                         [](unsigned char c) { return std::isdigit(c) != 0; }))
        text[0] = 'x';
    return "%" + text;
}

void
append_unique_reg(std::vector<std::string> &out, std::unordered_set<std::string> &seen,
                  reg_id_t reg)
{
    const auto formatted = format_reg(reg);
    if (formatted.empty())
        return;
    if (seen.insert(formatted).second)
        out.push_back(formatted);
}

std::string
join_regs(const std::vector<std::string> &regs, char sep)
{
    std::ostringstream oss;
    bool first = true;
    for (const auto &reg : regs) {
        if (!first)
            oss << sep;
        oss << reg;
        first = false;
    }
    return oss.str();
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

issue_group_t
classify_issue_group(const std::string &mnemonic)
{
    const std::string lower = to_lower_copy(mnemonic);
    if (lower.rfind("fc", 0) == 0 || lower.rfind("fm", 0) == 0 ||
        lower.rfind("fs", 0) == 0 || lower.rfind("fd", 0) == 0 ||
        lower.rfind("f", 0) == 0)
        return issue_group_t::FP;
    return issue_group_t::ALU;
}

branch_kind_t
classify_branch_kind(trace_type_t type, const std::string &mnemonic)
{
    if (type == TRACE_TYPE_INSTR_TAKEN_JUMP || type == TRACE_TYPE_INSTR_UNTAKEN_JUMP ||
        type == TRACE_TYPE_INSTR_CONDITIONAL_JUMP)
        return branch_kind_t::CONDITIONAL;
    if (type == TRACE_TYPE_INSTR_DIRECT_JUMP || type == TRACE_TYPE_INSTR_DIRECT_CALL)
        return branch_kind_t::DIRECT_UNCONDITIONAL;
    if (type == TRACE_TYPE_INSTR_INDIRECT_JUMP || type == TRACE_TYPE_INSTR_INDIRECT_CALL ||
        type == TRACE_TYPE_INSTR_RETURN)
        return branch_kind_t::INDIRECT;

    const std::string lower = to_lower_copy(mnemonic);
    if (lower.rfind("b.", 0) == 0 || lower == "cbz" || lower == "cbnz" || lower == "tbz" ||
        lower == "tbnz")
        return branch_kind_t::CONDITIONAL;
    if (lower == "b" || lower == "bl")
        return branch_kind_t::DIRECT_UNCONDITIONAL;
    if (lower == "br" || lower == "blr" || lower == "ret")
        return branch_kind_t::INDIRECT;
    return branch_kind_t::NONE;
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

class pseudo_view_tool_t : public analysis_tool_t {
public:
    explicit pseudo_view_tool_t(std::ostream &out, const Options &options)
        : out_(out)
        , options_(options)
    {
        if (options_.output_format == "compact-bin")
            out_.write("CTRCBIN1", 8);
    }

    std::string
    initialize_stream(memtrace_stream_t *serial_stream) override
    {
        if (serial_stream == nullptr)
            return "drmemtrace_pseudoview requires a serial memtrace stream";
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
            return emit_data_ref(entry.data);
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
        flush_all_pending();
        return true;
    }

private:
    DecodedInstruction
    decode_instruction(const _memref_instr_t &instr_memref)
    {
        const addr_t pc = instr_memref.addr;
        const bool refresh = instr_memref.encoding_is_new || decoded_cache_.find(pc) == decoded_cache_.end();
        if (!refresh)
            return decoded_cache_.at(pc);

        DecodedInstruction decoded;
        const bool emit_text = options_.output_format != "compact-bin";

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

            std::vector<std::string> uses;
            std::vector<std::string> defs;
            std::unordered_set<std::string> use_seen;
            std::unordered_set<std::string> def_seen;
            if (!emit_text) {
                uses.clear();
                defs.clear();
            }

            const int src_count = instr_num_srcs(&instr);
            for (int i = 0; i < src_count; ++i) {
                const opnd_t src = instr_get_src(&instr, i);
                const int reg_count = opnd_num_regs_used(src);
                for (int reg_idx = 0; reg_idx < reg_count; ++reg_idx) {
                    const auto reg = opnd_get_reg_used(src, reg_idx);
                    const int tracked_idx = tracked_reg_index(reg);
                    if (tracked_idx >= 0)
                        decoded.uses_mask |= (uint64_t{1} << tracked_idx);
                    if (emit_text)
                        append_unique_reg(uses, use_seen, reg);
                }
            }

            const int dst_count = instr_num_dsts(&instr);
            for (int i = 0; i < dst_count; ++i) {
                const opnd_t dst = instr_get_dst(&instr, i);
                if (opnd_is_memory_reference(dst)) {
                    const int reg_count = opnd_num_regs_used(dst);
                    for (int reg_idx = 0; reg_idx < reg_count; ++reg_idx) {
                        const auto reg = opnd_get_reg_used(dst, reg_idx);
                        const int tracked_idx = tracked_reg_index(reg);
                        if (tracked_idx >= 0)
                            decoded.uses_mask |= (uint64_t{1} << tracked_idx);
                        if (emit_text)
                            append_unique_reg(uses, use_seen, reg);
                    }
                    continue;
                }
                if (opnd_is_reg(dst)) {
                    const auto reg = opnd_get_reg(dst);
                    const int tracked_idx = tracked_reg_index(reg);
                    if (tracked_idx >= 0)
                        decoded.defs_mask |= (uint64_t{1} << tracked_idx);
                    if (emit_text)
                        append_unique_reg(defs, def_seen, reg);
                }
            }

            if (emit_text) {
                decoded.uses_csv = join_regs(uses, ',');
                decoded.defs_csv = join_regs(defs, ',');
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
        decoded_cache_[pc] = decoded;
        return decoded;
    }

    void
    flush_current(bool has_next_pc, addr_t next_pc)
    {
        if (!current_)
            return;

        auto &current = *current_;
        std::string taken_field;
        if (current.decoded.branch_taken_known) {
            taken_field = current.decoded.branch_taken ? "1" : "0";
        } else if (has_next_pc && current.decoded.has_direct_target) {
            taken_field = (current.decoded.direct_target == next_pc) ? "1" : "0";
        }

        ++emitted_instruction_count_;
        if (options_.output_format == "compact-bin") {
            compact_instr_record_t record;
            record.instruction_ordinal = current.instruction_ordinal;
            record.tid = static_cast<uint32_t>(current.tid);
            record.issue_group = static_cast<uint8_t>(current.decoded.issue_group);
            record.branch_kind = static_cast<uint8_t>(current.decoded.branch_kind);
            record.branch_taken =
                current.decoded.branch_taken_known ? (current.decoded.branch_taken ? 2 : 1) : 0;
            record.memop_count = static_cast<uint8_t>(std::min<size_t>(current.memops.size(), 255));
            record.pc = current.pc;
            record.uses_mask = current.decoded.uses_mask;
            record.defs_mask = current.decoded.defs_mask;
            out_.write(reinterpret_cast<const char *>(&record), sizeof(record));
            for (size_t i = 0; i < current.memops.size() && i < 255; ++i) {
                compact_memop_record_t mem_record;
                mem_record.kind = current.memops[i].kind == 'W' ? 1 : 0;
                mem_record.addr = current.memops[i].addr;
                mem_record.size = current.memops[i].size;
                out_.write(reinterpret_cast<const char *>(&mem_record), sizeof(mem_record));
            }
        } else {
            std::ostringstream memops_field;
            bool first = true;
            for (const auto &memop : current.memops) {
                if (!first)
                    memops_field << ';';
                memops_field << memop.kind << ',' << hex_string(memop.addr) << ',' << memop.size;
                first = false;
            }

            out_ << "I|" << current.instruction_ordinal << '|' << current.tid << '|'
                 << hex_string(current.pc) << '|' << current.decoded.mnemonic << '|'
                 << current.decoded.uses_csv << '|' << current.decoded.defs_csv << '|'
                 << taken_field << '|' << memops_field.str() << '\n';
        }
        current_.reset();
        if (options_.max_instructions > 0 && emitted_instruction_count_ >= options_.max_instructions)
            stop_processing_ = true;
    }

    void
    flush_all_pending()
    {
        flush_current(false, 0);
    }

    bool
    handle_instruction(const _memref_instr_t &instr_memref)
    {
        flush_current(true, instr_memref.addr);
        if (stop_processing_) {
            out_.flush();
            std::exit(0);
        }

        const auto instruction_ordinal = stream_->get_instruction_ordinal();
        const auto decoded = decode_instruction(instr_memref);
        current_.reset(new CurrentInstruction());
        current_->instruction_ordinal = instruction_ordinal;
        current_->tid = instr_memref.tid;
        current_->pc = instr_memref.addr;
        current_->decoded = decoded;
        return true;
    }

    bool
    emit_data_ref(const _memref_data_t &data_memref)
    {
        if (!current_)
            return true;
        MemOp memop;
        memop.kind = data_memref.type == TRACE_TYPE_WRITE ? 'W' : 'R';
        memop.addr = data_memref.addr;
        memop.size = static_cast<uint32_t>(data_memref.size);
        current_->memops.push_back(memop);
        return true;
    }

    std::ostream &out_;
    const Options &options_;
    memtrace_stream_t *stream_ = nullptr;
    std::unordered_map<addr_t, DecodedInstruction> decoded_cache_;
    std::unique_ptr<CurrentInstruction> current_;
    uint64_t emitted_instruction_count_ = 0;
    bool stop_processing_ = false;
};

void
print_usage(const char *argv0)
{
    std::cerr << "Usage: " << argv0
              << " --trace-dir <drmemtrace.dir|op_dir> [--output <path>|-]"
              << " [--output-format compact-text|compact-bin]"
              << " [--max-instructions N]\n";
}

bool
parse_args(int argc, char **argv, Options *options, bool *showed_help)
{
    *showed_help = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--trace-dir" && i + 1 < argc) {
            options->trace_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            options->output_path = argv[++i];
        } else if (arg == "--output-format" && i + 1 < argc) {
            options->output_format = argv[++i];
        } else if (arg == "--max-instructions" && i + 1 < argc) {
            options->max_instructions = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            *showed_help = true;
            return false;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }
    if (options->trace_dir.empty()) {
        print_usage(argv[0]);
        return false;
    }
    if (options->output_format != "compact-text" && options->output_format != "compact-bin") {
        std::cerr << "Unsupported --output-format: " << options->output_format << "\n";
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

    if (dr_standalone_init() == nullptr) {
        std::cerr << "Failed to initialize DynamoRIO standalone context\n";
        return 1;
    }

    std::unique_ptr<std::ostream> owned_stream;
    std::ostream *out = &std::cout;
    if (options.output_path != "-") {
        std::unique_ptr<std::ofstream> file(new std::ofstream(options.output_path));
        if (!*file) {
            std::cerr << "Failed to open output file: " << options.output_path << "\n";
            dr_standalone_exit();
            return 1;
        }
        out = file.get();
        owned_stream.reset(file.release());
    }

    pseudo_view_tool_t tool(*out, options);
    analysis_tool_t *tools[] = { &tool };
    analyzer_t analyzer(resolve_trace_input_path(options.trace_dir), tools, 1);
    if (!analyzer) {
        std::cerr << analyzer.get_error_string() << "\n";
        dr_standalone_exit();
        return 1;
    }
    if (!analyzer.run()) {
        std::cerr << analyzer.get_error_string() << "\n";
        dr_standalone_exit();
        return 1;
    }
    tool.print_results();
    out->flush();
    dr_standalone_exit();
    return 0;
}
