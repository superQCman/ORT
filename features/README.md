# ORT `features/*.csv` 字段说明

本文档以 `bs128_nip100.csv` 为例说明 `ORT/features/*.csv` 的字段语义。当前该文件共有 `333` 列；其它 `bs*_nip*.csv` 的主体结构相同，但 `instr_*` 这类按实际 trace 展开的列，可能会随算子执行到的 ARM 指令集合略有增减。

## 1. 行级语义

- 每一行都对应 `op_shapes_{batch}_{nip}.csv` 里的一个 ONNX 节点。
- `node_idx` / `node_name` / `op_type` 以 `op_shapes` 为准，是最终数据集里的规范节点标识。
- trace 特征来自 `dynamorio_tracing/trace_features_sweep/<combo>.csv`，通过 `op_idx == node_idx` 左连接。
- CPU profiling 特征来自整图 ORT profile，经 `build_training_features.py` 对齐并按节点聚合后，通过 `node_idx` 左连接。
- `has_trace_features=0` 表示这个节点没有成功生成单算子 trace 特征。
- `has_cpu_profile=0` 表示这个节点没有产出 ORT `thread_scheduling_stats`；`Constant` 节点最常见。
- 空值通常表示“该字段对当前节点未观测到”或“该提取器没有产出该列”，不要默认等同于严格的数值 `0`。

## 2. 基础字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `batch_size` | 生成当前样本时使用的 batch size。 | `build_training_features.py --batch-size` |
| `num_indices_per_lookup` | DLRM embedding lookup 的 `num_indices_per_lookup`。 | `build_training_features.py --num-indices-per-lookup` |
| `node_idx` | ONNX 节点在 `op_shapes` 中的规范索引，也是最终 join key。 | `op_shapes_*.csv` |
| `node_name` | ONNX 节点名，作为最终训练集中的规范节点名。 | `op_shapes_*.csv` |
| `op_type` | ONNX 算子类型，例如 `Gather`、`Gemm`、`Constant`。 | `op_shapes_*.csv` |
| `has_trace_features` | 是否成功匹配到单算子 DynamoRIO trace 特征。`1` 为有，`0` 为无。 | `trace_features_sweep/*.csv` 是否命中 |
| `has_cpu_profile` | 是否成功匹配到按节点聚合后的 ORT CPU thread profiling。`1` 为有，`0` 为无。 | `*_cpu_thread_node_aggregated.csv` 是否命中 |
| `cpu_profile_missing_reason` | `has_cpu_profile=0` 时的缺失原因。当前常见值有 `constant_or_no_thread_stats` 与 `no_cpu_thread_stats`。 | `build_training_features.py` |
| `trace_op_name` | 单算子 trace 侧的算子目录名，例如 `00000_Gather_emb_l0_Gather`。 | `trace_features_sweep/*.csv` |
| `trace_op_type` | 单算子 trace 侧解析出的算子类型。通常应与规范 `op_type` 一致。 | `trace_features_sweep/*.csv` |

## 3. DynamoRIO trace 特征

### 3.1 基础计数

| 字段 | 含义 |
| --- | --- |
| `total_instructions` | trace 中总取指次数。 |
| `unique_instructions` | trace 中去重后的取指数。 |
| `total_loads` | 数据 load 指令总数。 |
| `total_stores` | 数据 store 指令总数。 |
| `total_mem_ops` | 总内存访问数，等于 `total_loads + total_stores`。 |
| `load_store_ratio` | `total_loads / max(total_stores, 1)`。 |
| `mem_intensity` | `total_mem_ops / max(total_instructions, 1)`，表示访存强度。 |
| `total_prefetches` | 预取操作总数。 |
| `num_threads` | trace 中观测到的线程数。 |

### 3.2 Cache 统计

字段模板如下：

- `{scope}_{level}_{metric}`
- `scope` 可以是 `core0` / `core1` / `core2` / `core3` / `LLC` / `total`
- `level` 可以是 `L1D` / `L1I` / `L2`
- `metric` 可以是 `hits` / `misses` / `compulsory_misses` / `miss_rate_pct`

语义规则如下：

- `*_hits`: 命中次数。
- `*_misses`: miss 次数。
- `*_compulsory_misses`: compulsory miss 次数。
- `*_miss_rate_pct`: miss rate 百分比。
- `total_L1D_*` / `total_L1I_*` / `total_L2_*`: 对所有 core 对应 cache level 的求和或汇总 miss rate。

`bs128_nip100.csv` 当前包含这些 cache 列：

```text
LLC_compulsory_misses
LLC_hits
LLC_misses
core0_L1D_compulsory_misses
core0_L1D_hits
core0_L1D_misses
core0_L1I_compulsory_misses
core0_L1I_hits
core0_L1I_misses
core0_L2_compulsory_misses
core0_L2_hits
core0_L2_misses
core1_L1D_compulsory_misses
core1_L1D_hits
core1_L1D_misses
core1_L1I_compulsory_misses
core1_L1I_hits
core1_L1I_misses
core1_L2_compulsory_misses
core1_L2_hits
core1_L2_misses
core2_L1D_compulsory_misses
core2_L1D_hits
core2_L1D_misses
core2_L1I_compulsory_misses
core2_L1I_hits
core2_L1I_misses
core2_L2_compulsory_misses
core2_L2_hits
core2_L2_misses
core3_L1D_compulsory_misses
core3_L1D_hits
core3_L1D_misses
core3_L1I_compulsory_misses
core3_L1I_hits
core3_L1I_misses
core3_L2_compulsory_misses
core3_L2_hits
core3_L2_misses
total_L1D_hits
total_L1D_misses
total_L1I_hits
total_L1I_misses
total_L2_hits
total_L2_misses
LLC_miss_rate_pct
core0_L1D_miss_rate_pct
core0_L1I_miss_rate_pct
core0_L2_miss_rate_pct
core1_L1D_miss_rate_pct
core1_L1I_miss_rate_pct
core1_L2_miss_rate_pct
core2_L1D_miss_rate_pct
core2_L1I_miss_rate_pct
core2_L2_miss_rate_pct
core3_L1D_miss_rate_pct
core3_L1I_miss_rate_pct
core3_L2_miss_rate_pct
total_L1D_miss_rate_pct
total_L1I_miss_rate_pct
total_L2_miss_rate_pct
```

### 3.3 Reuse Time

| 字段 | 含义 |
| --- | --- |
| `reuse_time_total_accesses` | reuse time 分析看到的总访问数。 |
| `reuse_time_total_instructions` | reuse time 分析看到的总指令数。 |
| `reuse_time_mean` | 平均 reuse time。 |
| `reuse_time_access_per_instruction` | 平均每条指令对应多少次访问。 |
| `reuse_time_top10_cumulative_pct` | reuse time 距离桶 `1..10` 的累计覆盖比例。 |
| `reuse_time_bin_<n>_count` | reuse time 距离桶 `n` 的访问计数。 |
| `reuse_time_bin_<n>_pct` | reuse time 距离桶 `n` 的访问比例。 |

`bs128_nip100.csv` 当前包含这些 reuse time 列：

```text
reuse_time_total_accesses
reuse_time_total_instructions
reuse_time_mean
reuse_time_access_per_instruction
reuse_time_top10_cumulative_pct
reuse_time_bin_10_count
reuse_time_bin_10_pct
reuse_time_bin_1_count
reuse_time_bin_1_pct
reuse_time_bin_2_count
reuse_time_bin_2_pct
reuse_time_bin_3_count
reuse_time_bin_3_pct
reuse_time_bin_4_count
reuse_time_bin_4_pct
reuse_time_bin_5_count
reuse_time_bin_5_pct
reuse_time_bin_6_count
reuse_time_bin_6_pct
reuse_time_bin_7_count
reuse_time_bin_7_pct
reuse_time_bin_8_count
reuse_time_bin_8_pct
reuse_time_bin_9_count
reuse_time_bin_9_pct
```

### 3.4 Reuse Distance

| 字段 | 含义 |
| --- | --- |
| `reuse_distance_total_accesses` | reuse distance 分析中的总访问数。 |
| `reuse_distance_instruction_accesses` | 指令访问数。 |
| `reuse_distance_data_accesses` | 数据访问数。 |
| `reuse_distance_unique_accesses` | 去重后的唯一访问数。 |
| `reuse_distance_unique_cache_lines` | 访问过的唯一 cache line 数。 |
| `reuse_distance_mean` | 平均 reuse distance。 |
| `reuse_distance_median` | reuse distance 中位数。 |
| `reuse_distance_std` | reuse distance 标准差。 |
| `reuse_distance_unique_access_ratio` | `unique_accesses / total_accesses`。 |
| `reuse_distance_unique_cache_lines_per_k_accesses` | 每 1000 次访问对应的唯一 cache line 数。 |
| `reuse_distance_data_access_ratio` | `data_accesses / total_accesses`。 |
| `reuse_distance_distance_limit` | 计算 reuse distance 时使用的距离截断上限。 |
| `reuse_distance_pruned_address_hits` | 被裁剪地址仍命中的次数。 |
| `reuse_distance_pruned_addresses` | 因距离限制被裁剪的地址数。 |

当前包含这些 reuse distance 列：

```text
reuse_distance_total_accesses
reuse_distance_instruction_accesses
reuse_distance_data_accesses
reuse_distance_unique_accesses
reuse_distance_unique_cache_lines
reuse_distance_mean
reuse_distance_median
reuse_distance_std
reuse_distance_unique_access_ratio
reuse_distance_unique_cache_lines_per_k_accesses
reuse_distance_data_access_ratio
reuse_distance_distance_limit
reuse_distance_pruned_address_hits
reuse_distance_pruned_addresses
```

### 3.5 Opcode 类别混合

字段模板如下：

- `opc_<category>`: 某一类 opcode 的计数。
- `opc_<category>_ratio`: 该类别计数除以类别总指令计数后的比例。

当前类别含义：

| 类别 | 含义 |
| --- | --- |
| `branch` | 分支类指令计数或占比。 |
| `fp_convert` | 浮点转换类指令计数或占比。 |
| `fp_load_simd` | 浮点/SIMD load 类指令计数或占比。 |
| `fp_math` | 浮点运算类指令计数或占比。 |
| `fp_move` | 浮点 move 类指令计数或占比。 |
| `fp_store_simd` | 浮点/SIMD store 类指令计数或占比。 |
| `load` | 通用 load 类指令计数或占比。 |
| `math` | 通用算术/逻辑运算类指令计数或占比。 |
| `simd` | SIMD 类指令计数或占比。 |
| `store` | 通用 store 类指令计数或占比。 |

当前列如下：

```text
opc_branch
opc_fp_convert
opc_fp_load_simd
opc_fp_math
opc_fp_move
opc_fp_store_simd
opc_load
opc_math
opc_simd
opc_store
opc_branch_ratio
opc_fp_convert_ratio
opc_fp_load_simd_ratio
opc_fp_math_ratio
opc_fp_move_ratio
opc_fp_store_simd_ratio
opc_load_ratio
opc_math_ratio
opc_simd_ratio
opc_store_ratio
```

### 3.6 Instruction Mnemonic 计数

字段模板：

- `instr_<mnemonic>`

统一语义：

- 表示该单算子 trace 中，对应 ARM 指令助记符 `<mnemonic>` 的出现次数。
- 如果某个助记符在当前算子里没有被观测到，该列通常为空。
- `instr_load` / `instr_store` / `instr_math` / `instr_branch` / `instr_simd` 这类名字虽然看起来像类别，但在当前脚本里仍按 `instr_*` 普通列落盘。

`bs128_nip100.csv` 当前包含这些 `instr_*` 列：

```text
instr_add
instr_addp
instr_adds
instr_addv
instr_adr
instr_adrp
instr_and
instr_ands
instr_asrv
instr_autiasp
instr_b
instr_bcond
instr_bfm
instr_bic
instr_bics
instr_bit
instr_bl
instr_blr
instr_br
instr_branch
instr_bti
instr_casab
instr_casal
instr_cbnz
instr_cbz
instr_ccmn
instr_ccmp
instr_clz
instr_cmeq
instr_cmhs
instr_cmlt
instr_cnt
instr_csel
instr_csinc
instr_csinv
instr_csneg
instr_dc_zva
instr_dmb
instr_dup
instr_eor
instr_extr
instr_fadd
instr_fccmp
instr_fccmpe
instr_fcmp
instr_fcmpe
instr_fcsel
instr_fcvt
instr_fcvtn
instr_fcvtn2
instr_fcvtpu
instr_fcvtzs
instr_fcvtzu
instr_fdiv
instr_fmadd
instr_fmax
instr_fmla
instr_fmov
instr_fmul
instr_ins
instr_isb
instr_ld1
instr_ld1r
instr_ldadd
instr_ldaddal
instr_ldar
instr_ldarb
instr_ldaxr
instr_ldp
instr_ldr
instr_ldrb
instr_ldrh
instr_ldrsb
instr_ldrsh
instr_ldrsw
instr_ldur
instr_ldurb
instr_ldurh
instr_ldursb
instr_ldxr
instr_load
instr_lslv
instr_lsrv
instr_madd
instr_math
instr_movi
instr_movk
instr_movn
instr_movz
instr_mrs
instr_msr
instr_msub
instr_mvni
instr_nop
instr_orn
instr_orr
instr_paciasp
instr_prfm
instr_rbit
instr_ret
instr_rev
instr_sbc
instr_sbfm
instr_scvtf
instr_sdiv
instr_shl
instr_simd
instr_smaddl
instr_smulh
instr_sshr
instr_st1
instr_st4
instr_stlr
instr_stlrb
instr_stlxr
instr_store
instr_stp
instr_str
instr_strb
instr_strh
instr_stur
instr_sturb
instr_sturh
instr_stxr
instr_sub
instr_subs
instr_svc
instr_swpal
instr_swpalb
instr_tbnz
instr_tbz
instr_ubfm
instr_ucvtf
instr_udiv
instr_umaddl
instr_umaxp
instr_uminp
instr_umov
instr_umulh
instr_ushr
instr_zip1
instr_zip2
```

### 3.7 Trace 调试字段

| 字段 | 含义 |
| --- | --- |
| `drmem_dir` | 该单算子对应的 `drmemtrace.*.dir` 目录路径。用于回溯原始 trace。 |
| `error` | 提取 basic counts、cache sim、opcode mix、reuse time、reuse distance 时的错误信息；为空表示未报错。 |

## 4. ORT CPU 线程聚合特征

这些字段来自整图 ORT profiling JSON 中的 CPU 节点事件，经过 `build_training_features.py` 对齐到 `op_shapes` 并按 `node_idx` 聚合后，再统一加上 `cpu_` 前缀写入最终训练集。

### 4.1 非数值聚合字段

| 字段 | 含义 |
| --- | --- |
| `cpu_call_count` | 当前节点在整图 profiling 中成功匹配到的 CPU 事件次数。 |
| `cpu_provider` | CPU profiling 中记录到的 provider，通常是 `CPUExecutionProvider`。 |
| `cpu_cpu_profile_match_methods` | 该节点的对齐命中方式集合，可能包含 `node_index`、`node_name_op_type`、`node_name`、`node_index_fallback`。 |
| `cpu_main_cores` | 所有匹配调用里主线程使用到的 core ID 去重后列表，使用 `|` 分隔。 |
| `cpu_sub_cores` | 所有匹配调用里子线程使用到的 core ID 去重后列表，使用 `|` 分隔。 |
| `cpu_main_thread_used_count` | 主线程实际参与计算的调用次数。 |
| `cpu_main_thread_used_pct` | 主线程参与计算的调用比例，单位为百分比。 |

### 4.2 数值聚合字段模板

字段模板如下：

- `cpu_<metric>_avg`
- `cpu_<metric>_min`
- `cpu_<metric>_max`
- `cpu_dur_us_sum` 仅对 `dur_us` 额外提供总和

聚合后缀的含义：

| 后缀 | 含义 |
| --- | --- |
| `_avg` | 该节点所有匹配调用上的平均值。 |
| `_min` | 该节点所有匹配调用上的最小值。 |
| `_max` | 该节点所有匹配调用上的最大值。 |
| `_sum` | 该节点所有匹配调用上的总和；当前只用于 `dur_us`。 |

基础 metric 的含义：

| 基础 metric | 含义 |
| --- | --- |
| `dur_us` | 当前节点单次 CPU 事件的执行时长，单位微秒。 |
| `main_Distribution` | 主线程在线程调度统计里的 `Distribution` 时间。 |
| `main_DistributionEnqueue` | 主线程的 `DistributionEnqueue` 时间。 |
| `main_Run` | 主线程实际执行时间。 |
| `main_Wait` | 主线程等待时间。 |
| `main_WaitRevoke` | 主线程等待撤销时间。 |
| `num_sub_threads` | 子线程总数。 |
| `active_sub_threads` | 实际跑过任务的子线程数。 |
| `actual_threads_used` | 实际参与执行的线程总数，等于活跃子线程数加上是否启用主线程。 |
| `total_sub_runs` | 所有子线程的 `num_run` 总和。 |
| `sub_max_runs` | 单个子线程 `num_run` 的最大值。 |
| `output_size` | ORT profile 里记录的输出 tensor 大小。 |
| `activation_size` | ORT profile 里记录的 activation 大小。 |
| `parameter_size` | ORT profile 里记录的参数大小。 |

`bs128_nip100.csv` 当前包含这些 CPU 聚合列：

```text
cpu_call_count
cpu_provider
cpu_cpu_profile_match_methods
cpu_main_cores
cpu_sub_cores
cpu_main_thread_used_count
cpu_main_thread_used_pct
cpu_dur_us_avg
cpu_dur_us_min
cpu_dur_us_max
cpu_dur_us_sum
cpu_main_Distribution_avg
cpu_main_Distribution_min
cpu_main_Distribution_max
cpu_main_DistributionEnqueue_avg
cpu_main_DistributionEnqueue_min
cpu_main_DistributionEnqueue_max
cpu_main_Run_avg
cpu_main_Run_min
cpu_main_Run_max
cpu_main_Wait_avg
cpu_main_Wait_min
cpu_main_Wait_max
cpu_main_WaitRevoke_avg
cpu_main_WaitRevoke_min
cpu_main_WaitRevoke_max
cpu_num_sub_threads_avg
cpu_num_sub_threads_min
cpu_num_sub_threads_max
cpu_active_sub_threads_avg
cpu_active_sub_threads_min
cpu_active_sub_threads_max
cpu_actual_threads_used_avg
cpu_actual_threads_used_min
cpu_actual_threads_used_max
cpu_total_sub_runs_avg
cpu_total_sub_runs_min
cpu_total_sub_runs_max
cpu_sub_max_runs_avg
cpu_sub_max_runs_min
cpu_sub_max_runs_max
cpu_output_size_avg
cpu_output_size_min
cpu_output_size_max
cpu_activation_size_avg
cpu_activation_size_min
cpu_activation_size_max
cpu_parameter_size_avg
cpu_parameter_size_min
cpu_parameter_size_max
```

## 5. 读表时的几个注意点

- `Constant` 节点通常会有 trace 特征，但没有 CPU thread stats，因此常见组合是 `has_trace_features=1`、`has_cpu_profile=0`、`cpu_profile_missing_reason=constant_or_no_thread_stats`。
- 某些非 `Constant` 节点也可能 `has_cpu_profile=0`。这通常意味着它没有出现在 CPU provider 的 `thread_scheduling_stats` 里，或者实际运行在别的 provider 上。
- `instr_*`、`opc_*`、cache miss rate、reuse 特征都来自单算子 DynamoRIO 工具，因此和整图 ORT profiling 的时间列是两套独立来源。
- 如果之后修改了 `extract_trace_features.py` 的 opcode 分类、cache 配置或展开规则，这份字段清单也需要同步更新。
