# Qwen3.8-27B DSpark 四卡 Serving 与验证预算实验

实验日期：2026-10-03。代码：`benchmark/qwen38_multigpu`。

仓库迁移说明：本报告保留首轮实验结果，后续并发 16 直接对照见[补充报告](qwen38_dspark_closure_20261003.md)。压缩记录与校验清单位于 `results/qwen38_multigpu_20261003`；运行 `reproduce_results.py` 后可在 `outputs/qwen38-evidence` 查看解压及重新汇总的结果。模型、负载和策略字节哈希保持原实验记录。

原生 AR / DSpark 五轮对照、个人预算五轮消融、短任务检查、独立长序列评分与四个 Nsight 窗口均已完成。原生低并发吞吐为 AR 的 2.43× / 1.69×；个人改动相对同引擎原生 DSpark，在并发 4/16 下额外提升 6.3% / 5.7%。所有数字保留各自实验边界。

## 模型与测试边界

目标为 Qwen3.8-27B BF16，配套 RedHatAI DSpark 草稿约 1.988B 参数、5 层、7 个候选 token。目标模型每轮验证候选前缀并沿用 vLLM 原生拒绝采样。草稿模型不是本项目训练的。

本次实际使用 **4×NVIDIA A30、TP4、物理 GPU 1/5/6/7**。GPU 1 与其他三张卡的连接为 PHB，GPU 5/6/7 间为 PIX，没有 NVLink。其他卡存在任务，没有跑 TP8。草稿模型有 20 个 Q heads，不能把默认 draft TP8 支持视为已验证。

运行时为隔离的 vLLM 0.29.0 / FlashInfer 0.6.18，使用已有 Torch 2.13.0，GDN 状态显式 FP32。模型 revision、完整草稿权重 SHA-256、运行时文件哈希见 `results/qwen38_multigpu_20261003/environment.json`。没有覆盖先前 vLLM 安装。

原生 AR 与 DSpark 使用相同冻结 ShareGPT 输入、关闭 thinking 与 prefix cache、每卡 4.5 GiB KV 预算、相同 TP 拓扑、greedy 和固定 256 个实际输出 token。输入约 2K tokens；并发 1/4/16 分别完成 4/4/16 个请求。时间包含 HTTP 请求、prefill、decode 与队列，不包含模型加载、图捕获和预热。这是有限请求组，不是稳态线上 QPS。

每轮随机打乱 AR / DSpark 顺序，分别以新进程启动；每个条件五轮。加速统计为同轮吞吐比值的中位数，保留每轮范围。HTTP TPOT 按首尾非空 SSE 到达间隔 / 实际输出 token 数减一计算，不是纯 GPU kernel 时间。逐事件 token IDs、first-chunk token 数和完整输出均保留。

## 原生算法：五轮配对结果

来源为解压后的 `outputs/qwen38-evidence/native-paired-resumed.json`（`complete=true`，30 组测量，无失败）及其 `.summary.json`。吞吐比和 TPOT 降幅均为同轮比值的中位数；绝对吞吐、TPOT 为各端点的边际中位数，因此不应由表中绝对值重新计算配对比值。

| HTTP 并发 | AR / DSpark 吞吐 (token/s) | 配对加速中位数 | 五轮加速范围 | HTTP TPOT 降幅 | 平均接受长度（含 bonus） |
| --- | --- | --- | --- | --- | --- |
| 1 | 19.15 / 46.59 | **2.433×** | 2.342–2.456× | **63.87%** | 2.926 |
| 4 | 63.70 / 104.52 | **1.688×** | 1.384–1.786× | **55.87%** | 2.932 |
| 16 | 172.98 / 155.27 | **0.898×** | 0.817–1.024× | 27.27% | 3.321 |

所有测量组 HTTP 错误和 KV 抢占均为零。并发 16 的接受长度反而更长，却没有吞吐收益；活跃请求生成更快与整个有限请求组更快不是同一件事。表中范围保留了波动与回退，不能概括成并发 1–16 都有加速。

256-token greedy 完整序列相等次数分别为 10/20、1/20、9/80。这些分母包含重复轮次，不是独立质量任务。不能声称逐 token 一致，也不能仅凭序列差异宣布模型质量下降；独立质量与数值诊断见后续结果。

## 已发现的容量约束

高并发日志显示，DSpark 在相同字节预算下出现运行与等待请求同时存在。例如第二轮并发 16 的测量阶段，日志记录过 `Running: 8 / Waiting: 8`、`Running: 10 / Waiting: 4`，没有 KV 抢占。没有抢占不等于没有容量压力；请求可能在准入阶段排队。

从模型配置与 pinned runtime 源码能计算额外状态的量级：

- 目标有 48 层 GDN；TP4 下每卡 12 个 value heads，K/V 维度均为 128。
- 单层 FP32 recurrent state 为 `12 × 128 × 128 × 4 bytes = 0.75 MiB`。
- 48 层合计每个状态检查点约 **36 MiB / 请求 / 卡**。
- `MambaSpec.num_speculative_blocks` 由 `num_speculative_tokens` 决定；关闭 prefix cache 的默认 `mamba_cache_mode=none` 需 `1 + num_speculative_blocks` 个状态块。7-token 验证对应 8 块，仅 recurrent state 即约 **288 MiB / 请求 / 卡**，普通 AR 为约 36 MiB。

以上是根据配置和代码推导的 SSM 内容量，未包含卷积状态、page padding、目标 attention KV、草稿 KV 和模型 workspace，不是 profiler 显存测量。16 个请求仅 recurrent state 就约 4.5 GiB / 卡，因此不能由“输入总 token 数小于打印的 KV token 容量”断言 16 个请求都能同时驻留。

相关源码为 runtime 的 `model_executor/layers/mamba/mamba_utils.py`、`model_executor/layers/mamba/abstract.py` 和 `v1/kv_cache_interface.py`。完整分析须结合实际 Running / Waiting 和调度 trace，不能只把并发收益收窄归因于 GEMM。

## 个人实现与可拆分消融

`uniform_budget.py` 在异步调度阶段截取统一 K=1/2/4/7 验证前缀，草稿仍提出完整 block。使用新切片，保留原生拒绝采样、GDN 状态管理和请求生命周期。

vLLM 0.29 的 confidence-based adaptive verification 要求可变长度 ALWAYS CUDA Graph；本地 GDN 后端声明 UNIFORM_BATCH。因此本次研究统一 query width 的路径，没有把上游自适应开关当作个人实现。

`uniform_graphs.py` 为目标捕获 width 2/3/5/8、为草稿捕获 width 7；请求桶为 1/2/4/6/8/12/16/24/32。原始桶的 10 请求会补齐到 16，细分桶可以补齐到 12。两者共用捕获图池、模型和 KV 预算，在同一常驻引擎内切换可用图桶集合。

草稿本地 pinned `config.json` 明确为 `sample_from_anchor=true`，所以 7 个 query 均参与 next-token 草稿预测；目标还需 bonus token，验证 query width 为 K+1。第一次图适配误以为所有 speculators 格式都采用草稿 width 8，启动时被严格形状检查拦截。失败保留在 `budget.json` / `.log`，没有测量行；修正后重新运行到新的结果文件，未改变原生对照配置。

消融分别比较：完整验证 + 原始桶；完整验证 + 细分桶；校准验证预算 + 细分桶。先用前 16 条提示词校准，再冻结策略，换后 16 条做五轮随机交错验证。校准记录实际驻留 batch，避免把 HTTP 并发当成 GPU 执行 batch。原始 DSpark→AR 收益与这组个人增量单独统计。

图形适配的 CPU 检查使用真实 vLLM descriptor / dispatch，覆盖 224 个组合及 2 个原始/细分桶例子；这是形状检查，真实 GPU 数值与性能仍由运行数据验证。

修正后的运行文件为解压后的 `outputs/qwen38-evidence/budget-v2.json`。四个 rank 均实际捕获 36 张目标图与 9 张草稿图。24 组校准后，冻结映射为 `{"1":7,"3":4,"11":4}`；运行时按最近的实际活跃 batch 桶选择，同距离取较小桶。校准里的 HTTP 并发 4 对应主导纯验证 batch 3，说明二者不能直接等同。校准输入输出为约 2K / 96 tokens，独立验证为约 2K / 256 tokens；验证结果不能回填到校准策略。

分档捕获有启动与显存代价：该次适配引擎日志记录图捕获约 31 秒、增量 1.53 GiB/卡，原生第五轮 DSpark 约 9 秒、0.51 GiB/卡。它们是两次启动日志，不是重复配对的启动成本评测；大致说明多捕获不同宽度/请求桶需要额外资源。个人性能消融的三个模式均保留同一个 45 图池，因此未把不同图池容量混入同轮增量。

## 个人增量：五轮独立验证

`budget-v2.json` 已完成（24 组校准、45 组独立验证，`complete=true`，无失败）；汇总为 `budget-v2.summary.json`。这里的 native 是同一常驻引擎、同一 45 图池下仅选择原始桶及完整 K=7 的基线，与前面的新进程 AR 对照属于不同实验，不应把两个加速比相乘宣称新的端点加速。

| HTTP 并发 | 细分图桶 / native | 预算+图桶 / 细分图桶 | 预算+图桶 / native | 总增量五轮范围 |
| --- | --- | --- | --- | --- |
| 1 | +0.83% | +0.10% | **-0.42%** | -0.47% 至 +2.30% |
| 4 | -0.77% | +7.64% | **+6.30%** | -1.87% 至 +12.03% |
| 16 | +3.21% | +1.62% | **+5.68%** | -0.38% 至 +12.69% |

各列都是同轮比值中位数，不能由分列中位数相乘还原总增量。不能说所有轮次无退化。并发 1 的图形和验证策略实质不变；并发 4 不使用新增的 6/12/24 桶，图桶组变化主要反映有限请求组的准入时序、prefill 分块和输出数值路径波动，不能把它解释成图分桶收益。并发 16 中，细分图桶的五轮比值均大于 1，但预算相对细分图桶的增量更小且有回退。

调度/图计数给出了工作量变化的直接证据：并发 16 完整验证的 target padding ratio（捕获 token / 逻辑 token）从约 **1.413** 降到 **1.094**，draft 从约 **1.411** 降到 **1.096**。这是形状统计，不是 FLOPs 或 GPU 时间测量。

首轮并发 16 的纯图派发记录中，缩短前缀后 target 逻辑 token 从 8,696 降至 6,231（-28.3%）；target 图调用从 147 增至 158，draft 逻辑 token 从 8,785 增至 9,688，平均接受长度从 3.384 降至 3.060。更少的每步验证工作伴随更多步数和草稿工作，可以解释为何该轮预算相对细分图桶只有约 0.3% 的端点吞吐增量。整个策略仍按 K=7 分配最大状态块，未消除容量限制。

短任务检查中，native 与预算版本均为 **16/16** 严格答案匹配，16/16 自然停止，显式 newline 与 END 两项 stop 的文本和结束原因均符合预期，HTTP 错误为零。范围仅限该固定小集合，不代表完整业务质量、逐 token 等价或普遍无损。

## 正确性与 profiling 口径

长输出已观察到不同路径的 greedy 序列差异。独立目标模型 teacher forcing 同时检查 AR 与 DSpark 序列的 rank-1 一致率、logprob regret；它们是数值诊断，不是任务准确率，也不证明所有差异只来自舍入。

独立 AR 引擎已完成首轮长输出的 teacher forcing：69 条去重序列，涵盖普通 AR、自回归对照中的 DSpark，以及个人消融的三个模式。AR 自身控制组的 rank-1 一致率为 99.219–99.463%，原生 DSpark 为 99.121–99.512%，预算版本为 99.463–99.609%；各组平均 logprob regret 均小于 0.0008，最大观察值 0.375。AR 自身也未达到 100%，因此不能把所有序列差异归结为投机算法错误，亦不能证明全部仅来自浮点舍入。检查覆盖首轮长输出，不代表其余轮次或所有输入，指标不是任务准确率。完整每 token 数据见 `profiles.json` 的 `reference_scores`。

另有固定 16 个短回答任务和 2 个显式 stop 条件，检查自然 EOS、结束原因与严格答案匹配。这个小集合不代表完整业务质量验收。

Nsight 另行运行，性能数据来自未开启 profiler 的对照。采集已保留 prefill、target Decode 图、draft 全阶段与 draft 图的原始记录。累计 kernel 时间占比不等于请求延迟占比。当前驱动 `RmProfilingAdminOnly=1`，不依据缺少硬件计数器的 trace 宣称显存带宽饱和。

## Nsight 阶段记录

`profiles.json` 完整成功，AR / DSpark 各采集并发 1 和 16 两个窗口，每请求输出 64 tokens；自然任务各 16/16，stop 两项通过。CUDA device 0/1/2/3 在导出元数据中分别映射到物理 GPU 1/5/6/7，与性能实验一致。

分析通过 CUDA runtime launch correlation 关联 CPU NVTX 的最内层本地范围，每个 kernel 只归到一个阶段。`target_graph` 是目标 backbone Decode 图；目标 LM head 与采样在图之外，另设 `target_logits` 和 `target_sample`。`target_eager` 同时可能包含 prefill 与混合 query 的 eager 路径；草稿图和草稿图外 KV 准备分别记录。下表为 **target_graph 内累计 kernel 时间的分类比例**，先累加四卡同类 kernel，再除以四卡 kernel 总和。

| 模式 / HTTP 并发 | 矩阵计算 | NCCL | GDN 核 | Attention 核 |
| --- | --- | --- | --- | --- |
| AR / 1 | 54.5% | 27.7% | 1.3% | 1.6% |
| AR / 16 | 42.0% | 35.9% | 3.3% | 2.2% |
| DSpark / 1 | 30.1% | 55.3% | 2.4% | 2.6% |
| DSpark / 16 | 32.5% | 44.9% | 6.3% | 3.4% |

其余为 norm、卷积、逐元素与复制等 kernel。分类为基于名字的启发式，GDN 核包括 `gdn_decode_post_conv_mtp_kernel` 和 Triton recurrence；Attention 名字中的 cutlass 数据类型不应误归 GEMM。主要 NCCL 名称包括 `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`。

这份 trace 显示目标和草稿都包含矩阵计算及大量设备通信/等待，不能把全部开销只归于验证 GEMM。NCCL 时长含 peer 等待，profiler 也会改变 CPU 调度；图外目标 LM head 和 eager 阶段不在上表分母内。因此不拿该表当作端点延迟的百分比分解，不宣称 wire transfer 占比或带宽饱和，也不把 c=1 与 c=16 不同总工作量的累计毫秒直接相减作因果归因。高并发准入排队与状态容量的证据来自原始性能日志及缓存源码，图补齐减少的证据来自独立消融的分派计数。

原始 trace 为 `profiles.ar.1/2.nsys-rep` 和 `profiles.fixed.1/2.nsys-rep`；SQLite 导出与最终分类 `*.phases-v3.json` 均保留，汇总为 `profile-summary.json`，记录分析脚本 SHA-256。v1/v2 分类输出仅为分析过程记录，最终报告采用 v3。

## 下载与失败记录

运行时通过中科大 PyPI 镜像装好；HF 镜像的大权重链接最终回到官方 CDN，因此采用可续传并发分段下载，逐段校验 HTTP Range，完成后核验 pinned SHA-256。未使用来源不匹配的草稿权重。

休眠/唤醒预检后，正常 AR 回答变成重复 `duct`；这条路径未用于性能对照，`paired.py` 仅保留诊断。新增分析脚本最初与标准库 `profile` 同名，导致一轮服务启动失败；改名为 `capture_profile.py` 后继续。已完成请求组保留，失败日志不计入性能结论。

原始失败文件为 `native-paired.json`；恢复文件为 `native-paired-resumed.json`，记录恢复来源 SHA-256、旧源码哈希和原失败。两份文件都保留。最终有效数据应检查 `complete=true` 与无 `failure`；命令、GPU 快照、源码快照及每条请求的 token IDs 均在运行目录。
