# Qwen3.8-27B：DSpark 多卡推理与框架优化

[English](README.md) · 简体中文 · [项目首页](../../README.zh-CN.md)

在 4×A30 TP4 上接入匹配的 DSpark 草稿，完成自回归对照、统一验证预算、CUDA Graph 分档及高并发容量诊断。模型权重、拒绝采样和 GDN 状态管理来自上游；本目录提供调度与图捕获适配、实验驱动和分析工具。

目标模型约 27B，匹配的草稿约 1.99B 参数；草稿仓库名中的 `27B` 指它匹配的目标模型，不是草稿本身的参数量。

## 结果与个人改动

- 原生 DSpark 在请求并发 1 / 4 下，相对自回归生成吞吐为 **2.433× / 1.688×**，TPOT 降低 **63.9% / 55.9%**。
- `uniform_budget.py` 按实际活跃请求数选择统一验证前缀 K，草稿继续生成完整 7-token block；保留上游拒绝采样及状态管理。
- `uniform_graphs.py` 为目标 query width 2 / 3 / 5 / 8 和草稿 width 7 捕获图，增加 6 / 12 / 24 请求桶。图桶消融将目标 padding 比（捕获 token / 有效 token）从 **1.41 降至 1.09**。
- 冻结预算+图分档相对同图池原始 DSpark 策略，在并发 4 / 16 下增加吞吐 **6.3% / 7.8%**。c=4 来自首轮 held-out 消融，c=16 来自后续五轮直接对照。
- 最新并发 16 直接对照中，优化方案相对 AR 为 **0.995×**（范围 0.974–1.016×），基本持平，没有稳定吞吐加速。图桶 / 原始策略约 +0.9%，预算+图桶 / 仅图桶约 +8.2%；这是逐级消融，不能相加或相乘中位数。

完整说明：[首轮实验](../../reports/qwen38_dspark_tp4_20261003.md)、[直接对照与容量／数值诊断](../../reports/qwen38_dspark_closure_20261003.md)、[面试讲法](../../reports/qwen38_dspark_interview.md)。

## 不用 GPU，复核已记录结果

从仓库根目录运行，只有 Python 标准库依赖：

```bash
python3 benchmark/qwen38_multigpu/reproduce_results.py \
  --output outputs/qwen38-evidence
```

脚本核对 27 个 gzip 档案及解压后的 SHA-256，再将原服务器的文件引用移到新的输出目录。JSONL 字节保持不变，容量探针的窗口偏移因此仍有效。重新计算四份汇总，并要求其内容与提交的汇总完全相同（仅允许 `source` 路径不同）；核查 13,056 个实际输出决策及四个 TP rank 一致性。输出目录必须为空；再次运行时使用新的 `--output`。

数据在 [`results/qwen38_multigpu_20261003`](../../results/qwen38_multigpu_20261003)。gzip 中的记录是原始实验文件，不包含模型权重、安装后的运行时、服务日志或 Nsight 数据库。`bundle-manifest.json` 还保留原实验源码哈希，便于区别原测量代码与本次路径配置适配。公开 ShareGPT 负载的模型、数据集 revision、抽样及完整 tokenized prompts 在 `workload.json` 中，字节哈希保持为原实验值。

## GPU 实验环境

独立环境依赖见 [`requirements.txt`](requirements.txt)。原测量使用 vLLM **0.29.0**、Torch **2.13.0**、FlashInfer **0.6.18**；系统需配套 CUDA 驱动。适配依赖这一版 vLLM 的内部接口，不保证其他版本可直接使用。

```bash
python3 -m venv benchmark/qwen38_multigpu/.venv
source benchmark/qwen38_multigpu/.venv/bin/activate
python -m pip install -r benchmark/qwen38_multigpu/requirements.txt
```

准备固定 revision 的本地模型，配置和 tokenizer 文件应来自相同 revision：

| 模型 | Revision |
| --- | --- |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| `RedHatAI/Qwen3.8-27B-speculator.dspark` | `87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce` |

```bash
export DSPARK_MODEL=/absolute/path/Qwen3.8-27B
export DSPARK_DRAFT_MODEL=/absolute/path/Qwen3.8-27B-speculator.dspark
```

`download_draft.py` 可续传下载并按上游 LFS SHA-256 校验草稿权重，生成启动器所需 `download-manifest.json`；它只下载权重文件，配置/tokenizer 需另行准备。固定草稿权重为 3,976,869,890 bytes，SHA-256 为 `6cf6f33fbb2dfd7e74f8009132f0a2a2b4c71cd6cce9360a7d4145c9218f1781`。先下载配置、tokenizer 后运行：

```bash
python benchmark/qwen38_multigpu/download_draft.py
```

| 环境变量 | 用途 | 默认值 |
| --- | --- | --- |
| `DSPARK_PYTHON` | 服务进程解释器 | 当前 Python |
| `DSPARK_MODEL` | 目标模型目录 | `~/models/Qwen3.8-27B` |
| `DSPARK_DRAFT_MODEL` | 草稿模型目录 | `~/models/Qwen3.8-27B-speculator.dspark` |
| `DSPARK_WORKLOAD` | 冻结性能负载 | 仓库结果目录中的 `workload.json` |
| `DSPARK_QUALITY_CASES` | EOS / stop 检查样本 | 仓库结果目录中的 `quality-cases.json` |
| `DSPARK_RUNTIME` | 可选：独立安装的 Python 包目录 | 当前解释器中的安装 |
| `DSPARK_NSYS` | Nsight Systems 可执行文件 | PATH 中的 `nsys` |

原测试使用物理 GPU 1 / 5 / 6 / 7，无 NVLink，拓扑为 PHB / PIX 混合。复现时必须选择自己可用的四张卡；启动器核对空闲显存和实际计算进程，拒绝占用已有任务的卡，只停止自己创建的进程组。草稿有 20 个 attention heads，未测 TP8；更换 GPU 拓扑或缓存预算的结果应单独记录。

## 复现步骤

以下命令从仓库根目录运行，每个输出使用新文件名。所有 HTTP 服务绑定本机，诊断 RPC 只用于本地实验。

1. **原生 DSpark / AR 五轮对照。** 双方均为每卡 4.5 GiB 缓存预算，约 2K 输入、固定 256 实际输出 token；关闭 thinking/prefix cache。

   ```bash
   python benchmark/qwen38_multigpu/run.py \
     --gpus 1,5,6,7 --output outputs/qwen38-native.json \
     --modes ar,fixed --rounds 5 --concurrency 1,4,16 \
     --requests 4 --tokens 256 --kv-gib 4.5 --sample-offset 16
   python benchmark/qwen38_multigpu/summarize.py outputs/qwen38-native.json
   ```

2. **校准与冻结策略。** 前 16 提示词校准 K=1/2/4/7，后 16 提示词做五轮 held-out 消融。目录中的冻结历史策略来自实测，不应在验证集上重拟合。

   ```bash
   python benchmark/qwen38_multigpu/check_graphs.py
   python benchmark/qwen38_multigpu/calibrate.py \
     --gpus 1,5,6,7 --output outputs/qwen38-budget.json
   python benchmark/qwen38_multigpu/summarize_budget.py outputs/qwen38-budget.json
   ```

3. **并发 16 直接对照。** 每轮分别启动 AR 与 DSpark，后者在一个 45 图池中随机比较原始策略、仅细分图桶、预算+图桶。预热、加载、捕获和额外诊断不计入性能时间。

   ```bash
   python benchmark/qwen38_multigpu/threeway.py \
     --gpus 1,5,6,7 --rounds 5 --diagnostics \
     --profile configs/qwen38_uniform_budget.json \
     --output outputs/qwen38-threeway.json
   python benchmark/qwen38_multigpu/analyze_closure.py outputs/qwen38-threeway.json
   ```

4. **独立容量与数值诊断。** 记录真实分配返回值及 GDN/attention/draft 状态需求，另做 AR 串行/并发对照；探针运行吞吐不作为性能结论。

   ```bash
   python benchmark/qwen38_multigpu/capacity_probe.py \
     --gpus 1,5,6,7 --output outputs/qwen38-capacity.json
   python benchmark/qwen38_multigpu/analyze_capacity.py outputs/qwen38-capacity.json
   ```

5. **另行采集 Nsight。** `capture_profile.py` 与 `phases.py` 标记目标/草稿阶段，`analyze_profile.py` 分析导出的 SQLite；开启 profiler 的吞吐不进入性能对照。

   ```bash
   python benchmark/qwen38_multigpu/capture_profile.py \
     --gpus 1,5,6,7 --output outputs/qwen38-profiles.json
   python benchmark/qwen38_multigpu/analyze_profile.py --help
   ```

## 指标与边界

- 吞吐 = 实际输出 token 总数 / 有限请求组完成时间；这是 closed-loop 请求组，不是稳态生产 QPS。SSE chunk 个数不当成 token 数。
- 客户端 TPOT = 首尾非空 SSE 到达间隔 /（实际输出 token 数 − 1），包含调度与流式交付；不是纯 GPU step 时间。
- 加速为五轮同轮比值的中位数，不是端点中位数相除。不同实验或不同比值的中位数不能直接相乘。
- 最大 K=7 仍配置 8 个 GDN 状态块；缩短实际验证长度没有缩小这个预分配容量，因此没有解决所有高并发排队。
- 13,056 个决策符合各自目标 argmax，长序列仍有差异；已将所查首分歧定位到 BF16 logits 排序变化，未证明普遍逐 token 一致或全输入分布无损。
- 本目录的预算选择在一个 DSpark 引擎内调整验证前缀，与旧目录的 AR / speculative 路由器是不同策略；没有接入或验证 Qwen3.8 的自动 AR 回退。
