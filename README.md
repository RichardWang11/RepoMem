# LocAgent: Graph-Guided LLM Agents for Code Localization

<p align="center">
   📑&nbsp; <a href="https://arxiv.org/abs/2503.09089" target="_blank">Paper</a>
   | 📊&nbsp; <a href="https://huggingface.co/datasets/czlll/Loc-Bench_V1" target="_blank">Loc-bench</a>
   | 🤗&nbsp; <a href="https://huggingface.co/czlll/Qwen2.5-Coder-7B-CL" target="_blank">Qwen2.5-Coder-7B-CL</a>
   | 🤗&nbsp; <a href="https://huggingface.co/czlll/Qwen2.5-Coder-32B-CL" target="_blank">Qwen2.5-Coder-32B-CL</a>
</p>


## ℹ️ Overview
We introduce **LocAgent**, a framework that addresses code localization through graph-based representation.
By parsing codebases into directed heterogeneous graphs, LocAgent creates a lightweight representation that captures code structures and their dependencies, enabling LLM agents to effectively search and locate relevant entities through powerful multi-hop reasoning.
 <!-- <div align="center">
  <img src="./assets/overview.png" alt="Overview" width="800">
</div> -->
![MedAgents Benchmark Overview](assets/overview.png)

## ⚙️ Setup
1. Follow these steps to set up your development environment:
   ```
   git clone git@github.com:gersteinlab/LocAgent.git
   cd LocAgent

   conda create -n locagent python=3.12
   conda activate locagent
   pip install -r requirements.txt
   ```

## 🚀 Launch LocAgent
1. (Optional but recommended) Parse the codebase for each issue in the benchmark to generate graph indexes in batch.
   ```
   python dependency_graph/batch_build_graph.py \
         --dataset 'czlll/Loc-Bench_V1' \
         --split 'test' \
         --num_processes 50 \
         --download_repo
   ```
   - `dataset`: select the benchmark (by default it will be `SWE-Bench_Lite`); you can choose from `['czlll/SWE-bench_Lite', 'czlll/Loc-Bench_V1']`(adapted for code localization) and SWE-bench series datasets like `['princeton-nlp/SWE-bench_Lite', 'princeton-nlp/SWE-bench_Verified', 'princeton-nlp/SWE-bench']`
   - `repo_path`: the directory where you plan to pull or have already pulled the codebase
   - `index_dir`: the base directory where the generated graph index will be saved
   - `download_repo`: whether to download the codebase to `repo_path` before indexing

2. Export the directory of the graph indexes and the BM25 sparse index. If not generated in advance, the graph index will be generated during the localization process.
   ```
   export GRAPH_INDEX_DIR='{INDEX_DIR}/{DATASET_NAME}/graph_index_v2.3'
   export BM25_INDEX_DIR='{INDEX_DIR}/{DATASET_NAME}/BM25_index'
   ```

2. Run the script `scripts/run_lite.sh` to lauch LocAgent.
   ```
   python auto_search_main.py \
      --dataset 'czlll/SWE-bench_Lite' \
      --split 'test' \
      --model 'azure/gpt-4o' \
      --localize \
      --merge \
      --output_folder $result_path/location \
      --eval_n_limit 300 \
      --num_processes 50 \
      --use_function_calling \
      --simple_desc
   ```
   - `localize`: set to start the localization process
   - `merge`: merge the result of multiple samples
   - `use_function_calling`: enable function calling features of LLMs. If disabled, codeact will be used to support function calling
   -  `simple_desc`: use simplified function descriptions due to certain LLM limitations. Set to False for better performance when using Claude.

3. Evaluation
   After localization, the results will be saved in a JSONL file. You can evaluate them using `evaluation.eval_metric.evaluate_results`. Refer to `evaluation/run_evaluation.ipynb` for a demonstration.


## RepoMem Episodic Memory 扩展

### 是否修改了原始 LocAgent

是，但修改方式是**在原有 LocAgent localization pipeline 上增加一个可选的 memory 检索旁路**，而不是替换图索引、代码搜索或最终 localization/merge 流程。这里以加入 memory 前的提交 `4ea4d07` 为原始基线。

原始主链路仍然是：

```text
issue -> LLM agent loop -> tree/entity/graph/code search -> finish
      -> parse localization output -> merge/evaluate
```

启用 episodic memory 后，agent loop 中多出一条历史信息通道：

```text
issue -> SearchCommit(BM25 检索历史提交)
      -> ExamineCommit(读取提交信息、关联 issue 和历史 patch)
      -> 使用原有 LocAgent 工具检查当前 checkout
      -> finish -> 原有 parse/merge/evaluate
```

历史提交只作为定位提示。prompt 明确要求 agent 使用原有代码工具在当前代码库中验证候选文件，不能直接把历史 patch 当作最终答案。

### 为 episodic memory 修改或新增的核心部分

| 层次 | 文件 | 修改内容 | 对原机制的影响 |
|---|---|---|---|
| 离线 memory 构建 | `scripts/build_repomem_episodic_index.py` | 从本地 Git 历史提取 commit message、SHA、修改文件和 patch；可抓取关联 GitHub issue/Django ticket；生成 `commits.jsonl`、`issues.jsonl`、`meta.json` 和 `patches/` | 新增独立的历史 memory 索引构建阶段，不改变原 graph/BM25 code index 的内容 |
| memory 检索实现 | `plugins/location_tools/repo_ops/repo_ops.py` | 新增 memory index 加载与缓存、tokenization、BM25 排序、历史可见性控制、泄漏过滤，以及 `search_commit`、`examine_commit` | 扩展原工具层；正常的 tree/entity/graph/code-search 实现仍保留 |
| 工具 schema | `util/runtime/memory_tools.py` | 定义 `SearchCommit` 和 `ExamineCommit` 的 function-calling schema、参数和说明 | LLM 可像调用原 LocAgent 工具一样调用 memory 工具 |
| 工具注册 | `util/runtime/function_calling.py` | 将 memory 函数加入允许的函数集合，并根据开关把两个 episodic 工具加入 tool list | 改了 agent 可用工具集合，但 function-calling 模式下默认不启用 memory |
| CodeAct 执行环境 | `util/runtime/execute_ipython.py` | 将 `search_commit`、`examine_commit` 注入 IPython namespace | 不支持原生 function calling 的模型也可执行 memory 工具 |
| 主运行入口 | `auto_search_main.py` | 新增 `--use_repomem`、`--repomem_mode`、memory prompt、索引目录配置、工具选择和实验 metadata | 改了 pipeline 编排和 prompt；不启用开关时仍走原有主路径 |

### Episodic memory 的关键机制

- memory 以 repository 为单位加载，目录名使用 `owner__repo`。
- 检索对象主要是历史 commit message；`SearchCommit` 使用本地 BM25 排序，返回 SHA、提交主题和修改文件。
- 默认采用 fail-closed 的严格可见窗口。先把 `base_commit` 解析为 canonical SHA，再精确执行 `git rev-list --max-count=7001 <canonical-base-SHA>`，移除 base 自身并保留 Git 输出顺序，最终最多暴露 7000 个祖先 commit。base 或 manifest 无法解析时，memory 标记为 unavailable 并在调用 LLM 前终止，不再退化到索引末尾。
- 严格窗口只能从完整 Git object graph 构建：shallow clone、partial/promisor clone、replace refs、legacy grafts 或 `git fsck --full --connectivity-only` 失败都会拒绝构建；另行运行的 `git fsck --strict` 结果也会写入 audit（旧仓库的 `zeroPaddedFilemode`/`badTimezone` 等历史格式警告会记录但不冒充对象缺失）。manifest 还记录 `is_shallow=false` 等证据及 corpus/window SHA-256。
- 如果旧 `commits.jsonl` 没有收录 strict window 中的某些真实祖先，builder 会从完整 Git graph 补建 `visibility_commits.jsonl` 及对应 patch，并将其哈希写入 manifest。strict 运行同时加载该补充 corpus；legacy 对照仍仅使用原 `commits.jsonl`，避免“修正 strict”意外改变对照组。
- 默认启用 leakage filter：若历史 commit 关联的 issue 与当前 benchmark issue 高度重合，会从可见 memory 中排除。
- 当前 profile 默认限制 `SearchCommit` 最多返回 5 个结果、`ExamineCommit` 最多查看 3 个 commit，并截断过长 patch/issue 文本，避免 memory observation 挤占上下文。
- `ExamineCommit` 返回的是历史版本的行号和 patch；最终输出仍由原 LocAgent 的 `get_loc_results_from_raw_outputs()` 与 merge 逻辑处理。

严格窗口需要在运行前为数据集中的 base commits 构建一次。例如：

```bash
python scripts/build_repomem_visibility_windows.py \
  --dataset princeton-nlp/SWE-bench_Verified --split test \
  --index_dir repomem_index/verified_full500_episodic_with_issues_combined \
  --repo astropy/astropy=/path/to/full/astropy.git \
  --repo django/django=/path/to/full/django.git
```

上例为参数格式示意；对完整数据集必须重复 `--repo`，覆盖其全部 repository，否则 builder 会直接报错。

实际运行默认使用 `--repomem_visibility_policy strict`。只有复现实验中的旧行为对照才应显式传 `--repomem_visibility_policy legacy`；该模式会在日志和 trajectory audit 中标记为 unsafe。主入口会先对本批所有任务做 visibility preflight，任何缺失/损坏窗口都会在首次模型请求前失败。

### 当前工作树中的后续实验性改动

这些功能与 memory 实验有关，但不等同于最初的 episodic memory 本体：

- `--fixed_episodic_memory_manifest`：绕过 agent 主动检索，把冻结的 commit memory 直接注入任务 prompt，用于受控实验。
- `--repomem_tool_profile {current,paper-v1}`：切换工具 schema 和 observation 呈现方式，用于论文版本复现。
- semantic memory：`SearchSummary`/`ViewSummary` 和 `scripts/build_repomem_semantic_index.py` 是另一条文件摘要 memory 通道，不是 episodic commit memory。
- context guard、tool output bounds、provider fingerprint、sampling seed 等是运行稳定性和可复现性设施。
- ASQ search queue、disambiguation 和 entity grounding 是额外的 agent 搜索机制实验，不是加入 episodic memory 所必需。
- `build_bm25_index.py` 和 `dependency_graph/batch_build_graph.py` 对本地 JSONL 数据集及精确输出目录的支持，是实验运行便利性改动；没有改变原索引算法。
- `util/process_output.py` 的文件路径补捞，以及 `normalize_function_response()` 用 `ast.literal_eval` 替代 `eval`，属于输出/执行健壮性修复，不是 memory 检索机制。

### 兼容性边界

在常用的 function-calling 路径中，不传 `--use_repomem` 时不会把 episodic tools 加入模型的 tool list，因此原 LocAgent 的主要行为保持不变。需要注意：当前 CodeAct 路径会把 memory 函数注册到 IPython namespace，工具文档也可能包含 memory 工具；如果外部同时设置了 `REPOMEM_INDEX_DIR`，严格的 memory-off 对照实验应额外清理该环境变量，或进一步在 CodeAct 文档和 namespace 注册处增加开关隔离。

### 与 RepoMem 论文 v2 的实现差异

以下比较依据论文 [*Improving Code Localization with Repository Memory*, arXiv:2510.01003v2](https://arxiv.org/pdf/2510.01003v2)。结论是：当前实现复现了论文的总体设计，但**不是论文主实验的逐项等价实现**，因此不能直接把当前运行结果称为论文结果的严格 reproduction。

| 项目 | 论文 v2 | 当前实现 | 判断 |
|---|---|---|---|
| BM25 tokenizer | 主结果使用能识别代码实体名的 **LLM-based tokenizer**；论文另报告 whitespace tokenizer 消融 | 默认 `bm25-code-v2` 只是正则提取 token，并机械拆分 `snake_case`/`CamelCase`；没有调用 LLM，也没有论文 tokenizer 的 prompt、模型或缓存产物 | **确定不同，也是最重要的差异** |
| BM25 公式 | 论文只说明使用 BM25，没有公开公式、库、`k1`、`b` 和 IDF 细节 | 本地手写实现；`bm25-code-v2` 使用 `k1=1.5`、`b=0.75` 和始终为正的 IDF；另有 `bm25-experepair-v1` 采用另一种 IDF/floor | **无法证明相同**；即使 token 一样，排序也可能不同 |
| 默认检索数量 | 论文附录中的工具接口以 `SearchCommit(..., top_k=20)` 呈现 | `current` profile 的 schema 默认值和运行时 cap 都是 5；`paper-v1` 才把默认/cap 提升到至少 20 | **默认不同**；`paper-v1` 只修正这一部分 |
| SearchCommit 返回内容 | 论文称返回 ranked top-k commits，包括 SHA、commit messages 和 modified files | 每个 query 分别取 top-k，最多接受 3 个 query，因此一次调用可能返回多于 top-k 行且可能重复；通常只显示 subject，不显示完整 commit body；文件最多显示 12 个，并优先只保留 Python 文件 | **行为和信息量不同** |
| ExamineCommit 内容 | 论文描述为 complete code patch 和 linked issues/full context | `current` 默认 patch 最多 4000 字符、issue 最多 1200 字符、一次最多 3 个 commit、最多展示 3 个关联 issue；`paper-v1` 仍有 100000 字符 patch 上限和最多 3 个 issue | **不是严格的 complete context** |
| Agent task prompt | 论文附录明确说沿用原始 LocAgent task instruction，只扩展 action space/tool documentation | 当前 `get_task_instruction()` 额外要求“early”调用 `SearchCommit`、再调用 `ExamineCommit` 并验证当前代码；`paper-v1` 不会移除这段额外指令 | **确定不同**，会改变工具采用率和 agent policy |
| 7000-commit corpus | 论文对每个任务使用其 `base_commit` 之前的 7000 个 commits | 当前严格模式定义为 `W(b)=[git rev-list --max-count=7001 <canonical-base-SHA>]\{b}`，按 Git 输出顺序截到最多 7000；无法解析 base/window 时 fail closed | 已消除原 repo-wide `history_index`/索引末尾 fallback 的未来信息风险；`legacy` 仅保留用于配对对照 |
| 历史顺序/祖先关系 | 论文语义是当前 base commit 之前的历史 | manifest 从完整、非 shallow/partial 且无 replace/graft 的 object graph 构建，并通过 `git fsck`；运行时校验 manifest、corpus 和 window 哈希 | Git 祖先定义已严格化；仍不代表 BM25/tokenizer 等其他部分与论文相同 |
| 泄漏过滤 | 论文称移除与测试 instance 文本重叠的 issues，以及与其关联的 commits | 当前在运行时用 issue number、5-token n-gram Jaccard（默认阈值 0.20）或 80-token 连续重叠启发式过滤；论文没有公开其 overlap 算法 | 目标一致，但**无法证明过滤集合一致**；当前索引文件本身仍保留被过滤数据 |
| Commit–issue linking | 论文说明依赖显式链接，例如 `Fixes #123` | 当前正则除 `fix/close/resolve #N` 外，还接受 commit message 中任意 `#N`；而且 builder 默认不开 `--crawl_issues` | 当前链接范围更宽、可能更噪；未开 crawl 时缺少论文所用 issue context |
| 测试文件和非 Python 文件 | 论文正文只说 modified files，没有声明一律删除 test/non-Python paths | `paper-v1` 的 SearchCommit 只展示非测试 Python 文件；`current` 有 Python 文件时也省略非 Python 文件 | **额外过滤策略**，论文正文不足以支持其完全等价性 |
| 模型 | 论文所有实验使用 GPT-4o `2024-05-13` | strict-visibility pilot 请求当前 `openai/gpt-4o`，provider 每次实际回报 `gpt-4o-2024-11-20`；原定 `2024-05-13` deployment 在首次 case 前 preflight 返回 `DeploymentNotFound`，替换决定已写入不可变 amendment | **本次 pilot 确定与论文模型不同**；不将其数值称为论文主表复现 |
| 数据与现成索引覆盖 | 论文评估 SWE-bench-Verified 500 条、12 个仓库，以及 SWE-bench-Live 130 条、62 个仓库 | 自动 fallback 的 `verified_200_episodic_with_issues` 实际只含 `astropy` 和 `django`；另有通过 symlink 拼出的 Verified 12-repo combined index，但没有对应论文 SWE-bench-Live 的现成完整资产 | 默认资产**不覆盖论文评测范围** |
| 额外运行机制 | 论文描述的是原 LocAgent ReAct loop 加 memory actions | 当前若启用 fixed-memory injection、context guard、output bounds、ASQ、disambiguation 或 entity grounding，会进一步改变 prompt、observation 或动作调度 | 这些实验不能视为论文原始 RepoMem setting |

已有的 `--repomem_tool_profile paper-v1` 主要对齐论文附录里的工具参数和文本呈现，例如 `top_k=20`、隐藏 score、减少默认截断；它并不是“论文复现总开关”。它不会自动做到以下事项：

- 不会启用论文的 LLM-based tokenizer；
- 不会改变严格祖先窗口（它已由离线 manifest 固化并在运行时 fail closed）；
- 不会移除当前新增的 memory-specific task prompt；
- 不会锁定 GPT-4o `2024-05-13`；
- 不会保证 issue crawl、泄漏过滤结果和论文完全一致；
- 不会补齐 SWE-bench-Live 或默认 fallback index 的仓库覆盖。

论文也没有公开 LLM tokenizer 的完整构造方法、BM25 精确参数/库以及文本 overlap 的具体算法。因此，在没有作者代码或这些 artifacts 的情况下，这几项最多只能做近似复现，不能诚实地宣称 bit-for-bit 或 ranking-equivalent。当前实现最准确的定位是：**RepoMem 论文思想的独立重实现，并带有后续的安全、上下文控制和实验分析扩展；不是论文主表配置的原样实现。**

### Strict visibility 初步实验（2026-09-07）

在 SWE-bench Verified 500 条上先做了不调用 LLM 的全量静态对比：285/500 条的 legacy/strict commit set 不同；277 条 legacy window 含有非 base 祖先，282 条 strict window 含有原 corpus 漏掉的真实祖先。每题两组 set 的对称差均值为 439.898、中位数为 12、最大为 14000。这说明修正不是纯性能参数，确实改变了实验可见的历史边界。

随后使用 provider 实际回报的 GPT-4o `2024-11-20` 做了 12 题、24 个 arm run 的配对 pilot，每个 arm 两次成功采样：

| Cohort | n | Metric | Legacy | Strict | Strict − legacy |
|---|---:|---|---:|---:|---:|
| random | 2 | complete-file Acc@1/3/5 | 0.500 | 0.500 | +0.000 |
| sensitivity | 10 | complete-file Acc@1 | 0.400 | 0.600 | +0.200 |
| sensitivity | 10 | complete-file Acc@3/5 | 0.600 | 0.700 | +0.100 |

sensitivity 中有三个 strict-only metric flip：`pydata__xarray-3305` 在 Top-3/5 补回 gold 文件 `xarray/core/dataset.py`；`sphinx-doc__sphinx-11510` 和 `sympy__sympy-18763` 分别在 Top-1 把 gold 文件提到首位。没有 legacy-only win。但这只是小样本方向性信号；Acc@1 的 paired bootstrap 95% CI 为 `[0.0, 0.5]`、McNemar 双侧 `p=0.5`，Acc@3/5 的 CI 为 `[0.0, 0.3]`、`p=1.0`，**不支持统计显著提升的结论**。所有 24 个有效 arm 都观测到 `gpt-4o-2024-11-20`，visibility/provider 审计不变量失败为 0。纳入分析的 trajectory 成本为 `$27.23480`；另有两份因只成功 1 个 sample 而被排除、后来重跑的 trajectory，已记录成本 `$0.74584`，所以已知记录总成本为 `$27.98064`。其中一次 context-overflow 前的请求未落入成功 trajectory，真实 provider 账单可能略高，本地无法精确还原。完整机器可读结果在 `results/strict_visibility_pilot/summary.json`，人类可读报告在 `results/strict_visibility_pilot/report.md`。

该 pilot 与论文至少有四项明确差异：模型是 `2024-11-20` 而非 `2024-05-13`；BM25 使用 regex/code-token splitter 而非论文的 LLM tokenizer；`SearchCommit` cap 为 5 而非论文接口的 20；只是 12 题 pilot，而非论文的完整 benchmark。因此论文 LocAgent 的 67.8/72.4/74.3 只能作为背景，不能与上述 pilot 直接比较。

### E1：supplemental leakage propagation

默认 strict runtime 现在会把 overlap filter 判定出的 excluded issues 同时传播到原始 `commits.jsonl` 和 strict-only `visibility_commits.jsonl`；此前 supplemental rows 未参与这一步。`REPOMEM_SUPPLEMENTAL_LEAKAGE=off` 只保留为 E1 Control，不会恢复 legacy history fallback。

E1 使用 `scripts/prepare_e1_supplemental_leakage.py` 先构造并审计
`D_i=U_i∩C_supplemental∩L_i`，只选择 `|D_i|>0` 的 affected cases。运行器通过 `--e1_supplemental_leakage` 先执行一条共享 Control trajectory；逐个重放相同的 Memory call，observation 相同时继续共享，仅在第一条 differential Memory observation 处替换 Treatment observation 并 lazy fork。Primary 是该 observation 后的第一个 `tool name + canonicalized arguments`（或 finish files）是否改变；没有 exposure 的 affected case 按 ITT 记为 0。主报告只包含 Treatment-induced Next-Action Change Rate、Differential Exposure Rate 和 complete-file Acc@5。

冻结入口为：

```bash
bash scripts/run_e1_supplemental_leakage.sh --dry-run
bash scripts/run_e1_supplemental_leakage.sh
```

### E2：SearchCommit Top5 → Top20

E2 保持 strict history、已修复的 supplemental leakage propagation、同一
`current` tool schema、BM25、prompt、模型和 ExamineCommit 设置，只改变工具执行层的
SearchCommit 有效返回深度。共享轨迹固定为 Top5；同一 query 的 Top5/Top20 observation
相同时继续共享，第一次不同时才 lazy fork Top20 continuation。每个 query 都必须满足
Top5 结果是 Top20 的精确前缀。Primary 是 ITT 口径的 Memory-Adoption Change Rate，
secondary 是 Differential Exposure Rate 和 complete-file Acc@5。

当前实现的 `top_k` 是 per-query，因此最多三个 query 时 Top20 arm 可能返回 60 条；E2
不会同时修改这个聚合语义，也不会切换到会改变输出格式和文件过滤的 `paper-v1` profile。
12-case pilot 沿用 strict-visibility pilot 的已冻结 cases，只用于描述性机制验证，不代表
Verified 500 的总体效应。

```bash
bash scripts/run_e2_search_topk.sh --dry-run
bash scripts/run_e2_search_topk.sh
```

2026-09-10 的冻结 12-case pilot 得到 10 个有效 lazy-fork pair；Matplotlib 和
Sphinx case 在共享 Top5 prefix 阶段已经超过 GPT-4o 的 128K context，因而没有接受
Treatment，也没有被填成 Acc=0。有效 pair 中，Differential Exposure 和 Memory-Adoption
Change 均为 10/10；Treatment 在 9/10 中实际 Examine 了 rank 6–20 commit。Complete-file
Acc@5 为 Top5 `6/10`、Top20 `7/10`，paired delta `+10pp`（一题 Treatment-only win，
没有 Control-only win）。该结果是小型、非代表性 pilot，不能解释成 Verified 500 上的
总体提升。详见 `results/e2_search_topk/run/e2_summary.json`。

### E3：主 task prompt 的 Memory 指令

E3 将 baseline 冻结为 StrictHistory、修正后的 leakage propagation 和 E2 的 effective
Top20，并完整复用 E2 的 `current` tool schema。Control 保留当前主 task prompt 末尾
的 forced-early Memory 段落；Treatment 使用原始 LocAgent task prompt，但两组仍暴露
完全相同、仍推荐 early use 的 SearchCommit/ExamineCommit tools。Primary 定义为
SearchCommit 是否先于第一次 current-code exploration；secondary 为成功的
ExamineCommit adoption 和 complete-file Acc@5。

Pilot 使用 E2 中 10 个 context-feasible cases，并以 seed 44、45 各运行一次，两组共
40 个独立 trajectories。由于 prompt 从首次请求前已经不同，E3 不使用 frozen-prefix，
而采用 case/seed 内配对和 AB/BA counterbalancing。主指标只在两臂都有有效 trajectory
的 complete pairs 上计算；非对称失败单独报告，不把失败臂机械记为 Acc=0。

```bash
bash scripts/run_e3_prompt.sh --dry-run
bash scripts/run_e3_prompt.sh
```

在当前 Verified-500 strict corpus 上，静态审计 500/500 通过，但 affected count 为 0：虽然 231 题存在 excluded issues，未发现任何同时属于该题 strict window、supplemental-exclusive 且链接到 excluded issue 的 commit。根据预先定义的停止规则，当前 E1 会在任何付费模型调用前写出 audit/manifest 并停止；不会放宽 affected 定义来制造样本。

因此，准确的概括是：**改了 LocAgent 的工具层、prompt 和运行编排，增加了可选的历史检索机制；没有用 memory 替换原有 graph-guided localization，也没有因 episodic memory 改写核心 graph/parser/index 算法。**


## 📑 Cite Us

   ```
@inproceedings{chen-etal-2025-locagent,
    title = "{L}oc{A}gent: Graph-Guided {LLM} Agents for Code Localization",
    author = "Chen, Zhaoling  and
      Tang, Robert  and
      Deng, Gangda  and
      Wu, Fang  and
      Wu, Jialong  and
      Jiang, Zhiwei  and
      Prasanna, Viktor  and
      Cohan, Arman  and
      Wang, Xingyao",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.426/",
    doi = "10.18653/v1/2025.acl-long.426",
    pages = "8697--8727",
    ISBN = "979-8-89176-251-0",
    abstract = "Code localization{--}identifying precisely where in a codebase changes need to be made{--}is a fundamental yet challenging task in software maintenance. Existing approaches struggle to efficiently navigate complex codebases when identifying relevant code snippets.The challenge lies in bridging natural language problem descriptions with the target code elements, often requiring reasoning across hierarchical structures and multiple dependencies.We introduce LocAgent, a framework that addresses code localization through a graph-guided agent.By parsing codebases into directed heterogeneous graphs, LocAgent creates a lightweight representation that captures code structures and their dependencies, enabling LLM agents to effectively search and locate relevant entities through powerful multi-hop reasoning.Experimental results on real-world benchmarks demonstrate that our approach significantly enhances accuracy in code localization.Notably, our method with the fine-tuned Qwen-2.5-Coder-Instruct-32B model achieves comparable results to SOTA proprietary models at greatly reduced cost (approximately 86{\%} reduction), reaching up to 92.7{\%} accuracy on file-level localization while improving downstream GitHub issue resolution success rates by 12{\%} for multiple attempts (Pass@10). Our code is available at \url{https://github.com/gersteinlab/LocAgent}."
}
   ```
