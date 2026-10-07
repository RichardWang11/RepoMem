# SWE-bench Verified 跨仓库 Transfer Task 聚类与迁移实验

## Summary

- 研究问题：
  1. 能否基于修复机制，而非 issue 表面文本，可靠发现 same-pattern tasks？
  2. 同 repo 和跨 repo 的同 pattern 经验是否能提升代码定位与最终修复？
  3. Oracle 已知 pattern 有效时，实际的 issue-only retriever 能否找到这些经验？
- 数据使用本地 SWE-bench Verified 500 tasks、12 repos，以及 `repomem_index/verified_full500_episodic_with_issues_combined/` 中的 Git 历史。
- 当前数据高度不均衡：Django 231/500。所有 split、指标和置信区间按 task/repo 平衡，禁止普通随机切分。
- 实验分为：Task Signature → 聚类 → 双人验证 → repo-disjoint split → 跨 repo memory 实验 → 定位与修复评估。
- 只有同时满足“至少 3 repos、人工确认、split 无 repo 重叠”的 cluster 才能用于正式 cross-repo transfer claim。

## Task Signature 与聚类

- 先将 gold patch 与本地 Git commit 对齐：
  - 对 gold 修改文件计算稳定 patch-id。
  - 遍历对应 repo 的 commits，以 first-parent diff、仅保留 gold 文件后匹配 patch-id。
  - 多个匹配时优先：base commit 的后继、包含 instance issue/PR 编号、非 backport、时间最早。
  - 无精确匹配时 `commit_message=null`，不得用模糊匹配冒充 gold commit；主分析保留该 task，并增加“仅精确匹配 task”的敏感性分析。
- 使用 `gpt-4o`、temperature 0、JSON Schema，从 `problem_statement + gold patch + exact commit message` 生成两次独立 signature；字段冲突时运行第三次 adjudication。
- `TaskSignature` 固定字段：
  - `bug_category`
  - `observable_failure`
  - `trigger_condition`
  - `root_cause`
  - `violated_invariant`
  - `repair_action`
  - `repair_shape`
  - `affected_component_role`
  - `abstract_pattern`：不含 repo、路径、类名等专有标识的一句话
  - `evidence`、`confidence`、`commit_match_status`
- 只 embedding canonical signature，不直接 embedding 原始 issue。使用项目现有默认 `text-embedding-3-small`、1536 维、L2 normalization。
- 主聚类算法锁定为 sklearn HDBSCAN：
  - `metric="cosine"`、`algorithm="brute"`
  - `min_cluster_size=3`
  - `min_samples=2`
  - `cluster_selection_method="eom"`
  - 未归类 task 保留为 noise，不强行分配。
- 运行 100 次 90% task subsampling；cluster pair co-assignment stability 必须不低于 0.70。
- 输出两类候选：
  - Same-repo pattern：同 cluster、同 repo 至少 2 tasks。
  - Cross-repo pattern：同 cluster 至少 3 repos。

## 人工验证与 Transfer Split

- 两位标注者独立判断 task pair：
  - `same_transferable_pattern`
  - `related_but_not_transferable`
  - `different_pattern`
  - `unclear`
- “same”必须同时满足：核心 root-cause mechanism 相同，且 repair operation 能在抽象层面迁移；仅组件相似、关键词相似或都修改缓存不算。
- 固定 400-pair 验证集：
  - Top-20 largest clusters：200 个 medoid-member pairs。
  - 其他 cross-repo 候选：120 pairs，优先扩大 repo/cluster 覆盖。
  - Embedding 最近但不同 cluster 的 hard negatives：80 pairs。
- 标注包展示 issue、gold patch、commit message 和 signature，但隐藏 cluster ID、距离、模型判断和实验结果。
- 报告 raw agreement、Cohen’s κ、Gwet AC1、pair precision/recall 和 Wilson 95% CI。
- 聚类验证通过条件：
  - adjudicated candidate-pair precision ≥0.85；
  - Wilson lower bound ≥0.80；
  - AC1 ≥0.70 且 κ ≥0.60。
- 正式 transfer task 必须有至少两个被人工标为 `same_transferable_pattern` 的 source tasks，且来自两个不同的其他 repos。

- Cross-repo 主 split 使用 3-fold repository-disjoint CV：
  - 12 repos 通过确定性 greedy bin-packing 分为三组，按 eligible task 数平衡，seed `20260720`。
  - 每一 fold 的 test repos 完全不出现在 train memory bank。
  - 每个 test task 必须在 train repos 中有至少两个 validated sources。
  - 去除 issue 文本近重复、相同 patch fingerprint、相同 PR/commit 和派生 backport。
  - 每个 task 只作为一次 test；报告 task-macro、repo-macro 和 fold-macro 结果。
- Same-repo 辅助实验单独构造：每个 repo/cluster 按 `created_at` 做早期 source、后期 target，避免未来经验泄漏；不得与 cross-repo 主结论混合。
- 正式 cross-repo cohort 最低要求：40 test tasks、5 clusters、5 target repos；不足时结果标记为 exploratory。

## Memory 与 Agent 实验

- 新增跨 repo `ExperienceMemory`，不改变现有 repo-local `search_commit` 行为。每条 experience 包含：
  - source `instance_id`、repo、issue；
  - canonical signature 和 abstract repair summary；
  - exact commit message（若存在）；
  - gold changed files 和 gold patch；
  - cluster/fold/provenance，但 cluster 信息不暴露给 deployable retriever。
- 新增工具接口：
  - `search_experience(query_list, top_k=5)`
  - `examine_experience(experience_ids)`：按需展示 source issue、修复摘要和截断 patch。
- `auto_search_main.py` 增加：
  - `--use_transfer_memory`
  - `--transfer_memory_dir`
  - `--transfer_fold`
  - `--transfer_condition {placebo,oracle,retrieval}`
- 每个 cross-repo test task 跑四个条件、每条件 3 次：
  1. `no_memory`：LocAgent baseline。
  2. `placebo`：来自 train repos、不同 validated pattern、patch/token 长度匹配的 hard-negative memories。
  3. `oracle`：仅提供人工确认的跨 repo same-pattern memories，测迁移潜力上界。
  4. `retrieval`：搜索完整 train memory bank；query 只能使用 test issue，禁止 gold patch、gold signature、cluster ID。
- Deployable retrieval 固定为 `0.7 × issue-only signature cosine + 0.3 × BM25`，top-5；同时离线报告 Recall@1/5、MRR，并与 raw-issue embedding、纯 BM25 比较。
- 定位主指标：
  - File Acc@5 为 primary。
  - File Acc@1/3、MRR、retrieval Recall@5 为 secondary/diagnostic。
- 修复阶段：
  - 从 top-5 localized files 启动统一 repair prompt，memory 条件与定位阶段保持一致。
  - 最多生成一次 patch，加两次仅用于修复 diff 格式/`git apply --check` 的重试；不得利用测试失败反馈迭代。
  - 在隔离 worktree 应用 patch，用官方 SWE-bench Docker harness 计算 resolved rate。
  - Patch apply rate、FAIL_TO_PASS、PASS_TO_PASS 分别记录；resolved rate 为 secondary outcome。
- 两个预注册比较：
  - H1：`oracle - placebo`，检验同-pattern 跨 repo 经验是否有因果效用。
  - H2：`retrieval - placebo`，检验实际系统能否发现并利用经验。
- 每个 task 的 3 次重复先求均值，不能当作 3 个独立样本。使用 10,000 次 task-within-repo cluster bootstrap，报告 paired effect、95% CI；H1/H2 使用 Holm correction。
- “支持迁移”的门槛为 File Acc@5 提升至少 5 个百分点且校正后 95% CI 下界大于 0。Resolved rate 独立报告，不作为定位主结论的必要门槛。

## Artifacts、测试与假设

- 统一输出到 `results/transfer_task_clustering/v1/`：
  - commit mapping audit、signatures、embeddings、clusters；
  - annotation packets/labels/reliability；
  - frozen folds、memory banks、retrieval rankings；
  - localization/repair trajectories、bootstrap results、最终 Markdown 报告；
  - 所有冻结文件保存 SHA-256、模型名、prompt version、seed 和数据集 fingerprint。
- 实现入口采用一个可恢复的 pipeline CLI，子命令为：
  - `map-commits`
  - `extract-signatures`
  - `cluster`
  - `prepare-annotation`
  - `freeze-splits`
  - `build-memory`
  - `evaluate-retrieval`
  - `analyze`
- 测试覆盖：
  - signature JSON validation、API 重试和断点续跑；
  - gold-file patch-id 对齐、merge/backport/多匹配/无匹配；
  - cluster 和 split 在相同 seed 下完全确定；
  - train/test repo 零交集、test gold 不进入 memory/query；
  - 每个 cross-repo target 的两 source/two repo 条件；
  - duplicate patch/issue/backport leakage；
  - placebo 的 repo、数量和 token-budget 匹配；
  - 3 repeats 不被统计为独立 task；
  - 12-task/4-repo synthetic fixture 端到端 smoke test。
- 默认假设：
  - Signature 和 source experience 可以使用 gold patch，因为它们只用于离线 benchmark 构造和 train memory；test-time retrieval 严禁使用 test gold。
  - `gpt-4o` 与 `text-embedding-3-small` 沿用当前项目配置。
  - 完整修复评估需要额外安装官方 SWE-bench harness，并提供可运行 Docker 的机器；缺少 Docker 时不得用本地简化测试替代 resolved rate。
  - Same-repo 结果是辅助分析；论文的 cross-repo claim 只来自 repository-disjoint folds。
