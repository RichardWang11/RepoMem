# P0-1 人工标注说明

## 你需要填写的文件

只填写 `applicability_human_annotation_30.csv`。文件共有30行，每行是一条
issue–Memory candidate。请保持 `sample_id` 不变，不要删除、排序或新增行。

## 标注顺序

1. 阅读 `issue_description`，明确任务要求。
2. 阅读 `memory_subject`、`memory_message`、`memory_changed_files` 和
   `memory_patch`，理解历史经验做了什么。
3. 阅读 `current_code_context_json`，判断 Memory 引用的结构在当前 base
   commit 是否仍存在。这里的 current graph 只能证明“当前可解析”，不能证明
   历史关系从未变化。
4. 阅读 `gold_changed_files` 和 `gold_patch`。Gold 在本实验中是离线 oracle，
   用于判断 Memory 对真实修复是否适用。
5. 独立填写 Applicability、Validity 和 confidence。

## Applicability 标签（必填，三选一）

- `Direct-help`：Memory 提供可直接迁移到 Gold file、symbol 或目标行为的
  修复证据。在 evidence 中写明具体 transfer link。
- `Indirect-help`：没有直接给出修复，但正确缩小了模块、API、依赖或测试
  范围。在 evidence 中写明具体 bridge。
- `Irrelevant`：没有具体 transfer；属于错误子系统、表面词汇重合、泛化
  信息、过时信息或误导信息。

`applicability_evidence` 至少一句，不能只写“相关/不相关”。

## Validity 标签（必填，三选一）

- `Valid`：关键文件存在，至少一个关键 symbol/API anchor 可解析，且没有
  明显当前结构冲突。
- `Partially-valid`：部分路径或 symbol 已变化，但仍有可识别且有意义的
  anchor。
- `Invalid-obsolete`：关键 anchor 均无法解析，或当前代码明确否定 Memory
  所依赖的结构/API。

`validity_evidence` 至少一句，写清楚哪些 file/symbol/API 仍存在或已失效。

## Confidence（必填，三选一）

- `high`：证据明确，几乎没有其他合理解释。
- `medium`：总体可判，但存在一处不确定或间接推断。
- `low`：packet 信息不足、patch 截断或两种标签都合理。

不确定时仍选择最合理的标签，并在 `reviewer_notes` 解释；不要留空。

## 严禁在主标签冻结前查看

不要先看 `annotation_provenance_after_freeze.csv` 指向的完整 trajectory，也
不要查看 Agent adoption、最终 top-5、Acc@5、baseline outcome、paired harm、
sampling stratum 或机器 StructuralRelevant。它们会造成 outcome-informed bias。

完成30行后保存 UTF-8 CSV，然后运行：

```bash
source /data/wjl/RepoMem/openaigpt4o_env
/data/conda_envs/locagent/bin/python \
  scripts/validate_applicability_labels.py analyze
```

在 GPT judge 标签尚未完成时，脚本会正确停在 `AWAITING_GPT_LABELS`；Human
标签不会丢失。
