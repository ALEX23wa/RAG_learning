# 独立证据表示修订与冻结

2026-10-03状态：evidence-v2已经封存，8份首轮dev报告已完成replay；见 [重评分结果](archive/quality-v2-evidence-v2-replay.md) 与 [当前状态](quality-v2-status.md)。以下保留复现用法，既有输出目录不可覆盖，不能因后续模型结果调整已封存的alternatives。

`scripts/freeze_evidence_revision.py` 只处理原题集和独立来源审核记录，不调用检索、模型、解析服务或外部 API。v1 原字节与原成绩保持不变；evidence-v2 是新评测契约，不自动发布。开发集修订必须声明 `post_hoc_dev_representation_repair: true`，不能把重评分改善称为前瞻的模型收益。

每个正例原证据组必须恰有一条审核记录，状态为 `unchanged`、`equivalent_added` 或 `unresolved`。每组最多追加 3 个等价表示，原组全部条目原样排在最前。除 `evidence_groups` 外，题目、facts、answerable、locator、split、ID、history 以及其他 case 字段的 JSON 语义必须不变。工具不重写原 `evidence`。

审核 JSON 顶层为 `schema_version: evidence-equivalence-audit-v1`，包含 `split`、`parent_dataset_sha256`、`parent_manifest_sha256`、`reviewer_id`、带时区的 `reviewed_at_utc`、`model_outputs_used: false`。dev 还需 post-hoc 声明；blind 需 `blind_model_outputs_seen: false`。这里的模型输出指被评测系统的检索/问答结果；经批准的原资料解析缓存仍需来源哈希与原物理页核对，不自动成为权威金标。

`records` 每项包含 `case_id`、从 0 开始的 `group_index`、`status`、`alternatives` 和 `review`。组审核及每个替代项审核都必须记录：

- `source_checked: true`、`semantic_invariants_preserved: true`、`model_outputs_used: false`。
- `reviewer_id`、`reviewed_at_utc`、`method`、`rule_id`、`reason`。审核者必须与所属 split 一致，条目审核时间不得晚于整份审核时间。
- 替代项另需 `visual_checked: true` 或 `physical_page_verified: true`，说明原始物理页核验方法。

每个替代项必须同原条目的 `path`、`source_hash`、`locator`，具有 `span_text`、`span_literal_sha256`（原 UTF-8 字节）和 `span_sha256`（NFC 后去空白）。工具检查哈希、记录完整性和跨 split 的标准化 span、cluster、template 冲突；数学、单位、否定、条件和表格关系等语义是否等价仍是独立审核者的责任，工具不会声称机器已证明。

只核查一个 split，不读取另一个 split 的题目、不写入或封存：

```powershell
python scripts/freeze_evidence_revision.py --parent .rag/evaluation/quality-v2 --dev-audit .rag/evaluation/quality-v2-evidence-v2-drafts/dev-equivalence-audit.json --corpus papers --check-split dev
```

对 blind 作者可将 `dev` 换为 `blind`。单 split 验证不完成跨 split 审核，不能用于发布。

两份审核完成后，操作人提供独立声明 JSON，例如：

```json
{
  "declared_by": "actual-run-operator-id",
  "declared_at_utc": "2026-10-03T10:00:00+00:00",
  "blind_model_runs_started": false
}
```

必须填真实身份与时间；这是可追溯声明，不是对“从未发生盲测运行”的机器证明。封存前逐项核对源文件完整 SHA256，两份审核必须无 `unresolved`。运行：

```powershell
python scripts/freeze_evidence_revision.py --parent .rag/evaluation/quality-v2 --dev-audit .rag/evaluation/quality-v2-evidence-v2-drafts/dev-equivalence-audit.json --blind-audit .rag/evaluation/quality-v2-evidence-v2-drafts/blind-equivalence-audit.json --corpus papers --output .rag/evaluation/quality-v2-evidence-v2 --seal --before-blind-declaration .rag/evaluation/before-blind-declaration.json
```

输出必须是全新目录；工具拒绝覆盖 v1、已有 draft 或已有 revision。去掉 `--seal` 可生成独立 draft；draft 不能通过发布验证。封存产物包括原字节 `parent-v1/`、两份审核原字节、新 dev/blind 数据、逐组 change audit、源哈希核对、summary 和最后写入的 manifest。现有 `load_dataset` 可加载新数据并验证文件哈希。

发布会从父数据与审核重新构建所有 alternatives，校验全部非证据字段未变，并验证实际源文件。改回旧 `frozen` 状态不能绕过新增 alternatives 的审核闭环。blind 等价审核者必须进入 `gold_author_ids`，不得兼任最终答案审计者；内容冻结不替代独立答案、逐主张、数字、引用和性能验收。

重评分必须另存报告并绑定原响应 SHA256、新题集/manifest SHA256、评分器 SHA256。仅对原来真实返回的 evidence、candidates、context 评分；不能补检索、凭隐藏 parent 补证据或更改历史耗时。v1 与 evidence-v2 成绩须并列，后者标为开发集事后表示修订诊断。

封存后可用 `scripts/replay_retrieval_scores.py` 重评分首轮 8 份既有响应。脚本只读取 dev 的修订链路；它不打开 blind 题目、答案或等价审核正文。整包跨 split 规则由封存与发布工具核验，单独 replay 不能替代该步骤。

```powershell
python scripts/replay_retrieval_scores.py --dataset .rag/evaluation/quality-v2-evidence-v2/dev.json --output .rag/reports/quality-v2/replay-evidence-v2 --reports .rag/reports/quality-v2/old-bm25-v3.json .rag/reports/quality-v2/original-hybrid-v1.json .rag/reports/quality-v2/p0-hybrid-v1.json .rag/reports/quality-v2/parent-v4-bm25.json .rag/reports/quality-v2/parent-v4-hybrid-rrf10.json .rag/reports/quality-v2/parent-v4-hybrid-rrf30.json .rag/reports/quality-v2/parent-v4-hybrid-rrf60.json .rag/reports/quality-v2/parent-v4-rerank-rrf10.json
```

输出目录必须不存在。脚本首先要求原报告完整、原数据与父 manifest 哈希一致，并从实际保存响应重现逐题 v1 评分；重现失败时拒绝继续。每个新报告并列 `original_v1_metrics` 与 `evidence_revision_metrics`，保留历史 elapsed/RSS/cost，记录原响应的 canonical JSON 哈希、原报告哈希、修订 suite 哈希和实际评分代码哈希。为避免复制大段响应，新报告通过原报告位置与 SHA256 引用原响应。没有 trace 的历史候选/context 成绩仍为 unknown；不测量新检索性能，也不把 replay 时间算为用户等待时间。

所有输出固定为 `posthoc_diagnostic_only`，不能直接用于发布。冻结 evidence-v2 后新增/修改替代项必须另建版本并重新审核，不能在看到后续模型分数后为某个模型追加金标。
