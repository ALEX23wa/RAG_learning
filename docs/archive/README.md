# 历史实验归档

这些文件记录实验发生时的配置、成绩和判断，保留失败与中间版本，不能当作最终运行状态。最终结论见 [结项报告](../quality-v2-status.md)。原始答案、源审核、费用台账、代码快照和索引均保留在本机 `.rag/`，不纳入 Git 上传。

- [第一轮消融](quality-v2-first-round-ablation.md)
- [等价证据重评分](quality-v2-evidence-v2-replay.md)
- [Embedding 对照](quality-v2-embedding-comparison.md)
- [重排对照](quality-v2-reranker-comparison.md)
- [BGE 候选构建](quality-v2-bge-candidate.md)
- [查询规划对照](quality-v2-planner-comparison.md)
- [完整问答开发记录](quality-v2-answer-development.md)
- [阶段状态快照](quality-v2-status-history.md)

项目收尾移除了被 `environment.yml` 替代的 `environment_xxpp.yml`；旧文件可从 Git 历史恢复。保留全部仍由运行入口、测试或复现实验使用的代码，没有为减少文件数量删除功能模块。
