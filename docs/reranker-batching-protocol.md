# 重排序批处理与性能比较协议

## 当前协议

`scripts/benchmark_models.py rerank` 在新报告的 `settings.batching_protocol` 中记录
`native-predict-all-pairs-forward-rss-v1`。

每个问题只调用一次原生 `predict(all_pairs, batch_size=配置值)`。benchmark 不提前按候选顺序切批，也不自行重排输入：BGE CrossEncoder 保留已安装库的全局长度分组和结果逆排列；Qwen 适配器保留其原生顺序分批。分块、窗口、完整输入长度核验、模型精度和候选池保持各自冻结配置。

RSS 检查通过临时 forward hooks 执行。BGE 挂在 CrossEncoder 本身，Qwen 挂在适配器的 causal model；每次 forward 前后各调用一次资源检查。post hook 使用 `always_call=True`，模型 forward 失败时仍检查 RSS。发生超限、模型异常、hook 注册异常或其他异常，都会清理本轮 hooks 并恢复原有 `predict` 属性；预先存在的 hooks 保留。没有可用 hook 或预测未经过受保护 forward 时失败，不继续产出有效结果。

这些检查与后台采样仍产生监控开销。新协议对齐原生批次组成，并不把 benchmark 的重排序耗时变为 Service 的完整问答耗时。

## 为什么需要独立协议

此前 benchmark 先按原顺序把所有 pairs 切成小批，再分别调用 `CrossEncoder.predict`。本机 sentence-transformers 6.0.1 的原生 `predict` 会先对完整输入按长度估计排序，然后分批，最后恢复结果顺序。外层切批让 BGE 无法跨小批组合相近长度输入，并重复执行每次 `predict` 的准备工作。Qwen 当前适配器本来按原顺序切批，因此该差异对两个适配器不对称。

库的长度估计是字符数，并非实际 token 数。全局分组通常有利于减少 padding，但不能从源码推导每题必然更快或推断实际提升幅度。不能按候选数、padding 比例或窗口数线性换算历史耗时。

## 历史报告与新运行

- 旧 v3/v4 query commit 的读取算法不变；旧报告缺少新协议字段时，继续按其实际旧身份读取。
- 新协议属于 `settings`，已经进入 query commit 和 resume 身份。旧报告不能在新协议下续跑，即使其他字段一致。
- 使用新的输出路径保留新 epoch；不得覆盖、改写或拼接旧报告的记录。旧报告仍是原包装器下的质量与性能证据。
- 新旧批次组成可能引起浮点微差和相近分数的名次变化，应实际核查，不能替换旧分数或宣称完全等价。

## 后续实测口径

性能复测尚未由本次代码修改完成。固定模型文件、FP32、线程、候选池、问题窗口、token 预算和 inference batch size，预热后在相同环境测量。可记录每批实际 token 长度、padding 数量、forward 次数、耗时及服务与数据库的 RSS。若比较两种协议，采用交替顺序并单列监控开销和环境差异。

最终仍以真实 Service 全链路的完整问答 p95、峰值 RSS、费用和质量审核为验收依据；离线固定池指标、前缀候选事后消融和新批处理协议本身都不能代替该验收。
