# 第五批 T59–T64 开发与提交前验证

六题 generic 开发完成，最终共 **24 个测试方法、279 条 test/subTest 记录、141 次实际 kernel launch**，失败/错误/skip/xfail 均为 0。
六个规范 ZIP 已完成源码/回执/成员/哈希验签；状态为候选就绪、未提交平台。

| Task | 算子 | 测试方法 / kernel launch | 两个代理负载 speedup | source / verification |
| ---: | --- | ---: | --- | --- |
| 59 | [build_trtllm_mha_page_table](experiments/build_trtllm_mha_page_table.md) | 4 / 25 | 14.573x / 9.293x | `4836fe0` |
| 60 | [clamp_position](experiments/clamp_position.md) | 3 / 22 | 1.175x / 1.286x | `4836fe0` |
| 61 | [compute_src2dst](experiments/compute_src2dst.md) | 3 / 19 | 1.495x / 1.349x | `4836fe0` |
| 62 | [concat_mla_k](experiments/concat_mla_k.md) | 5 / 28 | 1.676x / 1.457x | `7b53fde` |
| 63 | [create_flashinfer_kv_indices](experiments/create_flashinfer_kv_indices.md) | 4 / 13 | 2.782x / 112.910x | `4836fe0` |
| 64 | [deepep_permute](experiments/deepep_permute.md) | 5 / 34 | 5.430x / 3.145x | `4836fe0` |

测速包含 wrapper，两个负载分别见各题账本；五轮交替 AB/BA 的配对比值取中位数。
NVIDIA RTX 5070 Ti / Python 3.12.13 / PyTorch 2.13.0+cu130 / Triton 3.7.1 / CUDA 13.0。
代理负载不等于平台隐藏测试，代理 speedup 不能替代八芯平均分。所有平台目标仍为 `target-runtime-unverified`。

## 已处理的契约边界

- T59：全部 4096 正约数页大小、页尾旧值、零长度、跨页/多 tile、整数索引与多维 stride；设备端读取长度。
- T60：int32/int64 极值（包含原类型减一的溢出语义）、空输入、尾块、stride、重复调用。
- T61：稳定路由排序的逆映射、随机/恒等/逆序排列、int32 输出、索引 stride。
- T62：bf16 原始位、head/tile/维度边界、空维度、多维 stride，以及 2^31 元素偏移的稀疏视图回归。
- T63：None/非零 start、长度 511/512/513、8193 长段、空段、前后及段间旧值保留。
- T64：fp16/bf16/fp32、输出类型转换、-1 跳过、未写行保留、topk_ids 不参与运算、长 hidden 和非连续视图。

## 证据与复现

- [契约与固定上游来源](strategy-batch5.md)。
- [完整证据清单](data/batch5-development-20260910.json)：全部 source/test/helper/runner/benchmark/ZIP 哈希、设备、测试数和原始测速样本。
- 证据清单 SHA-256：`74b37287f5649dc044696565306c97111117c991b3549393a88bcb703eba46b1`。
- 回执、完整日志、静态检查、launch PID/PGID、初版记录和解压清单位于 `artifacts/competition/batch5-development-20260910/`。
- 各题账本记录实际 ZIP 路径、完整源码 commit、verification commit 和重放命令；ledger commit 由本文件/账本 Git 历史定位。

## 提交边界

本轮未运行平台实时 preflight、未上传、未正式提交，不消耗比赛额度。
正式提交任务需在当时重新核对账号/团队、batch=5、Task、截止时间、全局额度和一次性提交状态；本地 release 回执不能替代这些实时条件。
当前没有 vendor 文件，也没有非 NVIDIA 目标 runtime 执行证据；具体芯片正确性与 0.1x 门槛由后续目标验证/平台评测确认。
