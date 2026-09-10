# 第五批首日六题开发契约

范围：T59–T64，batch 5；本轮开发、代理验证、不可变 ZIP，不提交平台。
官方完整题面位于 `tasks/batch-5/`，原始 API 记录位于
`data/batch5-development-contracts-20260910.json`。
六题均 exact，八芯均需达到 0.1x；截止 2026-09-17 19:59:59（北京时间）。
核心计算使用 Triton；不允许失败后 Torch fallback。

| Task / basename | 精确参数顺序 | 输入与输出契约 | 首版 |
| --- | --- | --- | --- |
| 59 build_trtllm_mha_page_table | req_to_token, req_pool_indices, cache_seqlens, page_table, page_size | 输出 [bs,max_pages] int32；有效页取 slot // page_size，尾部保留 page_table；page_size 为 4096 的正约数 | 一次遍历融合 gather、除法和尾部复制 |
| 60 clamp_position | seq_lens | 1D int32/int64；输出同 shape/dtype，先原类型减一再 clamp(min=0) | 单次融合 |
| 61 compute_src2dst | reorder_ids, num_toks | reorder_ids 为 [num_toks] int64 排列；输出 int32，out[reorder_ids[d]]=d | 直接 scatter，无排序、无原子 |
| 62 concat_mla_k | k, k_nope, k_rope | bf16；[T,H,N+R]、[T,H,N]、[T,1,R]；返回拼接张量；生产 H=128,N=128,R=64 | head 分组共享 RoPE，单次拷贝 |
| 63 create_flashinfer_kv_indices | req_to_token, req_pool_indices, page_kernel_lens, kv_indptr, kv_start_idx, kv_indices | pool 为 [max_batch,max_context] int32；indptr 为 [bs+1] int32；start 可 None；未写区保留 kv_indices | clone 保存底值，每请求循环复制 512 元素 |
| 64 deepep_permute | input, gateup_input, src2dst, topk_ids, topk, hidden_size | input [T,H]，gateup [T*K,H]，src2dst [T,K] int32；负目的地跳过；拷贝时转输出 dtype；topk_ids 不参与 reference | 每 token 的 512 元素 tile 读一次，多目的地写出 |

题面文字对 T59/T62 提及上游原地接口，但公开 reference 返回新张量。
本轮以可执行 reference 为验证契约：返回新张量，保持所有输入不变；
T59/T63/T64 的未写区域必须精确保留。T62 全量覆写，无需复制 k 的旧值。
T61 排列唯一；T64 的有效目的行唯一是路由置换前提，冲突目的地不声称有确定语义。

未公开：完整 benchmark shape、stride 范围、部分索引 dtype、T64 数据 dtype、
实际芯片型号/编译器。代理额外覆盖非连续输入、int32/int64 索引、空维度、
非整除 tile、页大小全部正约数，以及 T64 fp16/bf16/fp32 和输出类型转换；
这些是验证假设，不是隐藏测试事实。所有地址乘法在 int64 域完成。

固定复用来源：SGLang commit `8014d9d062c3cc5d393596ecdf2f7009191965df`。
T59 `python/sglang/kernels/ops/kvcache/trtllm_mha_page_table.py`；
T61/T64 `python/sglang/kernels/ops/moe/ep_moe_kernels.py`；
T63 `python/sglang/kernels/ops/kvcache/kv_indices.py`；
T60/T62 的 `python/sglang/kernels/jit/csrc/elementwise/` 下
`clamp_position.cuh` / `concat_mla.cuh` 提供计算及 head 共享结构，改用 Triton。
不引入 CUDA 私有指令、PDL、cache hint 或上游 SWA/量化额外分支。

开发顺序 60→61→59→63→62→64；首次正确成本预估依次为低、低、中低、中低、中、中。
所有目标芯 runtime 未验证；T60 小负载启动开销、T63 长段并行度、
T62 head 分组和 T64 路由写入是主要跨芯风险。
先取得正确基线，再以逐芯实测决定 vendor；T59/T63 为主要优化方向，
不把他队单次高分或 NVIDIA 代理速度视为本队八芯预测。

ZIP 每题仅含 `<operator>.py`；将来确有必要时才增加
`_amd/_ascend/_enflame/_hygon/_iluvatar/_kunlunxin/_metax/_nvidia` 后缀。
必要验证：完整 unittest、实际非空 kernel launch、commit 字节回执、
ZIP 成员/源码/canonical SHA-256 一致。平台实时 preflight 留到实际提交任务。
