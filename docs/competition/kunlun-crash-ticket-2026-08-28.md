# 昆仑芯评测崩溃工单(2026-08-28)

**队伍**:SoulCoder(team_id 2223)**账号**:15600308080 **赛季/批次**:第二季
race 782kzq4m / batch 3(Task 25–30)

## 现象

自 2026-08-28 00:08 起,我们在 batch 3 的提交在昆仑芯上几乎全部以同一
指纹失败,而同提交的其余七芯全部正常完成:

```
执行超时 (1830s/1800s),超时发生在: 验证执行阶段
子进程在超时前已退出(结果未能送达),疑似异常崩溃
Subprocess crash: Fatal Python error: Segmentation fault / Aborted
Thread ... torch/_inductor/compile_worker/subproc_pool.py:47 _recv_msg
环境: /root/miniconda/envs/python310_torch25_cuda python3.10
exec_ms 恒约 1833–1835s
```

## 关键时间线(按提交时间,均为本队)

| 时间 CST | Task | sub | 昆仑结果 |
| --- | --- | --- | --- |
| 00:08:11 | 29 gelu_and_mul | 5733 | 崩溃(上述指纹) |
| 00:10:57 | 30 interleaved_rope | 5735 | **通过**(该提交最终 8/8 valid) |
| 00:13:51 | 25 draft_topk1 | 5737 | 真实跑完,返回 correctness 结果(argmax mismatch,非崩溃) |
| 00:16–09:18 | 25/26/27/28/29 共 17 笔 | 5740…5811 | 全部同指纹崩溃;其中 5807 返回过"服务线程卡死自动恢复,请重新提交"(exec_ms=0) |
| 09:15/09:18 | 29/28 | 5810/5811 | 仍同指纹崩溃 |

涉及 submission_id:5733, 5737–5743, 5765–5769, 5802–5809, 5810, 5811。

## 请求

1. 请检查昆仑评测 worker/环境(torch 2.5 inductor 编译子进程池),
   该指纹与提交内容是否相关(我们怀疑与 erf/超越函数或 reference 编译
   路径相关,已在最新提交 5840 中规避 erf);
2. 若确认环境问题,请求对上表失败提交**免费 rerun** 昆仑芯;
3. 如可能,提供崩溃子进程的完整 stderr/native backtrace 与 worker/pod
   标识,便于我们区分内容触发与环境故障。

## 佐证

同日 00:10 我们的纯整数 kernel 在昆仑通过;00:13 的另一提交真实完成并
返回 case 级结果——评测器并非整体不可用。其余七芯对全部 20 笔提交均
正常完成评测。

## 后续实证更新(2026-08-28 12:00)

1. **T29 崩溃根因已定位为我方内容**:`tl.math.erf` 编译触发昆仑 worker
   崩溃(替换为 A&S 有理逼近后 sub 5840 昆仑 0.488x 通过、8/8 valid)。
   该类崩溃不再请求 rerun。
2. **核心请求(T28)**:sub 5845 与 sub 5861 选中的
   `gate_up_lora_b_kunlunxin.py` 字节完全相同,其余七芯全部通过;昆仑
   分别返回"服务线程卡死自动恢复"与 1833s compile-worker Aborted。
   请求:**在健康 worker 上仅对 sub 5861 重跑昆仑芯**(不耗团队额度、
   保留既有七芯结果);若无法重跑,请**返还一次因基础设施错误消耗的
   Task 28 提交机会**。
3. **请求提供的诊断字段**(任一即可大幅定位):
   - 崩溃时正在编译的 JIT kernel 名称与 specialization/case_idx;
   - 崩溃发生的编译 pass(TTIR/TTGIR/SDNN/LLVM/runtime);
   - 子进程退出信号与时间(先 SIGABRT 后 1800s,还是 watchdog 杀);
   - 完整 stderr 末 200 行 / native backtrace;
   - worker/pod/镜像标识、compiler cache key(hit/miss);
   - 其他队伍 T28 昆仑成功记录的时间分布(匿名即可):若均发生在
     00:14 前则支持服务事故,若按 pod 分组则可直接锁定坏 worker。
4. T25(sub 5852)与 T26/T27 同指纹失败仅作附证,不请求 rerun。


## 后续实证更新(2026-08-29 04:30)

 overnight 至今日的 8 笔 batch-3 提交中,昆仑 **7 崩 1 过**,且通过例
 是轻量算子、崩溃例覆盖四种截然不同的 kernel 结构:

| 时间 CST | Task | sub | 昆仑 | 选中文件/结构 |
| --- | --- | --- | --- | --- |
| 00:48 | 29 gelu_and_mul | 6127 | **通过 0.485x** | 平铺 elementwise(A&S,无 erf) |
| 00:58 | 28 gate_up_lora_b | 6135 | 崩(1833s Aborted) | host-resolved ieee dot v2 |
| 02:01 | 26 router_cudacore | 6187 | 崩(同指纹) | generic split-K ieee dot |
| 02:04 | 27 router_tensorcore | 6190 | 崩(同指纹) | generic split-K ieee dot |
| 02:5x | 28 gate_up_lora_b | 6194 | 崩(同指纹) | **无 dot FMA + stages=1 + int32 净室结构** |
| 03:2x | 26/27 | 6204/6212 | 崩(同指纹) | generic(6204 华为已通过、七芯仅剩昆仑) |

 结论更新:

 1. erf/超越函数假设已被 T29 反例双重排除(规避后通过);
 2. dot/stages/编译复杂度假设被 T28 6194 排除(FMA 无 dot 仍崩);
 3. 结构假设被排除:同日同 Task 六种结构、跨 Task 四类结构同指纹;
 4. 唯一通过例(gelu)为单 kernel 单 launch 轻量评测;崩溃例均为
    多 kernel 或多 launch 的较重评测——支持"worker 资源/池化在
    较重评测下进入死锁"的负载相关假说;
 5. T28 榜首 HelloWorldTJU 11.499x 为近期 8/8 达标,说明该题型在
    某 worker/时间窗上可正常评测,请求按提交时间对齐其他队伍的
    成功记录以定位健康窗口/pod。

 核心请求不变:对 sub 6223(T28 e8,七芯全过、华为 22.76x)与
 sub 6220(T27 e7,七芯全过、华为 0.515x)在健康 worker 上免费
 rerun 昆仑芯;或返还对应的基础设施错误消耗额度。
