# 第六批收官日作战记录（2026-09-24，窗口 19:59:59 关闭）

> 逐芯榜单快照 `data/batch6-leaderboard-20260924.json`（sha256
> `4add949b2a09f63fdcf35c2d00710e0de5505c89d21986d4ff9a20e134bd5f27`，
> observed 07:0x+08）。当日额度 30 发全局共享；昨夜并行会话已耗 6
> （T77 e7 1 发 + T80 重掷 5 发），本会话从 24/30 起步。

## 开局靶子分析（07:00-07:30）

- 逐芯可达性重算（彩票判据：榜首某芯 > 次优同芯 20x）：
  - **彩票榜首**（不可追）：T78（燧原 15268 vs 5.45=2800x）、
    T89（KLANG2026 全板 8k-36k 慢窗）、T85（华为 179 vs 8.39=21x）、
    T91（燧原 10.51 vs 1.62=6.5x，超 3.176 需 ≥10.29）。
  - **结构性均匀差距**：T84（全芯 1.6-2.2x，两队伍逐芯健康）、
    T77（大芯 1.5x + 燧原/华为 4.2x）、T92（b/华为/昆仑/沐曦
    1.5-2.7x）。
  - **多芯磨轴**：T80（rank3，四条 vendor 轴）、T86/T90/T87 薄差距。
- GitHub 扫描：flagos-ai/FlagGems-sglang 的 c2flowDS（=c2flow 队）PR
  只覆盖已收官批次（最新 #107 log_scaling_tau，09-23）；第六批 PR 要等
  窗口关闭后才会出现——赶不上，泄露通道今日无效。
- codex-ask（T84 三候选）：自旋 co-resident 死锁风险最高（普通 Triton
  launch 无 grid 驻留保证）；修正版 2-launch 零初始化可行但预估仅
  492→600-640（非 top1 量级）——T84 降级为次优先。

## 主攻线 1：T77 compute_position（TB 1100.74 → 1266.24，rank 10→8）

| 弹 | 结构 | 平台判决 |
| --- | --- | --- |
| e8 (20738) | 单 launch selfscan 融合（batch≤2048） | 7/8：昆仑 `tl.where(lanes<pid)` 标量广播数值错；**B +33%/A +21%/天数 +9.5% 验证方向** |
| e9 (20739) | +昆仑 vendor（e6 双 launch 字节） | **8/8 1233.71 新 TB**（昆仑 105.22 精确回水位） |
| e10 (20740) | tiled 2D grid 平铺（batch≤128） | valid 1192.73 判负：华为 139.6 持平（4.3x 缺口≠循环/CTA 形态）、天数 -11% |
| e11 (20741) | enflame 去 host-sync 探针 v1 | 7/8：int64 向量位运算撞 make_gcuir 编译墙 |
| e11r (20743) | int32-only 双布局分支 + 加法器进位公式 | **8/8 1266.24 新 TB：燧原 207.8（+80%，~100us/call D2H 探针是主因）** |
| e13 (20747) | 燧原 scan 融合进 fill（3→2 launch） | 燧原 156.8（-25%）判负：≤12 CTA 自扫成本 > 省 launch |
| e14 (20749) | ascend 双 int32 字 store（BLOCK 512） | valid 1246.80 判负：华为 98.6（-32%）——宽 store 假说证伪 |

- 关键机制发现：**vendor wrapper 的 host-sync 探针**（每 call
  `arange(4).view(int32)[:4].tolist()` ≈100us D2H）主导燧原 op 时间；
  in-kernel 读探针内容做 packed 判定后 +80%。全仓扫描确认第六批其余
  题无同类探针（其余 `.item()` 都在已收官批次）。
- 本地 release 矩阵两次拦截真 bug（进位公式 2^31 误判、GCU 编译墙前的
  int64 位运算），各省一发额度。

## 主攻线 2：T80 fixup_zero_kv（TB 569.13 → 581.23，rank 3 守住）

| 弹 | 结构 | 平台判决 |
| --- | --- | --- |
| e25 (20753) | 昆仑 vendor 去 per-call clone（+零步长视图守卫） | **8/8 570.31 新 TB**：昆仑 14.2→16.5（+16%，clone 只是小头） |
| e26 (20756) | 燧原解钉 warps + 沐曦 ot-cap | 550.56 判负：燧原 68.1（-38%，**pin 承重**）、沐曦 -10%；双回退 |
| e27 (20760) | 昆仑 BLOCK_V 16384 | 7/8：int64 大向量 XPU 数值崩（39% garbage） |
| e28 (20765) | 昆仑 BLOCK_V 4096 | **8/8 581.23 新 TB**（昆仑 16.9 微增 + 华为/沐曦水位上抬） |
| e29 (20776) | ascend vendor 1D-flat int32 重写 | 评测中 |

- e24 基线先从 codex/t80-top1 同步（五成员与 e24 ZIP 逐字节核对）。

## 主攻线 3：T92 unpad_draft_extend_output（TB 341.31 → 344.50，rank 6 守住）

| 弹 | 结构 | 平台判决 |
| --- | --- | --- |
| e24 (20768) | ascend vendor 向量管线 int64→int32（域守卫+宽回退核） | **8/8 344.50 新 TB**：华为 464.5（+4.6%，i64 向量只是小头）、昆仑 80.0（水位 +142%） |

## 横向知识（今日新增，跨题可迁移）

1. **host-sync 探针审计**：vendor wrapper 的 `.tolist()/.item()/.cpu()`
   在 wrapper-inclusive 计时下主导小 op——先扫这个再调内核。
2. **GCU warp pin 在宽 store 形态承重**（解钉 -38%）；与既有"不钉更好"
   实证并存——按 op store 形态区分，不可外推。
3. **XPU 数值雷**：`tl.where(lanes < scalar_pid)` 标量广播错算
   （T77 e8）；16384 宽 int64 向量 store 写坏（T80 e27）。
4. **Ascend 向量管线**：int64 向量 ALU 只值 +4.6%（T92）；2D i64 广播
   store 是最差形态（T80 e29 重写验证中）；UB 预算 1572864 bits 对
   BLOCK×双 store 敏感。
5. **水位论**：昆仑/沐曦/天数读数 ±10-35% 波动（T92 e24 昆仑水位
   +142% 白捡）；同字节重掷最多 2 次的纪律不变。
