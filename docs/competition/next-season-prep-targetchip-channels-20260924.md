# 下一赛季准备:目标芯免额度执行通道调研(2026-09-24)

> 来源:climb-loop 复盘第三条根因("为目标芯建立免额度执行通道")的
> 只读调研落盘。本调研不消耗平台额度、不启动任何生成/设备任务
> (KernelGen 调用纪律:纯审计只做服务发现)。

## 结论速览

| 通道 | 昆仑 XPU | 华为 Ascend | 燧原 GCU | 判定 |
| --- | --- | --- | --- | --- |
| KernelGen MCP(实测定存活) | ✅(设备表含 kunlun) | ✅(huawei) | ⚠️(设备表未列 enflame) | **昆仑/华为的唯一免额度通道** |
| gpu 容器(gpu-container-setup-flagos) | ❌(skill 明示不覆盖昆仑) | ✅(昇腾镜像族) | ❌(vendor hub 无燧原) | 华为可自建主机补强 |
| 现有 `gpu` SSH 主机 | ❌(NVIDIA) | ❌ | ❌ | 仅代理 |

实测记录(2026-09-24):`kernelgen_mcp.py list` 返回 4 工具
(generate/optimize/specialize/autotune),端点 `https://kernelgen.flagos.io/sse`
连通正常;2026-09-06 的 tools/list 设备描述含
nvidia/huawei/haiguang/tianshu/muxi/moore/sunrise/**kunlun**/amd,
具体排队/在线/执行能力以每次响应为准(skillhub-tools.md:36)。
燧原不在列表 → 燧原的 2.1x 结构缺口(T80 109vs235)暂无免额度通道,
需平台 raw_result 取证或厂商主机。

## 对 T84 昆仑 10x 缺口的具体启动协议(下一赛季开题即用)

1. 用 KernelGen `generate_kernel`(device=kunlun)携带 T84 契约 +
   e18r 两堵编译墙的完整报错(axis-0 归约禁令、tt.addptr 同编码)
   + s0 标量 vendor 源码,让其在真实 XPU 上迭代散射探针形态——
   每次迭代免额度;
2. 编译通过后再用一次平台发射银行化(预注册门沿用 e18 的
   kunlun ≥2.0 保留 / ≥5 轴确认);
3. 同协议适用于其它 XPU 假设(hash_topk 的标量串行限制等)。

## 优先级排序(按本季缺口×通道可得性)

1. **昆仑**(T84 10x + T78 昆仑环境族 + hash_topk XPU 限制)→
   KernelGen,通道现成;
2. **华为**(T80 564vs984、T77 4.2x、T92 435vs665)→ KernelGen +
   昇腾容器双通道,证据面最全;
3. **燧原**(T80 109vs235、T86 1.37vs2.7)→ 无免额度通道,只能靠
   平台 raw_result 逐发取证或申请厂商主机——列为通道建设缺口。
