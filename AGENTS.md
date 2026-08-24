# Project agent instructions

- 需要 GPU/runtime 的开发验证优先在远端后台运行，前台继续后续工作；只读查询、
  静态审计和现有产物验签不连接远端。
- 代码或竞赛账本改完后自动 commit、push；是否触发 CI 以目标分支和仓库规则为准。
- Lark/飞书相关操作使用对应 Lark CLI skill。
- 涉及 FlagOS 算子赛题调研、跨芯开发、ZIP 打包、平台提交、八芯评测或
  vendor 迭代时，先读取并遵循
  [`.agents/skills/flagos-operator-race/SKILL.md`](.agents/skills/flagos-operator-race/SKILL.md)。
- 用户已授予 FlagOS 竞赛候选持续自动提交权限：当前任务包含平台提交、完整闭环或
  继续既有竞赛闭环，且发布门禁、不可变 ZIP 验签和实时 preflight 全部通过时，立即
  执行 preflight 返回的一次性 submit 命令，无需再次询问。每个候选最多执行一次上传
  和正式提交；`sending`、`uncertain`、`stale_after_upload` 或已提交状态不得自动重试。
  用户明确暂不提交、请求只读或任一门禁不满足时不得提交。实际平台结果写入对应实验
  账本；只读状态查询不改文件。
