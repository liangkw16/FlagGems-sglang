# Project agent instructions

- 单元测试优先在远端后台运行，前台继续后续工作。
- 代码或竞赛账本改完后自动 commit、push，触发仓库既有流水线。
- Lark/飞书相关操作使用对应 Lark CLI skill。
- 涉及 FlagOS 算子赛题调研、跨芯开发、ZIP 打包、平台提交、八芯评测或
  vendor 迭代时，先读取并遵循
  [`.agents/skills/flagos-operator-race/SKILL.md`](.agents/skills/flagos-operator-race/SKILL.md)。
- 网页提交会消耗团队额度；只有用户针对明确 Task、ZIP 路径和哈希作出当次
  确认后才能提交。每次平台结果都写入对应实验账本。
