/* zcode-workflow
description: FlagOS 算子赛冲榜闭环 v2：刷新榜单→逐芯差距排序→逐题调研瓶颈（chip-rulesets.md
  证据底座+已上膛跳过回填）→统一分诊（逐芯算术）→开发+codex-review 独立评审（卡住走
  codex-ask）→远端回执+确定性验签上膛→额度前缀发射（uncertain/sending/stale
  即停）→中文冲榜报告。参数：researchLimit/maxBuilds/maxSubmits/reserveQuota/dryRun。
whenToUse: 需要对 FlagOS 竞赛批次跑一整轮"刷新榜单到发射候选"的闭环时；额度不足自动只上膛不发射，已上膛未发射的题会被跳过并回填其他题。
args:
  dryRun:
    type: boolean
    description: true 时只开发+上膛，不做平台提交
    default: false
  maxBuilds:
    type: number
    description: 最多开发的候选数
    default: 3
  maxSubmits:
    type: number
    description: 最多发射数（额外受剩余额度-保留额约束）
    default: 3
  reserveQuota:
    type: number
    description: 发射前保留的额度（给回执驱动的修复留弹药）
    default: 1
  researchLimit:
    type: number
    description: 进入调研的题目数（多研 2 题做跳过回填）
    default: 4
  maxRounds:
    type: number
    default: 2
    description: 单次 run 自动连打的轮数上限（每轮重刷榜单；额度触底可睡等到下一发射点再继续）
  waitOnQuota:
    type: boolean
    default: true
    description: 额度触底时睡等到下一个发射点继续下一轮（短片循环重读时钟，恢复后立即续跑）
  fireHour:
    type: number
    default: 4
    description: 每日发射点（本地 +08 时区小时数，默认 4 点避开 00:00 提交高峰）
*/

// FlagOS 冲榜循环：刷新榜单 -> 差距分析 -> 逐题调研 -> 分诊 -> 开发+独立评审
// (codex-review) -> 上膛 -> 额度门控发射 (uncertain 即停) -> 汇总


interface GapRow {
  task: number;
  operator: string;
  ourRank: number;
  ourBest: number;
  topTeam: string;
  topBest: number;
  gapPct: number;
}

interface Candidate {
  task: number;
  operator: string;
  /** structural=kernel/launch 结构, vendor=单芯 vendor, param=参数档位 */
  axis: "structural" | "vendor" | "param";
  /** 一句话结构假设 */
  hypothesis: string;
  /** 支持证据（逐芯读数/规则集/上游形态，含来源路径） */
  evidence: string;
  /** 要改的文件（仓库相对路径） */
  files: string[];
  /** 预计均值增量 */
  expectedAvgGain: number;
  /** 主要风险与预注册门 */
  risk: string;
}

interface ReviewVerdict {
  approved: boolean;
  mustFix: string[];
  summary: string;
}

interface Armed {
  task: number;
  operator: string;
  stage: string;
  commit: string;
  zipPath: string;
  receiptPath: string;
  gate: string;
}

interface SubmitOutcome {
  task: number;
  operator: string;
  attempted: boolean;
  submissionId: string;
  state: string;
  note: string;
}

interface ClimbFinding {
  task: number;
  operator: string;
  what: string;
  evidence: string;
  status: "verified" | "unconfirmed";
  severity: "low" | "medium" | "high";
}

interface WorkflowReport {
  conclusion: string;
  findings: ClimbFinding[];
  verified: string[];
  notCovered: string[];
}

const RACE = "782kzq4m";
const ACCOUNT = "15600308080";
const TEAM = "SoulCoder";
const CLI = ".agents/skills/flagos-operator-race/scripts/platform_cli.py";

const researchLimit = Math.max(1, Math.min(8, Number(args.researchLimit) || 4));
const maxBuilds = Math.max(1, Math.min(6, Number(args.maxBuilds) || 3));
const maxSubmits = Math.max(1, Math.min(6, Number(args.maxSubmits) || 3));
const reserveQuota = Math.max(0, Math.min(10, Number(args.reserveQuota) || 1));
const dryRun = args.dryRun === true;
const maxRounds = Math.max(1, Math.min(5, Number(args.maxRounds) || 2));
const waitOnQuota = args.waitOnQuota !== false;
const fireHour = Math.max(0, Math.min(23, Number(args.fireHour) || 4));
// 窗口关闭时刻（题面 2026-09-24 19:59:59 +08）
const WINDOW_END_S = Math.floor(Date.parse("2026-09-24T19:59:59+08:00") / 1000);

async function nowEpoch(): Promise<number> {
  const t = await world.run("date", ["+%s"]);
  return Number(t.stdout.trim()) || 0;
}

// 短片睡等：每 tick ≤300s 重读时钟——resume/重放后已过目标点则立即返回
async function sleepUntil(targetS: number): Promise<void> {
  for (let guard = 0; guard < 1200; guard++) {
    const now = await nowEpoch();
    if (now <= 0 || now >= targetS) return;
    const slice = Math.min(300, Math.max(1, targetS - now));
    await world.run("sleep", [String(slice)]);
  }
}

// 下一个本地(+08) fireHour 的 epoch 秒
function nextFirePoint(nowS: number): number {
  const dayStart = Math.floor((nowS + 8 * 3600) / 86400) * 86400 - 8 * 3600;
  let t = dayStart + fireHour * 3600;
  if (t <= nowS + 600) t += 86400;
  return t;
}

artifact.table("gaps", {
  title: "与榜首差距（实时）",
  key: "task",
  columns: [
    { field: "task", label: "题号" },
    { field: "operator", label: "算子" },
    { field: "ourRank", label: "我方名次" },
    { field: "ourBest", label: "我方最佳" },
    { field: "topTeam", label: "榜首" },
    { field: "gapPct", label: "差距%" },
  ],
});
artifact.board("cands", {
  title: "候选看板",
  key: "name",
  status: "stage",
  columns: ["调研", "开发评审", "已上膛", "已发射"],
  cardTitle: "operator",
  detail: [{ field: "hypothesis", label: "假设" }],
});

// 轮间共享：分诊员跨轮一致（EV 尺度统一）；findings/submissions 跨轮累积
const triage = agent("分诊员", "你在 FlagOS 冲榜循环里给优化候选排序：期望均值增量×成功率优先，结构性>vendor>参数；同一题只留最优一个。用中文给出理由。");
const findings: ClimbFinding[] = [];
const submissions: SubmitOutcome[] = [];
let stopNote = "";
for (let round = 1; round <= maxRounds && !stopNote; round++) {
  const roundNow = await nowEpoch();
  if (roundNow > 0 && roundNow >= WINDOW_END_S) {
    stopNote = "提交窗口已关闭（09-24 19:59:59 +08），轮次链终止";
    break;
  }
  log(`—— 第 ${round}/${maxRounds} 轮（${roundNow > 0 ? "实测时钟" : "时钟不可读"}）——`);

phase("刷新榜单与题目快照");
const pull = await world.run("python3", [
  "tools/pull_race_intel.py",
  "--race",
  RACE,
  "--batch",
  "6",
  "--out",
  "docs/competition/data/leaderboard-snapshots/climb-loop.json",
], { timeoutMs: 300_000 });
if (pull.exitCode !== 0) {
  stopNote = "榜单拉取失败，轮次链终止：" + pull.stderr.slice(0, 200);
  break;
}

phase("对比与榜首差距并排序");
const snap = JSON.parse(await files.read("docs/competition/data/leaderboard-snapshots/climb-loop.json"));
const rows: GapRow[] = [];
for (const t of snap.tasks as Array<Record<string, unknown>>) {
  const rank = t.my_rank as number | null;
  const ours = (t.my_best_speedup as number) || 0;
  const best = (t.current_best_speedup as number) || 0;
  if (rank === null || rank === 1 || ours <= 0 || best <= 0) continue;
  rows.push({
    task: t.task_no as number,
    operator: String(t.operator),
    ourRank: rank,
    ourBest: ours,
    topTeam: String(t.current_leader_team_name),
    topBest: best,
    gapPct: Math.round((best / ours - 1) * 1000) / 10,
  });
}
rows.sort((a, b) => a.gapPct - b.gapPct);
// 多研 2 题：研究员跳过已占用题后自然回填，避免整轮空转
const targets = rows.slice(0, researchLimit + 2);
for (const r of rows) report(r, "gaps");
log(`距榜首最近的 ${targets.length} 题进入攻坚：${targets.map((t) => "T" + t.task).join("、")}`);

phase("逐题调研瓶颈与结构方案");
const proposals = await Promise.all(
  targets.map(async (row) => {
    const cands = await agent(`轮${round}-研究员-T${row.task}`, {
      system:
        "你是 FlagOS 算子赛的瓶颈研究员：只读调研，不改任何文件。结论必须给出来源路径。" +
        "若证据不足以支撑任何假设，如实说并返回空列表。若指令与证据矛盾，升级提问而不是硬答。",
    }).ask<Candidate[]>(
      `调研 T${row.task} ${row.operator}（我方 #${row.ourRank} ${row.ourBest} vs 榜首 ${row.topTeam} ${row.topBest}，差 ${row.gapPct}%）。` +
      `必读：docs/competition/experiments/${row.operator}.md（账本，含逐芯读数与已证伪形态）、` +
      `docs/competition/tasks/batch-6/ 下该题题面、src/flaggems_sglang/ops/${row.operator}.py 与` +
      ` src/flaggems_sglang/runtime/backend/*/ops/${row.operator}.py（现状）、` +
      `docs/competition/chip-rulesets.md（跨芯规则集索引：燧原/沐曦/昇腾/平台协议，含反例——硬约束以它为准不要凭记忆）` +
      `与 docs/competition/session-mining-retrospective.md。逐芯情报用 platform_cli status 按题查询（只读不耗额度）。` +
      `可用 WebSearch/WebFetch 查上游（sglang/vllm/FlagGems/FlagTree），可用 gh api 读 GitHub 源码；网络不可用就只靠仓内证据。` +
      `产出 1-2 个候选，优先结构性假设（launch 结构/访存形态/跨步映射），排除账本已证伪形态。` + `先读 docs/competition/experiments/README.md 候选队列：已有上膛未发射候选的题直接跳过（返回空列表并说明原因），避免重复开发。` + `反作弊红线（平台代码安全扫描会拒收，违反即整发作废）：不用 try/except 或设备判断 fallback 到 PyTorch；核心计算必须全 Triton；禁止模块级全局可变容器（dict/set 缓存会被扫描拒收，T77 s0 实例）。用中文。`,
    );
    for (const c of cands) {
      report({ name: `T${c.task}-${c.axis}`, operator: c.operator, stage: "调研", hypothesis: c.hypothesis }, "cands");
    }
    return cands;
  }),
);
const allCandidates = proposals.flat().filter((c) => c && c.files && c.files.length > 0 && c.hypothesis);
if (allCandidates.length === 0) {
  stopNote = `第 ${round} 轮调研未产出可用候选（已上膛题被跳过或证据不足），轮次链终止`;
  break;
}

phase("统一分诊候选优先级");
const ranked = await triage.ask<Candidate[]>(
  `按 EV 排序并去重（同题最多留 1 个）。算术：单芯提升÷8 才是均值贡献；先核该芯是否已过/贴近 0.1 有效性门槛；区分抢榜收益与验证假设的信息收益，不给伪精确分数。返回排序后的完整候选列表：\n${JSON.stringify(allCandidates)}`,
);
const picked = ranked.filter((c) => c && c.files && c.hypothesis).slice(0, maxBuilds);
log(`分诊选出 ${picked.length} 个候选：${picked.map((c) => "T" + c.task + "/" + c.axis).join("、")}`);

const armed: Armed[] = [];

phase("并行开发各候选并过独立评审");
// 候选间互不相干（不同文件/账本）：开发+评审并行跑；
// 上膛含远端 release（远端纪律串行）放到汇合后的独立阶段
interface DevOutcome {
  /** 开发员自己创建的 commit 全长哈希（它提交后立即 git rev-parse HEAD 记下并返回；并行分支共享工作树，全局 HEAD 可能是别的候选的 commit，禁止用它绑定评审） */
  commit: string;
  /** 本轮做了什么/卡在哪，一句话 */
  note: string;
}
interface Reviewed {
  cand: Candidate;
  approved: boolean;
  commit: string;
  summary: string;
}
const allReviewed: Reviewed[] = await Promise.all(
    picked.map(async (cand): Promise<Reviewed> => {
      const dev = agent(`轮${round}-开发员-T${cand.task}`, {
        system:
          "你是 FlagOS 冲榜候选开发员，全程遵守 .agents/skills/flagos-operator-race/SKILL.md 的闭环纪律。" +
          "只改候选列出的文件与对应测试/账本；每步工具命令真实执行并引用输出；评审/外部主张必须逐条对源码核实后才采信（先例：声称改 2 条路径实为 3 条），核实后逐条修复才能进入下一步；" +
          "若门禁不可能通过（环境/权限/额度），升级说明而不是绕过。",
      });
      const reviewer = agent(`轮${round}-评审员-T${cand.task}`, {
        system:
          "你是独立评审员：没看过开发过程，只对 diff 与代码负责。运行 codex-review 脚本并叠加自己的判读；" +
          "要求找会出错的点而不是表态同意；不得修改任何文件；P1/P2 必须列入 mustFix。",
      });
      let feedback = "无（首轮）";
      let verdict: ReviewVerdict | null = null;
      let headCommit = "";
      for (let round = 1; round <= 2; round++) {
        const outcome = await dev.ask<DevOutcome>(
          `为 T${cand.task} ${cand.operator} 实现该候选（第 ${round} 轮）。假设：${cand.hypothesis}。证据：${cand.evidence}。` +
          `目标文件：${cand.files.join("、")}。预注册门：${cand.risk}。上轮评审意见：${feedback}。` +
          `要求：1) 写实现与测试（沿用 tests/test_${cand.operator}.py 的矩阵，新语义须加回归）；` +
          `2) python3 -m py_compile 全部触碰文件通过；3) git add 这些明确路径并 commit（--only 隔离）。` +
          `开工前先 git status --short：目标文件若有归属不明的既有改动，升级询问而不是覆盖。` +
          `改账本前先 git log -1 -- <账本路径> 并重读最新字节再编辑（多会话并行防线）。` +
          `新增回归测试必须同时列入该测试模块的 RELEASE_REQUIRED_TESTS。` +
          `最后返回你刚创建的 commit 全长哈希（git rev-parse HEAD，在你 commit 之后立刻读）。并行分支共享工作树，全局 HEAD 随时可能变成别的候选的 commit——评审只认你返回的这个哈希。`,
        );
        if (!outcome || !outcome.commit) {
          feedback = "开发员未返回自己的 commit 哈希，无法绑定评审";
          continue;
        }
        const compile = await world.run("python3", ["-m", "py_compile", ...cand.files], { timeoutMs: 60_000 });
        if (compile.exitCode !== 0) {
          feedback = "py_compile 失败：" + compile.stderr.slice(0, 400);
          continue;
        }
        headCommit = outcome.commit;
        verdict = await reviewer.ask<ReviewVerdict>(
          `评审 commit ${headCommit}（git show ${headCommit}）。上下文：候选假设=${cand.hypothesis}；契约=docs/competition/tasks/batch-6/ 下该题题面。` +
          `先跑：bash /Users/bytedance/.agents/skills/codex-review/scripts/run-review.sh --commit ${headCommit} --reasoning-effort high，` +
          `再自己读 diff 与周边调用方，合并成一份判决。用中文。`,
        );
        if (verdict && verdict.approved && verdict.mustFix.length === 0) break;
        feedback = verdict ? verdict.mustFix.join("；") : "评审无返回";
      }
      if (verdict && verdict.approved && verdict.mustFix.length === 0) {
        return { cand, approved: true, commit: headCommit, summary: verdict.summary };
      }
      const consult = await agent(`轮${round}-咨询员-T${cand.task}`, {
        system: "你是卡点咨询员：运行 codex-ask 脚本获取第二意见，核对后给出可执行的新方向。不改文件。",
      }).ask(
        `T${cand.task} 两轮评审未过（意见：${feedback}）。运行 ` +
        `/Users/bytedance/.agents/skills/codex-ask/scripts/ask-codex.sh --question-file <你写的问题文件> ` +
        `--context ${cand.files[0]} --reasoning-effort high，把候选假设、diff、评审意见、逐芯证据写进问题。` +
        `总结新方向入账本候选备注即可，本轮不再开发。`,
      );
      report({ name: `T${cand.task}-${cand.axis}`, operator: cand.operator, stage: "开发评审", hypothesis: cand.hypothesis }, "cands");
      return { cand, approved: false, commit: headCommit, summary: String(consult).slice(0, 300) };
    }),
);
const reviewed = allReviewed.filter((r) => r.approved);
for (const r of allReviewed) {
  if (!r.approved) {
    findings.push({
      task: r.cand.task, operator: r.cand.operator,
      what: "候选两轮评审未过，codex-ask 咨询已记录新方向",
      evidence: r.summary,
      status: "unconfirmed", severity: "medium",
    });
  }
}
for (const r of reviewed) {
  findings.push({
    task: r.cand.task, operator: r.cand.operator,
    what: "候选过两轮独立评审",
    evidence: `commit ${r.commit}；${r.summary}`,
    status: "verified", severity: "high",
  });
  report({ name: `T${r.cand.task}-${r.cand.axis}`, operator: r.cand.operator, stage: "开发评审", hypothesis: r.cand.hypothesis }, "cands");
}

phase("串行上膛：远端回执与不可变 ZIP");
// 远端 release 纪律串行（gpu 主机资源独占），按分诊序逐个上膛
for (const r of reviewed) {
  const cand = r.cand;
  const arm = await agent(`轮${round}-上膛员-T${cand.task}`, {
    system:
      "你负责把已过审的候选走完上膛闭环：远端 release 回执、不可变 ZIP、账本五元组。逐条真实执行并保留输出；" +
      "任何一步失败如实报告卡在哪，不伪造产物路径；额度类失败直接如实记录。",
  }).ask<Armed>(
    `为 T${cand.task} ${cand.operator} 完成上膛（工作目录 /Users/bytedance/ccc/flagos，远端 ssh 别名 gpu，` +
    `远端 python=/home/kevin/notebook/.venv/bin/python，按 SKILL.md 协议；注意与其他候选并行开发时只用 git --only 提交自己的路径）：` +
    `1) verify_release.py prepare ${cand.operator} --source-commit HEAD --verification-commit HEAD ` +
    `--proxy-vendor <该题现有全部 vendor> --directory /tmp/wf-${cand.operator}-release；` +
    `2) scp -q -r 该目录到 gpu:/tmp/wf-${cand.operator}-release（注意：远端若已有旧 verification.json 会拒覆盖，先 ssh 清目录）；` +
    `3) ssh gpu 'cd /tmp/wf-${cand.operator}-release && timeout 900 /home/kevin/notebook/.venv/bin/python ` +
    `.agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/wf-${cand.operator}-release'；exit 0 才继续；` +
    `4) scp 回执到 artifacts/competition/day5prep-20260921/${cand.operator}-wf/；` +
    `5) stage 编号从账本 CURRENT 块 candidate_stage 递增取下一个（保持记账连续；注意平台元组去重键是 zip_sha256 而非 stage——新候选必须产生新 ZIP 字节，同字节重掷需载体 commit），` +" +
    `build_submission.py ${cand.operator} --stage <该编号> --commit HEAD；` +" +
    `6) 账本 docs/competition/experiments/${cand.operator}.md CURRENT 块更新 + 追加段（五元组+预注册门），` +" +
    `五元组逐成员列出 ZIP 名单并与 zipfile 实际成员核对一致（防打包器夹带）；同步刷新 README 候选队列行。` +" +
    `7) python tools/gen_experiment_index.py + git commit --only 明确路径；push 前核对 @{upstream}..HEAD ` +" +
    `全部待推 commit，含无关既有 commit 则只本地 commit 不 push 并说明。` +" +
    `返回的 commit 字段必须等于回执的 verification_commit（否则发射 preflight 会拒）；prepare 前后若 HEAD 被其他提交改变，重跑 prepare 绑定新 HEAD。用中文。`,
  );
  if (arm && arm.zipPath) {
    const zipOk = await world.run("unzip", ["-t", arm.zipPath], { timeoutMs: 60_000 });
    const receiptOk = await world.run("python3", ["-c", "import sys,os;sys.exit(0 if os.path.exists(sys.argv[1]) and os.path.getsize(sys.argv[1])>0 else 1)", arm.receiptPath], { timeoutMs: 30_000 });
    if (zipOk.exitCode !== 0 || receiptOk.exitCode !== 0) {
      findings.push({
        task: cand.task, operator: cand.operator,
        what: "上膛产物确定性验签未过（unzip -t 或回执存在性失败）",
        evidence: `zip=${arm.zipPath} exit=${zipOk.exitCode}; receipt=${arm.receiptPath} exit=${receiptOk.exitCode}`,
        status: "unconfirmed", severity: "high",
      });
      continue;
    }
    armed.push(arm);
    findings.push({
      task: cand.task, operator: cand.operator,
      what: `候选上膛成功（${arm.stage}，门：${arm.gate}）`,
      evidence: `ZIP ${arm.zipPath}；回执 ${arm.receiptPath}；commit ${arm.commit}`,
      status: "verified", severity: "high",
    });
    report({ name: `T${cand.task}-${cand.axis}`, operator: cand.operator, stage: "已上膛", hypothesis: cand.hypothesis }, "cands");
  } else {
    findings.push({
      task: cand.task, operator: cand.operator,
      what: "上膛未完成（远端验证或打包失败）",
      evidence: String(arm).slice(0, 300),
      status: "unconfirmed", severity: "medium",
    });
  }
}

phase("平台发射（额度门控，单次尝试）");
if (dryRun) {
  log("dryRun=true：跳过平台提交");
} else if (armed.length === 0) {
  log("无可发射候选");
} else {
  const quotaOut = await world.run("python3", [CLI, "status", "--race", RACE, "--batch", "6", "--task", String(armed[0].task), "--operator", armed[0].operator], { timeoutMs: 120_000 });
  const quotaMatch = /"remaining":\s*(\d+)/.exec(quotaOut.stdout);
  const remaining = quotaMatch ? Number(quotaMatch[1]) : 0;
  const budget = Math.max(0, Math.min(remaining - reserveQuota, maxSubmits));
  const fireList = armed.slice(0, budget);
  if (quotaOut.exitCode !== 0 || fireList.length === 0) {
    log(`额度不足或预算为零（剩余 ${remaining}，保留 ${reserveQuota}，上限 ${maxSubmits}）——只上膛不发射`);
    for (const a of armed) submissions.push({ task: a.task, operator: a.operator, attempted: false, submissionId: "", state: "skipped", note: `额度剩余 ${remaining}（保留 ${reserveQuota}）` });
  } else {
    for (const a of armed.slice(fireList.length)) submissions.push({ task: a.task, operator: a.operator, attempted: false, submissionId: "", state: "skipped", note: "超出本轮发射预算前缀" });
    const launcher = agent(`轮${round}-发射员`, {
      system:
        "你是平台发射员：对每个已上膛候选执行且只执行一次 preflight+submit（间隔≥125 秒）。" +
        "preflight 参数照 CLI 约定（--account " + ACCOUNT + " --team " + TEAM + "），members=ZIP 全部成员；" +
        "遇 interval 按 status 返回的 minimum_interval_seconds 剩余秒数等待后重试 preflight（不要盲睡）；submit 成功后用返回的 watch_command 轮询终态（它绑定 file_url_sha256 与 " +
        "after-epoch，防止把同题旧终态误读成本次结果）；失败详情从 status 输出的 raw_result 字段读取。" +
        "任一 uncertain/sending 状态立即停止后续并如实上报，绝不重试（uncertain 元组会被 CLI 永久封锁；" +
        "同字节重掷需新载体 commit，只在用户明示时做）。",
    });
    for (const a of fireList) {
      const out = await launcher.ask<SubmitOutcome>(
        `发射 T${a.task} ${a.operator}：ZIP=${a.zipPath}，回执=${a.receiptPath}，commit=${a.commit}。` +
        `先 status 查逐题最新记录，再 preflight（--test-sha256 用 git show ${a.commit}:tests/test_${a.operator}.py 的 sha256），` +
        `拿 nonce 后 submit --confirm。轮询到终态，返回逐芯与均值。`,
      );
      const outcome: SubmitOutcome = out || {
        task: a.task, operator: a.operator, attempted: true,
        submissionId: "", state: "unknown", note: "发射员无返回",
      };
      submissions.push(outcome);
      if (["uncertain", "sending", "stale_after_upload", "unknown"].includes(outcome.state)) {
        stopNote = `第 ${round} 轮发射出现 ${outcome.state}，按纪律终止后续候选与后续轮次`;
        break;
      }
    }
  }
}

  // 轮末：额度触底且允许睡等且窗口未关 → 睡到下一个发射点继续下一轮
  if (!stopNote && round < maxRounds && waitOnQuota && !dryRun) {
    const q = await world.run("python3", [CLI, "status", "--race", RACE, "--batch", "6", "--task", "80", "--operator", "fixup_zero_kv"], { timeoutMs: 120_000 });
    const qm = /"remaining":\s*(\d+)/.exec(q.stdout);
    const qRemaining = qm ? Number(qm[1]) : 0;
    const nowS = await nowEpoch();
    if (qRemaining <= reserveQuota && nowS > 0 && nowS < WINDOW_END_S) {
      const target = nextFirePoint(nowS);
      log(`额度触底（剩余 ${qRemaining}），睡等到发射点 ${target}（约 ${Math.round((target - nowS) / 60)} 分钟）后继续第 ${round + 1} 轮`);
      await sleepUntil(target);
    }
  }
} // 轮次结束：回到循环顶重刷榜单；停止条件经 stopNote/break 跳出

phase("汇总产出冲榜报告");
const digest = await agent("报告员", "你把冲榜循环的结果写成给队友看的中文报告：结论先行，逐题列候选/门/终态/下一杆。不改文件。")
  .ask<{ summary: string; lines: string[] }>(
    `汇总：差距表=${JSON.stringify(rows.slice(0, 6))}；候选=${JSON.stringify(picked.map((c) => ({ task: c.task, axis: c.axis, hypothesis: c.hypothesis })))}；` +
    `上膛=${JSON.stringify(armed.map((a) => ({ task: a.task, stage: a.stage, gate: a.gate })))}；` +
    `发射=${JSON.stringify(submissions)}。轮次链终止原因：${stopNote || "正常完成 maxRounds 轮"}。`,
  );
await artifact.markdown(
  "climb-report",
  [
    "# FlagOS 冲榜循环报告",
    "",
    digest.summary,
    "",
    ...digest.lines,
    "",
    "## 与榜首差距（前 6）",
    ...rows.slice(0, 6).map((r) => `- T${r.task} ${r.operator}：#${r.ourRank} ${r.ourBest} vs ${r.topTeam} ${r.topBest}（+${r.gapPct}%）`),
    "",
    "## 发射结果",
    ...submissions.map((s) => `- T${s.task} ${s.operator}：${s.state}${s.submissionId ? "（sub " + s.submissionId + "）" : ""}——${s.note}`),
  ].join("\n"),
  { title: "冲榜循环报告", description: "差距、候选、发射与下一杆", primary: true },
);

const result: WorkflowReport = {
  conclusion: digest.summary,
  findings,
  verified: [
    "榜单刷新与差距解析由 pull_race_intel 实测驱动",
    ...armed.map((a) => `T${a.task} 远端 release 回执 + 不可变 ZIP（${a.zipPath}）`),
  ],
  notCovered: [
    "目标芯（燧原/昇腾等）上的实际性能——平台评测才裁决",
    ...(dryRun ? ["平台提交（dryRun）"] : []),
    ...(submissions.some((s) => !s.attempted) ? ["部分候选因额度不足未发射"] : []),
  ],
};
return result;
