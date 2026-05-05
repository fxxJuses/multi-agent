# multi-agent — 从零手写 Agent 范式（学习仓库）

一个用于**学习 LLM Agent 设计范式**的最小实现仓库。所有 Agent 都是**手写循环 + 正则解析 + 自管 scratchpad**，**不依赖 LangChain / LlamaIndex / AutoGen 等任何 Agent 框架**，便于看清每一种范式的本质。

底层只用一个轻量的 LLM 包装类（`src/llm/llm.py`），通过 OpenAI 兼容协议同时支持：

- OpenAI（GPT 系列）
- 阿里云百炼 / DashScope（Qwen 系列）
- 火山引擎方舟 / Ark（豆包等）

---

## 一、你能学到什么

仓库内置 3 种主流 Agent 范式，都基于同一套 LLM 客户端 + 同一组工具，方便横向对比：

| 范式 | 一句话描述 | LLM 调用次数（典型） | 适合场景 | 论文 |
|------|-----------|----------------------|---------|------|
| **ReAct** | "想-做-观察"循环：模型自己决定下一步 | N 步 = N 次（边走边看） | 不确定步数、需边推理边修正 | Yao et al., 2022 |
| **ReWOO** | 先一次性出完整 Plan，再批量执行工具，最后 Solver 总结 | 2 次 LLM + N 次工具 | 步骤可静态规划、想省 token | Xu et al., 2023 |
| **Plan-and-Execute** | 先出高层 Plan → 每步内部用 ReAct 执行 → 可触发 Replan → Synthesizer 合成 | Plan 1 + 每步若干 + Replan 若干 + Synthesize 1 | 步骤多、可能需中途调整计划 | Wang et al., 2023 |

每种范式的实现都集中在**单个文件**里（约 130~480 行），便于通读：

- `src/agents/react_agent.py`
- `src/agents/rewoo_agent.py`
- `src/agents/plan_execute_agent.py`

---

## 二、项目结构

```
multi-agent/
├─ .env.example          # 三家 LLM 的环境变量模板
├─ .gitignore
├─ requirements.txt      # 仅 openai + python-dotenv
├─ docs/                 # 教程与章节文档
├─ example/              # 与文档配套、可独立运行的小脚本（非核心 Agent 代码）
│  └─ mcp-jsonrpc-concept-demo.py
└─ src/
   ├─ __init__.py
   ├─ main.py            # 统一 CLI 入口（python -m src.main --agent ...）
   ├─ agents/
   │  ├─ react_agent.py            # ReAct 范式
   │  ├─ rewoo_agent.py            # ReWOO 范式
   │  └─ plan_execute_agent.py     # Plan-and-Execute 范式（含 Replan）
   ├─ llm/
   │  ├─ llm.py                    # OpenAI 兼容 LLM 客户端封装（支持流式）
   │  └─ qwen_stream_demo.py       # 流式调用 demo（Qwen）
   ├─ mcp/
   │  └─ client.py                 # 最小 MCP stdio client（initialize / tools/list / tools/call）
   └─ tools/
      └─ basic_tools.py            # calculator + fake_search（演示用）
```

> 设计原则：**Agent 与范式实现只放在 `src/`**；`example/` 仅放与 `docs/` 配套的演示脚本。根目录保持干净（配置文件与文档）。

---

## 三、环境搭建（推荐 Anaconda）

> 已假设你装好了 Anaconda 并执行过 `conda init cmd.exe`（或对应 shell）。

```bash
# 1) 创建虚拟环境
conda create -n multi-agent python=3.11 -y

# 2) 激活
conda activate multi-agent

# 3) 安装依赖
pip install -r requirements.txt
```

依赖非常轻：

```
openai>=1.30.0
python-dotenv>=1.0.0
```

---

## 四、配置 API Key

复制 `.env.example` 为 `.env`，**任选一家**填入即可（程序会按 OpenAI → 百炼 → 方舟 的顺序自动识别）：

```bash
cp .env.example .env     # Linux / macOS / Git Bash
copy .env.example .env   # Windows CMD
```

### 选项 A：阿里云百炼（推荐国内开发者）

```ini
DASHSCOPE_API_KEY=sk-你的key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus
```

### 选项 B：OpenAI

```ini
OPENAI_API_KEY=sk-你的key
OPENAI_MODEL=gpt-4o-mini
# 如需走代理可设
# OPENAI_BASE_URL=https://...
```

### 选项 C：火山方舟

```ini
ARK_API_KEY=你的key
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=ep-xxxxxxxxxxxxxxxx
```

> 也可以用 `LLM_MODEL` 这个公共变量统一覆盖模型名。详见 `src/llm/llm.py` 中 `_resolve_*` 三个解析函数。

---

## 五、运行

**所有命令都需要在仓库根目录 `multi-agent/` 下执行**，并使用 `-m` 模块化运行（不要 `cd src` 进去执行）。

### 1. 统一入口（推荐）

```bash
python -m src.main --agent react        --verbose
python -m src.main --agent rewoo        --verbose
python -m src.main --agent plan_execute --verbose
```

可选参数：

```bash
python -m src.main --help
```

```
--agent {react,rewoo,plan_execute}   选择 Agent 范式
--task TASK                          自定义任务（默认是一个数学+比较的小任务）
--verbose                            打印每一步中间过程（非常推荐学习时打开）
```

### 2. 单独运行某个 Agent

```bash
python -m src.agents.react_agent
python -m src.agents.rewoo_agent
python -m src.agents.plan_execute_agent
```

环境变量 `REACT_VERBOSE=1` / `REWOO_VERBOSE=1` / `PLAN_EXECUTE_VERBOSE=1` 可开启详细日志。

### 3. 流式调用 Qwen 的小 demo

```bash
python -m src.llm.qwen_stream_demo
```

---

## 六、三种 Agent 的设计对比（学习重点）

### 6.1 ReAct（`react_agent.py`）

**思路**：让模型每一步只输出一段：

```
Thought: ...
Action: <tool_name>
Action Input: <input>
```

主循环拿到这段后用正则解析、调用工具、把 `Observation: ...` 拼到 scratchpad 里，再让模型生成下一步，直到出现：

```
Thought: ...
Final Answer: ...
```

**关键点**：
- 单一 LLM 角色，**边推理边行动**
- `_parse()` 用正则区分 `Final Answer` / `Action` / 解析失败三种分支
- 解析失败时把错误反馈成新的 Observation，让模型自我修复（这是 ReAct 鲁棒性的来源）

### 6.2 ReWOO（`rewoo_agent.py`）

**思路**：先用一次 LLM 输出完整计划：

```
Plan: 计算 (128 * 7) / 2
#E1 = calculator[(128 * 7) / 2]
Plan: 把 #E1 与 500 比较
#E2 = calculator[#E1 - 500]
```

主程序按顺序执行所有 `#E*` 步骤（**不再调用 LLM 推理**，只调用工具），再用一次 LLM 作为 Solver 综合答案。

**关键点**：
- **两次** LLM 调用 + N 次工具调用 → 比 ReAct 省 token
- `_substitute_vars()`：后面的工具输入可以引用 `#E1` 等先前结果（变量替换）
- `_make_plan()` 含 3 次自动修复：发现 plan 不合规就把错误回灌给 LLM 重写
- **代价**：Plan 一旦写错，后续无法根据观察自动调整（不如 ReAct 灵活）

### 6.3 Plan-and-Execute（`plan_execute_agent.py`）

**思路**：四个 LLM 角色协作：

1. **Planner**：先出一个高层数字编号的 plan
2. **Executor**：每个 step 内部跑一个 mini-ReAct（最多 `max_step_turns` 轮工具）→ 输出 `Step Answer`
3. **Replanner**：每完成一步后判断 `CONTINUE` / `DONE` / `REPLAN`，必要时改写剩余计划
4. **Synthesizer**：所有步骤完成后合成最终答案

**关键点**：
- **结合了 Plan 的清晰结构 + ReAct 的灵活反应**
- 有 `max_replans` 上限防止无限改计划
- 最复杂的一个，建议放最后看，前面两个先吃透

---

## 七、内置工具（`src/tools/basic_tools.py`）

仅用于演示，全部本地实现、零外部依赖：

| 工具名 | 功能 |
|--------|------|
| `calculator` | 用 `ast` 安全求值 `+ - * / ( )`，**不支持变量、函数**，避免 `eval` 注入 |
| `fake_search` | 一个写死的迷你知识库（capital of france 等），**不是真搜索引擎** |

> 想加新工具？只需在 `basic_tools.py` 里写一个 `def tool_xxx(input: str) -> str:`，再加进末尾的 `TOOL_IMPL` 字典；同步更新各 Agent 提示词里的 `Available tools:` 段即可。

---

## 八、建议的学习路线

1. 先读 `src/llm/llm.py`，搞清楚 LLM 客户端是怎么封装的（多家厂商、流式、usage 提取）
2. 跑一遍 `python -m src.llm.qwen_stream_demo`，确认 API 通
3. 用 `--verbose` 跑一遍 `react`，对照 `react_agent.py` 一行行看 scratchpad 是怎么滚动起来的
4. 同样 `--verbose` 跑 `rewoo`，对比 ReAct 在调用次数和灵活性上的差别
5. 最后看 `plan_execute`，特别关注 Replan 决策怎么改写剩余 plan
6. **动手练习**（推荐）：
   - 把 `fake_search` 换成真实搜索（如 SerpAPI、Bocha、Tavily）
   - 加一个新工具（比如 `python_exec` 沙箱、`http_get`）
   - 给 ReAct 加上"自我反思"步骤（Reflexion 范式）
   - 把 ReWOO 的 Solver 换成 RAG 风格——把 evidence 当作检索结果

---

## 九、参考论文

- **ReAct** — Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022). https://arxiv.org/abs/2210.03629
- **ReWOO** — Xu et al., *ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models* (2023). https://arxiv.org/abs/2305.18323
- **Plan-and-Solve / Plan-and-Execute** — Wang et al., *Plan-and-Solve Prompting* (2023). https://arxiv.org/abs/2305.04091
- **Reflexion**（扩展阅读）— Shinn et al., 2023. https://arxiv.org/abs/2303.11366

---

## 十、常见问题

**Q: 提示 `Missing API key. Set OPENAI_API_KEY, DASHSCOPE_API_KEY, or ARK_API_KEY.`**
A: `.env` 没创建或 key 行还是注释掉的（开头有 `#`）。

**Q: 报 `ModuleNotFoundError: No module named 'src'`**
A: 你 `cd src` 进了子目录里执行。退回到仓库根目录 `multi-agent/`，用 `python -m src.xxx` 的方式跑。

**Q: 模型经常输出格式不对、解析失败？**
A: 这是 Agent 学习中最经典的鲁棒性问题。可以：
- 把 `temperature` 调更低（已默认 0.1~0.2）
- 看仓库里的"自动修复"逻辑（`ReWOO/_make_plan`、`PlanExecute/_make_plan`）：发现解析失败就把错误回灌给 LLM 重试
- 换更强的模型（如 `qwen-plus` → `qwen-max`，`gpt-4o-mini` → `gpt-4o`）

---

## License

MIT（仅用于学习与研究）。
