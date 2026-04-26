---
name: hello-agents-style
description: Write Chinese teaching content in the Hello-Agents (DataWhale) style for any agent-adjacent topic — agent paradigms (ReAct / Plan-and-Solve / Reflection / ReWOO), RAG variants, memory systems, tool / function-calling infrastructure, multi-agent collaboration, and evaluation methods. Style hallmarks: motivation-first reasoning, life-style analogies, formalized math after intuition, step-by-step (1)–(N) code dissection with annotated callouts, real run logs with semantic emoji, and closing characteristics / limitations / debugging sections. Use when writing or revising any tutorial markdown under multi-agent/docs/, when adding teaching-grade docstrings or section comments inside multi-agent/src/, when extending README.md with a new technique section, or when the user mentions "Hello-Agents 风格" / "教学风格" / "讲解" / "教程" / "让读者学会".
---

# Hello-Agents 教学风格写作 Skill

本 skill 提炼自 [Hello-Agents 第四章 智能体经典范式构建](https://hello-agents.datawhale.cc/#/./chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA)。它是一种 **写作风格**，与具体技术主题无关 —— 同样的骨架、同样的代码拆解节奏、同样的语言密度，可以用来讲：

- Agent 范式：ReAct / Plan-and-Solve / Reflection / ReWOO / Plan-and-Execute
- RAG 系统：朴素 RAG / HyDE / 重排 / GraphRAG / Self-RAG
- 记忆系统：短期 scratchpad / 长期向量记忆 / 记忆压缩 / 反思型记忆
- 工具系统：手写 function-calling / 工具检索 / MCP 集成
- 多智能体协作：辩论 / 角色扮演 / Manager-Worker
- 评估方法：端到端评测 / 步级别评测 / LLM-as-judge

适用产物：
- `multi-agent/docs/*.md` —— 章节式教程
- `multi-agent/src/**/*.py` —— 教学性 docstring / 段首注释
- `multi-agent/README.md` —— 新增技术小节

文中"主题"一词是占位符，按场景代入"范式 / 模块 / 系统 / 方法"。

## 一、五条核心写作原则

1. **先 Why 再 What 再 How，最后 Limits**。任何主题都先回答"现有方案有何不足、为什么需要它"，再给"核心思想 + 形式化"，再"动手写代码"，最后"特点 / 局限 / 调试技巧"。永远不要直接堆代码。
2. **代入读者，使用"我们"**。"我们将构建…"、"亲手实现…"、"跑一遍…" —— 让读者感觉是和作者一起从零造轮子。避免"开发者应该…"、"用户可以…"这种第三人称说明书口吻。
3. **每段代码都拆出来单独讲**。不要一次性贴一整个类。把一个主题拆成 **`(1)–(N)` 个组件小节**，每段代码后跟 1–3 行段落 + 加粗要点列表。具体拆几段、拆什么由主题决定（详见第三节）。
4. **运行实例必须真实可读**。实现章节末尾必须有一段贴实际 stdout 的"运行实例与分析"，并用 1–3 条结论解读它体现了主题的哪些特征。
5. **形式化只服务于直觉**。能用一两个数学符号（递推 / 检索打分 / 损失函数）说明清核心机制就用，但每个公式之前必须先用大白话讲过同一件事。公式不是炫技，是对前面叙述的精炼。

## 二、章节骨架（教程 markdown 必用）

每讲一个技术主题，固定按下面四个 phase 组织。第三个 phase 的 `(1)–(N)` 段数和小标题要按主题调整，本身没有固定模板。

```
## X.Y 主题名（中英文对照）

[开篇过渡段：承接上一节 + 类比一句话点题 + 抛出本章问题]

### X.Y.1 工作原理 / 核心思想
- 现有方案的不足（对比铺垫）
- 核心思想（含生活化类比）
- 形式化表达（LaTeX 数学公式）
- 适用场景（- 列表，3 条左右）
- 本章实战要解决的具体问题

### X.Y.2 关键基础组件设计（可选，复杂主题专用）
- 1–3 个独立组件，每个先讲设计动机再贴代码
- 代码后用「- 关键点：xxx」要点列表回扣
  例：Agent 章节里的 ToolExecutor、Reflection 章节里的 Memory 模块

### X.Y.3 主题名 的编码实现
（1）……
（2）……
（3）……
…
（N）运行实例与分析

### X.Y.4 特点、局限性与调试技巧
（1）主要特点（编号列表，3 条）
（2）固有局限性（编号列表，3 条）
（3）调试技巧（- 列表，可操作动作）
```

下一节开头要写一句过渡，把本节和后续内容串起来（如"在掌握了 X 之后，下一节我们将…"）。

完整可 fork 骨架与三类主题（Agent / RAG / 记忆）的对应实例见 [templates.md](templates.md)。

## 三、代码拆解模式

### 3.1 拆解段数由主题决定，但格式恒定

不同主题有不同的"自然组件"，拆解时按主题选择：

| 主题类型 | 推荐的 `(1)–(N)` 拆解顺序 |
|---------|--------------------------|
| **Agent 范式** | (1) 提示词模板 → (2) 核心循环 → (3) 输出解析器 → (4) 工具调用 → (5) 状态/历史整合 → (6) 运行实例 |
| **RAG 系统** | (1) 文档切分 → (2) Embedding 与索引 → (3) 检索器 → (4) 重排（可选） → (5) 生成器 → (6) 运行实例 |
| **记忆系统** | (1) 数据结构定义 → (2) 写入接口 → (3) 检索/查询接口 → (4) 与主流程对接 → (5) 运行实例 |
| **工具系统** | (1) 工具协议 → (2) 注册表 → (3) 调度器 → (4) 错误处理 → (5) 运行实例 |
| **多智能体** | (1) 角色提示词 → (2) 通信协议 → (3) 调度循环 → (4) 终止判定 → (5) 运行实例 |
| **评估方法** | (1) 数据集 → (2) 指标定义 → (3) 评估循环 → (4) 结果聚合 → (5) 报告输出 |

无论拆几段，**编号格式恒定**：用中文括号 `（1）（2）（3）`（不是 `1.` 列表，也不是 `### 标题`）。

### 3.2 每段代码后必跟"要点回扣"

代码块结束后，**至少要有一段自然语言**说明"这段代码在做什么 / 为什么这样设计"，并用加粗的"-"列表点出 2–4 个要点。模板：

```
这个 [组件名] 的设计要点：

- **要点术语1**：……
- **要点术语2 (`{placeholder}` / `xxx_func`)**：……
- **要点术语3**：这是最重要的部分，它……
- **要点术语4**：……
```

要点条目格式固定为 `**名词术语**：解释`。术语、占位符、函数名用反引号包起来。

### 3.3 提示词 / 配置 / Schema 永远独立成块

所有"配置型"声明（Prompt 模板、检索 Schema、JSON 结构、向量索引参数）必须**单独成一个代码块**先展示，再在下面解析。不要把它们嵌在主类的代码里一起贴。占位符必须用 `{name}` 风格，并在下方逐个解释。

例如 RAG 章节的检索 Schema：

```python
RAG_RETRIEVAL_SCHEMA = {
    "top_k": 5,
    "score_threshold": 0.6,
    "rerank": True,
}
```

不是和 `class Retriever` 揉在一起贴。

## 四、语言与排版规约

### 4.1 emoji 语义化（不是装饰）

固定语义，跨主题复用。**只在代码示例的 `print` 输出和运行实例 stdout 里出现，正文叙述里不放**。

| Emoji | 语义                                      |
| ----- | ----------------------------------------- |
| 🧠    | LLM 调用开始（`🧠 正在调用 ... 模型...`） |
| ✅    | 阶段成功                                  |
| ❌    | 报错或失败                                |
| ⚠️    | 警告（覆盖、降级）                        |
| 🤔    | 智能体的"思考"（Thought 解析后打印）       |
| 🎬    | 智能体的"行动"（Action 触发）              |
| 🔍    | 工具实际执行 / 检索器命中                 |
| 👀    | 观察 / 检索结果回写                        |
| 📝    | 写入记忆 / scratchpad / 索引              |
| 📚    | 加载文档 / 构建索引                       |
| 🧩    | 文档切分 / 组件拼装                       |
| 🎯    | 命中目标 / 检索召回                       |
| 🎉    | 最终答案 / 任务完成                       |

写新主题时若需要新 emoji，沿着"动作 → 视觉名词"的命名直觉来选，并在所属文档顶部小节首次使用时注释一行语义。

### 4.2 中英术语混排

直接保留英文术语首字母大写：`Thought` / `Action` / `Observation` / `Plan` / `Reflection` / `Embedding` / `Retriever` / `Reranker` / `Chunk` / `Memory`。首次出现写成"中文（英文）"形式：

- "反思 (Reflection)"
- "重排 (Re-ranking)"
- "切分 (Chunking)"
- "短期记忆 (Short-term Memory)"

### 4.3 数学形式化

用 LaTeX 表达递推 / 打分 / 损失。例：

- **Agent**：$(th_t, a_t) = \pi(q, (a_1, o_1), \dots, (a_{t-1}, o_{t-1}))$
- **RAG**：$\text{score}(q, d) = \cos(\mathbf{e}_q, \mathbf{e}_d)$，召回集 $D_q = \text{TopK}(\{\text{score}(q, d) : d \in \mathcal{D}\})$
- **记忆压缩**：$M_{t+1} = \text{Summarize}(M_t \oplus o_t)$ 当 $|M_t| > L_{\max}$

公式之前必须先用一段大白话讲过同一件事。

### 4.4 同一系列的最后一篇必有对比表

讲完同一系列的多个变种后，章末用一张表横向对比。

```
| 主题 | 一句话描述 | 关键代价 | 适合场景 | 典型局限 |
| ---- | --------- | ------- | ------- | ------- |
| ReAct | 边想边做  | N 次 LLM | 不确定步数 | 易循环 |
| ReWOO | 先规划后执行 | 2 次 LLM | 步骤可静态规划 | 计划易失效 |
| ...  | ...      | ...     | ...     | ...    |
```

至少包含「主要代价」「适合场景」「典型局限」三列；列名可按主题改名（RAG 系列里"代价"可能换成"延迟与召回率"）。

### 4.5 类比和比喻成对出现

讲新主题时强制配一个生活类比，并尽量与同系列其它变种形成对照：

- ReAct ↔ "经验丰富的侦探，根据现场蛛丝马迹一步步推理"
- Plan-and-Solve ↔ "建筑师，动工前先画完整蓝图"
- Reflection ↔ "学生写完初稿后校对、解完数学题后验算"
- 朴素 RAG ↔ "考前抱临时书翻一翻找答案"
- HyDE ↔ "先猜一个答案再去图书馆找佐证"
- 长期记忆 ↔ "日记本：写下来才不会忘"
- 记忆压缩 ↔ "做读书笔记：看完一章只记重点"

写新主题时遵循同样模式："如果说 X 像 A，那么 Y 更像 B"。

### 4.6 论文 / 来源引用

第一次出现该技术名时立即标注论文，格式："由 ××× 于 20xx 年提出[N]"。章末统一列参考文献：

```
## 参考文献

[1] Yao S, Zhao J, Yu D, et al. ReAct: Synergizing reasoning and acting in language models[C]//ICLR. 2023.
[2] Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks[J]. NeurIPS. 2020.
```

非论文来源（博客、官方文档）也按编号列出，URL 直接给。

## 五、运行实例的写法

每个实现章节末尾必须有"运行实例与分析"。结构：

1. 一句引导："下面是一次真实的运行记录。"
2. 一个代码块（标 `text` 或留空），原样贴 stdout，**保留所有 emoji 和分隔符**（`--- 第 N 步 ---`、`>>>`、`📚 加载 12 个文档`、`🎯 召回 5 条 evidence` 等）。
3. 紧跟一段总结："从上面的输出可以看到，[主题名] 清晰地展示了…"
4. 用编号列表点出 2–3 个该主题的特征。例：
   - **Agent**：哪一步体现了"动态调整"、"自我纠错"
   - **RAG**：哪一步体现了"召回质量"、"重排带来的提升"
   - **记忆**：哪一步体现了"跨轮对话保留信息"
5. 如果输出会因模型版本 / 数据集 / 网络环境而变化，**显式提醒**："由于模型与数据持续更新，你运行的结果可能与此不完全相同。"

## 六、代码内的教学性注释（src/**/*.py）

教程文档之外，源码也保持同样节奏：

1. **文件顶部 module docstring**：3–6 行讲清"这个文件实现什么 / 来源论文或博客 / 关键设计取舍"。例：
   ```python
   """ReAct 范式实现（Yao et al. 2022）。

   核心循环：Thought -> Action -> Observation 不断追加到 scratchpad，
   直到模型输出 Final Answer。鲁棒性来自：解析失败时把错误回灌为新的
   Observation，让模型自我修复。
   """
   ```

   ```python
   """朴素 RAG 检索器。

   流程：query -> embedding -> 余弦相似度 TopK -> 拼进 prompt。
   关键取舍：
   - 切分用固定 chunk_size 而非语义切分，简单但召回粗糙
   - 不做重排，保持 baseline 干净
   """
   ```

2. **关键方法 docstring** 写"它在主题中扮演什么角色"，而不是参数 / 返回机械描述。
3. **每个不直观的代码块前**用一行段首注释点题。例：
   ```python
   # 4) 把本轮 Action 与 Observation 追加到历史，形成下一轮上下文
   #    （这是 ReAct 闭环的关键）

   # 3) 余弦相似度 TopK 召回
   #    注意：query 和 doc 必须用同一个 embedding 模型，否则距离没有意义
   ```
4. 编号注释 `# 1) ... # 2) ... # 3) ...` 与文档里 `（1）（2）（3）` **一一对应**，让读者能在文档和代码间双向跳转。
5. **不要写废话注释**（如 `# 调用 LLM`、`# 返回结果`）。注释只在解释"为什么这么写"或"在主题中扮演什么角色"时出现。

## 七、反模式（出现任何一种就要返工）

- ❌ 一上来就贴完整 200 行类定义，没有 `（1）（2）（3）` 拆解。
- ❌ 代码后没有"要点回扣"段落直接进下一节。
- ❌ 全章节没有任何 stdout 真实运行实例。
- ❌ 没有"特点 / 局限 / 调试技巧"小节就结束。
- ❌ 用第二人称机械说明书口吻（"用户应当配置 .env 文件"）替代"我们一起做"语气。
- ❌ 在正文叙述里堆 emoji 装饰；emoji 只允许出现在 print / stdout。
- ❌ 数学公式先于直觉解释出现。
- ❌ 同一系列写完没有对比表。
- ❌ 抄网上的实现而不给出"为什么这么设计"的取舍说明。
- ❌ 把 Prompt 模板 / 检索 Schema / JSON 结构嵌在主类代码里一起贴出，没有独立讲解。
- ❌ 强行套用 Agent 的 6 段拆解去讲 RAG 或记忆 —— 拆解段数应由主题自然组件决定。

## 八、写作工作流（模型在被调用时按此执行）

当用户要求写一段教程或讲解性代码时，按下列顺序产出：

1. **确认目标产物**：是 `docs/*.md` 章节、还是 `src/**/*.py` 的注释升级、还是 `README.md` 小节。
2. **确认主题类型**：Agent 范式 / RAG / 记忆 / 工具 / 多智能体 / 评估，决定 `(1)–(N)` 拆解段数（参考第 3.1 节表格）。
3. **复用骨架**：从 [templates.md](templates.md) 复制对应骨架，先填充小节标题。
4. **先写"为什么"**：开篇过渡段 + 现有方案不足 + 类比，**不写代码**。
5. **形式化一次**：写一段大白话讲清核心机制，再补一个 LaTeX 公式（如适用）。
6. **拆代码**：按主题选定的 `(1)…(N)` 顺序，每段不超过 30 行，**每段后必须跟要点列表**。
7. **跑一遍**：贴一段真实 stdout（哪怕是模拟的，也要保留 emoji 节奏与分隔符）。
8. **收口**：写"特点 / 局限 / 调试技巧"，3+3+3 编号列表。
9. **校验反模式**：对照第七节自检，发现一条就返工。
10. **配套表格**：如果是同一系列的最后一篇，补上对比表。

完成以上 10 步才算交付。

## 九、附加资源

- [templates.md](templates.md) —— 可直接 fork 的章节骨架、Agent / RAG / 记忆三套实例化、运行实例模板、对比表模板
- 原始风格出处：[Hello-Agents 第四章](https://hello-agents.datawhale.cc/#/./chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA)
- 仓库内 `README.md` 已有的范式对比表是合规示例，新增章节请保持同样列结构
