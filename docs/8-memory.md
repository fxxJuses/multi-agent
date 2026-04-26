# 第八章 智能体的记忆系统：从短期记忆到长期记忆

在第七章里，我们从 KV 缓存出发，把"单次推理为什么会变慢、prompt 怎么写才能命中前缀缓存"讲透了。然而 KV 缓存解决的是**单轮内**的延迟问题——它解释不了一个更现实的需求：**当对话进入第 50 轮、第 500 轮，或者用户隔了两周再回来，Agent 凭什么还记得"上次说过我喜欢 Rust 不喜欢 TypeScript"？**

本章我们把视角拉远，从"prompt 怎么拼"放大到"轨迹怎么存"。我们会先把 **短期记忆 (Short-term Memory)**、**长期记忆 (Long-term Memory)**、**上下文工程 (Context Engineering)** 这三个常被混用的概念分清楚；然后**逐一拆解**短期与长期记忆的常见变体，给出代码骨架与典型局限；最后用一张表把它们的取舍并排放在一起，落到 LangGraph 这种成熟框架的实现上。

> 本章的概念地图直接借鉴了 Hu et al. *Memory in the Age of AI Agents* (NeurIPS 2025 Survey, arXiv:2512.13564) 给出的三视角分类法[1]：**形式 (Forms)** / **功能 (Functions)** / **动力学 (Dynamics)**。我们关心的"短期 / 长期"只是其中"功能"维度的一种切法，但它最贴近工程师日常面对的取舍。

---

## 8.1 概念辨析：短期记忆、长期记忆与上下文工程

在动手写任何代码之前，我们必须先把名词钉死。Agent 记忆领域有一个让人头大的现状：同一个东西，在不同的论文、不同的框架文档里有完全不同的命名[1]。下面这一节，我们把"短期记忆 / 长期记忆 / 上下文工程 / LLM 内置记忆"放进同一张坐标系，争取一次说清。

### 8.1.1 工作原理：四个常被混用的术语

在 Agent 的世界里，"模型能记住什么"其实由四股力量共同决定：

- **LLM 内置记忆 (Parametric / LLM Memory)**：模型权重本身——预训练时灌进去的"世界知识"。它从不变（除非微调），与具体某个用户、某个会话无关。
- **上下文工程 (Context Engineering)**：把"我们想让模型看到的全部内容"——系统提示、工具描述、历史、检索结果、当前问题——按某种策略**拼成一段 prompt**。它关心的是 *how* (怎么拼) 而不是 *what* (拼什么)。
- **短期记忆 (Short-term Memory)**：当前任务/当前会话内的轨迹——刚才用户说了什么、刚才工具返回了什么、刚才模型自己想了什么。生命周期 = 一条线程 / 一次任务 (thread-scoped)。
- **长期记忆 (Long-term Memory)**：跨会话、跨任务都能调出来的内容——用户偏好、过往经验、领域知识、反思总结。生命周期 = 持久化在外部存储里 (cross-thread)。

这四个术语的关系如果说成一句话：**LLM 内置记忆是"它本来就会的东西"，长期记忆是"我们要它每次都记得的东西"，短期记忆是"它正在做的事情"，而上下文工程是"把这三类拼成一段 prompt 的工艺"**。如果说 LLM 内置记忆像一个学生的常识储备，那么长期记忆更像他随身带的笔记本，短期记忆更像他正在写的草稿纸——上下文工程则是他每次答题前，决定从笔记本和草稿纸上抄哪几页到答题卡上的那套规则。

### 8.1.2 形式化：放进同一个坐标系

我们用两个轴把它们的关系画清楚——**存储介质**（在哪儿）和**生命周期**（活多久）。把任意一段记忆 $m$ 看成一个三元组：

$$m = (\text{content},\ \text{medium},\ \text{ttl})$$

- $\text{medium} \in \{\text{weights},\ \text{prompt},\ \text{external\_store}\}$
- $\text{ttl} \in \{\text{train-time},\ \text{thread},\ \text{cross-thread}\}$

可以一一对位：

| 名称           | medium                  | ttl              | 谁负责写                        |
| -------------- | ----------------------- | ---------------- | ------------------------------- |
| LLM 内置记忆   | `weights`               | `train-time`     | 预训练 / 微调 pipeline          |
| 短期记忆       | `prompt`                | `thread`         | Agent 主循环 (append history)   |
| 长期记忆       | `external_store`        | `cross-thread`   | 显式 `store.put` / 反思流程     |
| 上下文工程     | —— 不是一种记忆，而是函数 $f: \{m_i\} \to \text{prompt}$ | —— | Agent 工程师                    |

**最重要的一行**：**上下文工程不是一种记忆**——它是个函数 $f$，输入是"现在手头能拿到的所有 $m_i$"，输出是"这一轮要发给 LLM 的那串 token"。短期记忆和长期记忆是 $f$ 的输入；KV 缓存（第七章）是 $f$ 的输出走到推理引擎之后的物理优化。

$$\text{prompt}_t = f_{\text{ctx-eng}}(\text{system},\ M_{\text{long}}^{\text{retrieved}}(q_t),\ M_{\text{short}}^{1:t-1},\ q_t)$$

这个公式后面 8.2 / 8.3 节里所有"变体"，**本质上都是在改 $M_{\text{long}}$ 和 $M_{\text{short}}$ 的存法 + 取法**。

### 8.1.3 短期记忆 ≠ 上下文工程，但深度耦合

工程师最常踩的一个坑是：把"短期记忆"和"上下文工程"当成同一件事。它们确实都最终落到 prompt 里，但区别在于：

- **短期记忆是状态对象**：它是一个有结构的列表 / 字典 / 队列，有自己的"写入时机"和"读取时机"。LangGraph 把它表示成 `State` + `Checkpointer`[2]；MemGPT 把它表示成 `working context + FIFO queue`[3]。
- **上下文工程是拼接策略**：拿到状态对象之后，怎么把它压成 prompt——保留全量？滑窗？摘要？分块加 ID？这一层关心的是 token 经济学，不是"记不记得"。

举个能让人立刻感受到差异的例子：当一个对话超出上下文窗口时，

- **短期记忆**会问："我要保留哪些消息让 Agent 任务正确性不掉？"
- **上下文工程**会问："保留下来的这些消息，要不要再压一次缩减成小的版本，让 KV 缓存命中率更高？"

两个问题的答案可以独立选择——这就是为什么本章把它们分开讲，而不是塞在第七章里。

适用场景上的简单口诀：

- 任务横跨多轮但都在一个 session 里 → 主战场是**短期记忆 + 上下文工程**
- 任务跨 session、跨用户、跨星期 → 必然要**长期记忆**
- 你抱怨"模型基本知识都答错" → 那是 **LLM 内置记忆 / RAG** 的问题，不是 Agent 记忆能解决的

接下来 8.2 节我们先把短期记忆的变体逐个实现一遍，再到 8.3 节接长期记忆。

---

## 8.2 短期记忆的几种变体

短期记忆要解决的是一个非常朴素的问题：**模型每次推理都是无状态的，但任务要求它"接着上一步往下做"**。我们要把"上一步发生过什么"用某种结构存下来，并在下一次调用时塞回 prompt。

不同的"塞法"就形成了不同的变体。本节我们按照"代价从低到高、能力从弱到强"的顺序，把四种最常见的实现拆开来看。

### 8.2.1 工作原理：上下文窗口里的瞬时记忆

无论变体有多花哨，所有短期记忆都遵循同一个递推关系。设第 $t$ 步时，短期记忆为 $M^{\text{short}}_t$，新观测为 $o_t$，那么：

$$M^{\text{short}}_t = \mathcal{U}(M^{\text{short}}_{t-1},\ o_t)$$

其中 $\mathcal{U}$ 是**更新算子**——它就是各变体的核心区别。给定状态算 prompt 时：

$$\text{prompt}_t = \text{system} \oplus \mathcal{C}(M^{\text{short}}_t) \oplus q_t$$

$\mathcal{C}$ 是**压缩算子**——决定状态对象怎么变成 token。下表是四种变体在这两个算子上的不同设计（先看一眼，下面 4 段会逐个解释）：

| 变体             | $\mathcal{U}$ 更新算子               | $\mathcal{C}$ 压缩算子               |
| ---------------- | ------------------------------------ | ------------------------------------ |
| 朴素 ScratchPad  | append 全部                          | identity（原样拼回去）               |
| 滑动窗口         | append + 截断到最近 $N$ 条           | identity                             |
| 摘要缓冲         | append + 触发 LLM 摘要重写老内容     | identity                             |
| MemGPT 分层      | append + 主动 evict 到外部存储       | 从 working context 实时拉            |

一句话总结：**变体的差异 = 怎么管理"上下文窗口里这块有限的内存"**。

### 8.2.2 变体逐个拆解

（1）朴素 ScratchPad：把所有历史一字不漏地拼回去

最朴素的方案，也是 ReAct 论文的原始实现[4]。每一步把 `Thought / Action / Observation` 三件套追加到 history 里，下一轮把 history 整段拼进 prompt：

```python
class ScratchpadMemory:
    """ReAct 式短期记忆：append-only，原样拼回 prompt。"""

    def __init__(self):
        self.history: list[str] = []

    def observe(self, role: str, content: str):
        self.history.append(f"{role}: {content}")

    def render(self) -> str:
        return "\n".join(self.history)
```

这个 `ScratchpadMemory` 的设计要点：

- **append-only 语义 (`observe`)**：上一轮的 `Observation` 一旦写过就不再改。这是和第七章的 KV 缓存铁律严格一致——前缀稳定 = 缓存命中。
- **`render()` 等于 identity**：压缩算子是恒等函数，token 数完全等于历史长度。便宜在实现简单，贵在 token 数 $O(t)$。
- **任务正确性最高**：因为所有细节都在 prompt 里，模型不会因为"被压缩"而丢失关键中间结论。
- **代价**：当 $t > 100$ 步、或单步观测特别长（搜索结果、长代码、长日志），prompt 会迅速逼近上下文窗口上限。

（2）滑动窗口：只保留最近 N 条

`history` 长到一定程度，最朴素的解决办法是**只保留最近 $N$ 条**。在 LangChain 里这就是经典的 `ConversationBufferWindowMemory`：

```python
from collections import deque

class SlidingWindowMemory:
    """只保留最近 N 条记录，越老越早被 evict。"""

    def __init__(self, window: int = 20):
        self.buffer: deque[str] = deque(maxlen=window)

    def observe(self, role: str, content: str):
        self.buffer.append(f"{role}: {content}")

    def render(self) -> str:
        return "\n".join(self.buffer)
```

这段滑动窗口实现的设计要点：

- **`deque(maxlen=N)`**：这是 Python 标准库里最自然的"环形缓冲区"，满了自动从左侧 evict，连判断都不需要写。
- **简单但会"失忆" (Forget Anything Old)**：第 $N+1$ 条进来时，第 1 条直接消失。如果第 1 条是用户自我介绍——比如"我叫 Alex，我们用 Python"——那 Agent 在第 21 轮就完全忘了这件事。
- **对 KV 缓存极不友好**：每次 evict 都意味着前缀第一条消息变了——参考第七章的失配规则，**整段缓存作废**。所以**滑动窗口看起来省 token，实际上让缓存命中率断崖式下降**，TTFT 反而变差。
- **适合什么场景**：任务上下文极强、与"很久以前的话"完全无关的客服场景、轮次确定不长的工具调用场景。

（3）摘要缓冲 (Summary Buffer)：老的不删，压缩成摘要

要解决滑动窗口的"失忆"问题，最自然的想法就是：**把要被 evict 的老消息，先用 LLM 总结成一段摘要，再让摘要替代它们留在 prompt 里**。这是 LangChain 的 `ConversationSummaryBufferMemory` 做的事：

```python
class SummaryBufferMemory:
    """超过 token 阈值时，把最早的 N 条摘要成 1 段，替代它们。"""

    SUMMARY_PROMPT = (
        "请把以下对话压缩成不超过 200 字的中文摘要，保留所有事实、姓名、时间、决定：\n\n{messages}"
    )

    def __init__(self, llm, max_tokens: int = 4000):
        self.llm = llm
        self.max_tokens = max_tokens
        self.summary: str = ""
        self.recent: list[str] = []

    def observe(self, role: str, content: str):
        self.recent.append(f"{role}: {content}")
        if self._estimate_tokens() > self.max_tokens:
            self._compress()

    def _compress(self):
        # 1) 取前一半"老"消息，连同已有摘要一起重新摘要
        half = len(self.recent) // 2
        old = self.recent[:half]
        merged = (self.summary + "\n" + "\n".join(old)).strip()
        self.summary = self.llm.invoke(
            self.SUMMARY_PROMPT.format(messages=merged)
        )
        self.recent = self.recent[half:]

    def render(self) -> str:
        return f"[历史摘要]\n{self.summary}\n\n[最近对话]\n" + "\n".join(self.recent)

    def _estimate_tokens(self) -> int:
        return sum(len(s) for s in self.recent) // 2 + len(self.summary) // 2
```

这段 `SummaryBufferMemory` 的设计要点：

- **三段式 render (`[历史摘要]` + `[最近对话]`)**：这是把"压缩信息"和"高保真信息"分开摆——模型看摘要拿大局，看最近几条拿细节。
- **重新摘要而不是增量摘要**：每次摘要时**把已有摘要和新一批老消息一起喂给 LLM**，而不是只摘要新消息。这样能让"重要的旧事实"在多轮压缩里依然不丢。
- **触发阈值 = token 估算**：实际工程里 `_estimate_tokens` 应该用真正的 tokenizer，而不是字符数除二。这里只是为了示意。
- **致命陷阱（KV 缓存视角）**：每次 `_compress` 都改写了 prompt 中段——**前缀缓存全部失效**。这就是第七章末尾 7.1.4 节"调试技巧"特别提示的反例："不要在循环里压缩历史"。如果非要做摘要，正确做法是把摘要**追加在尾部**，让前面的历史保持原样。

> **深度解析：为什么是"重新摘要"而不是"增量摘要"？**
>
> 上面 `_compress` 第 1 行注释写的是"连同已有摘要一起重新摘要"——这是一个反直觉的设计。直觉上更省的做法是 **增量摘要 (incremental)**：只摘要本次要 evict 的那批新老消息，再把摘要 chunk 追加到老 `summary` 后面：
>
> ```python
> # 增量摘要（看起来更省, 实则会失控）
> chunk = LLM.summarize(本次要 evict 的老消息)
> self.summary = self.summary + "\n" + chunk
>
> # 重新摘要（我们采用的）
> self.summary = LLM.summarize(self.summary + 本次要 evict 的老消息)  # 整段覆盖
> ```
>
> **唯一的代码差别在于 `+=` (追加) 还是 `=` (覆盖)**，但跑久了之后 `summary` 字段的形态完全不同。下面是连续 N 次压缩后 `summary` 字段的状态变迁：
>
> ```text
> 做法 A (增量): summary = "S1\nS2\nS3"           (3 段)
> 做法 A (增量): summary = "S1\nS2\nS3\nS4"       (4 段)
> 做法 A (增量): summary = "S1\nS2\n...\nSN"      (N 段)   ← 线性增长, 终将吃光窗口
>
> 做法 B (重新): summary = "S_new_v2"             (始终一份, 50-60 token)
> 做法 B (重新): summary = "S_new_v3"
> 做法 B (重新): summary = "S_new_vN"             ← 长度恒定, 由 LLM 自适应取舍
> ```
>
> 增量摘要的失败模式有三：① `summary` 长度线性增长，最终把"最近对话"挤光；② LLM 每次只看到本批新消息，**完全不知道老 `summary` 里写过什么**，所以无法做"S1 里的 Linda 还重要吗"这种全局判断；③ 同一事实容易在不同子摘要里重复记录（"S2 提到 Linda、S5 又提到 Linda"），合并不掉。
>
> 重新摘要的代价是**摘要 LLM 调用的输入多了一份老 `summary`**（典型 ~50 token），但回报是主对话 prompt 里的 `[历史摘要]` 段长度始终可控、且每次压缩 LLM 都做一次"复习 + 取舍"。当对话超过 30 轮，这笔账几乎一定划算。

（4）MemGPT 风格的虚拟分层短期记忆

如果觉得"摘要 vs 滑窗"二选一都不优雅，MemGPT (Packer et al. 2023, arXiv:2310.08560)[3] 给出了第三条路：**把上下文窗口看成 RAM，把外部存储看成 disk，让 Agent 自己用工具调用在两者之间搬数据**。

它把"上下文窗口"分成三段，并通过一份**写得相当详细**的 system prompt 教会 LLM 该怎么用这些区域。注意：MemGPT 真正的 prompt 长达 2000+ token，下面是教学浓缩版，但保留了所有关键约束（提示词模板独立成块）：

```python
MEMGPT_SYSTEM_TEMPLATE = """\
[SYSTEM INSTRUCTIONS]
你是一个有自管记忆的智能体。你有三块存储：

1. WORKING CONTEXT (永远在 prompt 里, 极珍贵)
   规则: 单条字段 ≤ 100 字; 仅放"每次推理都用得上 + 周级别不变"的事实
   字段约定:
     - user_profile:  用户身份/偏好（姓名/语言/口吻偏好）
     - tech_stack:    长期技术栈
     - current_goal:  当前 1-3 个月主线目标
     - constraints:   硬约束（规则/禁忌/性能 SLO）

2. ARCHIVAL STORAGE (按需检索回 prompt)
   放: 详细技术决策、bug 修复经验、用户偶尔提到但非高频的事实
   通过 archival_memory_search(query) 取回

3. FIFO QUEUE (最近对话原文, 满了自动 evict 进 archival)

[何时该写 WORKING CONTEXT — 在线即写，不要等 MEMORY PRESSURE]
- 用户首次自我介绍/透露偏好 → core_memory_append("user_profile", ...)
- 用户提到正在做的项目或 deadline → current_goal
- 用户说"必须 / 不能 / 我们规定" → constraints
- 用户更正一条已有 WC 字段 → core_memory_replace(block, new)

[反例 — 绝对不要写 WC]
- 单次任务的临时焦点（如"正在调试 xxx.py"）→ 不写, 这是 FIFO 的活
- 偶尔提到的事实（如"我妈叫 Linda"）→ archival_memory_insert
- 详细代码 / 设计文档 → archival_memory_insert

[Few-shot 示例]
user: 你好我叫 Alex, 我们用 Python
→ core_memory_append("user_profile", "Alex")
→ core_memory_append("tech_stack", "Python")

user: 顺嘴一提我妈最近来看我了
→ archival_memory_insert("用户母亲近期来访")  # 不进 WC

user: 我们团队有铁律, 不能引入任何新框架
→ core_memory_append("constraints", "团队铁律: 禁止引入新框架")

[WORKING CONTEXT]
{working_context}

[FIFO QUEUE]
{recent_messages}
"""
```

这个 prompt 模板的设计要点：

- **三块存储 + 字段约定**：把 WC 切成 4 个命名固定的字段（user_profile / tech_stack / current_goal / constraints），让 LLM 不用临场发明字段名——临场发明是 WC 失控的头号原因。
- **"何时该写"和"反例"必须双向写**：只告诉模型"重要的就写 WC"远远不够；必须配上明确反例（"临时焦点不写"、"偶尔事实写 archival"）才能压住模型把啥都往 WC 里塞的冲动。
- **few-shot 比规则更有效**：上面三条 few-shot 覆盖了"该写 WC / 该写 archival / 强约束"三种典型分流。生产里通常还要再加 5-10 个反例 few-shot 才稳。
- **"在线即写，不要等 MEMORY PRESSURE"是关键一句**：缺了这句话，LLM 会拖到看见 [MEMORY PRESSURE] 才回头总结——但那时早期事实可能已经被 evict 了，提炼质量会大打折扣。

下面是 MemGPT "工具调用闭环 + memory pressure" 主循环的最小实现。我们把它拆成两段：先看主循环和工具分发，再看 evict / render 这些辅助方法。

```python
import time

TOOL_SCHEMA = [
    {"name": "core_memory_append",   "args": ["block", "content"]},
    {"name": "core_memory_replace",  "args": ["block", "new"]},
    {"name": "archival_memory_insert", "args": ["content"]},
    {"name": "archival_memory_search", "args": ["query"]},
]

class MemGPTMemory:
    """虚拟分层短期记忆 + tool dispatch 闭环。"""

    def __init__(self, llm, ctx_limit: int = 8000, warning_ratio: float = 0.7):
        self.llm = llm
        self.warning = int(ctx_limit * warning_ratio)
        self.working_context: dict[str, str] = {
            "user_profile": "", "tech_stack": "", "current_goal": "", "constraints": "",
        }
        self.fifo: list[dict] = []
        self.archival: list[dict] = []

    def step(self, user_msg: str) -> str:
        # 1) 用户消息入 FIFO; token 过线就触发 memory pressure
        self.fifo.append({"role": "user", "content": user_msg})
        if self._tokens() > self.warning:
            self._memory_pressure()

        # 2) 调 LLM；只要它返回工具调用就 dispatch + 回灌, 直到不再 call 为止
        response = self.llm.invoke(self.render(), tools=TOOL_SCHEMA)
        while response.tool_calls:
            for call in response.tool_calls:
                result = self._dispatch(call.name, call.args)
                self.fifo.append({"role": "tool", "name": call.name, "content": result})
            response = self.llm.invoke(self.render(), tools=TOOL_SCHEMA)

        # 3) 普通回复入 FIFO
        self.fifo.append({"role": "assistant", "content": response.content})
        return response.content

    def _dispatch(self, name: str, args: dict) -> str:
        # 这就是 LLM 真正能改 WC / archival 的地方
        if name == "core_memory_append":
            cur = self.working_context.get(args["block"], "")
            self.working_context[args["block"]] = (cur + " " + args["content"]).strip()
            return f"[OK] appended to {args['block']}"
        if name == "core_memory_replace":
            self.working_context[args["block"]] = args["new"]
            return f"[OK] replaced {args['block']}"
        if name == "archival_memory_insert":
            self.archival.append({"text": args["content"], "ts": time.time()})
            return "[OK] archived"
        if name == "archival_memory_search":
            return self._archival_search(args["query"], k=5)
        return f"[ERR] unknown tool {name}"
```

这段主循环 + tool dispatch 的设计要点：

- **`step()` 而不是 `observe()`**：旧版 `observe` 只是把消息塞进 FIFO，**没有真正调 LLM**——所以 LLM 永远没机会调用工具改 WC。`step` 才是 MemGPT 闭环：入队 → 调 LLM → 工具回灌 → 再调 LLM，直到模型不再 call。
- **`while response.tool_calls`**：这是 MemGPT 的"function chaining"原语[3]——一轮里允许多次工具调用，比如先 `archival_memory_search` 拿背景，再 `core_memory_append` 升级事实，最后才输出给用户。
- **`_dispatch` 是 WC 真正被改的唯一入口**：你之前那个观察"WC 在原代码里永远不变"——根本原因就是缺这个 dispatch。补上之后，每一次 `core_memory_append/replace` 才真正落到 `self.working_context` 上。
- **工具结果用 `role="tool"` 入 FIFO**：这与第七章 7.1.4 节"工具结果用 ID 标注、追加在末尾"的 KV 缓存铁律一致——不要回头改前面的 assistant message 假装"模型自己想到了这条结果"。

辅助方法（evict 与渲染）：

```python
    def _memory_pressure(self):
        # 1) 切掉前 50% (最早进的) 进 archival, 留下后 50%
        cut = len(self.fifo) // 2
        evicted, self.fifo = self.fifo[:cut], self.fifo[cut:]
        self.archival.extend(evicted)
        # 2) 在 FIFO 段最前面塞一条"系统中断", 让 LLM 下一轮一眼看到
        self.fifo.insert(0, {
            "role": "system", "transient": True,
            "content": "[MEMORY PRESSURE] 已 evict 早期消息到 archival。"
                       "请检查是否需要 core_memory_append / archival_memory_search。"
        })

    def render(self) -> str:
        return MEMGPT_SYSTEM_TEMPLATE.format(
            working_context="\n".join(f"{k}: {v}" for k, v in self.working_context.items()),
            recent_messages="\n".join(f"[{m['role']}] {m.get('content','')}" for m in self.fifo),
        )

    def _tokens(self) -> int:
        return sum(len(m.get("content", "")) for m in self.fifo) // 2
```

这段辅助方法的设计要点：

- **三层物理边界（`working_context` / `fifo` / `archival`）**：分别对应 OS 中的"寄存器 / RAM / 磁盘"。这是 MemGPT 把"OS 思想搬给 LLM"的精髓[3]。
- **`memory_pressure` 是事件而非定时器**：触发条件是真的快要爆窗口，避免了 KV 缓存的频繁失配（参考第七章 7.1.4 节）。
- **`insert(0, ...)` 不是 FIFO 入队，是渲染优先级**：把 [MEMORY PRESSURE] 放到 list 索引 0 是为了让 LLM **阅读 prompt 时第一眼看到**这条 interrupt——这是借用 list 位置语义而非队列语义。
- **`transient=True` 标记**：理论上这条系统中断是"消费一次就丢"的瞬时信号；下一次 evict 时应该跳过它（避免污染 archival）。教学版在 `_memory_pressure` 里没显式跳过，生产实现（如 Letta 的 `LettaMessageManager`）必须补上这一步。

> **深度解析：哪些信息该进 WORKING CONTEXT？**
>
> 上面的 system prompt 里写了"何时写 / 何时不写"，但很多读者会卡在"我手头这条信息到底算哪种"。下面给一份可直接套用的**判定清单 + 决策表**。
>
> 一条信息要进 WC，**必须同时满足**这四件事：
>
> 1. **每次推理几乎都用得上**——例如 `user_profile` 几乎每次回答都会被引用；如果一周才用一次，去 archival
> 2. **变化频率够低**——每周以上才变一次；如果一天变好几次（如"当前正在看的文件"），那是 FIFO 的活
> 3. **能用一行字说清**——WC 是"卡片"不是"文档"，长内容直接 archival
> 4. **丢了 Agent 就会答错**——例如团队规则丢了会推荐违禁工具 → 必须放
>
> 任何一条不成立，就去 archival 或 FIFO。具体场景对照：
>
> | 信息 | 进 WC？ | 该进哪 | 为什么 |
> | --- | --- | --- | --- |
> | Alex / Python / 后端 | ✅ | WC | 每次推理都要用、几乎不变 |
> | 当前项目"电商订单服务" | ✅ | WC | 整个 Q4 都不变 |
> | "团队禁用新框架" | ✅ | WC | 每次给建议都要遵守 |
> | "正在 review xxx.py" | ⚠️ | WC（但要短） | 任务期间高频引用，结束后就该 `replace` 掉 |
> | Alex 妈妈叫 Linda | ❌ | archival | 偶尔提及，无需常驻 |
> | "上周修了一个 N+1 bug" | ❌ | archival | 跨任务可能复用，但不是每次都要看 |
> | 详细 ER 图 / 设计文档 | ❌ | archival | 大块内容，需要时再 search |
> | 最近 5 轮对话原文 | ❌ | FIFO | 短期临时上下文 |
> | 工具上一次的输出 | ❌ | FIFO | 用完就丢 |
>
> **嗅觉测试**：把当前 WC 打印出来给一个新接手这个用户的同事看，如果他能在 30 秒内理解"这个用户是谁、在做什么、有什么硬约束"——那 WC 就设计对了。如果他需要看一堆"嗯这是某个临时变量"的东西才能拼出全景——那这些临时东西本来就不该在 WC 里。

> **深度解析：WC 改写的 KV 缓存代价**
>
> 这是 MemGPT 设计的**固有矛盾**：自管记忆的能力 ≠ KV 缓存友好。每次 `core_memory_append` / `core_memory_replace` 改写 WC，从 WC 段开始往后的所有 prompt token 都"变了"——按第七章 7.1.4 的铁律，**前缀缓存从那个失配点往后全部作废**。具体到时间轴：
>
> ```text
> 轮次 1:  [SYSTEM] + [WC: 空]      + [FIFO: m1]            → 全段首次, cache miss
> 轮次 2:  [SYSTEM] + [WC: 空]      + [FIFO: m1, m2]        → 命中 ✅ (前缀完全相同)
> 轮次 3:  [SYSTEM] + [WC: 空]      + [FIFO: m1, m2, m3]    → 命中 ✅
> 轮次 4:  模型主动 core_memory_append("user_profile", "Alex")
>          [SYSTEM] + [WC: Alex...] + [FIFO: ...]            → 从 [WC] 段起 ❌ 失配, 后段重算
> 轮次 5:  [SYSTEM] + [WC: Alex...] + [FIFO: ...m5]          → 命中到 [WC] 末尾 ✅, 仅 m5 是新的
> ```
>
> 注意 `[SYSTEM INSTRUCTIONS]` 段永远稳定，那部分缓存仍然命中——失配的只是 WC 起点开始往后的部分。这也是 MemGPT 故意把 SYSTEM 和 WC 拆成两段的原因：至少 system 段（典型 800-2000 token 的 prefill）能复用。
>
> Letta 的生产实现给出三条缓解策略：
>
> 1. **WC 切成多个独立 block，并按"改写频率从低到高"排序**：把最稳定的 `user_profile` 放最前，最易变的 `current_goal` 放最后。改后面的 block 时，前面 block 之后的 prompt 已经失配——但被失配的 token 数尽量少。
> 2. **system instructions 里直接告诉模型"昂贵操作慎用"**：Letta 的 default prompt 写了 *"core_memory_replace is expensive; only use it when truly necessary"*——把工程代价显式传达给 LLM 让它自己取舍。
> 3. **把易变信息全部赶去 archival**：当前任务进度、临时计算结果一律 `archival_memory_insert`；WC 只放真正"半永久"的事实。这样 WC 一周可能就改 2-3 次，每次失配也认了。
>
> 这其实是和 (3) 节 SummaryBufferMemory **同一种 KV 缓存反模式的两个不同面孔**：
>
> | | SummaryBufferMemory | MemGPT working_context |
> | --- | --- | --- |
> | 谁在改 prompt 中段 | Agent 框架（在 `_compress` 里） | LLM 自己（通过 function call） |
> | 触发频率 | 阈值触发（被动） | 模型自决（主动） |
> | 改写粒度 | 整段历史摘要重写 | 一个 block / 一行 |
> | KV 缓存代价 | system 之后**全部失配** | system + WC 前部命中、改写点之后失配 |
>
> MemGPT 比 SummaryBuffer 优势在改写粒度更细 + 时机由 LLM 决定，**但没有从根本上消除"中段改写 = 缓存失配"**——只是把代价转移到了"改写次数少 + 改写位置尽量靠后"上。

（5）短期记忆变体的运行实例与对比

下面是同一个 50 轮客服对话，分别跑四种变体后的记录（`📝` 表示写入，`⚠️` 表示压缩 / 失忆事件，`❌` 表示前缀缓存失配）：

```text
📚 共 50 轮对话，每轮平均 200 token

--- ScratchPad ---
📝 第  1 轮: prompt= 200 tokens, KV 命中率: -
📝 第 25 轮: prompt=5000 tokens, KV 命中率: ✅ 100%
📝 第 50 轮: prompt=10000 tokens, KV 命中率: ✅ 100%
🎯 任务正确性: 用户在第 1 轮自我介绍 Alex / Python 仍被记得 ✅
⚠️ token 成本: 累计 250k tokens, 接近 8k 上下文窗口上限

--- 滑动窗口 (N=10) ---
📝 第  1 轮: prompt= 200 tokens, KV 命中率: -
📝 第 11 轮: 第 1 条被 evict, 后续 KV 命中率: ❌ 0% （前缀变了）
📝 第 50 轮: prompt=2000 tokens, 始终 KV ❌
🎯 任务正确性: 第 30 轮被问到"我叫什么"时答错 ❌

--- 摘要缓冲 (max=4000) ---
📝 第 21 轮: 触发压缩, prompt 中段被改写, KV 命中率: ❌ 0%
📝 第 42 轮: 又一次压缩, 摘要里"用 Python"丢失 ⚠️
🎯 任务正确性: 第 50 轮答错语言偏好 ❌
⚠️ token 成本: 比 ScratchPad 省 60%, 但每次压缩多花 1 次 LLM 调用

--- MemGPT 分层 ---
📝 第  1 轮: 模型主动 core_memory_append("user_profile", "Alex / Python")
📝 第 28 轮: 触发 [MEMORY PRESSURE], evict 14 条到 archival
📝 第 28 轮: 模型主动 archival_memory_search("Python") → 召回 ✅
🎯 任务正确性: 50 轮全程"我叫什么 / 用什么语言"答对 ✅
⚠️ KV 命中: working_context 不变时 ✅, 模型改写它时 ❌
```

从上面的输出可以看到，四种变体清晰地展示了短期记忆设计上的几个核心权衡：

1. **任务正确性（不失忆）只有 ScratchPad 和 MemGPT 能保证**：滑动窗口和摘要都会因为"硬截断 / 信息有损压缩"而把第 1 轮的关键事实丢掉。
2. **token 成本与 KV 缓存命中率是负相关的**：朴素 ScratchPad token 最多但缓存命中率最高；摘要 / 滑窗 / MemGPT 都通过"改 prompt"来省 token，但每一次"改"都意味着至少局部的缓存失效。
3. **工业实践里没有银弹**：客服 / 单任务 → 滑窗够用；长链路 Agent / 跨轮反思 → MemGPT 模式或第七章末尾建议的"原历史 + 摘要追加"组合更稳。

由于模型与数据持续更新，你运行的结果可能与此不完全相同，但变体之间的相对关系是稳定的。

### 8.2.3 短期记忆变体的特点、局限性与调试技巧

通过亲手实现这四种变体，并把它们和 KV 缓存放在一起看，我们应该对短期记忆的内在机制有了更清楚的认识。

（1）短期记忆的主要特点

1. **天然 thread-scoped**：不需要持久化也能工作，重启进程就重置——这是它"轻"的一面。
2. **更新算子可以非常简单**：append-only 就能解决 80% 的需求；只有当上下文窗口真的紧张时才需要压缩或分层。
3. **直接决定 KV 缓存命中率**：从这一节起，你应该把"短期记忆设计"和"上下文工程"当成同一个工程指标的两面。

（2）短期记忆的固有局限性

1. **本质是有限缓冲区**：再聪明的压缩也无法在 8k token 里塞下 1000 轮真实对话。一旦任务跨度超过窗口，必须升级到长期记忆。
2. **每次"压缩 / 截断"都伤 KV 缓存**：第七章已经反复强调过——任何中段修改都会让前缀缓存大段失效。摘要缓冲是经典反例。
3. **信息损耗是单向的**：滑动窗口和摘要一旦丢了某条事实，后面再问就答不出来——除非升级到 MemGPT 这种"evict 但不删"的设计或外部长期记忆。

（3）调试技巧

- **打印 prompt 长度曲线**：横轴是轮次，纵轴是 prompt token 数。朴素 ScratchPad 是直线，滑窗是阶梯，摘要缓冲是锯齿，MemGPT 应该是平台 + 偶尔尖峰——曲线形状不对就是变体配错了。
- **构造"长程记忆"测试用例**：在第 1 轮埋一个独特事实（如"我妈名字是 Linda"），第 50 轮主动问。任何一种变体丢了它就要看是 evict 设错还是摘要 prompt 写漏。
- **追踪 KV 缓存命中率**：服务端通常会返回 `cached_tokens`。把它和 `prompt_tokens` 的比值画出来，每次断崖式下降都对应一次"prompt 中段被改"——这是排查"明明逻辑对、为什么慢"的最高效手段。
- **避免在 system prompt 里塞动态内容**：哪怕是当前时间、任务 ID、轮次数——一旦写进 system，就让所有后续轮次的缓存全部作废。把它们放在 user message 末尾，是仅次于"启用 KV 缓存"本身的最重要优化。
- **MemGPT 风格调试**：打印每次 `core_memory_*` 的调用栈。如果模型频繁改写 working_context 而不是用 archival，说明你给它的指令里"什么算重要事实"没说清楚——加几个 few-shot 例子会立竿见影。

在掌握了短期记忆的几种变体之后，下一节我们把视野推到"跨会话 / 跨任务"的尺度，看看真正的"长期记忆"长什么样。

---

## 8.3 长期记忆的几种变体

短期记忆解决的是"任务进行中怎么不丢东西"，长期记忆要解决的是另一个问题：**这个用户、这个 Agent，半个月之后再回来，凭什么还认识彼此？** 它不再活在 prompt 里，而是落到一个外部存储里——可能是向量库、关系数据库、知识图谱，甚至是模型自身的权重。

### 8.3.1 工作原理：跨会话的持久存储

我们把长期记忆抽象成一个键值存储 $\mathcal{S}$，它支持四个原子操作[5]：

$$\mathcal{S} = (\text{put},\ \text{get},\ \text{search},\ \text{evolve})$$

- **`put(namespace, key, value)`**：写入。`namespace` 一般是 `(user_id, topic)`。
- **`get(namespace, key)`**：精确取回。
- **`search(namespace, query, k)`**：按相关性 / 时间 / 重要性返回 Top-$k$。
- **`evolve(...)`**：可选——让已经写入的记忆**反过来被新记忆改写**（这是 A-MEM 的核心创新[6]）。

每一次新观测 $o_t$ 进来时，长期记忆要决定两件事：

$$\text{write}_t:\ \mathcal{S} \leftarrow \mathcal{S} \cup \{\phi(o_t)\}\quad\text{以及}\quad \text{read}_t:\ M^{\text{long, retrieved}}_t = \text{TopK}_\sigma(\mathcal{S},\ q_t)$$

其中 $\phi$ 是**抽取函数**（把"原始对话片段"变成"可索引的事实条目"），$\sigma$ 是**打分函数**（决定哪些条目最该被取出来）。各个变体的差异主要就在 $\phi$ 和 $\sigma$ 上。

> Hu et al. (2025)[1] 把上面的"动力学"概括为三个阶段：**Formation (形成)** / **Evolution (演化)** / **Retrieval (检索)**。本节五个变体可以一一对位到这三个阶段做了什么。

适用场景：

- **个性化助手**：用户偏好、聊天历史摘要、长期目标
- **长程任务 Agent**：跨任务的经验沉淀，"上次怎么写筛法"、"之前哪种 API 用法被反对过"
- **多智能体协作**：共享知识库，让 Agent A 写下的事实 Agent B 能看到

下面我们把五种最有代表性的变体拆开来看。

### 8.3.2 变体逐个拆解

（1）向量记忆：朴素 Embedding + TopK

最朴素也是工业上最常见的方案：把每条记忆做 embedding，存进向量库（FAISS / Chroma / pgvector）；查询时用 cosine 相似度 TopK 召回。这是 Mem0[7] / LangGraph `InMemoryStore`[2] 等几乎所有产品级方案的基线。

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class MemoryRecord:
    content: str
    namespace: tuple
    vec: np.ndarray
    created_at: float

class VectorMemory:
    """朴素长期记忆：每条记录一个 embedding，cosine TopK 召回。"""

    def __init__(self, embedder):
        self.embedder = embedder
        self.records: list[MemoryRecord] = []

    def put(self, namespace: tuple, content: str):
        vec = self.embedder.encode(content)
        self.records.append(MemoryRecord(
            content=content, namespace=namespace, vec=vec,
            created_at=time.time(),
        ))

    def search(self, namespace: tuple, query: str, k: int = 5) -> list[MemoryRecord]:
        # 1) namespace 精确过滤
        cand = [r for r in self.records if r.namespace == namespace]
        if not cand:
            return []
        # 2) cosine 打分 TopK
        q = self.embedder.encode(query)
        scored = [(self._cos(q, r.vec), r) for r in cand]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [r for _, r in scored[:k]]

    @staticmethod
    def _cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
```

这段 `VectorMemory` 的设计要点：

- **`namespace` 是必备的"租户"维度**：生产中至少要按 `(user_id, scope)` 切分，不然不同用户的记忆会串味。LangGraph 的 `BaseStore` 也强制要求 namespace[2]。
- **embedder 必须和检索时严格一致**：写入和读取必须用同一个模型——同一句话过两个不同的 embedder 出来的向量在不同坐标系里，欧氏距离/余弦相似度完全没有意义。换 embedder 时必须**全量重建索引**，不能"边换边查"。
- **优点**：实现极简、横向扩展容易（直接换成 pgvector / Pinecone 即可）、命中速度可控。
- **典型局限**：**召回完全靠语义相似度**——同义但不同主题的事实会互相干扰，例如"我喜欢咖啡"和"我喜欢茶"在向量空间里离得很近，但真实意图相反。Embedding 也不会自然衰减——一年前一句无关紧要的话，可能因为语义巧合排在 Top-1。这正是下一种变体要解决的问题。

（2）Generative Agents 记忆流：Recency × Importance × Relevance

Park et al. 2023 提出的"Generative Agents"[8] 把人类记忆的三个心理学因素直接搬进了 Agent 的检索打分：**最近的事更显眼、重要的事更难忘、相关的事最优先**。它的打分函数是上面三者归一化后的加权和：

$$\sigma(m, q, t) = \alpha_r \cdot \text{recency}(m, t) + \alpha_i \cdot \text{importance}(m) + \alpha_l \cdot \text{relevance}(m, q)$$

其中：

- $\text{recency}(m, t) = e^{-\lambda \cdot \Delta t}$ —— 自上次访问以来的指数衰减
- $\text{importance}(m) \in [0, 1]$ —— 写入时由 LLM 给出的"诗意分 (poignancy)"，平淡事 1 / 重大事 10
- $\text{relevance}(m, q) = \cos(\mathbf{e}_m, \mathbf{e}_q)$ —— 与朴素向量记忆相同的语义相似度

原论文的权重是 $\alpha_r{=}1$, $\alpha_i{=}1$, $\alpha_l{=}1$；社区开源实现里调成了 $0.5 / 2.0 / 3.0$ 这种加重 relevance 的版本[9]——**权重应该当超参看，而不是当真理看**。

```python
import time, math

class GenerativeAgentMemory(VectorMemory):
    """Recency + Importance + Relevance 三因子加权检索。"""

    def __init__(self, embedder, llm, decay_rate: float = 0.005,
                 alphas: tuple = (0.5, 2.0, 3.0)):
        super().__init__(embedder)
        self.llm = llm
        self.decay_rate = decay_rate
        self.alpha_r, self.alpha_i, self.alpha_l = alphas

    def put(self, namespace: tuple, content: str):
        # 1) 写入时让 LLM 打一次"重要性"，1-10 归一到 [0, 1]
        importance = self._score_importance(content) / 10.0
        super().put(namespace, content)
        self.records[-1].importance = importance
        self.records[-1].last_accessed = time.time()

    def _score_importance(self, text: str) -> float:
        prompt = (
            "以 1-10 给下列事件的重要性打分（1=平淡日常，10=人生大事），只回数字：\n"
            f"事件：{text}\n分数："
        )
        return float(self.llm.invoke(prompt).strip())

    def search(self, namespace, query, k=5, now=None):
        now = now or time.time()
        q = self.embedder.encode(query)
        cand = [r for r in self.records if r.namespace == namespace]
        scored = []
        for r in cand:
            recency = math.exp(-self.decay_rate * (now - r.last_accessed))
            relevance = self._cos(q, r.vec)
            score = (self.alpha_r * recency
                     + self.alpha_i * r.importance
                     + self.alpha_l * relevance)
            scored.append((score, r))
        scored.sort(reverse=True, key=lambda x: x[0])
        # 2) 命中即更新 last_accessed —— 这是 recency 衰减的关键
        for _, r in scored[:k]:
            r.last_accessed = now
        return [r for _, r in scored[:k]]
```

这段 `GenerativeAgentMemory` 的设计要点：

- **写入时多花一次 LLM 调用打 importance**：成本不可忽视，但它直接决定了"鸡毛蒜皮 vs 关键转折"能不能被区分开。在长程模拟里这一次调用的代价被回报远远摊平。
- **`last_accessed` 而不是 `created_at`**：被检索过的事会"刷新"它的 recency——这模拟了人类的"复习就不容易忘"。这一点是和朴素向量记忆最大的差别。
- **三因子互相覆盖**：纯相似度的失败模式（"我喜欢咖啡 / 我喜欢茶"误召回）会被 importance 拉开（其中一个被打过更高分）；纯重要性的失败模式（"重要但和当前任务无关"）会被 relevance 压下去。
- **典型局限**：importance 由 LLM 自评，受 prompt 偏置严重——同一件事在不同 system prompt 下打分可能差 3 分。生产里通常需要校准（用 reference 数据集做后处理回归）。

（3）反思型记忆：Reflection / Reflexion

到这里我们写下的还都是"原始事件"。但事件多了之后，真正影响行为的其实是**从这些事件里抽出来的总结、教训、规律**。这就是 **反思型记忆 (Reflective Memory)** 要做的事——它的代表论文是 Shinn et al. *Reflexion* (NeurIPS 2023)[10]，以及前面提到的 Generative Agents 中的"reflection"机制[8]。

反思的核心算子是：

$$\text{reflection}_t = \text{LLM}(\{m_i \in M^{\text{recent}}_t\},\ \text{prompt}_{\text{reflect}})$$

然后**反思本身被作为一条新的记忆 $\text{put}$ 回 $\mathcal{S}$**——这意味着下一次检索时，反思会和原始事件一起被排序，且因为它信息密度高、importance 通常被打得很高，往往压过原始事件被召回。

```python
REFLECT_PROMPT = """\
请根据以下最近的事件，提炼出 3 条更高层次的洞察 / 教训 / 规律。
每条要：
- 不超过 1 句话
- 体现因果或一般性，而不是事实复述
- 写得让"未来的你"读到就能直接拿来决策

最近的事件：
{events}

请输出 3 条编号洞察："""

class ReflectiveMemory(GenerativeAgentMemory):
    """在 Generative Agents 三因子的基础上，叠加反思生成。"""

    def __init__(self, *args, reflect_threshold: float = 5.0, **kw):
        super().__init__(*args, **kw)
        self.reflect_threshold = reflect_threshold
        self._importance_acc: float = 0.0

    def put(self, namespace, content):
        super().put(namespace, content)
        self._importance_acc += self.records[-1].importance
        # 1) 累计重要性达到阈值才反思 —— 频次自适应
        if self._importance_acc >= self.reflect_threshold:
            self._reflect(namespace)
            self._importance_acc = 0.0

    def _reflect(self, namespace):
        recent = [r for r in self.records[-50:] if r.namespace == namespace]
        if len(recent) < 5:
            return
        events_text = "\n".join(f"- {r.content}" for r in recent)
        insights = self.llm.invoke(REFLECT_PROMPT.format(events=events_text))
        for line in insights.strip().splitlines():
            line = line.strip().lstrip("0123456789. ").strip()
            if line:
                # 2) 反思条目作为新记忆写回，importance 默认设高
                super().put(namespace, f"[REFLECTION] {line}")
                self.records[-1].importance = 0.8
```

这段 `ReflectiveMemory` 的设计要点：

- **触发条件 = 累计重要性，不是固定轮次**：这是 Generative Agents 论文的关键设计[8]——平淡日子可以攒很久才反思一次，重大事件聚集时密集反思。
- **反思条目带前缀标记 (`[REFLECTION]`)**：后续检索一眼能看出"这是抽象洞察，不是原始事件"。如果要做"递归反思"（对反思再反思），这个前缀也方便筛掉避免无限套娃。
- **用同一份存储混排原始事件 + 反思**：检索时由打分函数决定谁更该被召回——这一行设计哲学决定了 Generative Agents 的智能体能在长时模拟里持续进化[8]。
- **典型局限**：反思 prompt 写得不好就会出"正确的废话"（"用户表现得很重视质量"——这条留下来对未来决策毫无帮助）。生产里要给 reflect_prompt 配 few-shot 示例，并周期性回看抽样反思的质量。

（4）A-MEM：Zettelkasten 风格的演化型记忆

到目前为止的所有变体都是 **append-only**——写下去就不再改。但人类做笔记时，新读到一段话会让我们**回去改写老笔记的边注、补一条新链接**。这就是 Xu et al. *A-MEM: Agentic Memory for LLM Agents* (NeurIPS 2025, arXiv:2502.12110)[6] 的核心创新：把卡片盒笔记法 (Zettelkasten) 搬进 Agent 记忆。

A-MEM 的关键不同有两点：

1. **每条记忆是结构化笔记**（不只是一段文本），写入时由 LLM 生成 `keywords / tags / context_description` 等多字段属性
2. **新记忆会触发老记忆的 `evolve`**——更新它们的 `tags / context`，让记忆网络持续重组

我们用一个简化但能跑的版本来体会：

```python
NOTE_SCHEMA_PROMPT = """\
将下列原始观察转换为结构化记忆笔记，输出 JSON：
{{
  "summary": "<= 2 句的要点",
  "keywords": ["<3-5 个关键词>"],
  "tags": ["<2-3 个高层标签，如: 偏好/经验/约束>"],
  "context": "这条记忆是在什么情境下产生的"
}}

原始观察：{observation}
JSON:"""

EVOLVE_PROMPT = """\
你刚刚记录了一条新笔记 [NEW]。请从下列 [CANDIDATES] 中找出与它强相关的老笔记，并对它们做 evolve（仅当确实需要更新时）。
[NEW]
{new}

[CANDIDATES]
{candidates}

为每条候选笔记输出 JSON：
{{"id": "...", "should_update": true|false, "new_tags": [...], "new_context": "..."}}
若 should_update 为 false，可省略其余字段。"""

class AMemMemory(VectorMemory):
    """Zettelkasten 风格的演化型长期记忆。"""

    def put(self, namespace, content):
        # 1) 用 LLM 把"原始观察"结构化成一条 note
        note = self._structurize(content)
        text = f"{note['summary']} | tags: {','.join(note['tags'])}"
        super().put(namespace, text)
        rec = self.records[-1]
        rec.note = note
        # 2) 找语义最近的 K 个老笔记，让 LLM 决定要不要 evolve 它们
        siblings = self._find_neighbors(rec, k=5, exclude_self=True)
        if siblings:
            self._evolve(rec, siblings)

    def _structurize(self, observation: str) -> dict:
        return json.loads(self.llm.invoke(NOTE_SCHEMA_PROMPT.format(observation=observation)))

    def _find_neighbors(self, rec, k=5, exclude_self=True):
        scored = [(self._cos(rec.vec, r.vec), r) for r in self.records
                  if r.namespace == rec.namespace and (not exclude_self or r is not rec)]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [r for s, r in scored[:k] if s > 0.6]

    def _evolve(self, new_rec, siblings):
        decisions = json.loads(self.llm.invoke(EVOLVE_PROMPT.format(
            new=json.dumps(new_rec.note, ensure_ascii=False),
            candidates="\n".join(f"id={id(r)}: {json.dumps(r.note, ensure_ascii=False)}" for r in siblings),
        )))
        # 3) 把决定写回老笔记的 tags / context
        for d in decisions:
            for r in siblings:
                if str(id(r)) == d["id"] and d.get("should_update"):
                    r.note["tags"] = list(set(r.note.get("tags", []) + d.get("new_tags", [])))
                    r.note["context"] = d.get("new_context", r.note["context"])
```

这段 `AMemMemory` 的设计要点：

- **结构化笔记 (`note` 字段)**：与朴素向量记忆只存一段 text 完全不同——结构化属性让"高层标签检索"成为可能（例如可以单独按 `tags` 过滤"约束类记忆"）。
- **`_find_neighbors` 的阈值 0.6**：太低会让 evolve 噪声爆炸，太高就触发不了 evolve。原论文也提到这是关键超参。
- **`_evolve` 的"only-if-needed"原则**：让 LLM 自己判断 should_update，而不是无脑改写。这避免了"每来一条新笔记就把所有相似笔记翻一遍"的成本失控。
- **代价与收益**：写入成本是普通向量记忆的 2-3 倍（一次 structurize + 一次 evolve）；但论文实测在六个底座模型上多跳推理任务上能比 MemGPT / 朴素 RAG 高出 10-30 个百分点[6]。
- **典型局限**：JSON 输出不稳定时整条流水线会断；需要一层强 schema 校验或重试。生产里通常用 function-calling / Structured Output 把它锁死。

（5）参数化长期记忆：把记忆烧进权重

前面四种变体记忆都活在外部存储里，称为 **token-level memory**[1]。还有一类完全相反的方案：**直接微调模型权重，把"用户事实"或"任务经验"烧进参数本身**。这一类在 Hu et al. (2025)[1] 的分类里叫 **parametric memory**，代表工作有 Memory-R1 (Yan et al. 2025)[11]、Self-RAG 中的"参数化背景知识"等。

它的好处显而易见：**推理时不用任何检索**。坏处也显而易见：**写记忆 = 反向传播 = 慢且贵**。一个最小化的 LoRA 注入式实现的伪代码（仅示意，不可直接跑）：

```python
class ParametricMemory:
    """把用户事实烧成 LoRA adapter，按 user_id 切换。"""

    def __init__(self, base_model, lora_config):
        self.base = base_model
        self.adapters: dict[str, "LoRAState"] = {}

    def put(self, user_id: str, facts: list[str]):
        # 1) 准备一个微型训练集，把 facts 写成 instruct 格式
        samples = [{"input": f"关于该用户：", "output": f"{f}"} for f in facts]
        adapter = self.adapters.get(user_id, LoRA.init(self.base, lora_config))
        # 2) 短训若干步（典型: 几百 step，几分钟）
        adapter = LoRA.train(adapter, samples, steps=200)
        self.adapters[user_id] = adapter

    def generate(self, user_id: str, query: str) -> str:
        # 3) 推理时按 user_id 临时挂上对应 LoRA
        with self.base.with_adapter(self.adapters[user_id]):
            return self.base.generate(query)
```

这段 `ParametricMemory` 的设计要点：

- **每用户一个 LoRA**：避免不同用户的记忆相互污染。生产里通常配合 multi-LoRA serving（如 LoRAX）按租户切换，开销可以做到接近单一基座。
- **写入是离线流程**：不可能在对话过程中实时反向传播——最终方案通常是"对话先用 token-level 记忆累积，定期（比如每 7 天）把用户事实蒸馏进 LoRA 一次"。
- **优点**：推理零额外 token，跨任务自然泛化（"用户的语气 / 偏好"会被基础语言模型整体内化，而不是依赖检索器恰好召回）。
- **致命局限**：**遗忘困难**——用户撤回某条事实时，要么重训整个 LoRA，要么靠 prompt 补丁覆盖；这与 GDPR 的"被遗忘权"严重冲突。生产里通常**只用它存"很难变的偏好"**（写作风格、专业术语），不存"易变事实"。

（6）运行实例与分析

我们让同一个 Agent 在三组任务上跑：① 第 1 天告诉它"我妈叫 Linda、我用 Python"；② 第 7 天问无关问题，掺一些噪声；③ 第 30 天问"我妈叫什么 / 我用什么语言"。下面是四种 token-level 长期记忆变体的输出对照（参数化记忆因为离线训练耗时太久，这里不放）：

```text
📚 Day 1: put("user_alex", "我妈叫 Linda")
📚 Day 1: put("user_alex", "我用 Python 写后端")
📚 Day 2-29: put 287 条噪声 (闲聊、无关吐槽、新闻评论...)
📚 Day 30: 提问 "我妈叫什么 / 我用什么语言"

--- 变体 A: 朴素向量记忆 ---
🎯 召回 5: ['某新闻里提到的一个 Linda 老师...', '吐槽过 Python 慢', ...]
❌ 答错: "你妈叫 Linda 老师？" (语义最近但是错的)
⚠️ 失败模式: 名字撞脸 / 同义干扰

--- 变体 B: Generative Agents 三因子 ---
🎯 召回 5: ['[importance=0.9] 我妈叫 Linda', '[importance=0.8] 我用 Python 写后端', ...]
✅ 答对: "你妈叫 Linda；你用 Python"
⚠️ 关键: importance 在 Day 1 写入时被打成 0.9 / 0.8，30 天后仍能压过噪声

--- 变体 C: 反思型记忆 (在 B 基础上) ---
📝 Day 7 触发反思: "[REFLECTION] 用户家庭情况: 母亲 Linda; 技术栈: Python 后端"
🎯 召回 Top-1: 直接命中那条反思
✅ 答对: "你妈叫 Linda；你用 Python" (Top-1 一条就够)
⚡ 召回成本: 比 B 少 3 条 evidence，token 也少 60%

--- 变体 D: A-MEM Zettelkasten ---
📝 Day 1 写入时被结构化为:
   { tags: [家庭, 偏好], keywords: [Linda, 母亲], context: ... }
📝 Day 5 evolve: 老笔记 'Python' 与新笔记 '后端开发' 被链接，tags 合并
🎯 按 tags=[家庭] 过滤 → 1 条精确召回
✅ 答对，且检索路径完全可追溯
```

从上面的输出可以看到，长期记忆的几种变体清晰地展示了它们的核心差异：

1. **朴素向量记忆在"长程 + 高噪声"场景下脆弱**：纯语义相似度无法区分"重要的事 vs 名字相似的噪声"。这是为什么生产里几乎没人只用它——必然要叠加 importance 或 metadata 过滤。
2. **三因子打分把"重要性"前置到写入时**：写入花一次 LLM，回报是检索时 30 天后还能压过 287 条噪声。这在长程 Agent 里几乎是 ROI 最高的一笔投资。
3. **反思 + 演化让"记忆密度"自动提升**：变体 C 直接把"我妈叫 Linda + 我用 Python"压成一条反思，变体 D 把它们链成一个家庭/技术栈知识网。两者都让"召回质量"随时间反而变高，而不是变差。

由于模型与数据持续更新，你运行的结果可能与此不完全相同；但相对趋势（朴素向量 < 三因子 < 反思 ≈ A-MEM）在多篇论文上是稳定的[6][8][10]。

### 8.3.3 长期记忆变体的特点、局限性与调试技巧

通过亲手实现这五种变体并把它们摆在一起对比，我们应该对长期记忆的设计空间有了清楚的认识。

（1）长期记忆的主要特点

1. **写入策略比检索策略更重要**：朴素向量记忆和三因子记忆用的是同一种"cosine 相似度"，但因为前者写入时不打 importance，后者打了，最终行为差出几个数量级。**先想清楚 $\phi$，再去优化 $\sigma$**。
2. **"只增不改" vs "可演化"是一条分水岭**：append-only 简单稳定但记忆密度上不去；A-MEM 风格能演化但工程复杂度陡增——选哪一边取决于任务的"长度 / 噪声比"。
3. **结构化属性是几乎免费的午餐**：哪怕只在向量记忆基础上加 `(user_id, topic, importance, created_at)` 四个字段，召回质量就能显著提升——比换更大的 embedding 模型 ROI 高得多。

（2）长期记忆的固有局限性

1. **Cold start 问题**：Agent 刚部署时 $\mathcal{S}$ 是空的，再聪明的检索也召回不出东西。必须有一段"积累期"或"冷启动种子库"。
2. **遗忘是被低估的难题**：删除一条错误事实并不等于"模型忘了"——对应的 reflection、evolve 出来的链接都要追着改。这就是为什么"GDPR 友好的 Agent 记忆"目前还没有任何成熟方案[1]。
3. **检索质量随条数增长而退化**：100 条记忆里 cosine TopK 很准，10 万条里就开始"语义噪声"主导。生产里必然要叠 metadata 过滤、HNSW + 重排、甚至按 namespace 分库。

（3）调试技巧

- **永远先打印 Top-K 召回的具体内容**：不要只看"答案对不对"。如果答案对但 Top-1 是"语义巧合的噪声"，下次换个 query 就会翻车——这是隐性的 bug。
- **用"对抗事实"测试集**：构造若干"语义近但事实反"的 pair（"我喜欢咖啡 / 我喜欢茶"），观察召回是否选对的那个。这是评测向量记忆稳健性的金标准。
- **importance 的分数分布要画直方图**：好的 importance 模型应该是右偏分布——绝大多数事打 1-3，少数大事打 8-10。如果你的分数全堆在 5 附近，说明 LLM 在"打不出区分度"，prompt 要重写。
- **反思条目周期性抽检**：随机抽 20 条 `[REFLECTION]` 让人评分（0=废话，1=有用）。低于 50% 有用率就要重写 reflect_prompt 或调阈值。
- **A-MEM 的 evolve 要打日志**：每次 evolve 改了哪条笔记的什么字段都要记录。否则一旦记忆网络变得"看起来不对但说不清哪儿不对"，根本无法回溯——这是 append-only 系统天然不会有的复杂性税。
- **冷启动期手写 seed 记忆**：`put("user_xxx", "已知偏好: ...")` 哪怕只放两三条，新用户体验就不会"前 5 轮像失忆"。

到这里我们已经把短期与长期记忆的常见变体讲完了。下一节我们落到一个真实框架——LangGraph——看看这一整套理论怎么用 50 行代码组合起来。

---

## 8.4 落到框架：LangGraph 中的短期 + 长期记忆

讲完一堆变体，工程师最关心的问题是：**这些东西在主流框架里到底怎么写？** 我们以 LangGraph[2] 为例——它是目前生产里把"短期 / 长期"分得最清楚的框架，正好可以把 8.1 节的概念图直接落到代码上。

LangGraph 的两大记忆原语：

- **Checkpointer（短期）**：thread-scoped；每个节点执行后自动保存 `State` 快照。生产用 `PostgresSaver`，开发用 `InMemorySaver`。
- **Store（长期）**：cross-thread；按 `namespace` 组织的键值存储，原生支持向量检索。生产用 `PostgresStore`，开发用 `InMemoryStore`。

下面是一个把两者一起用的最小骨架（来自官方文档的精简版[2][12]）：

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.graph import StateGraph, MessagesState

DB_URI = "postgresql://localhost/agent_memory?sslmode=disable"

def call_model(state: MessagesState, config, *, store):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "facts")

    # 1) 长期记忆：检索与本轮 query 相关的过往事实
    last_msg = state["messages"][-1].content
    memories = store.search(namespace, query=last_msg, limit=5)
    fact_block = "\n".join(f"- {m.value['text']}" for m in memories)

    # 2) 拼 prompt：稳定的 system + 长期记忆 + 短期 messages（短期记忆由 checkpointer 自动维护）
    system = f"已知用户事实：\n{fact_block}\n请基于事实作答。"
    response = llm.invoke([{"role": "system", "content": system}] + state["messages"])

    # 3) 让模型主动决定哪些新事实要写入长期记忆（function-calling）
    for fact in extract_facts(response):
        store.put(namespace, str(uuid.uuid4()), {"text": fact})

    return {"messages": [response]}

with PostgresSaver.from_conn_string(DB_URI) as checkpointer, \
     PostgresStore.from_conn_string(DB_URI) as store:
    graph = (StateGraph(MessagesState)
             .add_node(call_model)
             .add_edge("__start__", "call_model")
             .compile(checkpointer=checkpointer, store=store))

    # 同一个用户的两个不同会话
    cfg1 = {"configurable": {"thread_id": "session-A", "user_id": "alex"}}
    cfg2 = {"configurable": {"thread_id": "session-B", "user_id": "alex"}}
    graph.invoke({"messages": [HumanMessage("我用 Python，喜欢 FastAPI")]}, cfg1)
    # 切到全新会话，长期记忆仍能跨过来
    graph.invoke({"messages": [HumanMessage("帮我搭个后端")]}, cfg2)
```

这段 LangGraph 集成代码的设计要点：

- **`thread_id` 与 `user_id` 是两个独立维度**：前者是短期记忆的"会话 ID"，后者是长期记忆的"租户 ID"。同一用户开新会话，短期重置但长期延续——这正是 8.1.1 节四个术语关系的工程映射。
- **检索发生在 prompt 拼接前**：8.1 节的公式 $\text{prompt}_t = f(\dots, M_{\text{long}}^{\text{retrieved}}, M_{\text{short}}, q_t)$ 在这里就是一个 `store.search → 拼 system → invoke` 的简单串联。
- **写入时机是显式的**：LangGraph **不会自动**抽事实进 store——这一步必须由开发者写在节点里（通常是 LLM function-calling 抽取）。这与"长期记忆是显式工程产物"的认识完全一致[2]。
- **两个 backend 共用 Postgres 实例**：生产里这是最省事的部署——同一个 DB 既存 checkpoint 又存 store；只是 schema 不同。

这段代码本身已经能跑通短期 + 长期的最小闭环。如果想升级到 8.3.2 节的高级变体，**只需要替换 `store.put` / `store.search` 的内部实现**——把朴素向量召回换成三因子打分、或包一层 A-MEM 的 evolve 逻辑——主图代码完全不动。这种"插拔式"也是为什么我们把 LangGraph 单独拎出来做收口的原因。

---

## 8.5 横向对比：九种变体的选型决策

把短期 + 长期一共九种变体放进同一张表，方便做选型：

表 8.1 短期记忆与长期记忆变体的横向对比

| 变体 | 类型 | 一句话描述 | 主要代价 | 适合场景 | 典型局限 |
| --- | --- | --- | --- | --- | --- |
| **朴素 ScratchPad** | 短期 | 全历史 append + 原样塞回 prompt | $O(t)$ token | 短轮次任务、KV 缓存命中要求高 | 窗口爆炸 |
| **滑动窗口** | 短期 | 只保留最近 N 条 | KV 命中率断崖式下降 | 客服 / 单任务 | 失忆早期事实 |
| **摘要缓冲** | 短期 | 触发时 LLM 摘要老内容 | 每次压缩 1 次 LLM + 缓存失配 | 中长对话 | 压缩有损、缓存友好性差 |
| **MemGPT 分层** | 短期 | working_context + FIFO + archival，由模型自管 | 实现复杂、function-calling 成本 | 长链路 / 长文档分析 | 频繁改写 working_context 仍伤缓存 |
| **朴素向量记忆** | 长期 | embedding + cosine TopK | 写入便宜、检索 O(N) | 简单个性化 / Mem0 baseline | 同义干扰、无衰减 |
| **Generative Agents 三因子** | 长期 | recency × importance × relevance | 写入多 1 次 LLM 打分 | 长程模拟 / 个性化助手 | importance 受 prompt bias |
| **反思型记忆 (Reflexion)** | 长期 | 累计触发反思，反思条目同入库 | 周期性 LLM 调用 | 跨任务经验沉淀 | "正确的废话"风险 |
| **A-MEM (Zettelkasten)** | 长期 | 结构化笔记 + 演化 evolve | 写入 2-3× 普通向量 | 长程 + 高密度知识 | JSON 不稳 / 评估难 |
| **参数化记忆 (LoRA)** | 长期 | 用户事实烧进权重 | 离线训练、推理切 LoRA | 写作风格 / 专业术语 | 难遗忘、不能实时写 |

选型时建议按下面三条决策线走：

- **任务跨度 < 一次会话** → 短期记忆够，按 token 预算选 ScratchPad / 滑窗 / 摘要 / MemGPT。
- **任务跨会话但同一用户** → 长期记忆 token-level 即可，按"噪声密度"从朴素向量逐级升到 Generative Agents → 反思 → A-MEM。
- **跨用户且高度个性化（写作风格、品牌口吻）** → 在 token-level 之外配一份参数化记忆（LoRA / DPO 微调），定期蒸馏。

最后值得再次强调：**这九种变体不是互斥的**。生产级 Agent 几乎都是组合方案——例如 Letta (MemGPT)[3] 的 production 实现里，working_context 是短期分层、archival_memory 是 token-level 长期向量、recall_memory 是 FIFO 队列；A-MEM 的笔记机制可以叠在 archival 上做 evolve。把它们当成"乐高积木"而不是"路线选择"，会让架构设计自由度大得多。

---

## 参考文献

[1] Hu Y, Liu S, Yue Y, Zhang G, Liu B, Zhu F, et al. *Memory in the Age of AI Agents: A Survey*. arXiv:2512.13564, 2025-2026. <https://arxiv.org/abs/2512.13564>

[2] LangChain Inc. *LangGraph Memory: Short-term and Long-term Memory*. LangChain Documentation, 2025. <https://docs.langchain.com/oss/python/langgraph/add-memory>

[3] Packer C, Wooders S, Lin K, et al. *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560, 2023. <https://arxiv.org/abs/2310.08560>

[4] Yao S, Zhao J, Yu D, et al. *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR, 2023.

[5] Zhang Z, Bo X, Ma C, et al. *A Survey on the Memory Mechanism of Large Language Model Based Agents*. arXiv:2505.00675, 2025. <https://arxiv.org/abs/2505.00675>

[6] Xu W, Liang Z, Mei K, Gao H, Tan J, Zhang Y. *A-MEM: Agentic Memory for LLM Agents*. NeurIPS, 2025. arXiv:2502.12110. <https://arxiv.org/abs/2502.12110>

[7] Mem0 Team. *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*. 2025. <https://github.com/mem0ai/mem0>

[8] Park J S, O'Brien J, Cai C J, et al. *Generative Agents: Interactive Simulacra of Human Behavior*. UIST, 2023. arXiv:2304.03442. <https://arxiv.org/abs/2304.03442>

[9] AgentPatterns. *Generative Agents Memory Stream: Three-Layer Architecture for Long-Running Agent Sessions*. 2024. <https://agentpatterns.ai/agent-design/generative-agents-memory-stream/>

[10] Shinn N, Cassano F, Berman E, et al. *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS, 2023.

[11] Yan S, et al. *Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning*. 2025.

[12] Letta (formerly MemGPT). *Letta API Documentation*. 2025. <https://docs.letta.com/concepts/memgpt/>
