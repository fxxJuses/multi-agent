# 第九章 工具与上下文的统一插座：模型上下文协议（Model Context Protocol）

在第八章里，我们把「短期记忆 / 长期记忆 / 上下文工程」拆开了：Agent 能不能接着聊、隔周还记不记得偏好，本质上是**状态存在哪、怎么被 $f_{\text{ctx-eng}}$ 拼进 prompt**。然而还有一类问题被刻意留到了本章：**模型以外的世界**——文件系统、Git、Slack、内部 Wiki、业务数据库——要以什么**统一形状**暴露给「任意一个」LLM 应用？

如果每个产品各自定义一套 HTTP 接口、再写一遍鉴权与工具描述，我们会掉进经典的 **$N \times M$ 集成困局**：$N$ 个数据源要对接 $M$ 个 AI 客户端，维护成本随乘积爆炸。Anthropic 于 2024 年 11 月开源的 **模型上下文协议 (Model Context Protocol, MCP)**[1]，目标就是把「给模型接上下文」这件事，收成**一层协议 + 几类原语**，让数据源作者写一次 **MCP Server**，IDE / 桌面助手 / 企业网关等 **MCP Host** 侧各接一次 **MCP Client** 即可复用。

如果说第八章的长期记忆像「随身笔记本」，那么 MCP 更像**国标电源插座**：笔记本、台灯、充电器都长成同一种插头，酒店房间不用为每种电器改墙里的线。

---

## 9.1 模型上下文协议（Model Context Protocol）

### 9.1.1 工作原理

在 MCP 出现之前，工程上常见三类做法：

- **厂商绑定的函数调用 (Function Calling)**：能力强大，但工具 schema、鉴权、错误码往往与某一云 API 强耦合，迁移成本高。
- **Chat 插件 / 自定义 Action 框架**：体验接近「装扩展」，但协议分裂，很难让同一扩展同时跑在本地 IDE 与另一个聊天产品里。
- **手写 REST / gRPC**：灵活，但每个 Host 都要重复实现：发现工具有哪些、参数怎么校验、流式进度怎么传、取消怎么做。

MCP 的巧妙之处在于：**把「谁发起连接、谁暴露能力、消息长什么样」一次性写进开放规范**，传输层采用 **JSON-RPC 2.0** 消息，连接是**有状态的**（可协商版本与能力），并显式区分三个角色[2]：

- **Host (宿主)**：用户直接与之交互的 LLM 应用（例如带 Agent 能力的 IDE）。负责安全策略、用户同意、生命周期。
- **Client (客户端)**：跑在 Host 进程里的连接器，代表 Host 与某个 Server 维持一条会话。
- **Server (服务器)**：对外暴露「可被模型使用的上下文与能力」——资源、提示模板、可执行工具等。

我们可以把一次完整的「模型想用工具」抽象成状态机里的一条边：Host 先让用户**授权**某 Server；Client 与 Server 做完 **initialize** 能力协商；模型侧从 **tools/list** 拿到 schema；模型产出 **tools/call**；Server 执行后把结构化 **Observation** 经协议返回，再由 Host 写回对话历史。若用极简符号记 Host 在第 $t$ 轮选择的工具名为 $u_t$，参数为 $\theta_t$，Server 返回内容为 $o_t$，则与 ReAct 式循环自然衔接：

$$(u_t,\ \theta_t) = \pi_{\text{LLM}}(q,\ h_{1:t-1},\ \mathcal{T}_{\text{MCP}}),\quad o_t = \text{MCP\_Server}(u_t,\ \theta_t)$$

其中 $\mathcal{T}_{\text{MCP}}$ 表示当前已连接 Server 提供的工具清单与资源摘要（由 Client 在 prompt 组装阶段注入）。**公式只是第八章轨迹 $h$ 的「外部观测」来源被标准化了**——直觉仍是：模型提议动作，环境返回观察。

这种机制特别适用于：

- 希望在 **Claude Desktop / Cursor / ChatGPT** 等多 Host 间**复用同一套连接器**的团队场景；
- 需要把 **本地文件、Git、数据库** 以受控方式接到模型，又不想把业务逻辑写进 prompt 的工程场景；
- 正在从「手写 ToolExecutor」演进到「可插拔工具市场」的 Agent 平台。

本章我们要建立的心智模型是：**先理解协议角色与原语，再用手写 JSON-RPC 消息走通一遍「协商 → 列工具 → 调用」**，最后落到安全与排错。我们不依赖某一版 SDK 细节（SDK 会变），但会指向官方规范入口[2]。

### 9.1.2 Server 暴露的三类「货架商品」

规范把 Server 侧能力分成三块[2]，便于 Host 决定如何渲染给用户 / 如何写进模型上下文：

| 原语 | 英文 | 直觉 |
| ---- | ---- | ---- |
| 资源 | Resources | 像「只读文件句柄」：URI 标识的一段上下文（文档、配置片段），供模型或用户引用 |
| 提示模板 | Prompts | 预置工作流 / 提示词骨架，常带参数，由用户显式选用 |
| 工具 | Tools | 模型可发起的副作用操作，带 JSON Schema 形参，执行结果结构化返回 |

与之对称，**Client 也可向 Server 暴露能力**（例如 **Sampling**：Server 在受控条件下反向请求 Host 侧的 LLM 补全；**Roots**：约定文件系统边界；**Elicitation**：向用户追问额外信息）[2]。这些是「双向插头」里容易读漏的一面：MCP 不只是「模型调外部」，也支持「外部在授权下请模型帮忙」。

### 9.1.3 与 LSP / OpenAPI 的直觉对照

社区常把 MCP 比作 **OpenAPI**：都是「描述机器可消费的能力边界」。差别在于 OpenAPI 描述的是 **HTTP 资源与动词**，而 MCP 描述的是 **围绕 LLM 会话生命周期组织的能力**（协商、列表、调用、取消、进度、日志），且默认假设 **JSON-RPC 会话** 而非无状态 REST。

另一常见类比是 **语言服务器协议 (Language Server Protocol, LSP)**：LSP 把「编辑器 ↔ 语言语义」从 $N\times M$ 里解救出来；MCP 则在「**模型应用 ↔ 数据与工具**」这一侧做类似的事[3]。

---

## 9.2 关键基础：一条 JSON-RPC 消息长什么样

在接 SDK 之前，我们建议先在纸上**看懂 envelope**。下面是一条最小的 `initialize` 请求骨架（配置型声明独立成块，占位符用 `{name}` 风格）：

```json
{
  "jsonrpc": "2.0",
  "id": "{request_id}",
  "method": "initialize",
  "params": {
    "protocolVersion": "{mcp_version}",
    "capabilities": {},
    "clientInfo": { "name": "{host_name}", "version": "{semver}" }
  }
}
```

- **`jsonrpc`**：固定为 `"2.0"`，表示遵循 JSON-RPC 2.0。
- **`id`**：请求与响应的配对键；通知类消息可为 `null`（视方法而定）。
- **`method`**：MCP 定义的方法名（如 `initialize`、`tools/list`、`tools/call`）。
- **`params`**：方法参数对象；具体字段以你实现的协议版本为准[2]。

这条消息的设计要点：

- **要点 (`initialize`)**：这是**有状态会话**的起点；之后才能安全地假设 Server 已准备好处理 `tools/*` 等调用。
- **要点 (`protocolVersion`)**：规范持续演进（例如 2025 年 11 月里程碑版带来了 OAuth 流程简化、工具标准化命名、Sampling 与工具组合等面向生产的能力[4]），客户端与服务器必须对齐可接受的版本窗口。
- **要点 (错误形态)**：JSON-RPC 层错误与 MCP 业务错误都会在 `error` 或 `result.isError` 等字段体现；排错时要分清「传输失败 / 方法不存在 / 工具执行失败」。

---

## 9.3 模型上下文协议 的编码实现（概念 walkthrough）

本节不绑定某一语言 SDK，我们用仓库内一个小脚本，把 **Host → Client → Server** 的消息流打印出来，帮助建立「和 ReAct 的 Action / Observation 如何对应」的直觉。完整脚本见 `example/mcp-jsonrpc-concept-demo.py`。

（1）Host：用户同意与能力边界

Host 的职责无法被协议替代：**显式同意**、日志与审计、决定哪些 Server 可被连接。演示里用一行输出表示「用户已授权」。

（2）Client：`initialize` 与能力协商

```python
import json

init_req = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "demo-host", "version": "0.1"},
    },
}
print(json.dumps(init_req, ensure_ascii=False))
```

这段 `initialize` 请求的设计要点：

- **`clientInfo`**：帮助 Server 记日志、做兼容分支（类似 HTTP `User-Agent`）。
- **`capabilities`**：声明 Client 侧可选特性（例如是否支持 Sampling）；空对象表示「最小实现」。
- **`protocolVersion`**：必须与 Server 支持的区间有交集；否则应在响应里看到协商失败或可降级策略。

（3）Server：声明 `tools` 并响应 `tools/list`

Server 在 `initialize` 的 `result.capabilities` 里告知「我有哪些大类能力」，再通过 `tools/list` 返回每个工具的 **name / description / inputSchema**（供模型做结构化工具选择）。

（4）Client：发起 `tools/call` 并把结果交给 Host

当 Host 内嵌的 LLM 决定调用 `read_file` 时，Client 组装 `tools/call`，`arguments` 必须满足 `inputSchema`。Server 返回的 `content` 列表即是对话里下一轮的 **Observation**。

（5）运行实例与分析

下面是一次真实的运行记录（在仓库根目录执行 `python example/mcp-jsonrpc-concept-demo.py`）：

```text
--- MCP 概念演示：Host / Client / Server 消息流 ---
✅ Host 已获取用户对「本地 demo-server」的连接授权
🧠 Client 发送 initialize（能力协商）
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "demo-host", "version": "0.1"}}}
👀 Server 返回 initialize 结果（声明可提供 tools）
{"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "demo-server", "version": "0.1"}}}
🔍 Client 列出工具：tools/list
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
👀 Server：当前注册 1 个 tool → `read_file`
{"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "read_file", "description": "Read a UTF-8 text file by path", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]}}
🎬 Host 侧 LLM 决定调用工具（模型输出 tool call）
🔍 Client 执行 tools/call：read_file
{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "read_file", "arguments": {"path": "./README.md"}}}
👀 Server 返回文本内容（截断示意）
{"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "# multi-agent\n...(truncated)..."}], "isError": false}}
🎉 Host 将 Observation 写回对话，供下一轮 LLM 生成最终答案
```

从上面的输出可以看到，这条 walkthrough 把 MCP 放进了我们熟悉的 **Thought → Action → Observation** 叙事里：

1. **协商阶段**对应「开工前检查脚手架是否齐」：`initialize` 与 `tools/list` 把 $\mathcal{T}_{\text{MCP}}$ 固定下来。
2. **调用阶段**对应 ReAct 的 **Action**：`tools/call` 携带 $(u_t,\theta_t)$。
3. **返回阶段**对应 **Observation**：`content` 列表被 Host 写回 $h$，供下一轮的 $\pi_{\text{LLM}}$ 使用。

由于规范版本与 Host 实现持续更新，你在本机对接真实 Server 时的字段顺序、可选能力集合可能与演示不完全相同；请以当前规范[2]为准。

---

## 9.4 工程视角：MCP 到底帮 Agent 开发者省了什么

走完消息流之后，我们再把视角放回 Agent 开发者。很多人第一次接触 MCP 时，会把它理解成「又一种 Function Calling」。这个理解有一半是对的：它确实服务于工具调用；但更准确地说，**MCP 标准化的是工具发现、工具描述、工具调用与工具结果回传这整条链路**。

如果说手写 function call 像是在每个 Agent 项目里单独焊一个插头，那么 MCP 更像把插头形状定成标准件：Agent/Host 侧实现一个通用 MCP Client，工具提供方实现 MCP Server。这样，一套浏览器、数据库、GitHub、Slack 工具就可以被多个 Host 复用，而不是每个 Agent 框架重新写一遍适配层。

（1）Agent 如何知道 Server 里有哪些工具

Agent 并不是天然知道 MCP Server 里有哪些工具。完整流程是：

```text
Host 连接 MCP Server
  -> MCP Client 发送 initialize
  -> Server 返回 capabilities
  -> MCP Client 调用 tools/list
  -> Server 返回 name / description / inputSchema
  -> Host 把可用工具整理成 LLM 可读的 tool schema
  -> LLM 在这些工具中选择是否调用
```

这里的关键点是：**发现到工具 ≠ 全部暴露给 LLM**。Host 仍然可以用普通代码、配置和权限策略做筛选，例如只允许当前用户看到 `browser_screenshot`，暂时不暴露高风险的 `browser_evaluate_js`。

```python
all_tools = mcp_client.list_tools()

visible_tools = [
    tool
    for tool in all_tools
    if tool.name in allowed_tools
    and user_has_permission(tool)
    and tool.risk_level != "dangerous"
]

response = llm.chat(messages=messages, tools=visible_tools)
```

这段伪代码的设计要点：

- **`list_tools`**：对应 MCP 的工具发现阶段，拿到的是 Server 声明的能力边界。
- **`visible_tools`**：这是 Host 侧的安全与产品策略，不一定需要另一个 LLM 参与。
- **`llm.chat(..., tools=...)`**：LLM 只看到被 Host 允许看到的工具，而不是 Server 暴露的全部工具。

（2）工具到底在哪里执行

MCP 调用时，真正执行函数的一方是 **MCP Server 进程**。但 Server 可以跑在本地，也可以跑在远端：

| 部署方式 | 执行位置 | 典型例子 | 主要取舍 |
| ---- | ---- | ---- | ---- |
| 本地 MCP Server | 用户机器 / 本地容器 | 文件系统、Git、本地 SQLite、Playwright | 可离线、靠近本地资源，但要维护安装与权限 |
| 远端 MCP Server | 云端 / 企业内网 | GitHub、Slack、Notion、企业知识库 | 易共享、易集中治理，但依赖网络与服务可用性 |

因此，使用 MCP 并不意味着「把函数实现下载到本地」。更像是 RPC：Client 发 `tools/call`，Server 在自己所在的环境里执行，再把结构化结果返回给 Host。若你安装了一个本地 Playwright MCP Server，那是你把 Server package 装到了本机；协议本身并不会自动把远端函数源码下发回来。

（3）以 Playwright MCP 为例：截图数据怎么走

Playwright 这类浏览器工具很适合说明「执行环境」的重要性。若 Playwright MCP Server 在本地运行，那么浏览器实例、页面状态、截图都在本机，调用链大致是：

```text
Host / Agent
  -> MCP Client
  -> 本地 Playwright MCP Server
  -> 本地 Chromium / Firefox / WebKit
```

这时不是 Host 先截图再上传给 Server，而是 Server 本地控制浏览器、执行截图，再把截图结果作为工具返回值交给 Host。若 Playwright MCP Server 部署在远端，则截图发生在远端浏览器实例中，Server 再把图片或资源引用返回给本地 Host。

这也带来一个安全结论：**浏览器 MCP Server 拥有很强的环境控制权**。它可能点击页面、读取文本、截图、输入表单，甚至复用某些登录态。因此 Host 应该限制可访问域名、浏览器 profile、文件上传、脚本执行与敏感操作确认。

（4）MCP 不能消灭工具实现成本

MCP 的价值不是让工具实现成本消失，而是把成本从「每个 Agent 内部重复写工具适配」迁移到「MCP Server 侧统一实现」：

- **自己写 MCP Server**：仍然要维护 Playwright / Git / 数据库等具体 function call，只是额外包上一层标准协议。
- **使用现成 MCP Server**：不用维护工具函数本身，但要维护安装、启动、版本、权限、网络和日志。
- **只服务一个本地小项目**：直接手写 function call 可能更简单，MCP 未必划算。

所以更准确的判断是：**MCP 在“可复用、可共享、跨 Host、跨团队”的场景里收益最大；如果工具只在一个本地 Agent 里使用，且不需要标准化生态，手写 function call 仍然是合理选择**。

（5）离线与网络连通性

MCP 本身不是必须联网的协议。本地 LLM + 本地 Host + 本地 MCP Server 可以形成完全离线的闭环：

```text
本地 LLM
  -> 本地 Host / Agent
  -> 本地 MCP Client
  -> 本地 MCP Server
  -> 本地文件 / 浏览器 / 数据库
```

但一旦 Server 在远端，网络连通性就会变成 Agent 的新依赖。离线、DNS 故障、远端限流、Server 版本升级，都可能导致工具不可用或行为变化。因此在生产系统里，我们通常会把工具分层：

- **核心本地能力**：文件、Git、shell、本地数据库、本地浏览器，优先本地 MCP Server 或直接 function call。
- **可共享外部能力**：GitHub、Slack、Notion、云数据库、企业知识库，适合远端 MCP Server。
- **关键路径能力**：若断网也必须可用，就要本地化 Server，或保留本地 fallback。

这一节可以用一句话收束：**MCP 标准化的是工具接入方式，不保证工具所在环境的可用性；当 Server 在远端时，网络与服务状态会成为 Agent 的新依赖。**

---

## 9.5 模型上下文协议 的特点、局限性与调试技巧

通过把 MCP 放进「协议消息 + Agent 轨迹」同一套语言里描述，我们应该能回答：**它解决了什么、没解决什么、线上出问题先看哪**。

（1）主要特点

1. **开放标准与多实现**：规范与 SDK 生态持续发展，降低「每个数据源 × 每个客户端」的重复劳动[1][2]。
2. **围绕会话组织原语**：Resources / Prompts / Tools 覆盖「上下文只读引用、人类触发模板、模型触发副作用」三类需求[2]。
3. **双向能力**：Sampling / Roots / Elicitation 让「Server 侧编排 + Host 侧模型与用户」形成闭环，而不只是单向 RPC[2]。

（2）固有局限性

1. **安全无法在协议层完全保证**：任意数据访问与代码执行路径都需要 Host 做同意、鉴权与沙箱；规范明确列出隐私与工具安全方面的实现方责任[2]。
2. **不是「替你做 RAG」**：MCP 解决的是**集成接口**；检索质量、分块、重排仍回到第六章、第七章那套系统设计与评测。
3. **版本与扩展并存**：企业 OAuth、跨应用访问等能力通过 SEP 与扩展推进[4]；读文档时要分清「核心规范」与「扩展草案」的成熟度。

（3）调试技巧

- **先分三层看错误**：JSON-RPC 传输层 → MCP 方法语义 → 具体工具业务逻辑（文件不存在、SQL 拒绝等）。
- **打印 `initialize` 协商结果**：确认 `protocolVersion` 与 capabilities  bitmask 是否符合预期。
- **把 `tools/list` 的 `inputSchema` 原样喂给模型侧单元测试**：很多「模型乱传参」来自 schema 过宽或示例不足。
- **对本地 Server 用最小复现**：用官方示例 Server 或本仓库 `example/mcp-jsonrpc-concept-demo.py` 对照，排除 Host 配置问题。
- **区分本地不可用与远端不可用**：本地 Server 先看进程、依赖和权限；远端 Server 还要看网络、鉴权、限流与服务版本。

### 9.5.1 横向对比：集成方式怎么选

| 方式 | 一句话 | 主要代价 | 适合场景 | 典型局限 |
| ---- | ------ | -------- | -------- | -------- |
| MCP | 会话式 JSON-RPC 原语 | 学习规范 + Host 安全责任 | 多 Host 复用、本地+远程统一 | 实现质量依赖生态 |
| 云厂商 Function Calling | 与某一 API 强绑定 | 供应商锁定 | 快速上云、托管工具 | 迁移与多云成本高 |
| 手写 REST | 完全自定义 | 设计与文档成本 | 强定制后端、已有网关 | 每个客户端重复造轮子 |

在掌握了「把外部世界接进模型上下文」的一条标准路径之后，若你正在实现多 Agent 系统，下一类自然问题是：**多个 MCP Client / 多个 Server 在同一 Host 内如何编排优先级、如何做工具路由与配额**——这已经超出协议本体，而进入平台工程范畴；我们可以在后续章节或实战笔记里再展开。

---

## 参考文献

[1] Anthropic. Introducing the Model Context Protocol. 2024-11-25. https://www.anthropic.com/research/model-context-protocol

[2] Model Context Protocol. Specification (latest). https://modelcontextprotocol.io/specification/latest

[3] Wikipedia contributors. Model Context Protocol. https://en.wikipedia.org/wiki/Model_Context_Protocol

[4] Model Context Protocol Blog. One Year of MCP: November 2025 Spec Release. 2025-11-25. https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/
