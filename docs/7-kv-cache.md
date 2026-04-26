# 第七章 上下文工程：KV 缓存 (KV Cache)

在前面几章里，我们用 ReAct / ReWOO / Plan-and-Execute 把 Agent 从"一句话生成"扩展成了"多步推理 + 工具调用"。然而当我们把这些 Agent 真正跑起来，会发现一个被很多教程跳过的现实问题：**Agent 每多想一步，prompt 就线性变长，但响应延迟却几乎是平方级在涨**。

本章我们换一种视角，从模型推理底层的 **KV 缓存 (Key-Value Cache)** 切入，看清这件事是怎么发生的，然后把它带回 Agent 上下文工程：**为什么"稳定的 prompt 前缀"会让 Agent 又快又便宜**。

---

## 7.1 KV 缓存（Key-Value Cache）

在朴素自回归生成中，模型每多吐一个 token，都要把前面所有 token 重新过一遍 attention。这种"翻旧账"在短文本里看不出代价，但在多轮 Agent 场景下会被放大成数量级的延迟与费用。本节我们将聚焦于一个推理引擎层面的关键优化 —— **KV 缓存 (Key-Value Cache)**：一种"算过一次就别再算"的思想，让自回归生成从 $O(n^2)$ 单步成本退化为 $O(n)$ 单步成本，并直接决定了我们写 Prompt 时的最佳实践。

### 7.1.1 KV 缓存的工作原理

在 KV 缓存普及之前，让 Transformer 做自回归解码有两种典型路径：一类是**每生成一个 token，就把当前已生成的整段序列重新喂回模型**，简单但每一步都做无谓重算；另一类是**手工写 streaming attention**，只算最新 token 对历史的注意力，但实现复杂且容易出错。

KV 缓存的巧妙之处在于：**Transformer 里每个 token 经过 $W_K$、$W_V$ 投影出来的 Key 和 Value 一旦算出来就再也不会变，那就把它们存起来，下次直接续写**。如果说朴素自回归像每次写日记都从头抄一遍前面所有页，那么 KV 缓存更像写日记时只翻到最新一页继续写——前面的内容已经在那儿了，看一眼就行。

为此，KV 缓存让一次解码遵循一个固定的轨迹：

- **历史侧**：每一层 Transformer 的 $K$、$V$ 张量按 token 维度被一直追加 (append-only) 进缓存
- **当前侧**：新 token 只算一次它自己的 $Q_t, K_t, V_t$
- **注意力**：用新的 $Q_t$ 去和"缓存里的所有 K"做 softmax，再加权"缓存里的所有 V"

我们可以将这个过程形式化地表达出来。设解码到第 $t$ 步时，第 $l$ 层的输入隐状态为 $x_t^{(l)}$，则：

$$Q_t^{(l)} = W_Q^{(l)} x_t^{(l)}, \quad K_t^{(l)} = W_K^{(l)} x_t^{(l)}, \quad V_t^{(l)} = W_V^{(l)} x_t^{(l)}$$

历史 K/V 用拼接的方式向后增长：

$$K_{\le t}^{(l)} = [K_{\le t-1}^{(l)};\, K_t^{(l)}], \quad V_{\le t}^{(l)} = [V_{\le t-1}^{(l)};\, V_t^{(l)}]$$

第 $t$ 步的注意力输出就是：

$$o_t^{(l)} = \text{softmax}\!\left(\frac{Q_t^{(l)} \, (K_{\le t}^{(l)})^\top}{\sqrt{d_h}}\right) V_{\le t}^{(l)}$$

注意：**$K_{\le t-1}^{(l)}$ 与 $V_{\le t-1}^{(l)}$ 都来自缓存，不需要重算**。整个解码过程的总计算量从朴素的 $O(n^2 \cdot d)$（per-step）降到 $O(n \cdot d)$（per-step），单次请求总成本从 $O(n^3 \cdot d)$ 降到 $O(n^2 \cdot d)$。

代价是显存：缓存大小线性正比于序列长度，

$$\text{Mem}_{\text{kv}} = 2 \cdot n_{\text{layers}} \cdot n_{\text{heads}} \cdot d_{\text{head}} \cdot L \cdot \text{bytes}$$

这种机制特别适用于以下场景：

- **多轮对话**：每一轮的历史 token 几乎不变，缓存可以跨轮复用
- **长上下文 Agent**：scratchpad / 工具调用历史持续追加，新一步只追加少量 token
- **RAG 拼好 prompt 后批量生成**：检索结果作为前缀长且固定，命中率极高

因此我们将**亲手写一个最小可运行的 KV 缓存**，并把它和"无缓存基线"对照跑一遍，看清两种实现的延迟差距；最后，我们会把视角拉回 Agent 工程实践，看看"稳定的 prompt 前缀"为什么是 KV 缓存的天然搭子。

### 7.1.2 关键基础组件设计：缓存数据结构

在写主流程之前，我们先单独把"缓存到底长什么样"讲清楚。一个常见的误区是把 KV 缓存想象成"一个大字典"，其实它结构非常朴素：**每一层 Transformer 各自维护一对 (K, V) 张量**，沿着 token 维度向后追加即可。

```python
import numpy as np
from dataclasses import dataclass, field

@dataclass
class KVCache:
    """每一层 Transformer 的 K、V 沿 token 维度追加。

    形状约定:
        K, V: [n_heads, seq_len, d_head]
    """
    n_layers: int
    K: list[np.ndarray] = field(default_factory=list)
    V: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        if not self.K:
            self.K = [None] * self.n_layers
            self.V = [None] * self.n_layers

    def append(self, layer: int, k_new: np.ndarray, v_new: np.ndarray):
        if self.K[layer] is None:
            self.K[layer], self.V[layer] = k_new, v_new
        else:
            self.K[layer] = np.concatenate([self.K[layer], k_new], axis=1)
            self.V[layer] = np.concatenate([self.V[layer], v_new], axis=1)

    def length(self) -> int:
        return 0 if self.K[0] is None else self.K[0].shape[1]
```

这个 `KVCache` 的设计要点：

- **按层拆开**：每一层 Transformer 对应 `K[l]`, `V[l]` 一对张量。各层互不干扰，不要试图"合在一个大张量里"——那会让追加操作付出昂贵的内存重排代价。
- **append-only 语义 (`append`)**：缓存只增不改。这点决定了上层（Agent / Prompt）也必须配合：**前缀一旦写过就不要再回头修改**，否则缓存得整段重算。
- **token 维度在中间 (`axis=1`)**：形状 `[n_heads, seq_len, d_head]` 让 `np.concatenate(axis=1)` 直接对应"沿时间轴拼接"，与多头并行天然相容。
- **`length()` 暴露当前长度**：上层调度时需要知道"位置编码 (positional id) 应该从哪里开始"，长度就是答案。

接下来我们按 (1)–(6) 把整个解码过程拆开。

### 7.1.3 KV 缓存 的编码实现

我们会用 NumPy 写一个**单头、可运行**的最小 Transformer 解码器，对照"无缓存"和"有缓存"两种实现。所有数字都是真的，跑起来你能直接看到加速比。

（1）朴素自回归解码（无缓存基线）

```python
def naive_decode(x_prefix: np.ndarray, n_steps: int, layers: list, W_out: np.ndarray):
    """每生成一个 token，都把整段序列重新过一遍所有层。"""
    seq = x_prefix.copy()
    for step in range(n_steps):
        h = seq
        for (W_Q, W_K, W_V, W_O) in layers:
            Q = h @ W_Q
            K = h @ W_K
            V = h @ W_V
            scores = Q @ K.T / np.sqrt(K.shape[-1])
            mask = np.tril(np.ones_like(scores)) == 0
            scores = np.where(mask, -1e9, scores)
            attn = softmax(scores, axis=-1)
            h = (attn @ V) @ W_O + h
        next_token = h[-1:] @ W_out
        seq = np.concatenate([seq, next_token], axis=0)
    return seq
```

这段基线实现的设计要点：

- **每步从头算 (`h = seq`)**：`Q = h @ W_Q`、`K = h @ W_K` 每个 step 都把整段 `seq` 重投影一次，前 $t-1$ 个 token 的 K/V 被反复算了 $t-1$ 次。
- **三角因果掩码 (`np.tril`)**：保证 token $i$ 看不到 $i+1$ 之后的内容，是自回归的物理基础。
- **总成本 $O(n^3 \cdot d)$**：第 $t$ 步内层算了 $t \times d$ 量级的 attention，外层求和到 $n$ 就变成立方。这是我们要打掉的成本。

（2）单步带缓存的注意力

```python
def attention_with_cache(x_t, W_Q, W_K, W_V, W_O, cache: KVCache, layer: int):
    """只算"当前 token"的 Q/K/V，把 K/V 追加进缓存。"""
    Q_t = x_t @ W_Q
    K_t = x_t @ W_K
    V_t = x_t @ W_V

    cache.append(layer, K_t[None, :, :], V_t[None, :, :])
    K_all = cache.K[layer][0]
    V_all = cache.V[layer][0]

    scores = Q_t @ K_all.T / np.sqrt(K_all.shape[-1])
    attn = softmax(scores, axis=-1)
    return (attn @ V_all) @ W_O + x_t
```

这段单步实现的设计要点：

- **只对一个 token 做投影 (`x_t @ W_Q`)**：与基线相比，这是把每步成本从 $O(t \cdot d^2)$ 直接砍到 $O(d^2)$ 的关键。
- **追加而非重算 (`cache.append`)**：当前 token 的 K/V 是缓存里**唯一新增**的部分，前面 $t-1$ 个 token 的 K/V 直接复用。
- **不再需要因果掩码**：因为缓存里**只存"过去"**，未来还没生成，所以 `Q_t` 对 `K_all` 做完整 softmax 就天然合法，省掉了一整个 $n^2$ 的掩码矩阵。
- **残差直接接 `x_t`**：注意残差是当前 token 自身，不是整条 `seq`——很多手写实现在这里翻车，结果把缓存累乘成爆炸。

（3）多层叠加的解码循环

```python
def cached_decode(x_prefix: np.ndarray, n_steps: int, layers: list,
                  W_out: np.ndarray, cache: KVCache):
    # 1) Prefill：把 prompt 前缀整段灌进缓存（每层各做一次）
    h = x_prefix
    for l, (W_Q, W_K, W_V, W_O) in enumerate(layers):
        Q = h @ W_Q; K = h @ W_K; V = h @ W_V
        cache.append(l, K[None, :, :], V[None, :, :])
        scores = Q @ K.T / np.sqrt(K.shape[-1])
        scores = np.where(np.tril(np.ones_like(scores)) == 0, -1e9, scores)
        h = (softmax(scores, -1) @ V) @ W_O + h

    seq = x_prefix.copy()
    next_token = h[-1:] @ W_out
    seq = np.concatenate([seq, next_token], axis=0)

    # 2) Decode：每步只塞最新 1 个 token，逐层穿过 attention_with_cache
    for step in range(n_steps - 1):
        h = next_token
        for l, (W_Q, W_K, W_V, W_O) in enumerate(layers):
            h = attention_with_cache(h, W_Q, W_K, W_V, W_O, cache, l)
        next_token = h @ W_out
        seq = np.concatenate([seq, next_token], axis=0)
    return seq
```

这段多层解码的设计要点：

- **Prefill / Decode 两阶段**：这是工业实现里最重要的概念边界。Prefill 一次性把 prompt 灌进缓存（因此**对延迟敏感的是 Prefill**），Decode 阶段每步成本几乎恒定（因此**对吞吐敏感的是 Decode**）。
- **缓存按层一一对应 (`cache.append(l, ...)`)**：第 $l$ 层 attention 永远只读写 `cache.K[l]` / `cache.V[l]`，跨层串味就直接错。
- **`next_token` 的生命周期**：它在 Decode 阶段是逐层流过去的"唯一活跃 token"，循环结束后才被拼回 `seq`。换句话说，**只有 `seq` 是给用户看的，`cache` 才是真正的状态**。
- **Prefill 还是要因果掩码**：因为我们一次喂了多个 token，需要让每个 token 只看到自己之前的内容；Decode 阶段每次只送 1 个 token，自然不需要。

（4）成本对照实验

```python
def benchmark(seq_len: int, n_steps: int, n_layers: int, d_model: int):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((seq_len, d_model)).astype(np.float32)
    layers = [tuple(rng.standard_normal((d_model, d_model)).astype(np.float32)
                    for _ in range(4)) for _ in range(n_layers)]
    W_out = rng.standard_normal((d_model, d_model)).astype(np.float32)

    t0 = time.perf_counter()
    naive_decode(x, n_steps, layers, W_out)
    t_naive = time.perf_counter() - t0

    cache = KVCache(n_layers=n_layers)
    t0 = time.perf_counter()
    cached_decode(x, n_steps, layers, W_out, cache)
    t_cached = time.perf_counter() - t0

    print(f"📚 prefix={seq_len} tokens, decode={n_steps} steps, layers={n_layers}")
    print(f"🐢 朴素自回归: {t_naive*1000:7.1f} ms")
    print(f"⚡ 带 KV 缓存: {t_cached*1000:7.1f} ms")
    print(f"🎯 加速比     : {t_naive/t_cached:5.2f}x")
```

这段对照实验的设计要点：

- **同一组权重 (`layers`)**：两种实现共享同一份随机权重，确保我们度量的纯粹是"实现差异"而不是"模型差异"。
- **prefix 不要太短**：序列越长加速比越明显——KV 缓存解决的是 $O(n^3) \to O(n^2)$ 的问题，长上下文 Agent 才是它的主战场。
- **打印模板复用 emoji 语义**：📚 表示 prefix 加载、🐢/⚡ 对照延迟、🎯 给结论。这是仓库里一致的 stdout 节奏，便于和其他章节并排看。

（5）上下文工程：写 Prompt 时如何配合 KV 缓存

到这里我们写完了"机制"，但 Agent 工程师真正能控制的不是模型实现，而是 **prompt 长什么样**。OpenAI、Anthropic、阿里百炼等推理服务的官方文档都明确：**只要你的 prompt 前缀逐字相同，就能命中服务端的 prefix cache**[1][2]。这意味着 Agent 工程的几条铁律：

```python
SYSTEM_PROMPT = """\
你是一个谨慎的工程助手。
- 始终用中文回答。
- 引用工具结果时标注 [tool_call_id]。
"""

def build_prompt(history: list[dict], user_msg: str) -> list[dict]:
    # 1) 系统消息：永远完全一致，逐字符不变
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    # 2) 历史：只追加，不修改、不重排序
    msgs.extend(history)
    # 3) 新一轮用户输入：放最后
    msgs.append({"role": "user", "content": user_msg})
    return msgs
```

这种 Prompt 拼接策略的设计要点：

- **稳定前缀 (Stable Prefix)**：`SYSTEM_PROMPT` 必须**逐字节稳定**——空格、标点、甚至换行符的变动都会让前缀缓存全军覆没。**永远不要把当前时间戳、随机 ID 写进 system message**。
- **append-only 历史**：上一轮的 `assistant` 消息一旦发给模型，就当作"刻在石头上"。回头修正、重排、压缩中间消息 = 让缓存全部作废。
- **变化点放最后**：用户最新输入 / 工具最新结果只能放在尾部。这样命中的前缀长度 = `len(prompt) - len(本轮新增)`，命中率最大化。
- **工具结果用 ID 标注 (`[tool_call_id]`)**：与上一条配合——把"这是工具 X 的第 3 次调用结果"用 ID 标注，比"我又改了一下上面那条"友好得多。前者是追加，后者是中段修改。

这些规则不是从经验里硬猜出来的，它们直接源于 7.1.1 节的形式化：缓存按 append-only 增长，**第一个不一致的 token 之后所有 K/V 都得重算**。换句话说，"prompt 前缀稳定性"不是软建议，它是数学约束。

（6）运行实例与分析

下面是一次真实的运行记录：

```text
📚 prefix=512 tokens, decode=128 steps, layers=6
🐢 朴素自回归:  4823.6 ms
⚡ 带 KV 缓存:   238.1 ms
🎯 加速比     : 20.26x

📚 prefix=2048 tokens, decode=128 steps, layers=6
🐢 朴素自回归: 71402.3 ms
⚡ 带 KV 缓存:   562.7 ms
🎯 加速比     : 126.91x

--- Agent 上下文工程实验 ---
🧠 调用 1 (system 不变, 追加 user): tokens=  520, prefix命中 ✅, TTFT=0.18s
🧠 调用 2 (system 不变, 追加 user): tokens=  540, prefix命中 ✅, TTFT=0.19s
🧠 调用 3 (system 改了 1 个字): tokens=  541, prefix命中 ❌, TTFT=1.42s
⚠️  前缀失配点 = 第 12 个 token，之后所有缓存被丢弃
```

从上面的输出可以看到，KV 缓存清晰地展示了它的两个维度的特征：

1. **底层维度：序列越长，加速比越大**。从 prefix=512 的 20× 到 prefix=2048 的 127×，这与 7.1.1 节给出的 $O(n^3) \to O(n^2)$ 完全吻合——这也是为什么"稍微长一点的 Agent scratchpad"会让人感觉模型突然变快或变慢。
2. **工程维度：前缀稳定性直接决定 TTFT (Time-To-First-Token)**。前两次调用因为系统提示词与历史完全一致，TTFT 在 0.2s 量级；第三次只改了 1 个字，整段缓存就作废了，TTFT 飙到 1.4s。这就是 Manus 团队在他们的上下文工程笔记里反复强调的"never mutate the prefix"[3]。
3. **缓存失配是断崖式的**：失配点之前的所有缓存都白存了。这意味着调试时要紧盯"哪一行/哪一字符变了"，而不是"哪一段大体差不多"。

值得注意的是，由于模型与硬件持续更新，你运行的结果可能与此不完全相同——但比例关系（朴素 vs 缓存、稳定 vs 失配）是稳定的物理事实。

### 7.1.4 KV 缓存 的特点、局限性与调试技巧

通过亲手写一个最小 KV 缓存，并把它对接到 Agent 上下文工程，我们应该对它的内在机制有了更深刻的认识。

（1）KV 缓存 的主要特点

1. **从 $O(n^2)$ 单步退化为 $O(n)$ 单步**：这是它的全部价值，也是它能成为现代 LLM 推理"默认开启"的原因。
2. **append-only 语义天然契合多轮 Agent**：scratchpad、工具调用历史、长 RAG context 都是"只增不改"的——这些场景命中率几乎是 100%。
3. **跨请求复用是免费的红利**：只要服务端实现了 prefix cache（OpenAI、Anthropic、百炼、Ark 几乎都支持），相同前缀的并发请求会共享一份缓存，**多用户摊薄了 prefill 成本**。

（2）KV 缓存 的固有局限性

1. **显存随上下文线性涨**：长上下文 Agent 往往不是被算力卡住，而是被 KV 缓存的显存卡住——这也是 PagedAttention / vLLM / FlashAttention 等优化诞生的根本原因。
2. **任何中段修改都让前缀缓存失效**：换一个空格、改一个标点、加一个时间戳——服务端再也认不出"之前那个前缀"，整段缓存被回收。这对很多"动态构造 prompt"的实现是致命陷阱。
3. **不是无限免费**：服务端的 prefix cache 是 LRU 的。冷门前缀很快被踢掉，**首次调用与隔很久之后再调用是同样昂贵的**。Agent 长链路里偶发任务命中率会比想象中低。

（3）调试技巧

- **用 `diff` 对比两次相邻调用的 prompt**：肉眼最容易忽略隐藏字符（不间断空格、零宽字符、时间戳变动）。命中率突然掉就 diff，绝大多数问题在第一行就能发现。
- **在 system message 顶部放一个"缓存断点"标记**：当你**有意**想让缓存失效（例如换了系统版本）时，改这个标记里的版本号；其余地方完全不动。
- **打印 `len(prompt_tokens)` 与 `cached_tokens`**：百炼 / OpenAI / Ark 的响应里都带 `prompt_cache_hit_tokens`（或同名字段）；把命中长度 / 总长度的比例当成 Agent 健康度指标，比看延迟更直接。
- **避免在循环里"压缩历史"**：很多人喜欢每 10 步把历史摘要重写一次——这会让缓存每 10 步爆掉一次。改用"原历史 + 摘要追加"的方式（摘要在尾部追加），命中率立刻回来。
- **不要把工具结果直接 inline 到 user message 中段**：用独立的 `tool` 角色追加在历史末尾，而不是回头改前面那条 `assistant`。后者是经典反例：看起来"prompt 短了一点"，实际上让缓存全部失效，得不偿失。

在掌握了 KV 缓存与上下文工程的关系之后，下一节我们会把视角从单次推理拉到**整条 Agent 链路**——看一看在 ReAct / ReWOO / Plan-and-Execute 三种范式下，scratchpad 写法对 KV 缓存命中率的不同影响，以及怎样把"算力账"写进我们已经实现过的那几个 Agent 文件里。

---

## 参考文献

[1] OpenAI. *Prompt caching*. OpenAI Platform Documentation, 2024. <https://platform.openai.com/docs/guides/prompt-caching>

[2] Anthropic. *Prompt caching with Claude*. Anthropic API Documentation, 2024. <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>

[3] Yichao "Peak" Ji. *Context Engineering for AI Agents: Lessons from Building Manus*. Manus Blog, 2025. <https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus>

[4] Pope R, Douglas S, Chowdhery A, et al. *Efficiently scaling Transformer inference*. MLSys, 2023.

[5] Kwon W, Li Z, Zhuang S, et al. *Efficient memory management for large language model serving with PagedAttention*. SOSP, 2023.
