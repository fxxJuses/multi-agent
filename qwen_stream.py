from llm import LLM


def main() -> None:
    llm = LLM(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        system_prompt="You are a helpful assistant.",
    )

    print("AI: ", end="", flush=True)
    response = llm.stream_chat("请介绍一下自己")

    if response.usage:
        print("\n--- 请求用量 ---")
        print(f"输入 Tokens: {response.usage.prompt_tokens}")
        print(f"输出 Tokens: {response.usage.completion_tokens}")
        print(f"总计 Tokens: {response.usage.total_tokens}")

    # print(f"\n--- 完整回复 ---\n{response.content}")


if __name__ == "__main__":
    main()