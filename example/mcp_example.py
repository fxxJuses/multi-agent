from fastmcp import FastMCP

mcp = FastMCP("演示 🚀")

@mcp.tool
def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b

if __name__ == "__main__":
    mcp.run()