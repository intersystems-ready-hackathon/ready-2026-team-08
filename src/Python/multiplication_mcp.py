import logging
import sys
import iris
from mcp.server.fastmcp import FastMCP

# ---- MCP STDIO RULE ----
# stdout MUST contain ONLY MCP protocol JSON
# so move everything else off it
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP('multiplication', log_level="ERROR")

@mcp.tool()
async def multiply(a: int, b: int) -> int:
    """Multiply 2 integers and return the result.
    Args:
        a: First integer to multiply
        b: Second integer to multiply"""
    return a * b


if __name__ == '__main__':
    mcp.run(transport='stdio')
