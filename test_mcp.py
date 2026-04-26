import asyncio
import base64
from langchain_mcp_adapters.client import MultiServerMCPClient


AUTH_HEADER = base64.b64encode(b"SuperUser:SYS").decode("utf-8")
async def get_tools():
    client = MultiServerMCPClient(
        {
            "minimal": {
                "transport": "http",
                "url": "http://localhost:8080/mcp/sample",
                "headers": {"Authorization": f"Basic {AUTH_HEADER}"},
            }
        }
    )

    tools = await client.get_tools()
    for tool in tools: 
        print("- ", tool.name)
    print("Available MCP tools:")
    
    for tool in sorted(tools, key=lambda item: item.name):
        if tool.name == "mcp_sample_GetPeopleYoungerThan":
            try:
                out = await tool.ainvoke({"Age": 30})
                print(out)
            except Exception as e:
                print(f"Error invoking {tool.name}: {e}")
        elif tool.name == "mcp_sample_multiply":
            try:
                out = await tool.ainvoke({"a": 5, "b": 7})
                print(out)
            except Exception as e:
                print(f"Error invoking {tool.name}: {e}")
        




    
    return tools 

async def main():
    tools = await get_tools()
    # print(tools)
    if tools== [] or tools is None:
        print("No tools available. Exiting.")
        return


if __name__ == "__main__":
    asyncio.run(main())




