Class Sample.ToolSet Extends %AI.ToolSet [DependsOn=Sample.Tools]
{
    XData Definition
    {
        <ToolSet>
            <Description>Sample Toolset</Description>
            <Include Class="Sample.Tools"></Include>
            <MCP Name="PythonServer">
                <Stdio Executable="/usr/irissys/bin/irispython"
                Args="/home/irisowner/dev/src/Python/multiplication_mcp.py" />
            </MCP>
            <MCP Name="EventLogServer">
                <Stdio Executable="/usr/irissys/bin/irispython"
                Args="/home/irisowner/dev/src/Python/event_log_mcp.py" />
            </MCP>
        </ToolSet>
    }    
}

