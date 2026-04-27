/// Wraps IRIS AI Hub to answer natural-language questions about system logs.
/// Called by the Python MCP tool (event_log_mcp.py) via iris.cls().
Class Sample.LogAnalyzer Extends %AI.Agent
{

Parameter PROVIDER = "openai";

Parameter MODEL = "gpt-5-nano";

/// No toolsets needed — this agent only answers questions about supplied text.
Parameter TOOLSETS = "";

XData INSTRUCTIONS [ MimeType = text/markdown ]
{
# IRIS Log Analyst

You are an InterSystems IRIS database administrator assistant.
You will be given recent IRIS system log entries and a question.
Answer in plain English. Highlight errors, warnings, and important events.
If you see a root cause or a repeated pattern, explain it clearly.
Keep your answer concise and actionable.
}

Method %CreateProvider() As %Status
{
    Set apiKey = $System.Util.GetEnviron("OPENAI_API_KEY")
    If apiKey = "" Quit $$$ERROR($$$GeneralError, "OPENAI_API_KEY is not set")

    Set sc = $$$OK
    Try {
        Set settings = {"api_key": (apiKey)}
        Set ..Provider = ##class(%AI.Provider).Create(..#PROVIDER, settings)
    } Catch ex {
        Set sc = $$$ERROR($$$GeneralError, "Failed to create provider: " _ ex.DisplayString())
    }
    Quit sc
}

/// Entry point called from Python: returns AI answer as a plain string.
ClassMethod QueryLogs(query As %String, logs As %String) As %String
{
    Set agent = ##class(Sample.LogAnalyzer).%New()
    Set sc = agent.%CreateProvider()
    If $$$ISERR(sc) Quit "Error creating AI provider: " _ $SYSTEM.Status.GetErrorText(sc)

    Set fullPrompt = "Question: " _ query _ $CHAR(10,10)
                   _ "IRIS System Logs:" _ $CHAR(10) _ logs

    Set response = ""
    Try {
        Set response = agent.Chat(fullPrompt)
    } Catch ex {
        Set response = "Error during AI call: " _ ex.DisplayString()
    }
    Quit response
}

}
