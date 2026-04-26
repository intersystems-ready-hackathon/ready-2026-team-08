Class Sample.Agent Extends %AI.Agent
{
    /// LLM Provider
    Parameter PROVIDER = "openai";

    /// LLM Model
    Parameter MODEL = "gpt-5-nano";

    /// Toolsets that the agent can use
    Parameter TOOLSETS = "Sample.ToolSet";
    
    
    /// System Prompt
    XData INSTRUCTIONS [ MimeType = text/markdown ]
    {
    # Sample Assistant

    You are a helpful assistant with access to a set of tools to interact with a database of people.
    }

    Method %CreateProvider() As %Status {
        Set apiKey = $System.Util.GetEnviron("OPENAI_API_KEY")
        If apiKey = "" {
            Quit $$$ERROR($$$GeneralError, "OPENAI_API_KEY is not set")
        }

        Set sc = $$$OK
        Try {
            Set settings = {"api_key": (apiKey)}
            Set ..Provider = ##class(%AI.Provider).Create(..#PROVIDER, settings)
        } Catch ex {
            Set sc = $$$ERROR($$$GeneralError, "Failed to create provider '" _ ..#PROVIDER _ "': " _ ex.DisplayString())
        }

        If $$$ISERR(sc) Quit sc
        Quit $$$OK
    }


}
