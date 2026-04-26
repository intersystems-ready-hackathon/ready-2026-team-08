# TEAM 08 - [ Name of Project ] 

## Project Summary 

[ Add a 1-2 sentence summary of your project here - i,e. what is the high level goal of your project ] 

## Technical Details

[ Add details on the technical implementation of your project here ] 

## Setup Instructions [Optional]

[ If you have time, please add details on how someone else can start using your project here ]

## Publicly accessible statement

**Delete as appropriate**: 
- We are happy for our project to be publicly visable after the event (you will remain repo admins) 
- We would like our project to be made private at the end of the event


# Template Instructions (feel free to delete)


This repo provides a template to kickstart development with AI Hub. 

## Contents

- **./skills** - agent skills with information on using AI hub for AI agents. Move these to a suitable location for your preferred AI coding agent. 
- **./src/Sample** - Basic sample classes for tools, toolsets, agents and MCP servers. These are installed with zpm when the container is build.  
- **./src/Python** - An example stdio MCP server defined in Python and used in the IRIS Toolsets 
- **Datasets.md** - Notes on some datasets or tools to import datasets available on Open Exchange for easy install 

## Using the template

### Download AI Hub Container


1. Download an AI Hub container from the [Early Access Program Portal](https://evaluation.intersystems.com/Eval/early-access/AIHub). The docker-containers end with `docker.tar.gz`, ensure you choose the version suitable for your operating system (arm64 for macOS).

OR 

1. Copy AI Hub Container from your Flash Drive

2. Load the image with: 

    ```bash
    docker load -i /path/to/irishealth-community-2026.2.0AI.158.0-docker.tar.gz
    ```

    Once it's complete you should see `Loaded image: docker.iscinternal.com/docker-intersystems/intersystems/iris-community:2026.2.0AI.158.0` (if not you can use `docker images` to find the image name). 

3. Change the Image name in the [Dockerfile](./Dockerfile) to match your version and operating system (image name printed above).



### Build Template Repo

4. Clone this repo: 

```bash
git clone https://github.com/intersystems-community/ai-hub-dev-template
cd ai-hub-dev-template
```

5. (Optional) Add an OPENAI_API_KEY to a file called .env in this repo. You can see an example in .env.example.

6. Build the container with: 

```bash
docker-compose up -d --build 
```

## Using IRIS AI Hub Container

### Accessing IRIS 

You can find the Management Portal at http://localhost:52773/csp/sys/UtilHome.csp.

Login with: 
    - SuperUser / SYS

You can access the IRIS Terminal with:

```objectscript
docker-compose exec -it iris iris session iris
```

or the bash terminal with:

```bash
docker-compose exec -it iris iris session iris
```

### Testing Sample agent 

There is a basic agent in src/Sample.Agent, a simple way to use it from objectscript is to run the following (note this does require an OPENAI_API_KEY to be added to .env before running th container). 

```objectscript
set $NAMESPACE= "IRISAPP"
Set agent = ##class(Sample.Agent).%New()
Set sc = agent.%Init()
write:sc'=1 $SYSTEM.Status.GetErrorText(sc), !

Set session = agent.CreateSession()

// This requires using both tools defined in Sample.Tools and packaged in Sample.ToolSet
Set request = "Add a person named Alice aged 30, and then get people younger than 35."
Set response = agent.Chat(session, request)
write response.content
```

### Test MCP Server

The build process installs an MCP server web application at http://localhost:52773/mcp/sample. You can check this MCP server is running by going to http://localhost:52773/mcp/sample/v1/services. 

For the MCP Server to be usable, there is an additional step of starting this via a Rust binary which connects to IRIS through the web gateway protocol. The Binary is installed in `/usr/irissys/bin` (should already be in PATH).  

A sample configuration is shown in [config.toml](./config.toml), which serves a remote HTTP server on port 8080 (which is exposed by the docker-compose file). 

To start the transport, open a bash terminal within the container: 

```bash
docker-compose exec -it iris bash 
```

Then start the `iris-mcp-server`

```bash 
iris-mcp-server -c config.toml run 
```

You can now connect the MCP server to your MCP Client of choice (e.g. coding agents like claude code) using the address: http://localhost:8080/mcp/sample. 

An example python MCP client is shown in test_mcp_connection.py, which uses Langchain's MCP adapters module. To try this, run: 

```bash
pip install langchain-mcp-adapters
python test_mcp.py
```
