from fastapi import FastAPI
from agents.rag_agent import agent_executor
from models.rag_query import QueryInput, QueryOutput
from utils.async_utils import async_retry


app = FastAPI(
    title="LLM API",
    description="An API for LLMs",
    version="0.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    agent = agent_executor()
    return await agent.ainvoke({"input": query})

@app.get("/")
async def root():
    return {"status": "Runing..."}

@app.post("/rag-agent")
async def query_rag_agent(query: QueryInput) -> QueryOutput:
    output = await invoke_agent_with_retry(query=query.query)
    output["intermediate_steps"] = [
        str(s) for s in output["intermediate_steps"]
    ]

    return output