import asyncio
import time
import httpx


CHATBOT_URL = "http://127.0.0.1:8000/rag-agent"

questions = [
   "What is the current wait time at Wallace-Hamilton hospital?",
   "Which hospital has the shortest wait time?",  
   "What are patients saying about the nursing staff at Castaneda-Hardy?",
   "What was the total billing amount charged to each payer for 2023?",
   "How many patients has Dr. Ryan Brown treated?",
   "Which physician has the lowest average visit duration in days?", # <--ini error
   "How many visits are open and what is their average duration in days?",
   "Have any patients complained about noise?",
   "How much was billed for patient 789's stay?",
   "Which physician has billed the most to cigna?",
]


async def make_async_post(url, data):
    timeout = httpx.Timeout(timeout=120)
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, timeout=timeout)
        return response

async def make_bulk_requests(url, data):
    tasks = [make_async_post(url, payload) for payload in data]
    responses = await asyncio.gather(*tasks)
    outputs = [r.json()["output"] for r in responses]
    return outputs

def run_asyncio_task(task):
    if asyncio.get_event_loop().is_running():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        result = new_loop.run_until_complete(task)
        asyncio.set_event_loop(asyncio.get_event_loop())
    else:
        result = asyncio.run(task)
    return result

request_bodies = [{"query": q} for q in questions]

start_time = time.perf_counter()
outputs = run_asyncio_task(make_bulk_requests(CHATBOT_URL, request_bodies))
end_time = time.perf_counter()

print(f"Run time: {end_time - start_time} seconds")

## 