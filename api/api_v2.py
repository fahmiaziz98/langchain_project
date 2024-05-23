from fastapi import FastAPI
import asyncio
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
from threading import Thread
from queue import Queue

from langchain.schema.messages import HumanMessage, AIMessage
from lang import get_response



app = FastAPI()

# Creating a Streamer queue
streamer_queue = Queue()


def generate(query):
    
    llm = get_response()
    llm.invoke([HumanMessage(content=query)])


def start_generation(query):
    # Creating a thread with generate function as a target
    thread = Thread(target=generate, kwargs={"query": query})
    # Starting the thread
    thread.start()


async def response_generator(query):
    # Start the generation process
    start_generation(query)

    # Starting an infinite loop
    while True:
        # Obtain the value from the streamer queue
        value = streamer_queue.get()
        # Check for the stop signal, which is None in our case
        if value == None:
            # If stop signal is found break the loop
            break
        # Else yield the value
        yield value
        # statement to signal the queue that task is done
        streamer_queue.task_done()

        # guard to make sure we are not extracting anything from 
        # empty queue
        await asyncio.sleep(0.1)


@app.post("/stream/")
async def stream(request: Request):
    req = await request.json()
    user_query = req["user_query"]
    return StreamingResponse(response_generator(user_query), media_type='text/event-stream')