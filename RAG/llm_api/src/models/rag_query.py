from pydantic import BaseModel
from typing import List

class QueryInput(BaseModel):
    query: str

class QueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: List[str]