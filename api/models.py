from pydantic import BaseModel

class ChatModelSchema(BaseModel):
    message: str