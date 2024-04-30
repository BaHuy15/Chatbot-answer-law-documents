from pydantic import BaseModel, Field

class QAquestion(BaseModel):
    query: str =Field(...,title="Query",description="The search query")

class QARespond(BaseModel):
    rag_prompt : str = Field(...,title="Input prompt", description = "Question and content retrieved by RAG method")
    answer : str = Field(...,title="Assistant answer",description="This is answer from chatbot")
    # source: List =Field(...,title="Retrieval source")