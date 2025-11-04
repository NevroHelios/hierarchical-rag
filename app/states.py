from typing import TypedDict, Dict, List, Tuple, Literal, Annotated
from pydantic import BaseModel, Field
from operator import add


class MasterAgentState(TypedDict):
    question: str
    queries: Dict[Literal['clinic', 'research', 'book'], str]
    contexts: Annotated[List[Tuple[str, str]], add]
    answer: str
    
class MasterQuery(BaseModel):
    queries: Dict[Literal['clinic', 'research', 'book'], str] = Field(description="Dictionary of {agent_type : query}. query will be passed to the vector store of respective agent_type")