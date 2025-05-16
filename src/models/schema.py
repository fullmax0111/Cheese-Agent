from typing import List, TypedDict, Optional
from pydantic import BaseModel, Field

class PlanExecute(TypedDict):
    curr_state: str
    message: List[str]
    query_to_retrieve_or_answer: str
    curr_context: str
    aggregated_context: str
    tool: str
    answer_quality: str
    human_feedback: str
    reasoning_steps:List[dict]

class MongoQuery(BaseModel):
    """Schema for MongoDB query generation."""
    filter_conditions: dict = Field(description="MongoDB filter conditions")
    sort_conditions: Optional[dict] = Field(description="MongoDB sort conditions", default=None)
    projection: Optional[dict] = Field(description="Fields to include in results", default=None)

class reasoningOutput(BaseModel):
    """Output schema for the task handler."""
    query: str = Field(description="The specific query or question to be used")
    analysis: str = Field(description="Brief explanation of why this tool was chosen")
    curr_context: str = Field(description="The context to use")
    tool: str = Field(description="The tool to be used should be either MongoDB_retrieval, pinecone_retrieval.")