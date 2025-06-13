from pydantic_ai import Agent
from pydantic import BaseModel
from typing import List, Dict, Any
import json
from langchain.schema import Document
import os


class LogAnalysisResponse(BaseModel):
    answer: str
    relevant_logs: List[Dict[str, Any]]
    total_logs_analyzed: int

# Initialize PydanticAI Agent
log_analysis_agent = Agent(
    'openai:gpt-4o',
    output_type=LogAnalysisResponse,
    system_prompt="""
    You are an intelligent log analysis assistant. Your job is to:
    
    1. Analyze the provided log entries to answer the user's question
    2. Provide a clear, comprehensive answer based on the log data
    3. Identify patterns, trends, errors, or specific information requested
    4. Include relevant log entries that support your analysis
    5. Be specific and cite actual data from the logs when possible
    
    Always provide:
    - A detailed answer to the user's question
    - The relevant log entries that support your answer
    - The total number of logs you analyzed
    
    Make your analysis thorough and insightful.
    """
)

def analyze_logs_with_agent(question: str, relevant_chunks: List[Document], api_key: str) -> LogAnalysisResponse:
    try:
        # Prepare context from chunks
        context_logs = []
        for doc in relevant_chunks:
            try:
                log_data = json.loads(doc.page_content)
                context_logs.append(log_data)
            except json.JSONDecodeError:
                # If not valid JSON, include as text
                context_logs.append({"raw_content": doc.page_content})
        
        # Create the prompt with context
        context_text = f"""
        Question: {question}
        
        Log Entries to Analyze:
        {json.dumps(context_logs, indent=2, default=str)}
        
        Please analyze these logs to answer the question. Provide insights, patterns, and specific findings.
        """

        os.environ["OPENAI_API_KEY"] = api_key

        result = log_analysis_agent.run_sync(context_text)
        return result.output
        
    except Exception as e:
        # Return error response in expected format
        return LogAnalysisResponse(
            answer=f"Error analyzing logs: {str(e)}",
            relevant_logs=[],
            total_logs_analyzed=0
        )