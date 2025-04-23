import os
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
from tools import list_tables_tool, get_schema_tool, db_query_tool, query_check, db, llm

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a SQL expert with a strong attention to detail.
Given an input question, generate a syntactically correct SQLite query to run, then execute it and return the answer.
When generating the query:
- Limit to at most 15 results unless specified.
- Order results by a relevant column.
- Query only relevant columns.
- Rewrite queries if errors occur or results are empty.
- Do not make DML statements (INSERT, UPDATE, DELETE, DROP).
- Use cleaned column names (e.g., 'operatsionnye_raskhody_mln_tenge' instead of 'Операционные_расходы_млн_тенге').
- Use the query_check tool to validate and correct queries before execution.
- Use list_tables_tool and get_schema_tool to understand the database structure if needed.
- Always provide the final answer in a clear, concise format.
- If an error occurs, explain it and suggest a corrected query or action.
{context}"""), 
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "{input}"),
])

tools = [list_tables_tool, get_schema_tool, db_query_tool, query_check]

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    prompt=agent_prompt,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

def run_agent(query: str, use_history: bool = False, chat_history: list = None) -> dict:
    try:
        context = ""
        if use_history and chat_history:
            context = "Previous Q&A for context:\n" + "\n".join(
                [f"Q: {qa['question']}\nA: {qa['answer']}" for qa in chat_history]
            ) + "\nUse this context only if relevant to the current query."
        result = agent.invoke({
            "input": query,
            "context": context
        })
        return {
            "success": True,
            "output": result["output"],
            "intermediate_steps": result.get("intermediate_steps", [])
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}"
        }

if __name__ == "__main__":
    query = "пример запроса для теста"
    result = run_agent(query)
    if result["success"]:
        print("Result:", result["output"])
        print("Intermediate Steps:", result["intermediate_steps"])
    else:
        print(result["error"])