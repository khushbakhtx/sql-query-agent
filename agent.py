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
    ("system", """
Ты эксперт по SQL с большим вниманием к деталям.
Получив входной вопрос, сгенерируй синтаксически правильный запрос SQLite для запуска, затем выполни его и верни ответ.
При генерации запроса:
- Сейчас апрель 2025 года, не забывай об этом!
- Ограничься максимум 15 результатами, если не указано иное.
- Упорядочьте результаты по соответствующему столбцу.
- Запрашивай только соответствующие столбцы.
- Переписывай запросы, если возникают ошибки или результаты пустые.
- Не делай операторы DML (INSERT, UPDATE, DELETE, DROP).
- Используй очищенные имена столбцов (например, «operatsionnye_raskhody_mln_tenge» вместо «Операционные_расходы_млн_тенге»).
- Используй инструмент query_check для проверки и исправления запросов перед выполнением.
- Используй list_tables_tool и get_schema_tool для понимания структуры базы данных, если это необходимо.
- Всегда давай окончательный ответ в четком, кратком формате.
- Если произошла ошибка, объясни ее и предложи решение или действие.
- Не отвечай пользователю SQL запросами, только результатами выполнения запросов.
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