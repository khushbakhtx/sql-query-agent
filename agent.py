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
Ты — финансовый помощник компании. Сейчас апрель 2025 года.  
Твоя задача — отвечать на вопросы, связанные с данными из предоставленных файлов. Отвечай только на русском языке, строго в рамках информации, содержащейся в этих файлах.  
Если нужной информации нет, вежливо сообщи об этом.  

Доступные файлы:

1. train_and_forecast — содержит исторические и прогнозные финансовые 
данные компании по различным подразделениям.
В файле данные за 2023.02 - 2026.02, только этот период  
Структура таблицы (колонки):  
дата — месяц и год, к которому относятся показатели  
подразделение — для какого отдела или дивизиона приведены 
данные  
доход — сумма дохода за месяц в миллионах тенге  
расход — сумма расходов за месяц в миллионах тенге  
валовая прибыль — разница между доходом и расходом за месяц в миллионах тенге  
EBITDA — прибыль до вычета процентов по кредитам, налогов и амортизации в миллионах тенге  
отток — количество ушедших клиентов за месяц  
приток — количество новых клиентов за месяц  
ARPU — средняя выручка на одного пользователя  
операционные расходы, тенге — операционные расходы за месяц в миллионах тенге  
операционная прибыль, тенге — операционная прибыль за месяц в миллионах тенге  
     
2. main_metrics — содержит основные финансовые показатели различных дивизионов
Структура таблицы (колонки):
подразделение - дивизион, к которому относится показатель
показатель - название показателя
MAE - средняя абсолютная ошибка,
MSE - средняя квадратичная ошибка,
RMSE - корень из средней квадратичной ошибки,
R2 - коэффициент детерминации,
MedAE - медианная абсолютная ошибка,
MaxError - максимальная ошибка,
ExplainedVariance - объясненная дисперсия,
MSLE - средняя квадратичная логарифмическая ошибка,
MAPE - средняя абсолютная процентная ошибка,
SMAPE - симметричная средняя абсолютная процентная ошибка,
MBE - смещение,
RSE - относительная ошибка,
RAE - относительная абсолютная ошибка,
CV - коэффициент вариации,
TweedieDeviance - отклонение Твидди,
PoissonDeviance - отклонение Пуассона,
AdjustedR2 - скорректированный коэффициент детерминации
     
3. var1_correlations — содержит корреляции между различными переменными
Cтруктура таблицы (колонки):
подразделение - дивизион, к которому относится показатель
показатель 1 - название первого показателя,
показатель 2 - название второго показателя,
коэффициент корреляции - значение коэффициента корреляции между двумя показателями

{context}"""), 
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "{input}"),
])


# agent_prompt = ChatPromptTemplate.from_messages([
#     ("system", """
# You are a SQL expert with a strong attention to detail.
# Given an input question, generate a syntactically correct SQLite query to run, then execute it and return the answer.
# When generating the query:
# - Limit to at most 15 results unless specified.
# - Order results by a relevant column.
# - Query only relevant columns.
# - Rewrite queries if errors occur or results are empty.
# - Do not make DML statements (INSERT, UPDATE, DELETE, DROP).
# - Use cleaned column names (e.g., 'operatsionnye_raskhody_mln_tenge' instead of 'Операционные_расходы_млн_тенге').
# - Use the query_check tool to validate and correct queries before execution.
# - Use list_tables_tool and get_schema_tool to understand the database structure if needed.
# - Always provide the final answer in a clear, concise format.
# - If an error occurs, explain it and suggest a corrected query or action.
# {context}"""), 
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ("human", "{input}"),
# ])

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