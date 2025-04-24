import streamlit as st
from agent import run_agent

st.set_page_config(page_title="Epsilon LangChain SQL Agent", page_icon=":database:", layout="wide")

st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e6f3ff;
        text-align: right;
    }
    .agent-message {
        background-color: #f0f0f0;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.header("LangChain SQL Agent")
st.sidebar.markdown("""
Приветствуем в LangChain SQL Agent! Задавайте вопросы связанные с тремя файлами:\n - train_and_forecast\n - main_metrics\n - var1_correlations\n 
""")

def should_use_history(query: str) -> bool:
    history_keywords = ["those", "these", "previous", "last", "same", "that", "them"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in history_keywords)

st.title("Epsilon LangChain Agent")
st.markdown("Чат с LangChain SQL Agent для запроса базы данных. Результаты и SQL-запросы отображаются под каждым сообщением.")

chat_container = st.container()

with chat_container:
    for idx, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**You**: {chat['question']}")
            if chat["use_history"]:
                st.markdown("*Using chat history*")
        with st.chat_message("assistant", avatar="🤖"):
            if chat["success"]:
                st.markdown(f"**Agent**: {chat['answer']}")
                with st.expander("Show SQL Queries"):
                    for step in chat["intermediate_steps"]:
                        action = step[0]
                        output = step[1]
                        if action.tool in ["db_query_tool", "query_check"]:
                            st.code(output, language="sql")
                        else:
                            st.write(f"**{action.tool}**: {output}")
            else:
                st.error(f"**Error**: {chat['error']}")

query = st.chat_input("Задайте вопрос о базе данных...")

if query:
    use_history = should_use_history(query)

    with chat_container:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**You**: {query}")
            if use_history:
                st.markdown("*Использование истории чата*")

    with st.spinner("Processing query..."):
        result = run_agent(query, use_history=use_history, chat_history=st.session_state.chat_history)

        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                if result["success"]:
                    st.markdown(f"**Agent**: {result['output']}")
                    with st.expander("Show SQL Queries"):
                        for step in result["intermediate_steps"]:
                            action = step[0]
                            output = step[1]
                            if action.tool in ["db_query_tool", "query_check"]:
                                st.code(output, language="sql")
                            else:
                                st.write(f"**{action.tool}**: {output}")
                else:
                    st.error(f"**Error**: {result['error']}")

        st.session_state.chat_history.append({
            "question": query,
            "answer": result["output"] if result["success"] else result["error"],
            "success": result["success"],
            "intermediate_steps": result["intermediate_steps"],
            "use_history": use_history
        })

if len(st.session_state.chat_history) > 50:
    st.session_state.chat_history = st.session_state.chat_history[-50:]