import streamlit as st
from agent import run_agent

st.set_page_config(page_title="SQL Query Agent", page_icon=":database:", layout="wide")

st.title("SQL Query Agent")
st.markdown("Enter a natural language query to retrieve data from the SQLite database. The agent will generate and execute a SQL query, displaying the results and execution details.")

with st.form(key="query_form"):
    query = st.text_input("Enter your query:", placeholder="e.g., Ваш запрос...?")
    submit_button = st.form_submit_button(label="Run Query")

if submit_button and query:
    with st.spinner("Processing query..."):
        result = run_agent(query)
        
        if result["success"]:
            st.subheader("Result")
            st.success(result["output"])
            
            if result["intermediate_steps"]:
                st.subheader("Execution Details")
                for step in result["intermediate_steps"]:
                    action = step[0]
                    output = step[1]
                    if action.tool == "db_query_tool":
                        st.code(output, language="sql")
                    elif action.tool == "query_check":
                        st.code(output, language="sql")
                    else:
                        st.write(f"**{action.tool}**: {output}")
        else:
            st.subheader("Error")
            st.error(result["error"])

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Enter a natural language query in the text box.
2. Click "Run Query" to execute.
3. View the results and any generated SQL queries below.
4. The agent will automatically validate and correct SQL queries if needed.
5. Ensure queries are safe (no INSERT, UPDATE, DELETE, DROP).
""")