import os
import ast
import uuid
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

st.set_page_config(
    page_title="CSV AI Agent",
    page_icon="📊",
    layout="wide"
)

_df: pd.DataFrame | None = None


if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()


def set_df(df: pd.DataFrame):
    global _df
    _df = df



@tool
def get_data_info():
    """
    Get columns, data types, shape,
    and first 5 rows of the uploaded CSV.
    """

    if _df is None:
        return "No CSV loaded."

    if _df.empty:
        return "The CSV contains no rows."

    return f"""
    Shape: {_df.shape}

    Columns and Types:
    {_df.dtypes.to_string()}

    First 5 Rows:
    {_df.head().to_string()}
    """



@tool
def filter_tool(condition: str):
    """
    Filter rows using a pandas query condition.

    Example:
    Age > 30 and Department == 'Engineering'
    """

    if _df is None:
        return "No CSV loaded."

    if _df.empty:
        return "The CSV contains no rows."

    try:

        result = _df.query(condition)

        return (
            f"Found {len(result)} rows:\n"
            f"{result.to_string()}"
        )

    except Exception as e:

        return f"Error filtering data: {e}"


ALLOWED_METHODS = {
    "groupby",
    "mean",
    "median",
    "sum",
    "min",
    "max",
    "count",
    "std",
    "var",
    "value_counts",
    "nunique",
    "describe",
    "head",
    "tail",
    "sort_values",
    "agg",
    "size",
    "reset_index",
    "quantile"
}


def validate_expression(node):

    if isinstance(node, ast.Expression):

        validate_expression(node.body)

    elif isinstance(node, ast.Name):

        if node.id != "df":
            raise ValueError(
                "Only dataframe 'df' is allowed."
            )

    elif isinstance(node, ast.Constant):

        if not isinstance(
            node.value,
            (str, int, float, bool)
        ):
            raise ValueError("Unsupported constant.")

    elif isinstance(node, ast.Subscript):

        validate_expression(node.value)
        validate_expression(node.slice)

    elif isinstance(node, ast.Attribute):

        if node.attr not in ALLOWED_METHODS:
            raise ValueError(
                "This dataframe attribute is not allowed."
            )

        validate_expression(node.value)

    elif isinstance(node, ast.Call):

        if not isinstance(node.func, ast.Attribute):
            raise ValueError(
                "Only approved dataframe methods are allowed."
            )

        if node.func.attr not in ALLOWED_METHODS:
            raise ValueError(
                "This dataframe method is not allowed."
            )

        validate_expression(node.func)

        for arg in node.args:
            validate_expression(arg)

        for keyword in node.keywords:
            validate_expression(keyword.value)

    elif isinstance(node, (ast.List, ast.Tuple)):

        for element in node.elts:
            validate_expression(element)

    elif isinstance(node, ast.Load):

        pass

    else:

        raise ValueError("Unsupported expression.")


@tool
def analyze_data(expression: str):
    """
    Run approved pandas expressions on dataframe df.

    Examples:

    df['Salary'].mean()

    df.groupby('Department')['Salary'].mean()

    df['City'].value_counts()
    """

    if _df is None:
        return "No CSV loaded."

    if _df.empty:
        return "The CSV contains no rows."

    try:

        df = _df

        parsed = ast.parse(expression, mode="eval")

        validate_expression(parsed)

        result = eval(
            compile(parsed, "<analysis>", "eval"),
            {"__builtins__": {}},
            {"df": df}
        )

        return str(result)

    except Exception as e:

        return f"Error analyzing data: {e}"


all_tools = [
    get_data_info,
    filter_tool,
    analyze_data
]



system_prompt = """
You are a professional data analyst assistant.

You help users explore and analyze CSV datasets.

Available tools:

1. get_data_info:
   Get columns, data types and sample rows.

2. filter_tool:
   Filter rows using pandas query conditions.

3. analyze_data:
   Perform calculations, statistics,
   aggregations and group analysis.

Instructions:

- Always use get_data_info first when column names
  or data types are unknown.
- Use appropriate tools to answer questions.
- Never invent statistics or results.
- Base answers only on the uploaded dataset.
- Explain results in simple language.
- If a requested column does not exist,
  inform the user.
- Present numerical results clearly.
"""


def create_csv_agent():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing from your .env file."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=system_prompt,
        checkpointer=st.session_state.memory
    )

    return agent

def run_agent(query: str):

    agent = create_csv_agent()

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }
    )

    answer = response["messages"][-1].content

    return answer


st.header("📊 CSV AI Agent")

st.write(
    "Upload your CSV dataset and ask questions "
    "to analyze your data using AI."
)


file = st.file_uploader(
    label="Select your CSV File",
    type="csv"
)

if file:

    try:

        df = pd.read_csv(file)

        set_df(df)

        st.dataframe(df.head())

        st.success(
            f"Loaded {df.shape[0]} rows."
        )

    except Exception as e:

        st.error(f"CSV loading error: {e}")


for msg in st.session_state.messages:

    role = msg["role"]

    content = msg["content"]

    with st.chat_message(role):

        st.markdown(content)



if _df is not None and not _df.empty:

    query = st.chat_input(
        "Ask anything about your CSV data..."
    )

    if query:


        with st.chat_message("user"):

            st.markdown(query)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        # Generate AI response

        with st.chat_message("assistant"):

            with st.spinner("Analyzing your data..."):

                try:

                    res = run_agent(query)

                    st.markdown(res)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": res
                        }
                    )

                except Exception as e:

                    error_message = f"Error: {e}"

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message
                        }
                    )

else:

    st.info(
        "Please upload a CSV file to start chatting."
    )