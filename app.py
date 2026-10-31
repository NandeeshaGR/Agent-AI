import os
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


def get_api_key():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    return api_key


if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "df" not in st.session_state:
    st.session_state.df = None

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None


def create_csv_agent(df):

    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Configure it in .env or Streamlit Secrets."
        )

    @tool
    def get_data_info() -> str:
        """Get dataset shape, column names, data types, missing values and sample records."""

        return f"""
        Dataset Information:

        Total Rows: {df.shape[0]}

        Total Columns: {df.shape[1]}

        Column Names:
        {list(df.columns)}

        Data Types:
        {df.dtypes.to_string()}

        Missing Values:
        {df.isnull().sum().to_string()}

        First Five Records:
        {df.head().to_string()}
        """

    @tool
    def filter_data(column: str, operator: str, value: str) -> str:
        """Filter the dataset using a column, operator and comparison value. Supported operators: ==, !=, >, <, >=, <=, contains."""

        if column not in df.columns:
            return f"Column '{column}' does not exist."

        series = df[column]

        if operator == "contains":
            result = df[
                series.astype(str).str.contains(
                    value,
                    case=False,
                    na=False,
                    regex=False
                )
            ]

        elif operator in ["==", "!="]:
            numeric_value = pd.to_numeric(value, errors="coerce")

            if pd.api.types.is_numeric_dtype(series) and pd.notna(numeric_value):
                comparison = series == numeric_value
            else:
                comparison = (
                    series.astype(str).str.lower() == value.lower()
                )

            if operator == "!=":
                comparison = ~comparison

            result = df[comparison]

        elif operator in [">", "<", ">=", "<="]:
            numeric_series = pd.to_numeric(series, errors="coerce")

            try:
                numeric_value = float(value)
            except ValueError:
                return "Please provide a numeric comparison value."

            if operator == ">":
                result = df[numeric_series > numeric_value]
            elif operator == "<":
                result = df[numeric_series < numeric_value]
            elif operator == ">=":
                result = df[numeric_series >= numeric_value]
            else:
                result = df[numeric_series <= numeric_value]

        else:
            return "Unsupported operator."

        return f"""
        Matching Rows: {len(result)}

        Results:
        {result.head(20).to_string()}
        """

    @tool
    def analyze_data(
        operation: str,
        column: str,
        group_by: str = ""
    ) -> str:
        """Perform statistical analysis. Supported operations: mean, median, sum, min, max, count, value_counts and describe. Optional group_by column for grouped analysis."""

        if column not in df.columns:
            return f"Column '{column}' does not exist."

        valid_operations = [
            "mean",
            "median",
            "sum",
            "min",
            "max",
            "count",
            "value_counts",
            "describe"
        ]

        if operation not in valid_operations:
            return f"Unsupported operation. Choose from {valid_operations}"

        series = df[column]

        if operation in ["mean", "median", "sum"]:
            numeric_series = pd.to_numeric(series, errors="coerce")

            if numeric_series.notna().sum() == 0:
                return "This column has no usable numeric values."

            if group_by:
                if group_by not in df.columns:
                    return f"Group column '{group_by}' does not exist."

                result = (
                    df.assign(_analysis_value=numeric_series)
                    .groupby(group_by)["_analysis_value"]
                    .agg(operation)
                )

                return result.to_string()

            return str(getattr(numeric_series, operation)())

        if operation == "min":
            return str(series.min())

        if operation == "max":
            return str(series.max())

        if operation == "count":
            return str(series.count())

        if operation == "value_counts":
            return series.value_counts(
                dropna=False
            ).head(20).to_string()

        if operation == "describe":
            return series.describe().to_string()

    @tool
    def check_data_quality() -> str:
        """Check duplicate rows, missing values and dataset quality."""

        return f"""
        DATA QUALITY REPORT

        Total Rows: {len(df)}

        Total Columns: {len(df.columns)}

        Duplicate Rows: {df.duplicated().sum()}

        Total Missing Values: {df.isnull().sum().sum()}

        Missing Values By Column:

        {df.isnull().sum().to_string()}
        """

    all_tools = [
        get_data_info,
        filter_data,
        analyze_data,
        check_data_quality
    ]

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key
    )

    system_prompt = """
    You are an intelligent CSV Data Analysis Agent.

    Your job is to analyze the uploaded CSV dataset
    and answer user questions accurately.

    Rules:

    1. Always use get_data_info first when column names or data types are unknown.
    2. Use appropriate tools to answer questions.
    3. Never invent statistics or results.
    4. Base answers only on the uploaded dataset.
    5. Explain results in simple language.
    6. If a requested column does not exist, inform the user.
    7. Present numerical results clearly.
    8. Use check_data_quality for data quality questions.
    9. Use filter_data for filtering requests.
    10. Use analyze_data for statistical calculations.
    """

    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=system_prompt,
        checkpointer=st.session_state.memory
    )

    return agent


def run_agent(query: str, df):

    agent = create_csv_agent(df)

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

    messages = response.get("messages", [])

    if not messages:
        return "The agent did not return a response."

    answer = messages[-1].content

    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict)
            else str(item)
            for item in answer
        )

    return answer


with st.sidebar:

    st.header("📊 Available Features")

    st.markdown("""
    - 📋 Dataset Information
    - 🔍 Data Filtering
    - 📈 Statistical Analysis
    - 📊 Group Analysis
    - 🧹 Data Quality Checks
    """)

    st.divider()

    st.markdown("Powered by Groq, LangChain and LangGraph.")


st.title("📊 CSV AI Agent")

st.markdown(
    """
    ### Intelligent CSV Data Analysis

    Upload your CSV dataset and ask questions in natural language.
    """
)

st.write("")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader(
        "📂 Upload Your CSV Dataset",
        type=["csv"],
        key="main_csv_uploader"
    )

st.write("")


if uploaded_file is not None:

    try:
        file_signature = (
            uploaded_file.name,
            uploaded_file.size
        )

        if file_signature != st.session_state.file_signature:

            df = pd.read_csv(uploaded_file)

            st.session_state.df = df
            st.session_state.file_signature = file_signature
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())

        else:
            df = st.session_state.df

        st.success("CSV uploaded successfully!")

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Duplicate Rows", int(df.duplicated().sum()))

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()


if st.session_state.df is not None:

    st.divider()

    st.subheader("💬 Chat With Your Dataset")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input(
        "Ask something about your CSV dataset..."
    )

    if query:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner("Analyzing your data..."):

                try:
                    answer = run_agent(
                        query,
                        st.session_state.df
                    )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    error_message = f"Error: {str(e)}"

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message
                        }
                    )

else:

    st.info("Please upload a CSV file to start chatting.")