import os

import streamlit as st
from dotenv import load_dotenv
from pandasai import Agent
from pandasai.connectors import PandasConnector
from pandasai.llm import OpenAI
from pandasai.responses import StreamlitResponse

from value_dashboard.pipeline.ih import load_data
from value_dashboard.utils.config import get_config


def get_agent(data, llm):
    agent = Agent(
        data,
        config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse},
        memory_size=10,
        description=get_config()["chat_with_data"]["agent_prompt"],
    )

    return agent


load_dotenv()
st.title("Chat With Your Data")
if "data_loaded" not in st.session_state:
    st.warning("Please configure your files in the `data import` tab.")
    st.stop()

# Sidebar for API Key settings
with st.sidebar:
    st.header(
        "Set your API Key",
        help="You can get it from [OpenAI](https://platform.openai.com/account/api-keys/).",
    )
    # Get API base from input or environment variable
    api_base_input = st.text_input(
        "Enter API Base (Leave empty to use environment variable)",
        value=os.environ.get("OPENAI_API_BASE"),
    )

    # Get API key from input or environment variable
    api_key_input = st.text_input(
        "Enter API Key (Leave empty to use environment variable)",
        type="password",
        value=os.environ.get("OPENAI_API_KEY"),
    )
    # Add css to hide item with title "Show password text"
    st.markdown(
        """
    <style>
        [title="Show password text"] {
            display: none;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Set OpenAI API key
    openai_api_base = (
        api_base_input if api_base_input else os.environ.get("OPENAI_API_BASE")
    )
    openai_api_key = (
        api_key_input if api_key_input else os.environ.get("OPENAI_API_KEY")
    )

    # Create llm instance
    llm = OpenAI(api_token=openai_api_key, temperature=0, seed=31)
    llm.api_base = openai_api_base
    if llm:
        metrics_data = load_data()
        metrics_descs = get_config()["chat_with_data"]["metric_descriptions"]
        data_list = []
        for metric in metrics_data.keys():
            if (
                metric.startswith("eng")
                | metric.startswith("conv")
                | metric.startswith("exp")
            ):
                df = metrics_data[metric].to_pandas()
                pconnector = PandasConnector(
                    config={"original_df": df},
                    name=metric,
                    description=metrics_descs[metric],
                )
                data_list.append(pconnector)
        analyst = get_agent(data_list, llm)

    def clear_chat_history():
        st.session_state.messages = []

    st.button("Clear chat 🗑️", on_click=clear_chat_history)


def chat_window(analyst):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "question" in message:
                st.markdown(message["question"])
            elif "response" in message:
                st.write(message["response"])
            elif "error" in message:
                st.text(message["error"])

    if prompt := st.chat_input("What would you like to know? "):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "question": prompt})

        try:
            st.toast("Getting response...")
            response = analyst.chat(prompt)
            st.session_state.messages.append(
                {"role": "assistant", "response": response}
            )
            st.toast("Getting explanation...")
            explanation = analyst.explain()

            with st.chat_message("assistant"):
                st.write(response)
                with st.status("Show explanation", expanded=False):
                    st.write(explanation)
                    st.code(analyst.last_code_generated, line_numbers=True)
                if os.path.exists("exports/charts/temp_chart.png"):
                    st.image("exports/charts/temp_chart.png")
                    os.remove("exports/charts/temp_chart.png")
        except Exception as e:
            st.write(e)
            error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"


chat_window(analyst)
