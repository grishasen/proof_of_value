import base64
import os
from io import BytesIO

import pandasai as pai
import plotly.io as pio
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from pandasai import Agent
from pandasai.core.response import ChartResponse, DataFrameResponse
from pandasai_openai import OpenAI

from value_dashboard.pipeline.ih import load_data
from value_dashboard.utils.config import get_config

pio.kaleido.scope.default_format = "jpeg"
pio.kaleido.scope.default_scale = 2
pio.kaleido.scope.default_height = 640


def get_agent(data, llm) -> Agent:
    agent = Agent(
        data,
        config={"llm": llm, "verbose": True},
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
    openai_api_key = (
        api_key_input if api_key_input else os.environ.get("OPENAI_API_KEY")
    )

    if not openai_api_key:
        st.error("Please configure API key.")
        st.stop()

    # Create llm instance
    llm = OpenAI(
        api_token=openai_api_key,
        temperature=0,
        model="gpt-4o"
    )
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
                df = pai.DataFrame(
                    metrics_data[metric].to_pandas(),
                    name=metric,
                    description=metrics_descs[metric]
                )
                data_list.append(df)
        analyst = get_agent(data_list, llm)
        analyst.start_new_conversation()


    def clear_chat_history():
        st.session_state.messages = []
        analyst.start_new_conversation()


    st.button("Clear chat 🗑️", on_click=clear_chat_history)


def print_response(message):
    if "question" in message:
        st.markdown(message["question"])
    elif "response" in message:
        if message["type"] == 'img':
            st.image(Image.open(BytesIO(base64.b64decode(message["response"]))))
        elif message["type"] == 'data':
            st.dataframe(message["response"]['value'])
        else:
            st.write(message["response"])
    elif "error" in message:
        st.text(message["error"])


def chat_window(analyst):
    new_chat = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
        new_chat = True

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            print_response(message)

    if prompt := st.chat_input("What would you like to know? "):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "question": prompt})

        try:
            st.toast("Getting response...")
            response = analyst.chat(prompt) if new_chat else analyst.follow_up(prompt)
            saved_resp = ''
            resp_type = 'str'
            if isinstance(response, ChartResponse):
                saved_resp = response.get_base64_image()
                resp_type = 'img'
            elif isinstance(response, DataFrameResponse):
                saved_resp = response.to_dict()
                resp_type = 'data'
            else:
                saved_resp = response
                resp_type = 'str'

            last_msg = {
                "role": "assistant",
                "response": saved_resp,
                "type": resp_type
            }
            st.session_state.messages.append(last_msg)

            with st.chat_message("assistant"):
                print_response(last_msg)
                with st.status("Show explanation", expanded=False):
                    st.code(analyst.last_generated_code, line_numbers=True)
                if os.path.exists("exports/charts/temp_chart.png"):
                    os.remove("exports/charts/temp_chart.png")
        except Exception as e:
            st.write(e)
            error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"


chat_window(analyst)
