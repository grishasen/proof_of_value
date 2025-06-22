import base64
import os
import traceback
from io import BytesIO

import pandasai as pai
import plotly.io as pio
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from pandasai import Agent
from pandasai.core.response import ChartResponse, DataFrameResponse
from pandasai_openai import OpenAI

from value_dashboard.metrics.clv import rfm_summary
from value_dashboard.pipeline.holdings import load_holdings_data as load_holdings_data
from value_dashboard.pipeline.ih import load_data as ih_load_data
from value_dashboard.utils.config import get_config

pio.defaults.default_scale = 4
pio.defaults.default_height = 480
pio.defaults.default_width = 1280


def get_agent(data) -> Agent:
    agent = Agent(
        data,
        memory_size=10,
        description=get_config()["chat_with_data"]["agent_prompt"],
    )

    return agent


def clear_chat_history(analyst_ref):
    st.session_state.messages = []
    analyst_ref.start_new_conversation()


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

    openai_api_key = (
        api_key_input if api_key_input else os.environ.get("OPENAI_API_KEY")
    )

    if not openai_api_key:
        st.error("Please configure API key.")
        st.stop()

    model_choice = st.selectbox(
        "Choose Model",
        options=OpenAI._supported_chat_models,
        index=OpenAI._supported_chat_models.index("gpt-4.1-mini")
    )

    # Create llm instance
    llm = OpenAI(
        api_token=openai_api_key,
        temperature=0,
        model=model_choice
    )
    if llm:
        metrics_data = ih_load_data() if st.session_state.get('data_loaded', default=False) else {}
        clv_data = load_holdings_data() if st.session_state.get('holdings_data_loaded', default=False) else {}
        metrics_descs = get_config()["chat_with_data"]["metric_descriptions"]
        data_list = []
        for metric in metrics_data.keys():
            if metric.startswith(("engagement", "conversion", "experiment")):
                df = pai.DataFrame(
                    metrics_data[metric].to_pandas(),
                    name=metric,
                    description=metrics_descs[metric]
                )
                data_list.append(df)
        for metric in clv_data.keys():
            if metric.startswith(("clv")):
                m_config = get_config()['metrics'][metric]
                totals_frame = rfm_summary(clv_data[metric], m_config)
                df = pai.DataFrame(
                    totals_frame.to_pandas(),
                    name=metric,
                    description=metrics_descs[metric]
                )
                data_list.append(df)
        pai.config.set({
            "llm": llm,
            "verbose": True
        })
        analyst = get_agent(data_list)
        analyst.start_new_conversation()

    c1, c2 = st.columns([0.5, 0.5], vertical_alignment="center")
    with c1:
        st.button("Clear chat 🗑️", on_click=lambda: clear_chat_history(analyst), use_container_width=True)
    with c2:
        if "messages" in st.session_state:
            if st.session_state.messages:
                chat_log = "\n\n".join(
                    f"{msg['role'].capitalize()}: {msg.get('question') or msg.get('response') or msg.get('error')}"
                    for msg in st.session_state.messages
                )
                chat_log_bytes = BytesIO(chat_log.encode("utf-8"))
                st.download_button(
                    label="Save chat 📨",
                    data=chat_log_bytes,
                    file_name="chat_log.txt",
                    mime="text/plain",
                    use_container_width=True
                )


def print_previous_response(message):
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


def print_response(message):
    if "question" in message:
        st.markdown(message["question"])
    elif "response" in message:
        if message["type"] == 'img':
            if message.get('last_generated_code', None):
                if 'plotly_chart' in message['last_generated_code']:
                    return
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
            print_previous_response(message)

    if prompt := st.chat_input("What would you like to know? "):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "question": prompt})

        try:
            st.toast("Getting response...")
            response = analyst.chat(prompt) if new_chat else analyst.follow_up(prompt)
            path = ''
            if isinstance(response, ChartResponse):
                saved_resp = response.get_base64_image()
                resp_type = 'img'
                path = response.value
            elif isinstance(response, DataFrameResponse):
                saved_resp = response.to_dict()
                resp_type = 'data'
            else:
                saved_resp = response
                resp_type = 'str'

            last_msg = {
                "role": "assistant",
                "response": saved_resp,
                "type": resp_type,
                "path": path,
                "last_generated_code": analyst.last_generated_code
            }
            st.session_state.messages.append(last_msg)

            with st.chat_message("assistant"):
                print_response(last_msg)
                with st.status("Show explanation", expanded=False):
                    st.code(analyst.last_generated_code, line_numbers=True)
                if os.path.exists(last_msg["path"]):
                    os.remove(last_msg["path"])
        except Exception as e:
            print(traceback.format_exc())
            error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"
            st.write(error_message)
            st.write(e)


chat_window(analyst)
