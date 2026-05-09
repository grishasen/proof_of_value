import traceback
from io import BytesIO

import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv

from value_dashboard.ai.code_agent import DataCodeAgent
from value_dashboard.ai.data_context import DataChatDataset
from value_dashboard.ai.responses import DataFrameResponse, PlotlyResponse
from value_dashboard.metrics.clv import rfm_summary
from value_dashboard.pipeline.holdings import load_holdings_data as load_holdings_data
from value_dashboard.pipeline.ih import load_data as ih_load_data
from value_dashboard.utils.config import get_config
from value_dashboard.utils.llm_utils import render_litellm_sidebar

pio.defaults.default_scale = 4
pio.defaults.default_height = 480
pio.defaults.default_width = 1280

st.set_page_config(page_title="✨Chat With Data", layout="wide")
st.markdown(
    """
<style>
    .stMainBlockContainer {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
    """,
    unsafe_allow_html=True
)


def get_agent(data, llm) -> DataCodeAgent:
    agent = DataCodeAgent(
        datasets=data,
        llm=llm,
        memory_size=10,
        description=get_config()["chat_with_data"]["agent_prompt"],
    )

    return agent


def clear_chat_history(analyst_ref):
    st.session_state.messages = []
    analyst_ref.start_new_conversation()


def metric_description(metric: str, descriptions: dict) -> str:
    if metric in descriptions:
        return descriptions[metric]
    for prefix in ("engagement", "conversion", "experiment", "clv"):
        if metric.startswith(prefix):
            return descriptions.get(prefix, "")
    return ""


def message_log_text(message: dict) -> str:
    if "question" in message:
        return message["question"]
    if "error" in message:
        return message["error"]

    summary = message.get("summary")
    if summary:
        return summary
    if message.get("type") == "plotly":
        return "[Plotly chart]"
    if message.get("type") == "data":
        return "[Dataframe response]"
    return str(message.get("response", ""))


def messages_to_agent_history(messages: list[dict]) -> list[dict[str, str]]:
    history = []
    for message in messages:
        if "question" in message:
            history.append({"role": "user", "content": message["question"]})
        elif "response" in message:
            history.append({"role": "assistant", "content": message_log_text(message)})
    return history


load_dotenv()
st.title("Chat With Your Data")
if "data_loaded" not in st.session_state:
    st.warning("Please configure your files in the `data import` tab.")
    st.stop()

# Sidebar for API Key settings
with st.sidebar:
    model: str = "gpt-5.5"
    reasoning_effort = "low"  # "minimal" | "low" | "medium" | "high"
    verbosity = "low"  # "low" | "medium" | "high"
    llm = render_litellm_sidebar(
        key_prefix="chat_with_data",
        default_model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        missing_key_message="Please configure API key.",
    )
    if llm:
        metrics_data = ih_load_data() if st.session_state.get('data_loaded', default=False) else {}
        clv_data = load_holdings_data() if st.session_state.get('holdings_data_loaded', default=False) else {}
        metrics_descs = get_config()["chat_with_data"]["metric_descriptions"]
        data_list = []
        for metric in metrics_data.keys():
            if metric.startswith(("engagement", "conversion", "experiment")):
                df = DataChatDataset(
                    name=metric,
                    dataframe=metrics_data[metric].to_pandas(),
                    description=metric_description(metric, metrics_descs),
                )
                data_list.append(df)
        for metric in clv_data.keys():
            if metric.startswith(("clv")):
                m_config = get_config()['metrics'][metric]
                totals_frame = rfm_summary(clv_data[metric], m_config)
                df = DataChatDataset(
                    name=metric,
                    dataframe=totals_frame.to_pandas(),
                    description=metric_description(metric, metrics_descs),
                )
                data_list.append(df)
        analyst = get_agent(data_list, llm)

    c1, c2 = st.columns([0.5, 0.5], vertical_alignment="center")
    with c1:
        st.button("Clear chat 🗑️", on_click=lambda: clear_chat_history(analyst), width='stretch')
    with c2:
        if "messages" in st.session_state:
            if st.session_state.messages:
                chat_log = "\n\n".join(
                    f"{msg['role'].capitalize()}: {message_log_text(msg)}"
                    for msg in st.session_state.messages
                )
                chat_log_bytes = BytesIO(chat_log.encode("utf-8"))
                st.download_button(
                    label="Save chat 📨",
                    data=chat_log_bytes,
                    file_name="chat_log.txt",
                    mime="text/plain",
                    width='stretch'
                )


def print_previous_response(message):
    if "question" in message:
        st.markdown(message["question"])
    elif "response" in message:
        if message["type"] == 'plotly':
            st.plotly_chart(message["response"], width='stretch', theme="streamlit")
        elif message["type"] == 'data':
            st.dataframe(message["response"])
        else:
            st.write(message["response"])
    elif "error" in message:
        st.text(message["error"])


def print_response(message):
    if "question" in message:
        st.markdown(message["question"])
    elif "response" in message:
        if message["type"] == 'plotly':
            st.plotly_chart(message["response"], width='stretch', theme="streamlit")
        elif message["type"] == 'data':
            st.dataframe(message["response"])
        else:
            st.write(message["response"])
    elif "error" in message:
        st.text(message["error"])


def chat_window(analyst):
    new_chat = False
    if "messages" not in st.session_state or not st.session_state.messages:
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
            with st.spinner('Getting response...'):
                analyst.set_history(messages_to_agent_history(st.session_state.messages[:-1]))
                response = analyst.chat(prompt) if new_chat else analyst.follow_up(prompt)
            if isinstance(response, PlotlyResponse):
                saved_resp = response.value
                resp_type = 'plotly'
            elif isinstance(response, DataFrameResponse):
                saved_resp = response.value
                resp_type = 'data'
            else:
                saved_resp = response.value
                resp_type = 'str'

            last_msg = {
                "role": "assistant",
                "response": saved_resp,
                "type": resp_type,
                "summary": response.summary,
                "last_generated_code": analyst.last_generated_code
            }
            st.session_state.messages.append(last_msg)

            with st.chat_message("assistant"):
                print_response(last_msg)
                with st.status("Show explanation", expanded=False):
                    st.code(analyst.last_generated_code, line_numbers=True)
        except Exception as e:
            print(traceback.format_exc())
            error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"
            st.write(error_message)
            st.write(e)


chat_window(analyst)
