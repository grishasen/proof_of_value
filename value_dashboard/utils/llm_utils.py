import os

import streamlit as st
from pandasai_litellm import LiteLLM

SUPPORTED_LITELLM_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.4",
    "gpt-5.4-pro",
]
SUPPORTED_REASONING_EFFORTS = ["minimal", "low", "medium", "high"]
SUPPORTED_VERBOSITY_LEVELS = ["low", "medium", "high"]

_PASSWORD_TOGGLE_STYLE = """
<style>
    [title="Show password text"] {
        display: none;
    }
</style>
"""


def render_litellm_sidebar(
        key_prefix: str,
        default_model: str,
        reasoning_effort: str,
        verbosity: str,
        api_key_label: str = "Enter API Key (Leave empty to use environment variable)",
        model_label: str = "Choose Model",
        missing_key_message: str = "Please configure API key.",
        env_var_name: str = "OPENAI_API_KEY",
        supported_models: list[str] | None = None,
        require_api_key: bool = True,
) -> LiteLLM:
    """Render common LiteLLM sidebar controls and return a configured client."""
    models = supported_models or SUPPORTED_LITELLM_MODELS
    selected_default_model = default_model if default_model in models else models[0]
    selected_default_reasoning = (
        reasoning_effort
        if reasoning_effort in SUPPORTED_REASONING_EFFORTS
        else SUPPORTED_REASONING_EFFORTS[0]
    )
    selected_default_verbosity = (
        verbosity
        if verbosity in SUPPORTED_VERBOSITY_LEVELS
        else SUPPORTED_VERBOSITY_LEVELS[0]
    )

    api_key_input = st.text_input(
        api_key_label,
        type="password",
        value=os.environ.get(env_var_name),
        key=f"{key_prefix}_api_key",
    )
    st.markdown(_PASSWORD_TOGGLE_STYLE, unsafe_allow_html=True)

    api_key = api_key_input if api_key_input else os.environ.get(env_var_name)
    if not api_key:
        if require_api_key:
            st.error(missing_key_message)
            st.stop()
        st.info("Add an API key when you are ready for AI-assisted generation.")
        return None

    model_choice = st.selectbox(
        model_label,
        options=models,
        index=models.index(selected_default_model),
        key=f"{key_prefix}_model",
    )
    selected_reasoning_effort = st.selectbox(
        "Reasoning Effort",
        options=SUPPORTED_REASONING_EFFORTS,
        index=SUPPORTED_REASONING_EFFORTS.index(selected_default_reasoning),
        key=f"{key_prefix}_reasoning_effort",
        help="Higher reasoning effort may improve difficult tasks but usually increases latency.",
    )
    selected_verbosity = st.selectbox(
        "Verbosity",
        options=SUPPORTED_VERBOSITY_LEVELS,
        index=SUPPORTED_VERBOSITY_LEVELS.index(selected_default_verbosity),
        key=f"{key_prefix}_verbosity",
        help="Controls how detailed the model response should be.",
    )

    return LiteLLM(
        model=model_choice,
        api_key=api_key,
        reasoning_effort=selected_reasoning_effort,
        verbosity=selected_verbosity,
    )
