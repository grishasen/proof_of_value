import os
import tomllib
from pathlib import Path
from typing import Any

import streamlit as st

from value_dashboard.ai.litellm_client import LiteLLMClient

LLM_CONFIG_ENV_VAR = "VALUE_DASHBOARD_LLM_CONFIG"
DEFAULT_LLM_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "llm_config.toml"
DEFAULT_LLM_SETTINGS: dict[str, Any] = {
    "model": "gpt-5.5",
    "api_key_env_var": "OPENAI_API_KEY",
    "api_key_label": "Enter API Key (Leave empty to use environment variable)",
    "api_key_required": True,
    "api_key_fallback": "",
    "missing_key_message": "Please configure API key.",
    "missing_key_stops_page": True,
    "reasoning_effort": "low",
    "verbosity": "low",
    "extra_params": {},
}
LITELLM_PASSTHROUGH_KEYS = {
    "api_base",
    "api_version",
    "custom_llm_provider",
    "max_tokens",
    "num_retries",
    "temperature",
    "timeout",
}

_PASSWORD_TOGGLE_STYLE = """
<style>
    [title="Show password text"] {
        display: none;
    }
</style>
"""


def _has_value(value: Any) -> bool:
    """Return whether a TOML value should be sent to LiteLLM."""
    return value is not None and value != ""


def _optional_string(value: Any) -> str | None:
    """Normalize optional TOML strings while preserving unset values."""
    if not _has_value(value):
        return None
    return str(value)


def _coerce_bool(value: Any) -> bool:
    """Coerce config values that may come from TOML or old call-site defaults."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _load_litellm_config(config_path: str | None = None) -> dict[str, Any]:
    """Load local LiteLLM settings from TOML."""
    session_config_path = st.session_state.get("llm_config", "")
    resolved_config_path = Path(
        config_path
        or session_config_path
        or os.environ.get(LLM_CONFIG_ENV_VAR)
        or DEFAULT_LLM_CONFIG_PATH
    ).expanduser()
    try:
        with open(resolved_config_path, mode="rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        if config_path or os.environ.get(LLM_CONFIG_ENV_VAR):
            st.warning(f"LLM config file not found: {resolved_config_path}")
        return {}
    except tomllib.TOMLDecodeError as exc:
        st.error(f"LLM config file is not valid TOML: {exc}")
        return {}


def _merge_litellm_settings(key_prefix: str, fallback_settings: dict[str, Any]) -> dict[str, Any]:
    """Merge default, file-level, section-level, and call-site LLM settings."""
    config = _load_litellm_config()
    settings = dict(DEFAULT_LLM_SETTINGS)
    extra_params: dict[str, Any] = {}

    for source in (
        fallback_settings,
        config.get("defaults", {}),
        config.get(key_prefix, {}),
    ):
        source_extra_params = source.get("extra_params", {}) or {}
        if isinstance(source_extra_params, dict):
            extra_params.update(source_extra_params)
        settings.update({
            key: value
            for key, value in source.items()
            if key != "extra_params"
        })

    for key in LITELLM_PASSTHROUGH_KEYS:
        if _has_value(settings.get(key)):
            extra_params[key] = settings[key]

    settings["extra_params"] = extra_params
    return settings


def render_litellm_sidebar(
        key_prefix: str,
        default_model: str | None = None,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
        api_key_label: str | None = None,
        model_label: str | None = None,
        missing_key_message: str | None = None,
        env_var_name: str | None = None,
        supported_models: list[str] | None = None,
        require_api_key: bool | None = None,
) -> LiteLLMClient | None:
    """Render the API-key field and return a LiteLLM client configured from local TOML."""
    _ = model_label, supported_models
    fallback_settings: dict[str, Any] = {}
    if default_model is not None:
        fallback_settings["model"] = default_model
    if reasoning_effort is not None:
        fallback_settings["reasoning_effort"] = reasoning_effort
    if verbosity is not None:
        fallback_settings["verbosity"] = verbosity
    if api_key_label is not None:
        fallback_settings["api_key_label"] = api_key_label
    if missing_key_message is not None:
        fallback_settings["missing_key_message"] = missing_key_message
    if env_var_name is not None:
        fallback_settings["api_key_env_var"] = env_var_name
    if require_api_key is not None:
        fallback_settings["missing_key_stops_page"] = require_api_key

    settings = _merge_litellm_settings(key_prefix, fallback_settings)
    model = _optional_string(settings.get("model"))
    if not model:
        st.error("LLM config must define a model.")
        return None

    configured_api_key_label = (
        _optional_string(settings.get("api_key_label"))
        or DEFAULT_LLM_SETTINGS["api_key_label"]
    )
    configured_env_var = _optional_string(settings.get("api_key_env_var")) or "OPENAI_API_KEY"

    api_key_input = st.text_input(
        configured_api_key_label,
        type="password",
        value=os.environ.get(configured_env_var, ""),
        key=f"{key_prefix}_api_key",
    )
    st.markdown(_PASSWORD_TOGGLE_STYLE, unsafe_allow_html=True)

    api_key = (
        api_key_input
        or os.environ.get(configured_env_var)
        or _optional_string(settings.get("api_key_fallback"))
    )
    if not api_key and _coerce_bool(settings.get("api_key_required", True)):
        if _coerce_bool(settings.get("missing_key_stops_page", True)):
            st.error(_optional_string(settings.get("missing_key_message")) or "Please configure API key.")
            st.stop()
        st.info("Add an API key when you are ready for AI-assisted generation.")
        return None

    return LiteLLMClient(
        model=model,
        api_key=api_key,
        reasoning_effort=_optional_string(settings.get("reasoning_effort")),
        verbosity=_optional_string(settings.get("verbosity")),
        extra_params=settings["extra_params"],
    )
