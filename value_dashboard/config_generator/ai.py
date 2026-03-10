import copy
import os
import pprint
import tomllib
import uuid

import polars as pl
import tomlkit
from jinja2 import Environment
from pandasai.core.prompts import BasePrompt

from value_dashboard.utils.config import set_config


def _safe_prompt_config_block(config: dict) -> str:
    """Render config text for the prompt without failing on unsupported nested values."""
    try:
        return tomlkit.dumps(config)
    except Exception:
        return pprint.pformat(config, sort_dicts=False, width=120)


def build_ai_config_prompt(
        file_name: str,
        approved_schema,
        approved_fields: list[str],
        template_config: dict,
        ih_config: dict,
) -> str:
    """Build the constrained AI prompt for metrics and report generation."""
    with pl.Config(tbl_cols=len(approved_schema), tbl_rows=len(approved_schema)):
        return f"""
You generate only valid TOML.

Task:
Using the approved Interaction History preprocessing config and the approved working schema,
generate only the TOML sections needed to configure dashboard behavior after data ingestion.

Output contract:
1. Output valid TOML only.
2. Do not include markdown, comments, prose, or code fences.
3. Output only these sections:
   - [metrics]
   - [reports]
   - [variants]
4. Do not output [ih], [holdings], [ux], [copyright], or [chat_with_data].
5. Never invent fields.
6. Only use approved fields listed below.
7. Use the template as the style and recipe baseline.
8. Keep only reports that can be mapped safely.
9. You may add useful reports when the approved schema clearly supports them.
10. Every referenced field must exist in the approved field list.

File name:
{file_name}

Approved fields:
{approved_fields}

Approved schema:
{approved_schema}

Approved IH config:
{_safe_prompt_config_block({"ih": ih_config})}

Template baseline:
{_safe_prompt_config_block(template_config)}

Hard rules:
1. Include Day, Month, Year, Quarter where relevant for time-based metrics and reports.
2. Keep metrics.global_filters limited to safe low-cardinality business dimensions from approved fields.
3. Keep metrics and reports internally consistent.
4. Remove any report that depends on a missing field.
5. If experiment fields are not available, omit experiment metrics and reports.
6. If CLV fields are not available, omit clv metrics and reports.
7. If descriptive properties are missing, keep descriptive only when supported by approved fields.

Final self-check:
1. Every metric group_by field exists in approved fields.
2. Every report metric exists in [metrics].
3. Every report field exists in approved fields or is a known metric score.
4. Output valid TOML only.
"""


def generate_ai_sections(llm, prompt: str) -> dict:
    """Call the LLM and parse the returned TOML sections."""
    env = Environment()
    instruction = BasePrompt()
    instruction.prompt = env.from_string(prompt)
    response_text = llm.call(instruction=instruction)
    response_text = response_text.replace("```toml", "").replace("```", "").strip()
    return tomllib.loads(response_text)


def build_final_config(template_config: dict, ih_config: dict, generated_sections: dict) -> dict:
    """Merge AI-generated metrics/report sections with the fixed template sections."""
    final_config = copy.deepcopy(template_config)
    final_config["ih"] = ih_config
    if "metrics" in generated_sections:
        final_config["metrics"] = generated_sections["metrics"]
    if "reports" in generated_sections:
        final_config["reports"] = generated_sections["reports"]
    if "variants" in generated_sections:
        final_config["variants"] = generated_sections["variants"]
    return final_config


def save_generated_config(config: dict) -> tuple[str, str]:
    """Persist a generated config to a temp file and activate it in the app."""
    os.makedirs("temp_configs", exist_ok=True)
    cfg_file_name = os.path.join("temp_configs", f"config_studio_{uuid.uuid4().hex}.toml")
    toml_text = tomlkit.dumps(config)
    with open(cfg_file_name, "w") as handle:
        handle.write(toml_text)
    set_config(cfg_file_name)
    return cfg_file_name, toml_text
