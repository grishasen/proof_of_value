from functools import lru_cache

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


@lru_cache(maxsize=1)
def init_plotly_theme():
    adjusted_colors = [
        "#3498db", "#2ecc71", "#e74c3c", "#9b59b6",
        "#f39c12", "#d35400", "#1abc9c", "red", "green", "blue", "#7b3f00",
        "purple", "yellow", "black", "darkblue", "darkgreen", "darkred", "#4b006e",
        "steelblue", "lightgreen", "lightblue", "#fffacd"
    ]

    dark_palette = [
        "#3498DB", "#C9D1D9", "#E74C3C", "#F39C12", "#9B59B6",
        "#1ABC9C", "#D35400", "#4682B4", "#4B006E", "#7B3F00",
        "#FFFF00", "#ADD8E6"
    ]

    light_palette = [
        "#2C82C9", "#2ECC71", "#D64541", "#8E44AD", "#E67E22",
        "#16A085", "#2C3E50", "#B9770E", "#4B006E", "#1F618D",
        "#117A65", "#8B0000"
    ]

    base = pio.templates.default
    base_st = st.get_option("theme.base") or "light"
    colorway = dark_palette if base_st == "dark" else light_palette
    new_template = pio.templates[base].update(layout=go.Layout(colorway=colorway))
    pio.templates["cdhvd"] = new_template
    pio.templates.default = "cdhvd"
