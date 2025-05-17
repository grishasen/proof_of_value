from functools import lru_cache

import plotly.graph_objects as go
import plotly.io as pio


@lru_cache(maxsize=1)
def init_plotly_theme():
    adjusted_colors = [
        "#3498db", "#2ecc71", "#e74c3c", "#9b59b6",
        "#f39c12", "#d35400", "#1abc9c", "red", "green", "blue", "#7b3f00",
        "purple", "yellow", "black", "darkblue", "darkgreen", "darkred", "#4b006e",
        "steelblue", "lightgreen", "lightblue", "#fffacd"
    ]

    base = pio.templates["plotly_white"]
    new_template = base.update(layout=go.Layout(colorway=adjusted_colors))
    pio.templates["cdhvd"] = new_template

    pio.templates.default = "cdhvd"
