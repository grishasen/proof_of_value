import os

import streamlit as st
from PIL import Image

from value_dashboard.utils.config import get_config

cols = st.columns([0.1, 0.9])
image = Image.open(os.path.join(os.path.dirname(__file__), '../img/logo2.png'))
cols[0].image(image, use_container_width="always")
cols[1].title(get_config()['copyright']['name'])

f"""
Welcome to the early alpha version of the CDH value reporting application.

### About
The proposed solution involves an open-source reporting and dashboard application prototype, designed for discovery and insights' generation flexibility. 
This dashboard does not replace existing CDH reporting tools but complements them by offering customizable metrics and reports. 
The technology stack includes components like Streamlit, Polars, Plotly, and DuckDB to ensure efficient data processing and interactive visualizations.

Rigorous design of Key Performance Indicators (KPIs) and attribution models is crucial for linking marketing actions to business outcomes, such as customer lifetime value (CLV) and revenue/conversion rates. 

The dashboard aims to support decision-making by providing a clear visualization of data to secure stakeholders' buy-in and demonstrate tangible value.
### How To
To use the application, go to the **Configuration** page to edit config file or to the **Data Import** page directly 
where you can upload your data by selecting and loading 
ZIP or Parquet files. Once the data is imported, navigate to the Dashboard page to view the results. Here, you can 
apply various filters to interactively explore and analyze the visualized data.

"""
st.info(get_config()['variants']['description'])

footer_html = """<div style='text-align: center;'>
  <p>Developed with ❤️ by EMEA MDA Team</p>
  <p>Version """ + get_config()['copyright']['version'] + """</p>
</div>"""

st.markdown(footer_html, unsafe_allow_html=True)
