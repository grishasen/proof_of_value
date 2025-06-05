import os
from typing import Dict, Any

import pandas as pd
import streamlit as st
from PIL import Image
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

im = Image.open(os.path.join(os.path.dirname(__file__), "../img/favicon.ico"))


def get_page_configs():
    kwargs = {"layout": "wide",
              "page_icon": im}
    return kwargs


def highlight_and_format(val):
    """
    Highlight negative values in red and format with thousands separator.
    Display None or NaN values with dark grey background.
    """
    if pd.isna(val):
        bg = 'background-color: lightgrey;'
        theme = st.session_state['theme']
        if theme is None:
            bg = 'background-color: lightgrey;'
        else:
            if theme['base'] == 'dark':
                bg = 'background-color: darkgrey;'
        return bg
    elif isinstance(val, (int, float)):
        color = f'color: red;' if val < 0 else ''
        return color
    else:
        return ''


def align_column_types(df: pd.DataFrame):
    df = df.reindex(sorted(df.columns), axis=1)
    return df.convert_dtypes()


def format_dates(df: pd.DataFrame):
    for col in df.columns:
        if 'Month' == col:
            df['Month'] = pd.to_datetime(df['Month'], format='mixed')
        if 'Day' == col:
            df['Day'] = df["Day"].dt.strftime("%Y-%m-%d")
    df = df.style.format({'Month': '{:%b %Y}'}).background_gradient(cmap='YlGn')
    return df


def filter_dataframe(df: pd.DataFrame, case: bool = True) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        :param df: Original dataframe
        :param case: ignore case

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    random_key_base = pd.util.hash_pandas_object(df)

    df = df.copy()

    # Try to convert datetimes into standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect(
            "Filter dataframe on",
            df.columns,
            key=f"{random_key_base}_multiselect",
            label_visibility='collapsed',
            placeholder="Choose a column to filter dataframe on",
        )
        filters: Dict[str, Any] = dict()
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            if isinstance(df[column], pd.CategoricalDtype) or df[column].nunique() < 15:
                left.write("↳")
                filters[column] = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                    key=f"{random_key_base}_{column}",
                )
                df = df[df[column].isin(filters[column])]
            elif is_numeric_dtype(df[column]):
                left.write("↳")
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                filters[column] = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                    key=f"{random_key_base}_{column}",
                )
                df = df[df[column].between(*filters[column])]
            elif is_datetime64_any_dtype(df[column]):
                left.write("↳")
                filters[column] = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                    key=f"{random_key_base}_{column}",
                )
                if len(filters[column]) == 2:
                    filters[column] = tuple(map(pd.to_datetime, filters[column]))
                    start_date, end_date = filters[column]
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                left.write("↳")
                filters[column] = right.text_input(
                    f"Pattern in {column}",
                    key=f"{random_key_base}_{column}",
                    help="Use regular expression filtering for text values. E.g. ^(?!N/A) - values NOT EQUAL 'N/A'"
                )
                if filters[column]:
                    df = df[df[column].str.contains(filters[column], case=case)]

    return df
