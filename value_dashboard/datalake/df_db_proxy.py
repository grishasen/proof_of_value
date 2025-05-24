import os
from typing import Optional

import duckdb
import polars as pl

from value_dashboard.utils.config import get_config


class PolarsDuckDBProxy:
    def __init__(self):
        try:
            os.makedirs("db")
        except FileExistsError:
            pass
        variant = get_config()['variants']['name']
        self.connection = duckdb.connect('db/pov_data_' + variant + '.duckdb')
        self._tables = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def sql(self, query: str, params: Optional[list] = None):
        return self.connection.sql(query, params=params)

    def close(self):
        if hasattr(self, "connection") and self.connection:
            self.connection.close()
            self.connection = None
            self._tables.clear()

    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        import re
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
            raise ValueError("Invalid table name.")
        return name

    def store_dataframe(self, df: pl.DataFrame, table_name: str):
        table_name = self._sanitize_identifier(table_name)
        if not self.is_dataframe_exist(table_name):
            self.connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        else:
            self.connection.execute(f"INSERT INTO {table_name} SELECT * FROM df")
        self._tables.add(table_name)

    def drop_dataframe(self, table_name: str):
        table_name = self._sanitize_identifier(table_name)
        self.connection.execute(f"DROP TABLE IF EXISTS {table_name}")
        if table_name in self._tables:
            self._tables.remove(table_name)

    def get_dataframe(self, table_name: str) -> pl.DataFrame:
        table_name = self._sanitize_identifier(table_name)
        return self.connection.execute(f"SELECT * FROM {table_name}").pl()

    def is_dataframe_exist(self, table_name: str):
        exists = self.connection.execute(f"""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = '{table_name}'
        """).fetchone()[0] > 0
        return exists
