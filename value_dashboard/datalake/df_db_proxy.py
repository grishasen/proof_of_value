import os
from typing import Optional

import duckdb
import polars as pl


class PolarsDuckDBProxy:
    def __init__(self):
        try:
            os.makedirs("db")
        except FileExistsError:
            pass
        self.connection = duckdb.connect("db/pov_data.duckdb")
        self._tables = set()

    def __del__(self):
        self.close()

    def sql(self, query: str, params: Optional[list] = None):
        return self.connection.sql(query, params=params)

    def close(self):
        if hasattr(self, "conn") and self.connection:
            self.connection.close()
            self.connection = None
            self._tables.clear()

    def store_dataframe(self, df: pl.DataFrame, table_name: str):
        self.connection.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
        self.connection.execute(f"INSERT INTO {table_name} SELECT * FROM df")
        self._tables.add(table_name)

    def drop_dataframe(self, table_name: str):
        self.connection.execute(f"DROP TABLE IF EXISTS {table_name}")
        if table_name in self._tables:
            self._tables.remove(table_name)

    def get_dataframe(self, table_name: str) -> pl.DataFrame:
        return self.connection.execute("SELECT * FROM " + table_name).pl()

    def is_dataframe_exist(self, table_name: str):
        exists = self.connection.execute(f"""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = '{table_name}'
        """).fetchone()[0] > 0
        return exists
