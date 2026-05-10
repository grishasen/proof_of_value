import os
from typing import Optional

import duckdb
import polars as pl

from value_dashboard.utils.config import get_config


class PolarsDuckDBProxy:
    """Persist and retrieve Polars dataframes through a DuckDB database file."""
    def __init__(self):
        """Open the variant-specific DuckDB database and initialize table tracking."""
        try:
            os.makedirs("db")
        except FileExistsError:
            pass
        variant = get_config()['variants']['name']
        self.connection = duckdb.connect('db/pov_data_' + variant + '.duckdb')
        self._tables = set()

    def __enter__(self):
        """Enter the DuckDB proxy context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the DuckDB proxy when leaving a context manager block."""
        self.close()

    def sql(self, query: str, params: Optional[list] = None):
        """Execute a SQL statement against the DuckDB connection."""
        return self.connection.sql(query, params=params)

    def close(self):
        """Close the DuckDB connection and clear cached table names."""
        if hasattr(self, "connection") and self.connection:
            self.connection.close()
            self.connection = None
            self._tables.clear()

    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        """Validate a table identifier before interpolating it into SQL."""
        import re
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
            raise ValueError("Invalid table name.")
        return name

    def store_dataframe(self, df: pl.DataFrame, table_name: str):
        """Store or append a Polars dataframe in a DuckDB table."""
        table_name = self._sanitize_identifier(table_name)
        if not self.is_dataframe_exist(table_name):
            self.connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        else:
            self.connection.execute(f"INSERT INTO {table_name} SELECT * FROM df")
        self._tables.add(table_name)

    def drop_dataframe(self, table_name: str):
        """Drop a dataframe table from DuckDB if it exists."""
        table_name = self._sanitize_identifier(table_name)
        self.connection.execute(f"DROP TABLE IF EXISTS {table_name}")
        if table_name in self._tables:
            self._tables.remove(table_name)

    def get_dataframe(self, table_name: str) -> pl.DataFrame:
        """Load a DuckDB table as a Polars dataframe."""
        table_name = self._sanitize_identifier(table_name)
        return self.connection.execute(f"SELECT * FROM {table_name}").pl()

    def is_dataframe_exist(self, table_name: str):
        """Return whether a DuckDB table exists for the dataframe name."""
        exists = self.connection.execute(f"""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = '{table_name}'
        """).fetchone()[0] > 0
        return exists
