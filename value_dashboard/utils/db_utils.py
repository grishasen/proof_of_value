import logging
from datetime import datetime

from value_dashboard.datalake.df_db_proxy import PolarsDuckDBProxy
from value_dashboard.utils.logger import get_logger

logger = get_logger(__name__, logging.DEBUG)


def save_file_meta(proxy: PolarsDuckDBProxy, file_name: str):
    proxy.sql("CREATE TABLE IF NOT EXISTS metadata (filename TEXT PRIMARY KEY, timestamp DATETIME)", [])
    ts = datetime.now()
    proxy.sql("INSERT OR REPLACE INTO metadata (filename, timestamp) VALUES (?, ?)", [file_name, ts])


def get_file_meta(proxy: PolarsDuckDBProxy):
    proxy.sql("CREATE TABLE IF NOT EXISTS metadata (filename TEXT PRIMARY KEY, timestamp DATETIME)", [])
    return proxy.get_dataframe('metadata')


def drop_file_meta(proxy: PolarsDuckDBProxy):
    proxy.sql("DROP TABLE IF EXISTS metadata", [])


def drop_all_tables(proxy: PolarsDuckDBProxy):
    tables = proxy.sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main';", []).fetchall()
    for (table_name,) in tables:
        logger.info("Dropping table", extra={"table_name": table_name})
        proxy.sql(f"DROP TABLE IF EXISTS {table_name};", [])
