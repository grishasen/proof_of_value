import gzip
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import polars as pl

from value_dashboard.utils.logger import get_logger
from value_dashboard.utils.timer import timed

logger = get_logger(__name__)

@timed
def extract_compressed_file(file_path) -> str:
    """
    Extracts .gz, .gzip, .tar.gz, or .tgz files.
    For .gz/.gzip: extracts to a file with same name minus extension.
    For .tar.gz/.tgz: extracts all files to a directory with same name as archive.
    """
    file_path = Path(file_path)
    suffixes = file_path.suffixes
    file_name_no_ext = file_path.with_suffix('')

    if suffixes[-2:] == ['.tar', '.gz'] or suffixes[-1] == '.tgz':
        extract_dir = file_name_no_ext.with_suffix('')
        extract_dir.mkdir(exist_ok=True)
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)
        return extract_dir

    elif suffixes[-1] in ['.gz', '.gzip']:
        output_path = file_name_no_ext
        with gzip.open(file_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return output_path
    else:
        raise Exception(f"File cannot be extracted: {file_path}")

@timed
def read_dataset_export(
        file_names,
        src_folder=".",
        tmp_folder=None,
        lazy=False,
        verbose=False
):
    if isinstance(file_names, str):
        file_names = [file_names]
    if not file_names:
        raise ValueError("No files provided.")

    tmp_folder = tmp_folder if tmp_folder else tempfile.gettempdir()
    ext = os.path.splitext(file_names[0])[1].lower()
    df = pl.DataFrame() if not lazy else pl.LazyFrame()

    def resolve_path(f):
        if os.path.exists(f):
            return f
        elif os.path.exists(os.path.join(src_folder, f)):
            return os.path.join(src_folder, f)
        return None

    if ext in [".json"]:
        files = [resolve_path(f) for f in file_names]
        files = [f for f in files if f]
        if not files:
            raise Exception("No valid JSON files found.")
        if verbose:
            logger.info("Reading JSON files:", files)
        if lazy:
            df = pl.scan_ndjson(files)
        else:
            df = pl.read_ndjson(files)
    elif ext in [".parquet"]:
        files = [resolve_path(f) for f in file_names]
        files = [f for f in files if f]
        if not files:
            raise Exception("No valid Parquet files found.")
        if verbose:
            logger.info("Reading Parquet files:", files)
        if lazy:
            df = pl.scan_parquet(files, cache=False, missing_columns='insert', extra_columns='ignore')
        else:
            df = pl.read_parquet(files, missing_columns='insert', allow_missing_columns=True)
    elif ext in [".gzip", ".gz", ".zip"]:
        extracted_files = []
        for f in file_names:
            full_f = f if os.path.exists(f) else os.path.join(src_folder, f)
            if not os.path.exists(full_f):
                continue
            if ext in [".gzip", ".gz"]:
                extracted_path = extract_compressed_file(full_f)
                extracted_files.append(extracted_path)
            elif ext == ".zip":
                with zipfile.ZipFile(full_f, 'r') as zip_ref:
                    json_files = [name for name in zip_ref.namelist() if name.endswith('.json')]
                    for json_name in json_files:
                        zip_ref.extract(json_name, tmp_folder)
                        extracted_files.append(os.path.join(tmp_folder, json_name))
        if not extracted_files:
            raise Exception("No valid extracted files found in compressed archives.")
        if lazy:
            df = pl.scan_ndjson(extracted_files, infer_schema_length=100000)
        else:
            df = pl.read_ndjson(extracted_files, infer_schema_length=100000)
            for f in extracted_files:
                os.remove(f)
    else:
        raise Exception(f"Unsupported file extension: {ext}")

    return df


def detect_delimiter(filename: str, n=2):
    sample_lines = head(filename, n)
    common_delimiters = [',', ';', '\t', ' ', '|', ':']
    for d in common_delimiters:
        ref = sample_lines[0].count(d)
        if ref > 0:
            if all([ref == sample_lines[i].count(d) for i in range(1, n)]):
                return d
    return ','


def head(filename: str, n: int):
    try:
        with open(filename) as f:
            head_lines = [next(f).rstrip() for x in range(n)]
    except StopIteration:
        with open(filename) as f:
            head_lines = f.read().splitlines()
    return head_lines
