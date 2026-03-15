import atexit
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
_DATASET_EXPORT_TEMP_DIRS = set()


def _cleanup_dataset_export_temp_dir(temp_dir: str | None):
    if not temp_dir:
        return
    shutil.rmtree(temp_dir, ignore_errors=True)
    _DATASET_EXPORT_TEMP_DIRS.discard(temp_dir)


@atexit.register
def _cleanup_dataset_export_temp_dirs():
    for temp_dir in list(_DATASET_EXPORT_TEMP_DIRS):
        _cleanup_dataset_export_temp_dir(temp_dir)


def _create_dataset_export_temp_dir(tmp_folder: str) -> str:
    temp_dir = tempfile.mkdtemp(prefix="dataset_export_", dir=tmp_folder)
    _DATASET_EXPORT_TEMP_DIRS.add(temp_dir)
    return temp_dir


def _copy_stream_with_newline(src, dst):
    last_byte = None
    while True:
        chunk = src.read(1024 * 1024)
        if not chunk:
            break
        dst.write(chunk)
        last_byte = chunk[-1]
    if last_byte not in (None, ord("\n")):
        dst.write(b"\n")


def _normalized_archive_output_path(temp_dir: str, archive_path: str, index: int) -> str:
    archive_name = Path(archive_path).name
    for suffix in (".tar.gz", ".tgz", ".gzip", ".gz", ".zip"):
        if archive_name.endswith(suffix):
            archive_name = archive_name[: -len(suffix)]
            break
    safe_name = archive_name.replace(os.sep, "_") or f"archive_{index}"
    return os.path.join(temp_dir, f"{index:04d}_{safe_name}.ndjson")


def _normalize_gzip_archive(file_path: str, output_path: str):
    with gzip.open(file_path, "rb") as src, open(output_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


def _normalize_zip_archive(file_path: str, output_path: str):
    with zipfile.ZipFile(file_path, "r") as zip_ref, open(output_path, "wb") as dst:
        json_members = sorted(
            name
            for name in zip_ref.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in {".json", ".ndjson"}
        )
        if not json_members:
            raise Exception(f"No JSON members found in zip archive: {file_path}")
        for member_name in json_members:
            with zip_ref.open(member_name, "r") as src:
                _copy_stream_with_newline(src, dst)


def _normalize_tar_gz_archive(file_path: str, output_path: str):
    with tarfile.open(file_path, "r:gz") as tar_ref, open(output_path, "wb") as dst:
        json_members = sorted(
            member for member in tar_ref.getmembers()
            if member.isfile() and Path(member.name).suffix.lower() in {".json", ".ndjson"}
        )
        if not json_members:
            raise Exception(f"No JSON members found in tar archive: {file_path}")
        for member in json_members:
            src = tar_ref.extractfile(member)
            if src is None:
                continue
            with src:
                _copy_stream_with_newline(src, dst)


def _normalize_compressed_exports(file_paths: list[str], temp_dir: str) -> list[str]:
    normalized_files = []
    for index, file_path in enumerate(file_paths):
        suffixes = Path(file_path).suffixes
        output_path = _normalized_archive_output_path(temp_dir, file_path, index)
        if suffixes[-2:] == [".tar", ".gz"] or (suffixes and suffixes[-1] == ".tgz"):
            _normalize_tar_gz_archive(file_path, output_path)
        elif suffixes and suffixes[-1] in [".gzip", ".gz"]:
            _normalize_gzip_archive(file_path, output_path)
        elif suffixes and suffixes[-1] == ".zip":
            _normalize_zip_archive(file_path, output_path)
        else:
            raise Exception(f"Unsupported compressed file extension: {file_path}")
        normalized_files.append(output_path)
    return normalized_files


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

    files = [resolve_path(f) for f in file_names]
    files = [f for f in files if f]
    if not files:
        raise Exception("No valid input files found.")

    if ext in [".json"]:
        if verbose:
            logger.info("Reading JSON files:", files)
        if lazy:
            df = pl.scan_ndjson(files)
        else:
            df = pl.read_ndjson(files)
    elif ext in [".parquet"]:
        if verbose:
            logger.info("Reading Parquet files:", files)
        if lazy:
            df = pl.scan_parquet(files, cache=False, missing_columns='insert', extra_columns='ignore')
        else:
            df = pl.read_parquet(files, missing_columns='insert', allow_missing_columns=True)
    elif ext in [".gzip", ".gz", ".zip", ".tgz"]:
        temp_dir = _create_dataset_export_temp_dir(tmp_folder)
        try:
            normalized_files = _normalize_compressed_exports(files, temp_dir)
            if not normalized_files:
                raise Exception("No valid extracted files found in compressed archives.")
            if lazy:
                df = pl.scan_ndjson(normalized_files, infer_schema_length=100000)
            else:
                df = pl.read_ndjson(normalized_files, infer_schema_length=100000)
                _cleanup_dataset_export_temp_dir(temp_dir)
        except Exception:
            _cleanup_dataset_export_temp_dir(temp_dir)
            raise
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
