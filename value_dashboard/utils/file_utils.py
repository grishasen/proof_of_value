import concurrent.futures
import gzip
import os
import queue
import shutil
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

import polars as pl


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


def read_dataset_export(file_name, src_folder=".",
                        tmp_folder=None,
                        lazy=False,
                        verbose=False):
    export_file = None
    error_reason = ""
    tmp_folder = tmp_folder if tmp_folder else tempfile.gettempdir()

    if file_name.endswith(".json"):
        error_reason = "Error reading JSON file"
        if os.path.exists(file_name):
            export_file = file_name
        elif os.path.exists(os.path.join(src_folder, file_name)):
            export_file = os.path.join(src_folder, file_name)
        if export_file and verbose:
            print(error_reason, export_file)
        if export_file:
            if lazy:
                df = pl.scan_ndjson(export_file)
            else:
                df = pl.read_ndjson(export_file)
    elif file_name.endswith(".parquet"):
        error_reason = "Error reading PARQUET file"
        if os.path.exists(file_name):
            export_file = file_name
        elif os.path.exists(os.path.join(src_folder, file_name)):
            export_file = os.path.join(src_folder, file_name)
        if export_file and verbose:
            print(error_reason, export_file)
        if export_file:
            if lazy:
                df = pl.scan_parquet(export_file)
            else:
                df = pl.read_parquet(export_file)
    elif file_name.endswith(".gzip") or file_name.endswith(".gz"):
        error_reason = "Error reading GZIP file"
        if os.path.exists(file_name):
            export_file = file_name
        elif os.path.exists(os.path.join(src_folder, file_name)):
            export_file = os.path.join(src_folder, file_name)
        if export_file and verbose:
            print(error_reason, export_file)
        if export_file:
            output_file_or_dir = extract_compressed_file(export_file)
            if lazy:
                df = pl.read_ndjson(output_file_or_dir, infer_schema_length=100000)
                os.remove(output_file_or_dir)
                df = df.lazy()
            else:
                df = pl.read_ndjson(output_file_or_dir, infer_schema_length=100000)
                os.remove(output_file_or_dir)
    else:
        zip_file = file_name
        if file_name.endswith(".zip"):
            error_reason = "Error reading ZIP file"
            if os.path.exists(file_name):
                zip_file = file_name
            elif os.path.exists(os.path.join(src_folder, file_name)):
                zip_file = os.path.join(src_folder, file_name)
            if verbose:
                print(error_reason, zip_file)

            if os.path.exists(zip_file):
                error_reason = "Error extracting data.json"
                if verbose:
                    print(error_reason, zip_file)

                export_file = os.path.join(tmp_folder, "data.json")
                if os.path.exists(export_file):
                    os.remove(export_file)

                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    all_zip_entries = zip_ref.namelist()
                    json_file_in_zip = [s for s in all_zip_entries if "data.json" in s]
                    if verbose:
                        print("data.json in zip file:", json_file_in_zip, zip_file)

                    for file in json_file_in_zip:
                        zip_ref.extract(file, tmp_folder)
                        export_file = os.path.join(tmp_folder, file)

                if not os.path.exists(export_file):
                    raise Exception(f"Dataset zipfile {zip_file} does not have \"data.json\"")
                if lazy:
                    df = pl.scan_ndjson(export_file, infer_schema_length=100000)
                else:
                    df = pl.read_ndjson(export_file, infer_schema_length=100000)
                    os.remove(export_file)

    if export_file is None:
        raise Exception(f"Dataset export not found {error_reason}")
    return df


class PooledFileReader:
    def __init__(self, num_slots, file_type):
        self.queue = queue.Queue(maxsize=num_slots)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_slots)
        self.file_paths = []
        self.shutdown_flag = False
        self.file_type = file_type

    def read_file(self, file_path):
        if self.file_type == 'parquet':
            ih = pl.read_parquet(file_path)
        elif self.file_type == 'pega_ds_export':
            ih = read_dataset_export(file_path)
        else:
            raise Exception("File type not supported")
        return ih

    def worker(self, file_path):
        try:
            file_content = self.read_file(file_path)
            self.queue.put(file_content, block=True)  # This will block if the queue is full
        except Exception as e:
            self.queue.put(f"Error reading file {file_path}: {e}")

    def submit_files(self, file_paths):
        self.file_paths.extend(file_paths)

    def process_files(self):
        while not self.shutdown_flag or self.file_paths:
            if self.file_paths and not self.queue.full():
                file_path = self.file_paths.pop(0)
                self.executor.submit(self.worker, file_path)
            else:
                time.sleep(0.1)  # Sleep for 100 ms

    def get_result(self):
        return self.queue.get(block=True)  # This will block if the queue is empty

    def shutdown(self):
        self.shutdown_flag = True
        self.executor.shutdown(wait=True)


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
