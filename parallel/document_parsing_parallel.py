import json
import csv
import concurrent.futures
import multiprocessing

def parse_json(file_path):
    """parsing json files"""
    with open(file_path, 'r') as f:
        return json.load(f)
def parse_csv(file_path):
    """parsing csv files"""
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        return [row for row in reader]
def parallel_document_parsing(file_paths, parser_function, num_threads=None):
    if not file_paths:
        return []

    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    chunk_size = max(1, len(file_paths) // num_threads)
    chunks = [file_paths[i:i + chunk_size] for i in range(0, len(file_paths), chunk_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(lambda chunk: [parser_function(path) for path in chunk], chunks)

    return [item for sublist in results for item in sublist]