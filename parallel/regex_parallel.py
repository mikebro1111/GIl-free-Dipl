import re
import concurrent.futures
import multiprocessing

def regex_search(pattern, texts):
    return [re.findall(pattern, text) for text in texts]
def parallel_regex_search(pattern, texts, num_threads=None):
    if not texts:
        return []
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()
    chunk_size = max(1, len(texts) // num_threads)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(lambda chunk: regex_search(pattern, chunk), chunks)
    return [item for sublist in results for item in sublist]