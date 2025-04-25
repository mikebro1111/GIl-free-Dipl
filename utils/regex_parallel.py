import re
import time
from multiprocessing import Pool, cpu_count


def read_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def find_dates(text):
    pattern = r'\b(?:\d{1,2}[\/\-\.]){2}\d{2,4}\b'
    return re.findall(pattern, text)


def chunkify(data, n):
    k, m = divmod(len(data), n)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def process_chunk(chunk):
    return find_dates(chunk)


def sequential_regex(text):
    start = time.time()
    matches = find_dates(text)
    end = time.time()
    return matches, end - start


def parallel_regex(text):
    start = time.time()
    chunks = chunkify(text, cpu_count())
    with Pool(cpu_count()) as pool:
        results = pool.map(process_chunk, chunks)
    matches = [match for sublist in results for match in sublist]
    end = time.time()
    return matches, end - start


if __name__ == '__main__':
    text = read_text_file('data/test_data.txt')
    seq_matches, seq_time = sequential_regex(text)
    par_matches, par_time = parallel_regex(text)
    print(f'Sequential: {len(seq_matches)} matches in {seq_time:.4f}s')
    print(f'Parallel:   {len(par_matches)} matches in {par_time:.4f}s')