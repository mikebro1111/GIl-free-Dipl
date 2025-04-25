from parallel.regex_parallel import parallel_regex_search
from parallel.document_parsing_parallel import parallel_document_parsing
from parallel.document_parsing_parallel import parse_csv
import csv
import os
import time

def test_regex_parallel():
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is great for data science and machine learning.",
        "Regular expressions are powerful tools for text processing."
    ]
    pattern = r'\b\w+\b'
    start_time = time.time()
    result = parallel_regex_search(pattern, texts)
    print("Regex Parallel Search Result:", result)
    print("Time taken for regex:", time.time() - start_time)

def test_document_parsing_parallel():
    file_paths = ['file1.csv', 'file2.csv']
    for i, path in enumerate(file_paths):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'age'])
            writer.writerow([f'Name {i+1}', f'{20+i}'])
    
    start_time = time.time()
    result = parallel_document_parsing(file_paths, parse_csv)
    print("Document Parsing Result:", result)
    print("Time taken for document parsing:", time.time() - start_time)
    for path in file_paths:
        os.remove(path)

if __name__ == "__main__":
    print("Testing Regex Parallel:")
    test_regex_parallel()
    
    print("\nTesting Document Parsing Parallel:")
    test_document_parsing_parallel()