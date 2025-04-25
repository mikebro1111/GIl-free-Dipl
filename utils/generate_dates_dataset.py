import os
import argparse
import random
from datetime import datetime, timedelta

def generate_random_date(start_year=1900, end_year=2100):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    random_date = start + timedelta(days=random_days)
    return random_date.strftime('%d.%m.%Y')

def generate_dataset(n):
    return [generate_random_date() for _ in range(n)]

def save_to_file(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for date in data:
            f.write(f"{date}\n")

def main():
    parser = argparse.ArgumentParser(description="generator of data for parsing")
    parser.add_argument('--num', type=int, default=1000, help='data')
    parser.add_argument('--output', type=str, default='data/test_data.txt', help='file for saving')

    args = parser.parse_args()
    dataset = generate_dataset(args.num)
    save_to_file(dataset, args.output)
    print(f"Saved {args.num} data in {args.output}")

if __name__ == '__main__':
    main()
