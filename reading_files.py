import argparse
import cv2
import os
import re
import numpy as np
import time 
import pstats
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from processing import recieve_centers
# from centers_analisis import calculate_simple_rms, show_stat


def natural_sort_key(filename):

    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', filename)]


def _process_pair(light_file, dark_file, folder_path, dir_name):
    print(f"Processe: {light_file} и {dark_file}")

    dark_path = os.path.join(folder_path, dark_file)
    light_path = os.path.join(folder_path, light_file)

    dark_pupil_frame = cv2.imread(dark_path)
    bright_pupil_frame = cv2.imread(light_path)

    if dark_pupil_frame is None or bright_pupil_frame is None:
        return {
            'processed': False,
            'pair_key': light_file,
            'error': ["file upload error!"]
        }

    diff = cv2.subtract(bright_pupil_frame, dark_pupil_frame)
    error = recieve_centers(diff, dark_pupil_frame, bright_pupil_frame, dir_name)

    return {
        'processed': True,
        'pair_key': light_file,
        'error': error
    }

def process_folder_images(folder_path, num_workers):

    # Функция открывает файл и считывает файлы парами

    all_files = os.listdir(folder_path)
    bmp_files = [f for f in all_files if f.lower().endswith('.bmp')]
    
    bmp_files.sort(key=natural_sort_key)
    
    print(f"Find {len(bmp_files)} BMP files")

    i = 0
    amount = 0
    errors = 0
    error_stat = {}
    stat = dict()
    dir_name = os.path.basename(folder_path)
    pairs = []

    while i < len(bmp_files) - 1 and i < 20000:
        if i + 1 >= len(bmp_files):
            break
        pairs.append((bmp_files[i], bmp_files[i + 1]))
        i += 2

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_process_pair, light, dark, folder_path, dir_name)
            for light, dark in pairs
        ]

        for future in as_completed(futures):
            result = future.result()
            if not result['processed']:
                print("file upload error!")
                errors += 1
                error_stat[result['pair_key']] = result['error']
                continue

            amount += 1

            if result['error'] != []:
                error_stat[result['pair_key']] = result['error'].copy()
                errors += 1

    print(error_stat)
    print(f"{amount} frames was sent for processing")
    print(f"{errors} frames doesn't process")
    file = 'results/' + dir_name + '.csv'
    stat[dir_name] = {
    'error': errors,
    'error_stat': error_stat}
    
    return stat

 
def main():

    # На вход принимается путь до папки с записями в формате bmp
    parser = argparse.ArgumentParser(description="Process bright/dark BMP pairs in parallel.")
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=14,
        help="Количество потоков для обработки пар файлов (по умолчанию 4)."
    )
    parser.add_argument(
        "-f", "--folders",
        nargs="+",
        default=["dataset"],
        help="Пути до папок с BMP-файлами."
    )
    args = parser.parse_args()

    num_workers = max(1, args.workers)

    all_stat = {}

    for f in args.folders:
        stat  = process_folder_images(f, num_workers)
        all_stat.update(stat)
    # show_stat(all_stat, total_files=900, save_to_file=False)

    print("Processing of all files completed")

if __name__ == "__main__":
    main()
