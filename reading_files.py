import cv2
import os
import re
import numpy as np
import time 

from processing import recieve_centers
from centers_analisis import calculate_simple_rms, show_stat


def natural_sort_key(filename):

    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', filename)]

def process_folder_images(folder_path):

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

    while i < len(bmp_files) - 1 and i < 100000:
        if i + 1 >= len(bmp_files):
            break
        
        # Точно знаем что 1 файл имеет светлый зрачок
        light_file = os.path.join(folder_path, bmp_files[i])
        dark_file = os.path.join(folder_path, bmp_files[i + 1])
        
        print(f"Processe: {bmp_files[i]} и {bmp_files[i + 1]}")
        

        dark_pupil_frame = cv2.imread(dark_file)
        bright_pupil_frame = cv2.imread(light_file)
        
  
        if dark_pupil_frame is None or bright_pupil_frame is None:
            print(f"file upload error!")
            continue
        
        # вычитаем файлы для получения области зрачка, весь фон будет темный и только зрачки яркие
        diff = cv2.subtract(bright_pupil_frame, dark_pupil_frame)

        # основная функция для получения центров зрачков, возращает ошибки из-за которых не были найдены зрачки.
        # Данная функция записывает всю статистку (центры зрачков, отбелсков) в csv файл, имя файла такое же как название папки с кадрами 
        error = recieve_centers(diff, dark_pupil_frame, bright_pupil_frame, dir_name)

        
        amount += 1

        if error != []:
            error_stat[bmp_files[i]] = error.copy()
            errors += 1

        i += 2

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
    folders = ['/Users/user/Desktop/Skoltech/python_project/dataset']


    all_stat = {}

    for f in folders:
        stat  = process_folder_images(f)
        all_stat.update(stat)
    show_stat(all_stat, total_files=900, save_to_file=False)

    print("Processing of all files completed")

if __name__ == "__main__":
    main()