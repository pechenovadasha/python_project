import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def save_centers_to_file(pupil_centers, glints_center, reference_centers, filename="pupil_centers_2.csv"):
    """
    Сохраняет координаты центров в CSV файл
    pupil_centers = [pupilL, pupilR] где pupilL = (x, y)
    reference_centers = [(x1, y1), (x2, y2)]
    """
    pupilL, pupilR = pupil_centers
    refL, refR = reference_centers
    glintL, glintR = glints_center

    
   
    if  pupilL is not None and len(pupilL) >= 2:
        pupilL_x, pupilL_y = float(pupilL[0]), float(pupilL[1])
    else:
        pupilL_x, pupilL_y = 0.0, 0.0  

    if  pupilR is not None and len(pupilR) >= 2:
        pupilR_x, pupilR_y = float(pupilR[0]), float(pupilR[1])
    else:
        pupilR_x, pupilR_y = 0.0, 0.0  

    if  glintL is not None and len(glintL) >= 2:
        glintL_x, glintL_y = float(glintL[0]), float(glintL[1])
    else:
        glintL_x, glintL_y = 0.0, 0.0  

    if  glintR is not None and len(glintR) >= 2:
        glintR_x, glintR_y = float(glintR[0]), float(glintR[1])
    else:
        glintR_x, glintR_y = 0.0, 0.0  

    if  refL is not None and len(refL) >= 2:
        refL_x, refL_y = float(refL[0]), float(refL[1])
    else:
        refL_x, refL_y = 0.0, 0.0  

    if  refR is not None and len(refR) >= 2:
        refR_x, refR_y = float(refR[0]), float(refR[1])
    else:
        refR_x, refR_y = 0.0, 0.0  
    

    diffL = ((pupilL_x - refL_x)**2 + (pupilL_y - refL_y)**2)**0.5
    diffR = ((pupilR_x - refR_x)**2 + (pupilR_y - refR_y)**2)**0.5
    total_diff = (diffL + diffR) / 2  # средняя разница
    
    # Записываем в файл
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Если файл пустой, добавляем заголовок
        if file.tell() == 0:
            writer.writerow([
                'timestamp', 
                'pupilL_x', 'pupilL_y', 'refL_x', 'refL_y', 'diffL', 'glintL_x', 'glintL_y',
                'pupilR_x', 'pupilR_y', 'refR_x', 'refR_y', 'diffR', 'glintR_x', 'glintR_y',
                'total_diff'
            ])
        
        writer.writerow([
            datetime.now().isoformat(),
            pupilL_x, pupilL_y, refL_x, refL_y, diffL, glintL_x, glintL_y,
            pupilR_x, pupilR_y, refR_x, refR_y, diffR, glintR_x, glintR_y,
            total_diff
        ])
    
    return total_diff


def calculate_simple_rms(csv_file_path):
    """
    Упрощенная версия для вычисления только RMS ошибок
    """
    df = pd.read_csv(csv_file_path)
    
    # Вычисляем ошибки для каждого кадра
    errors_L = np.sqrt((df['pupilL_x'] - df['refL_x'])**2 + (df['pupilL_y'] - df['refL_y'])**2)
    errors_R = np.sqrt((df['pupilR_x'] - df['refR_x'])**2 + (df['pupilR_y'] - df['refR_y'])**2)
    
    # RMS ошибки
    rms_L = np.sqrt(np.mean(errors_L**2))
    rms_R = np.sqrt(np.mean(errors_R**2))
    
    return rms_L, rms_R

def plot_eyes_separate(csv_file, timestamp_start=None, timestamp_end=None, separator=100):
    """
    Отрисовывает графики координат левого и правого глаза на отдельных фигурах
    
    Parameters:
    csv_file (str): путь к CSV файлу
    timestamp_start (float): начальное время для отображения (опционально)
    timestamp_end (float): конечное время для отображения (опционально)
    """
    
    # Чтение данных
    df = pd.read_csv(csv_file)


    x = range(len(df))
    max_x = len(df)

    plt.figure(figsize=(12, 6))
    for i in range(0, max_x, separator):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    
    # ФИГУРА 1: Левый глаз

    plt.plot(x, df['pupilL_x'], label='pupilL_x', color='yellow', marker='.', linestyle='', markersize=2)
    plt.plot(x, df['pupilL_y'], label='pupilL_y', color='red',  marker='.', linestyle='', markersize=2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    

    # ФИГУРА 2: Правый глаз
    plt.plot(x, df['pupilR_x'], label='pupilR_x', color='black',  marker='.', linestyle='', markersize=2)
    plt.plot(x, df['pupilR_y'], label='pupilR_y', color='brown',  marker='.', linestyle='', markersize=2)
    plt.title('Eyes - x and y coordinate')
    plt.xlabel('Frame number')
    plt.ylabel('Coordinate')
    plt.legend()
    plt.tight_layout()
    plt.show()



def show_saved_stats(filename, columns_to_show=None):

    if columns_to_show is None:
        columns_to_show = ['Folder', 'Processed %', 'Not processed', 'RMS Left', 'RMS Right']
    

    df = pd.read_csv(filename)
    

    display_df = df[columns_to_show]
    

    print("Saved Statistics:")
    print("=" * 60)
    print(display_df.to_string(index=False, float_format="%.4f"))


def show_stat(stat, total_files=2700, save_to_file=True):
    data = []
    total_files = 2700
    for dir_name, metrics in stat.items():
        processed_files = metrics.get('error', 0)
        percentage = (100 - (processed_files / total_files) * 100) if total_files > 0 else 0
        data.append({
            'Folder': dir_name,
            'Processed %': f"{percentage:.1f}%",
            'Not processed': metrics.get('error', 'N/A'),
            'RMS Left': metrics.get('rms_left', 'N/A'),
            'RMS Right': metrics.get('rms_right', 'N/A'),
            'Error Stats': metrics.get('error_stat', 'N/A')
        })
    
    df = pd.DataFrame(data)
    
    # Select only the columns you want to display
    columns_to_show = ['Folder', 'Processed %', 'Not processed', 'RMS Left', 'RMS Right']
    display_df = df[columns_to_show]
    
    # Display settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print("Statistics Table:")
    print("=" * 60)
    print(display_df.to_string(index=False))

    if save_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        csv_filename = f"statistics/statistics_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\nData saved to: {csv_filename}")



if __name__ == "__main__":
    filename = "results/dataset.csv"
    plot_eyes_separate(filename)

   