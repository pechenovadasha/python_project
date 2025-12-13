import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import json
import os
from datetime import datetime

import json
import os
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime
import pandas as pd
from scipy import stats

def save_to_json(left_diameters, right_diameters):
    """
    Сохраняет массивы диаметров левого и правого зрачков в JSON файл.
    
    Args:
        left_diameters: список диаметров левого зрачка
        right_diameters: список диаметров правого зрачка
    """
    folder_path = "result"
    filename = f"pupil_diameters_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)
    
    # Если файл существует, загружаем данные
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Если файл поврежден или пустой, создаем новую структуру
            data = {
                "measurements": [],
                "session_info": {}
            }
    else:
        # Создаем новую структуру данных
        data = {
            "measurements": [],
            "session_info": {}
        }
    
    # Проверяем, что массивы одинаковой длины
    if len(left_diameters) != len(right_diameters):
        print(f"⚠️  Предупреждение: массивы разной длины (левый: {len(left_diameters)}, правый: {len(right_diameters)})")
        min_length = min(len(left_diameters), len(right_diameters))
        left_diameters = left_diameters[:min_length]
        right_diameters = right_diameters[:min_length]
    
    # Добавляем все измерения
    for i, (left_dia, right_dia) in enumerate(zip(left_diameters, right_diameters)):
        measurement = {
            "timestamp": datetime.now().isoformat(),
            "measurement_index": i,
            "left_pupil_diameter": float(left_dia) if left_dia is not None else None,
            "right_pupil_diameter": float(right_dia) if right_dia is not None else None,
            "average_diameter": (float(left_dia) + float(right_dia)) / 2 if left_dia is not None and right_dia is not None else None,
            "difference": abs(float(left_dia) - float(right_dia)) if left_dia is not None and right_dia is not None else None
        }
        data["measurements"].append(measurement)
    
    # Добавляем информацию о сессии
    data["session_info"] = {
        "session_start": datetime.now().isoformat(),
        "total_measurements": len(data["measurements"]),
        "left_diameters_count": len(left_diameters),
        "right_diameters_count": len(right_diameters),
        "file_created": datetime.now().isoformat()
    }
    
    # Сохраняем обновленные данные
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Сохранено {len(data['measurements'])} измерений в: {filepath}")
    metrics = analyze_pupil_data(filepath)
    if 'error' not in metrics:
        print_basic_metrics(metrics)
    else:
        print(f"Ошибка: {metrics['error']}")
    # return filepath


def analyze_pupil_data(json_filepath):
    """
    Анализирует данные зрачков и возвращает базовые метрики с оценкой точности.
    
    Args:
        json_filepath: путь к JSON файлу с данными
        
    Returns:
        Словарь с метриками и оценкой точности
    """
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        measurements = data['measurements']
        if not measurements:
            return {"error": "Нет данных для анализа"}
        
        # Извлекаем диаметры
        left_diameters = []
        right_diameters = []
        
        for meas in measurements:
            if meas['left_pupil_diameter'] is not None:
                left_diameters.append(meas['left_pupil_diameter'])
            if meas['right_pupil_diameter'] is not None:
                right_diameters.append(meas['right_pupil_diameter'])
        
        # Рассчитываем базовые метрики
        metrics = {
            'left_eye': calculate_metrics_with_accuracy(left_diameters, 'Левый'),
            'right_eye': calculate_metrics_with_accuracy(right_diameters, 'Правый'),
            'comparison': calculate_comparison_metrics(left_diameters, right_diameters),
            'overall_accuracy': calculate_overall_accuracy(left_diameters, right_diameters),
            'sample_size': len(measurements)
        }
        
        # Строим простой график
        plot_basic_data(left_diameters, right_diameters, json_filepath)
        
        return metrics
        
    except Exception as e:
        return {"error": f"Ошибка при анализе: {str(e)}"}

def calculate_metrics_with_accuracy(diameters, eye_name):
    """Рассчитывает базовые метрики для одного глаза с оценкой точности"""
    if not diameters:
        return {"error": f"Нет данных для {eye_name} глаза"}
    
    arr = np.array(diameters)
    mean = np.mean(arr)
    std_dev = np.std(arr)
    
    # Расчет показателей точности
    cv = (std_dev / mean) * 100 if mean != 0 else 0  # Коэффициент вариации (%)
    precision_1sd = (1 - (std_dev / mean)) * 100 if mean != 0 else 0  # Точность в пределах 1 SD
    precision_2sd = (1 - (2 * std_dev / mean)) * 100 if mean != 0 else 0  # Точность в пределах 2 SD
    
    return {
        'eye': eye_name,
        'mean': float(mean),                   # Среднее значение
        'median': float(np.median(arr)),       # Медиана
        'variance': float(np.var(arr)),        # Дисперсия
        'std_dev': float(std_dev),             # Стандартное отклонение
        'min': float(np.min(arr)),             # Минимальное значение
        'max': float(np.max(arr)),             # Максимальное значение
        'range': float(np.ptp(arr)),           # Размах (max - min)
        'max_deviation': float(np.max(np.abs(arr - mean))),  # Максимальное отклонение от среднего
        
        # Показатели точности в процентах
        'coefficient_of_variation': float(cv),  # CV (%)
        'precision_1sd': float(precision_1sd),  # Точность в пределах 1 SD (%)
        'precision_2sd': float(precision_2sd),  # Точность в пределах 2 SD (%)
        'stability_score': calculate_stability_score(cv)  # Оценка стабильности
    }

def calculate_stability_score(cv):
    """Оценивает стабильность измерений по коэффициенту вариации"""
    if cv < 5: return "Отличная"
    elif cv < 10: return "Хорошая"
    elif cv < 15: return "Удовлетворительная"
    elif cv < 20: return "Низкая"
    else: return "Очень низкая"

def calculate_comparison_metrics(left, right):
    """Рассчитывает метрики сравнения между глазами"""
    if not left or not right:
        return {"error": "Недостаточно данных для сравнения"}
    
    min_len = min(len(left), len(right))
    left_arr = np.array(left[:min_len])
    right_arr = np.array(right[:min_len])
    
    differences = np.abs(left_arr - right_arr)
    mean_diff = np.mean(differences)
    mean_size = (np.mean(left_arr) + np.mean(right_arr)) / 2
    
    # Точность синхронности (чем ближе к 100%, тем лучше)
    sync_accuracy = (1 - (mean_diff / mean_size)) * 100 if mean_size != 0 else 0
    
    return {
        'mean_difference': float(mean_diff),      # Средняя разница
        'max_difference': float(np.max(differences)),  # Максимальная разница
        'correlation': float(np.corrcoef(left_arr, right_arr)[0, 1]),  # Корреляция
        'synchronization_accuracy': float(sync_accuracy)  # Точность синхронности (%)
    }

def calculate_overall_accuracy(left, right):
    """Рассчитывает общую оценку точности системы"""
    if not left or not right:
        return {"error": "Недостаточно данных"}
    
    left_arr = np.array(left)
    right_arr = np.array(right)
    
    # Средний коэффициент вариации
    cv_left = (np.std(left_arr) / np.mean(left_arr)) * 100 if np.mean(left_arr) != 0 else 0
    cv_right = (np.std(right_arr) / np.mean(right_arr)) * 100 if np.mean(right_arr) != 0 else 0
    mean_cv = (cv_left + cv_right) / 2
    
    # Общая точность системы
    overall_accuracy = 100 - mean_cv
    
    return {
        'mean_coefficient_of_variation': float(mean_cv),
        'overall_system_accuracy': float(overall_accuracy),
        'accuracy_rating': get_accuracy_rating(overall_accuracy)
    }

def get_accuracy_rating(accuracy):
    """Оценивает общую точность системы"""
    if accuracy >= 95: return "Отличная"
    elif accuracy >= 90: return "Очень хорошая"
    elif accuracy >= 85: return "Хорошая"
    elif accuracy >= 80: return "Удовлетворительная"
    elif accuracy >= 70: return "Низкая"
    else: return "Неудовлетворительная"

def plot_basic_data(left_diameters, right_diameters, json_filepath):
    """Строит базовый график"""
    plt.figure(figsize=(12, 6))
    
    # Временной ряд
    plt.subplot(1, 2, 1)
    if left_diameters:
        plt.plot(left_diameters, 'b-', label='Левый зрачок', alpha=0.7)
    if right_diameters:
        plt.plot(right_diameters, 'r-', label='Правый зрачок', alpha=0.7)
    plt.title('Динамика диаметров зрачков')
    plt.xlabel('Измерение')
    plt.ylabel('Диаметр (пиксели)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    data_to_plot = []
    labels = []
    if left_diameters:
        data_to_plot.append(left_diameters)
        labels.append('Левый')
    if right_diameters:
        data_to_plot.append(right_diameters)
        labels.append('Правый')
    
    if data_to_plot:
        plt.boxplot(data_to_plot, labels=labels)
        plt.title('Распределение диаметров')
        plt.ylabel('Диаметр (пиксели)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем в папку results
    plot_folder = "result_plots"
    os.makedirs(plot_folder, exist_ok=True)
    filename = os.path.basename(json_filepath).replace('.json', '_plot.png')
    plt.savefig(os.path.join(plot_folder, filename), dpi=150, bbox_inches='tight')
    plt.close()

def print_basic_metrics(metrics):
    """Красиво выводит метрики с оценкой точности"""
    print("\n" + "="*60)
    print("МЕТРИКИ ДИАМЕТРОВ ЗРАЧКОВ С ОЦЕНКОЙ ТОЧНОСТИ")
    print("="*60)
    
    print(f"\n📊 Общее количество измерений: {metrics['sample_size']}")
    
    # Метрики для левого глаза
    if 'error' not in metrics['left_eye']:
        print(f"\n--- ЛЕВЫЙ ЗРАЧОК ---")
        left = metrics['left_eye']
        print(f"Среднее значение: {left['mean']:.2f} пикс.")
        print(f"Стандартное отклонение: {left['std_dev']:.2f} пикс.")
        print(f"Коэффициент вариации: {left['coefficient_of_variation']:.1f}%")
        print(f"Точность (1σ): {max(0, left['precision_1sd']):.1f}%")
        print(f"Точность (2σ): {max(0, left['precision_2sd']):.1f}%")
        print(f"Стабильность: {left['stability_score']}")
    
    # Метрики для правого глаза
    if 'error' not in metrics['right_eye']:
        print(f"\n--- ПРАВЫЙ ЗРАЧОК ---")
        right = metrics['right_eye']
        print(f"Среднее значение: {right['mean']:.2f} пикс.")
        print(f"Стандартное отклонение: {right['std_dev']:.2f} пикс.")
        print(f"Коэффициент вариации: {right['coefficient_of_variation']:.1f}%")
        print(f"Точность (1σ): {max(0, right['precision_1sd']):.1f}%")
        print(f"Точность (2σ): {max(0, right['precision_2sd']):.1f}%")
        print(f"Стабильность: {right['stability_score']}")
    
    # Сравнение глаз
    if 'error' not in metrics['comparison']:
        print(f"\n--- СРАВНЕНИЕ ГЛАЗ ---")
        comp = metrics['comparison']
        print(f"Средняя разница: {comp['mean_difference']:.2f} пикс.")
        print(f"Точность синхронности: {comp['synchronization_accuracy']:.1f}%")
        print(f"Корреляция: {comp['correlation']:.3f}")
    
    # Общая точность системы
    if 'error' not in metrics['overall_accuracy']:
        print(f"\n--- ОБЩАЯ ТОЧНОСТЬ СИСТЕМЫ ---")
        acc = metrics['overall_accuracy']
        print(f"Средний CV: {acc['mean_coefficient_of_variation']:.1f}%")
        print(f"Общая точность: {acc['overall_system_accuracy']:.1f}%")
        print(f"Оценка: {acc['accuracy_rating']}")
    
    print("="*60)