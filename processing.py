from center_detection import process_frame
import time 
from centers_analisis import save_centers_to_file



def recieve_centers(diff, dark_pupil_frame, bright_pupil_frame, dir_name):

    # получаем центры зрачков и отблесков и обрабаьываем возможные ошибки
    start = time.time()
    csv_file_name  = 'results/' + dir_name + '.csv'
    pupil_left, pupil_right, glint_left, glint_right = process_frame(diff, dark_pupil_frame, bright_pupil_frame, dir_name)


    error = []
    if pupil_left == (None, None):
        print("Left pupil not found")
        pupil_left = (0, 0)
        error.append("Left pupil not found")
    if pupil_right == (None, None):
        pupil_right = (0, 0)
        print("Right pupil not found")
        error.append("Right pupil not found")
    if glint_left == (None, None):
        glint_left = (0, 0)
        print("Left glint not found")
        error.append("Left glint not found")
    if glint_right == (None, None):
        glint_right = (0, 0)
        print("Right glint not found")
        error.append("Right glint not found")
        
    save_centers_to_file([pupil_left, glint_left],[pupil_right, glint_right], [(0, 0), (0, 0)], csv_file_name)

    return error