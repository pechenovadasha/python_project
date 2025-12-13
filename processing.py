from center_detection import process_frame


def recieve_centers(diff, dark_pupil_frame, bright_pupil_frame, dir_name):

    # получаем центры зрачков и отблесков и обрабаьываем возможные ошибки
    pupil_left, pupil_right, glint_left, glint_right = process_frame(diff, dark_pupil_frame, bright_pupil_frame, dir_name)
    
    error = []
    if pupil_left == (None, None):
        print("Left pupil not found")
        error.append("Left pupil not found")
    if pupil_right == (None, None):
        print("Right pupil not found")
        error.append("Right pupil not found")
    if glint_left == (None, None):
        print("Left glint not found")
        error.append("Left glint not found")
    if glint_right == (None, None):
        print("Right glint not found")
        error.append("Right glint not found")
        


    if pupil_left == None or pupil_right == None or glint_left == None or glint_right == None:
        return error
    
    return error