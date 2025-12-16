import threading
from center_detection import PupilTracker


_thread_local = threading.local()

def _get_tracker():
    """Return a tracker bound to the current thread to keep processing thread-safe."""
    tracker = getattr(_thread_local, "tracker", None)
    if tracker is None:
        tracker = PupilTracker()
        _thread_local.tracker = tracker
    return tracker



def recieve_centers(diff, dark_pupil_frame, bright_pupil_frame, dir_name):

    # получаем центры зрачков и отблесков и обрабаьываем возможные ошибки
    csv_file_name  = 'results/' + dir_name + '.csv'
    tracker = _get_tracker()
    pupil_left, pupil_right, glint_left, glint_right = tracker.process(diff, bright_pupil_frame, dir_name)


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
    
    return error
