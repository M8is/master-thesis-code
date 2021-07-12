from time import process_time


class Stopwatch:
    def __init__(self):
        time = process_time()
        self.__start_time = time
        self.__pause_start = time
        self.__paused_time = 0

    def resume(self) -> None:
        self.__paused_time += process_time() - self.__pause_start

    def pause(self) -> float:
        pause_start = process_time()
        self.__pause_start = pause_start
        return pause_start - self.__start_time - self.__paused_time
