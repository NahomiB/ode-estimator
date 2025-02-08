import multiprocessing
import psutil

def get_n_process():
    total_cores = multiprocessing.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=1)  # Check CPU usage

    if cpu_usage < 50:
        n_process = total_cores - 1
    elif cpu_usage < 75:
        n_process = max(1, total_cores // 2)
    else:
        n_process = 1  # Avoid overloading

    return n_process