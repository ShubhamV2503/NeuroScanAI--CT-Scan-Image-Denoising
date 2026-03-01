import multiprocessing
import os

PORT = int(os.getenv("PORT", 10000))
bind = f"0.0.0.0:{PORT}"

workers = multiprocessing.cpu_count() * 2 + 1
threads = 2
timeout = 120

worker_class = "gthread"
