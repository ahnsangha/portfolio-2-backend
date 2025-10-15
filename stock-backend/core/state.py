# 전역 상태(STATE)
# 멀티 프로세스/스레드 공유 자원 한곳에 모아둠

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)
analysis_tasks: dict[str, dict] = {}