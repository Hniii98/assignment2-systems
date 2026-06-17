import torch
from typing import Callable
import timeit
import statistics

def benchmarck(description: str, run: callable, num_warmups: int = 1, num_trials: int =3, memory_profile: bool=False):
	assert num_trials >= 1, f"num_trials must be >= 1, got {num_trials}"
	
	# Warmup
	for _ in range(num_warmups):
		run()

	
	if torch.cuda.is_available():
		torch.cuda.synchronize()

	times: list[float] = []
	# Execution
	if memory_profile:
		torch.cuda.memory._record_memory_history(max_entries=1000000)
		record_once = True

	for trial in range(num_trials):
		start_time = timeit.default_timer()
		run()

		if torch.cuda.is_available():
			torch.cuda.synchronize()
		
		end_time = timeit.default_timer()
		times.append((end_time - start_time) * 1000 )

		if memory_profile and record_once:
			torch.cuda.memory._dump_snapshot(f"{description}_one_run_memory_snapshot.pickle")
			torch.cuda.memory._record_memory_history(enabled=False)
			record_once = False
	

	return statistics.mean(times) if num_trials > 1 else times[0] , times