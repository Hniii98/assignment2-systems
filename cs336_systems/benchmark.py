import torch
from typing import Callable
import timeit
import statistics

def benchmarck(description: str, run: callable, num_warmups: int = 1, num_trials: int =3):
	assert num_trials >= 1, f"num_trials must be >= 1, got {num_trials}"
	
	# Warmup
	for _ in range(num_warmups):
		run()

	
	if torch.cuda.is_available():
		torch.cuda.synchronize()

	times: list[float] = []

	for trial in range(num_trials):
		start_time = timeit.default_timer()

		run()

		if torch.cuda.is_available():
			torch.cuda.synchronize()
		
		end_time = timeit.default_timer()

		times.append((end_time - start_time) * 1000 )

	return statistics.mean(times) if num_trials > 1 else times[0] , times