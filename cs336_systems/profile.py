from typing import Callable
import torch
import os

from torch.profiler import ProfilerActivity

def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
	# Warm up

	for _ in range(num_warmups):
		run()

	if torch.cuda.is_available():
		# Wait for CUDA threads to finish
		torch.cuda.synchronize()

	with torch.profiler.profile(
		activities= [
			ProfilerActivity.CPU,
			ProfilerActivity.CUDA
		],
		# Output stack trace for visualization
		with_stack=with_stack,
		profile_memory=True,
		record_shapes=True,
		# Needed to export stack trace for visualization
		experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
	) as prof:
		
		run()

		if torch.cuda.is_available():
			torch.cuda.synchronize()

	# Print out table
	table = prof.key_averages().table(sort_by="self_cuda_time_total",
								   	  max_name_column_width=80,
									  row_limit=20)
	
	# Write stack trace visualization
	if with_stack:
		# text_path = f"var/stacks_{description}.txt"
		# svg_path = f"var/stacks_{description}.svg"
	
		output_path = os.path.dirname(os.path.abspath(__file__))
		
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		#prof.export_stacks(output_path, "self_cuda_time_total")
		prof.export_chrome_trace(os.path.join(output_path, f"{description}.json"))

	return table

	