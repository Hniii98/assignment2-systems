import argparse
import torch
import statistics



def validate_args(args):
	if args.device == "cuda":
		if torch.cuda.is_available() is False:
			raise ValueError(f"No available cuda device!")
		
def parse_args():
	"""
	Parse argument from command line.
	"""
	parser = argparse.ArgumentParser(description="Benchmark and profile options")
	
	parser.add_argument("--batch_size", type=int, default=2)
	parser.add_argument("--context_length", type=int, default=512)
	parser.add_argument("--vocab_size", type=int, default=10000)
	parser.add_argument("--rope_theta", type=float, default=10000)

	parser.add_argument("--warm_up", type=int, default=1, help="Warm up steps before measuring time")
	parser.add_argument("--execution", type=int, default=10, help="Executions steps for geting average result")

	
	
	parser.add_argument("--model_size", choices=["small", "medium", "large", "xl", "10B" ], default="small")
	parser.add_argument("--run_mode", choices=["forward", "backward", "full"], default="forward",
						help=(
							"Profiling mode: forward, backward, or full. "
							"forward: only forward pass; "
							"backward: forward + backward; "
							"full: forward + backward + optimizer step"
						))
		
	parser.add_argument("--mixed_precision", action="store_true", default=False) # using bf16
	parser.add_argument("--memory_profile", action="store_true", default=False)
	parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
	parser.add_argument("--torch_compile", action="store_true", default=False)

	args = parser.parse_args()

	validate_args(args)

	return args






		

	
	
