import argparse
import cs336_basics
import torch
import timeit
import sys

import statistics



def parse_args():
	"""
	Parse argument from command line.
	"""
	parser = argparse.ArgumentParser(description="Model hyperparameters")
	
	parser.add_argument("--batch_size", type=int, default=2)
	parser.add_argument("--context_length", type=int, default=512)
	parser.add_argument("--vocab_size", type=int, default=10000)
	parser.add_argument("--rope_theta", type=float, default=10000)
	parser.add_argument("--warm_up", type=int, default=1, help="Warm up steps before measuring time")
	parser.add_argument("--execution", type=int, default=10, help="Executions steps for geting average result")

	# Using small size model parameters as default.
	parser.add_argument("--d_model", type=int, default=768,
					 	help="The dimensionality of the model embeddings and sublayer outputs")
	parser.add_argument("--num_layers", type=int, default=12,
					 	help="The number of Transformer layers to use")
	parser.add_argument("--num_heads", type=int, default=12,
					 	help="Number of heads to use in multi-headed attention. `d_model` must be evenly divisible by `num_heads`.")
	parser.add_argument("--d_ff", type=int, default=3072,
					 	help="Dimensionality of the feed-forward inner layer")
	
	parser.add_argument("--mode", choices=["forward", "backward", "full"], default="forward",
						help=(
							"Profiling mode: forward, backward, or full. "
							"forward: only forward pass; "
							"backward: forward + backward; "
							"full: forward + backward + optimizer step"
						))
	parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
	
	args = parser.parse_args()

	return args

def validate_args(args):
	if args.d_model % args.num_heads != 0:
		raise ValueError(f"`d_model`: {args.d_model} must be evenly divisible by `num_heads`: {args.num_heads}")
	if args.device == "cuda":
		if torch.cuda.is_available() is False:
			raise ValueError(f"No available cuda device!")
		

def	set_up(args):
	model = cs336_basics.model.BasicsTransformerLM(
				vocab_size = args.vocab_size,
				context_length = args.context_length,
				d_model = args.d_model,
				num_layers = args.num_layers,
				num_heads = args.num_heads,
				d_ff = args.d_ff,
				rope_theta = args.rope_theta
	)
	return model.to(args.device)

def run(model, args):
	# Generate random input and output
	x = torch.randint(low=0, high=args.vocab_size, 
					  size=(args.batch_size, args.context_length),
					  device=args.device)
	
	y = torch.randint(low=0, high=args.vocab_size, 
					  size=(args.batch_size, args.context_length),
					  device=args.device)
	
	if args.mode == "full":
		optimizer = cs336_basics.optimizer.AdamW(model.parameters())
	
	
	# Warm up 
	for _ in range(args.warm_up):
		logits = model(x)

		if args.mode == "backward" or args.mode == "full":
			
			model.zero_grad(set_to_none=False)
			loss = cs336_basics.nn_utils.cross_entropy(logits, y)
			loss.backward()

		if args.mode == "full":
			optimizer.step()
	
	# Wait warm up finished
	torch.cuda.synchronize() 

	# Execution
	time_each_step = []
	total_start = timeit.default_timer()
	for _ in range(args.execution):
		torch.cuda.synchronize()
		step_start = timeit.default_timer()
		# Every mode do the forward
		logits = model(x)

		if args.mode == "backward" or args.mode == "full":
			model.zero_grad(set_to_none=False)
			loss = cs336_basics.nn_utils.cross_entropy(logits, y)
			loss.backward()

		if args.mode == "full":
			optimizer.step()
		torch.cuda.synchronize()
		
		step_end = timeit.default_timer()

		time_each_step.append(step_end - step_start)

	torch.cuda.synchronize()
	total_end = timeit.default_timer()
	
	return (total_end- total_start) / args.execution, time_each_step


def main():
	args = parse_args()
	validate_args(args)
	
	model = set_up(args)
	time, time_steps = run(model=model, args=args)
	print(f"Average use {time:.4f} s for {args.mode} mode. Best {min(time_steps):.4f} s. Worst {max(time_steps):.4f} s")
	
	print(f"std {statistics.stdev(time_steps):.4f}")
	print(f"raw test time list: {time_steps}")
	return 0

if __name__ == "__main__":
	
	sys.exit(main())





		

	
	
