from cs336_systems.utils import run_forward, run_backward, run_full, get_model, get_device
from cs336_systems.benchmark import benchmarck
from cs336_systems.parser import parse_args
import torch
import statistics


fn_dict = {
	"forward": run_forward,
	"backward": run_backward,
	"full": run_full,
}


def main():

	args = parse_args()

	# Build model
	model = get_model(vocab_size=args.vocab_size, context_length=args.context_length, rope_theta=args.rope_theta,
				   	  model_size=args.model_size, device=args.device)
	
	#Get forward only、 backward、 full run closure.
	fn = fn_dict[args.run_mode]

	run = fn(
		model=model, 
		batch_size=args.batch_size, 
		vocab_size=args.vocab_size, 
		context_length=args.context_length,
		device=args.device,
		mixed_precision=args.mixed_precision)
	
	avg_time, times = benchmarck("Model {args.run_mode} run", run = run, num_warmups=args.warm_up, num_trials=args.execution)

	print(f"{args.run_mode} run use {avg_time}  average time in {args.execution} loops, std: {statistics.stdev(times)}")
	
	

if __name__ == "__main__":
	main()
	
	
