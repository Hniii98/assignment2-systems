from cs336_systems.utils import run_forward, run_backward, run_full, get_model, get_device
from cs336_systems.parser import parse_args

from cs336_systems.benchmark import benchmarck
from cs336_basics import BasicsTransformerLM
from torch import nn
from torch import Tensor
import torch
from torch.utils.checkpoint import checkpoint
import statistics

from cs336_systems.profile_script import profile

fn_dict = {
	"forward": run_forward,
	"backward": run_backward,
	"full": run_full,
}

class CheckpointWrapper(nn.Module):
	def __init__(
		self,
		model: nn.Module,
		block_size: int,
	):
		super().__init__()
		self.model = model
		self.block_size = block_size

	def group_run(self, group, x: torch.Tensor):
		for layer in group:
			x = layer(x)
		return x
	
	def forward(self, x: torch.Tensor):
		_, sequence_length = x.size()

		embedded_tokens = self.model.token_embeddings(x)

		x = embedded_tokens

		for i in  range(0, len(self.model.layers), self.block_size):
			group = self.model.layers[i:i+self.block_size]
			x = checkpoint(self.group_run, group, x, use_reentrant=False)
		
		x = self.model.ln_final(x)
		logits = self.model.lm_head(x)

		return logits
	

def main():
	args = parse_args()

	# Build model
	model = get_model(vocab_size=args.vocab_size, context_length=args.context_length, rope_theta=args.rope_theta,
					model_size=args.model_size, device=args.device)
	

	for block_size in [2, 3, 4, 5, 6, 7, 8]:
		torch.cuda.empty_cache()
		torch.cuda.reset_peak_memory_stats()
		wrapper = CheckpointWrapper(model=model, block_size=block_size)
		
		#Get forward only、 backward、 full run closure.
		fn = fn_dict[args.run_mode]

		run = fn(
			model=wrapper, 
			batch_size=args.batch_size, 
			vocab_size=args.vocab_size, 
			context_length=args.context_length,
			device=args.device,
			mixed_precision=args.mixed_precision)
		
		avg_time, times = benchmarck(f"{args.run_mode}_mode_{block_size}", run = run, num_warmups=args.warm_up, num_trials=args.execution, memory_profile=args.memory_profile)

		peak = torch.cuda.max_memory_allocated()

		print(f"block_size={block_size}: {peak / 1024**3:.2f} GB")
		# print(f"{args.run_mode} run use {avg_time}  average time in {args.execution} loops")
		del wrapper, run

	

if __name__ == "__main__":
	main()
		

		
		
