from typing import Callable
import torch
from cs336_systems.configs import CONFIGS
import cs336_basics
from contextlib import nullcontext



def get_device(device: str):
	return torch.device(device=device)

def get_model(vocab_size: int, context_length: int, rope_theta: float, model_size: str, device: str, torch_compile: bool):
	model_hyper = CONFIGS.get(model_size, CONFIGS["small"])


	model = cs336_basics.model.BasicsTransformerLM(
		vocab_size=vocab_size,
		context_length=context_length,
		rope_theta=rope_theta,
		d_model=model_hyper["d_model"],
		num_layers=model_hyper["num_layers"],
		num_heads=model_hyper["num_heads"],
		d_ff=model_hyper["d_ff"]

	)

	if torch_compile:
		model = torch.compile(model)

	return model.to(get_device(device=device))



def run_forward(model: torch.nn.Module, batch_size: int, vocab_size: int, context_length: int, device: str, mixed_precision: bool = False ):
	
	ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if mixed_precision else nullcontext()
	input_tensor = torch.randint(low=0, high=vocab_size, 
					  size=(batch_size, context_length),
					  device=device)
	def run():
		with ctx:
			logits = model(input_tensor)
	
	return run

def run_backward(model: torch.nn.Module, batch_size: int, vocab_size: int, context_length: int, device: str, mixed_precision: bool = False ):
	
	ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if mixed_precision else nullcontext()
	input_tensor = torch.randint(low=0, high=vocab_size, 
					  size=(batch_size, context_length),
					  device=device)
	
	target = torch.randint(low=0, high=vocab_size, 
					  size=(batch_size, context_length),
					  device=device)
	def run():
		with ctx:
			logits = model(input_tensor)
			model.zero_grad(set_to_none=False)
			
			loss = cs336_basics.nn_utils.cross_entropy(logits, target)
			loss.backward()
	
	return run	

def run_full(model: torch.nn.Module, batch_size: int, vocab_size: int, context_length: int,  device: str, mixed_precision: bool = False ):
	
	ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if mixed_precision else nullcontext()
	optimizer = cs336_basics.optimizer.AdamW(model.parameters())
	input_tensor = torch.randint(low=0, high=vocab_size, 
					  size=(batch_size, context_length),
					  device=device)
	
	target = torch.randint(low=0, high=vocab_size, 
					  size=(batch_size, context_length),
					  device=device)

	def run():
		with ctx:
			logits = model(input_tensor)
			model.zero_grad(set_to_none=True)
			
			loss = cs336_basics.nn_utils.cross_entropy(logits, target)
			loss.backward()

			optimizer.step()
	
	return run	




