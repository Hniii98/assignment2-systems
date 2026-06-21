from cs336_basics.model import scaled_dot_product_attention, RotaryEmbedding, Linear
from torch import nn
import torch

import timeit

class CasualMultiHeadSelfAttention(nn.Module):
	def __init__(
		self,
		d_model,
		position_encoder: RotaryEmbedding | None = None,
	):
		super().__init__()
		self.d_model = d_model
		self.d_k = d_model
		self.d_v = d_model

		self.q_proj = Linear(self.d_model, self.d_k)
		self.k_proj = Linear(self.d_model, self.d_k)
		self.v_proj = Linear(self.d_model, self.d_v)

		self.output_proj = Linear(self.d_model, self.d_model)
		self.positional_encoder = position_encoder

	def forward(
		self,
		x,
		token_positions
	):
		*batch_dims, sequence_length, d_model = x.size()
		assert d_model == self.d_model

		Q = self.q_proj(x)
		K = self.k_proj(x)
		V = self.v_proj(x)


		mask = torch.tril(torch.ones(sequence_length, sequence_length, device=x.device)).bool()
		attn_output = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)

		output = self.output_proj(attn_output)
		return output


def main():
	# Fixed batch size
	batch_size = 8
	embedding_dims = [16, 32, 64, 128]
	sequence_lens = [256, 1024, 4096, 8192, 16384]
	num_trials = 100
	num_warmups = 2
	device = torch.device("cuda")
	

	

	for seq_len in sequence_lens:
		for d_model in embedding_dims:
			input = torch.randn(size=(batch_size, seq_len, d_model), device=device)
			token_ids =torch.tensor( [id for id in range(seq_len)] , device=device)
			model = CasualMultiHeadSelfAttention(d_model=d_model).to(device)

			# Warm up
			for _ in range(num_warmups):
				x = model(input, token_ids)
				x.sum().backward()

			# Reset and record start
			torch.cuda.reset_peak_memory_stats()
			start_time = timeit.default_timer()
			
			# Test forward pass
			for i in range(num_trials):
				x = model(input, token_ids)
			
			torch.cuda.synchronize()
			end_time = timeit.default_timer()
			mem_in_use = torch.cuda.memory_allocated()
			

			print(f"{end_time - start_time:.4f} s for 100 passes forward with seq_len: {seq_len} d_model: {d_model}, use {mem_in_use / 1024**3:.4f} GiB memory")

			start_time = timeit.default_timer()
			for i in range(num_trials):
				x.mean().backward(retain_graph=True)
			torch.cuda.synchronize()
			end_time = timeit.default_timer()

			print(f"{end_time - start_time:.4f} s for 100 passes backward with seq_len: {seq_len} d_model: {d_model}")
				
if __name__ == "__main__":
	main()
	

			
