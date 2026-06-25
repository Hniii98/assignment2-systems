import torch
from einops import rearrange
import math


class FlashAttenPytorch(torch.autograd.Function):
	@staticmethod
	def forward(ctx, Q, K, V, is_casual=False):
		NB, NQ, D = Q.shape
		_, NK, _ = K.shape
		Q_TILE_SIZE = 16
		K_TILE_SIZE = 16

		O = torch.zeros_like(Q)
		L = torch.zeros((NB, NQ), device=Q.device)
		
		for b in range(NB):
			for i in range((NQ + Q_TILE_SIZE - 1) // Q_TILE_SIZE):
				# Q : [Q_TILE_SIZE, D]
				start_q = i*Q_TILE_SIZE
				end_q = i*Q_TILE_SIZE + Q_TILE_SIZE
				Qi = Q[b, start_q:end_q, :]	# Load Q
				Oi_pre = torch.zeros_like(Qi) 
				li_pre = torch.zeros((Q_TILE_SIZE,), device=Q.device)
				mi_pre = torch.full((Q_TILE_SIZE,), float('-inf'), device=Q.device, dtype=Q.dtype)

				for j in range((NK + K_TILE_SIZE - 1) // K_TILE_SIZE):
					# Kj : [K_TILE_SIZE, D]
					# Vj : [K_TILE_SIZE, D]
					start_k = j*K_TILE_SIZE
					end_k = j*K_TILE_SIZE + K_TILE_SIZE
					Kj = K[b, start_k:end_k,:]
					Vj = V[b, start_k:end_k,:]
				
					# Compute tile fo pre-softmax attention score Sij
					tile_attn_score = (Qi @ Kj.T)	/ math.sqrt(D)
					# Compute tile row max 
					#print(tile_attn_score.shape)
					tile_row_max = torch.max(tile_attn_score, dim=1).values
					# Get row max between current tile and previous tile 
					mi = torch.maximum(tile_row_max, mi_pre) # [Q_TILE_SIZE]
					local_exp = torch.exp(tile_attn_score - mi[:, None]) # [Q_TILE_SIZE, K_TILE_SIZE]
					row_sum_local_exp = torch.sum(local_exp, dim=1) # [Q_TILE_SIZE]

					scales = torch.exp(mi_pre - mi)
					li = scales * li_pre + row_sum_local_exp # [Q_TILE_SIZE]
					Oi = scales[:, None] * Oi_pre + local_exp @ Vj # [Q_TILE_SIZE, D]

					# Update row parameters
					Oi_pre = Oi
					li_pre = li
					mi_pre = mi
				
				Oi = Oi_pre / li_pre[:, None]
				li = mi_pre + torch.log(li_pre)
				O[b, start_q:end_q,:] = Oi
				L[b, start_q:end_q] = li

		ctx.save_for_backward(L, Q, K, V, O)
		return O
	
	@staticmethod
	def backward(ctx, grad_output):
		raise NotImplementedError
	







				
				


			
		