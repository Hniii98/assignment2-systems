import torch
from einops import rearrange
import math


#@torch.compile
def _flash_attn_bwd_impl(Q, K, V, O, grad_output, L):
	NB, NQ, DIM = Q.shape
	_, NK, _  = K.shape

	Q_TILE_SIZE = 16
	K_TILE_SIZE = 16
	scale = math.sqrt(DIM)

	dQ = torch.zeros_like(Q)
	dK = torch.zeros_like(K)
	dV = torch.zeros_like(V)

	for	b in range(NB):
		D = torch.sum(O[b, :, :] * grad_output[b, :, :], dim=-1, keepdim=True) # [NQ, 1]
		for i in range((NQ + Q_TILE_SIZE - 1) // Q_TILE_SIZE):
			start_q = i*Q_TILE_SIZE
			end_q = i*Q_TILE_SIZE + Q_TILE_SIZE

			Qi = Q[b, start_q:end_q, :]	# Load Qi: [Q_TILE_SIZE, DIM]
			dOi = grad_output[b, start_q:end_q, :] # Load dOi: [Q_TILE_SIZE, DIM]
			
			Li = L[b, start_q:end_q] 	# Load Li:  [Q_TILE_SIZE]
			Di = D[start_q:end_q, :] 	# [Q_TILE_SIZE, 1]
			for j in range((NK + K_TILE_SIZE - 1) // K_TILE_SIZE):
				start_k = j*K_TILE_SIZE
				end_k = j*K_TILE_SIZE + K_TILE_SIZE

				Kj = K[b, start_k:end_k, :]# Load Kj: [K_TILE_SIZE, DIM]
				Vj = V[b, start_k:end_k,:]# Load Vj: [K_TILE_SIZE, DIM]

				Sij =  (Qi @ Kj.T) / scale # [Q_TILE_SIZE, K_TILE_SIZE]
				Pij = torch.exp(Sij - Li[:, None]) # [Q_TILE_SIZE, K_TILE_SIZE]
				dV[b, start_k:end_k, :] += Pij.T @ dOi # [K_TILE_SIZE, DIM]

				dPij = dOi @ Vj.T # [Q_TILE_SIZE, K_TILE_SIZE]
				dSij = Pij*(dPij - Di) # [Q_TILE_SIZE, K_TILE_SIZE]

				dQ[b, start_q:end_q, :] += (dSij @ Kj) / scale # [Q_TILE_SIZE, DIM]
				dK[b, start_k:end_k, :] += (dSij.T @ Qi) / scale# [K_TILE_SIZE, DIM]
	return dQ, dK, dV


class FlashAttenPytorch(torch.autograd.Function):
	@staticmethod
	def forward(ctx, Q, K, V, is_casual=False):
		NB, NQ, DIM = Q.shape
		_, NK, _ = K.shape
		Q_TILE_SIZE = 16
		K_TILE_SIZE = 16

		O = torch.zeros_like(Q)
		L = torch.zeros((NB, NQ), device=Q.device)
		
		for b in range(NB):
			for i in range((NQ + Q_TILE_SIZE - 1) // Q_TILE_SIZE):
				# Q : [Q_TILE_SIZE, DIM]
				start_q = i*Q_TILE_SIZE
				end_q = i*Q_TILE_SIZE + Q_TILE_SIZE
				Qi = Q[b, start_q:end_q, :]	# Load Q
				Oi_pre = torch.zeros_like(Qi) 
				li_pre = torch.zeros((Q_TILE_SIZE,), device=Q.device)
				mi_pre = torch.full((Q_TILE_SIZE,), float('-inf'), device=Q.device, dtype=Q.dtype)

				for j in range((NK + K_TILE_SIZE - 1) // K_TILE_SIZE):
					# Kj : [K_TILE_SIZE, DIM]
					# Vj : [K_TILE_SIZE, DIM]
					start_k = j*K_TILE_SIZE
					end_k = j*K_TILE_SIZE + K_TILE_SIZE
					Kj = K[b, start_k:end_k,:]
					Vj = V[b, start_k:end_k,:]
				
					# Compute tile fo pre-softmax attention score Sij
					tile_attn_score = (Qi @ Kj.T)	/ math.sqrt(DIM)
					# Compute tile row max 
					#print(tile_attn_score.shape)
					tile_row_max = torch.max(tile_attn_score, dim=1).values
					# Get row max between current tile and previous tile 
					mi = torch.maximum(tile_row_max, mi_pre) # [Q_TILE_SIZE]
					local_exp = torch.exp(tile_attn_score - mi[:, None]) # [Q_TILE_SIZE, K_TILE_SIZE]
					row_sum_local_exp = torch.sum(local_exp, dim=1) # [Q_TILE_SIZE]
					scales = torch.exp(mi_pre - mi)
					li = scales * li_pre + row_sum_local_exp # [Q_TILE_SIZE]
					Oi = scales[:, None] * Oi_pre + local_exp @ Vj # [Q_TILE_SIZE, DIM]

					# Update row parameters
					Oi_pre = Oi
					li_pre = li
					mi_pre = mi
				
				Oi = Oi_pre / li_pre[:, None]
				# LSE of #Q_TILE_SIZE row, for recovering after softmax logits without saving huge matrix
				li = mi_pre + torch.log(li_pre)
				O[b, start_q:end_q,:] = Oi
				L[b, start_q:end_q] = li

		ctx.save_for_backward(Q, K, V, O, L)
		return O
	
	@staticmethod
	def backward(ctx, grad_output):
		Q, K, V, O, L = ctx.saved_tensors

		dQ, dK, dV = _flash_attn_bwd_impl(Q, K, V, O,grad_output, L)
					
		
		return dQ, dK, dV, None
				



				
				


			


		


		
	







				
				


			
		