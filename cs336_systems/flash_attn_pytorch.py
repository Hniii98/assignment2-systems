import torch

class FlashAttenPytorch(torch.autograd.function):
	@staticmethod
	def forward(ctx, Q, K, V, is_casual=False):
		NQ, D = Q.shape()
		NK = K.shape()[0]
		Q_TILE_SIZE = 16
		K_TILE_SIZE = 16

		O = torch.zeros_like(Q)
		L = torch.zeros(NQ, device=Q.device)
		
		for i in range((NQ + Q_TILE_SIZE - 1) // Q_TILE_SIZE):
			# Q : [Q_TILE_SIZE, D]
			Qi = Q[i*Q_TILE_SIZE : i*Q_TILE_SIZE+Q_TILE_SIZE]	# Load Q
			Oi_pre = torch.zeros_like(Qi) 
			li_pre = torch.zeros((Q_TILE_SIZE,), device=Q.device)
			mi_pre = torch.full((Q_TILE_SIZE,), float('-inf'), device=Q.device)

			for j in range((NK + K_TILE_SIZE + 1) // K_TILE_SIZE):
				# Kj : [K_TILE_SIZE, D]
				# Vj : [K_TILE_SIZE, D]
				Kj = K[j*K_TILE_SIZE : j*K_TILE_SIZE + K_TILE_SIZE]
				Vj = V[j*K_TILE_SIZE : j*K_TILE_SIZE + K_TILE_SIZE]

				# Compute tile fo pre-softmax attention score Sij
				tile_attn_score = (Qi * Kj.T)	/ torch.sqrt(D)
				# Compute tile row max 
				tile_row_max = torch.max(tile_attn_score, dim=0)
				# Get row max between current tile and previous tile 
				mi = torch.max(tile_row_max, mi_pre) # [Q_TILE_SIZE]
				local_exp = torch.exp(tile_attn_score - mi[:, None]) # [Q_TILE_SIZE, K_TILE_SIZE]
				row_sum_local_exp = torch.sum(local_exp, dim=0) # [Q_TILE_SIZE]
				li = torch.exp(mi_pre - mi) * li_pre + row_sum_local_exp # [Q_TILE_SIZE]
				Oi = torch.diag(torch.exp(mi_pre - mi)) @ Oi_pre + local_exp @ Vj # [Q_TILE_SIZE, D]

				# Update row parameters
				Oi_pre = Oi
				li_pre = li
				mi_pre = mi
			
			denominator = torch.diag(1.0 / li_pre)
			Oi_pre = denominator @ Oi_pre
			li_pre = mi_pre + torch.log(li_pre)

			O[i*Q_TILE_SIZE : i*Q_TILE_SIZE+Q_TILE_SIZE] = Oi_pre
			L[i*Q_TILE_SIZE : i*Q_TILE_SIZE+Q_TILE_SIZE] = li_pre
		
		ctx.save_for_backward(L, Q, K, V, O)
		return O, L
	
	@staticmethod
	def backward(ctx, grad_output):
		raise NotImplementedError
	







				
				


			
		