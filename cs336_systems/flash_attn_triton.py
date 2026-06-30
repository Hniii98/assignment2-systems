import triton
import triton.language as tl	

import torch

@triton.jit
def flash_fwd_kernel(
	Q_ptr, K_ptr, V_ptr,
	O_ptr, L_ptr,
	stride_qb, stride_qq, stride_qd,
	stride_kb, stride_kk, stride_kd,
	stride_vb, stride_vk, stride_vd,
	stride_ob, stride_oq, stride_od,
	stride_lb, stride_lq,
	N_QUERIES, N_KEYS,
	scale,
	DIM: tl.constexpr,
	is_causal: tl.constexpr,
	Q_TILE_SIZE: tl.constexpr,
	K_TILE_SIZE: tl.constexpr,
):
	# Program indices
	query_tile_index = tl.program_id(0)
	batch_index = tl.program_id(1)

	# Offset each pointer with corresponding batch index
	# multiplied with the batch stride for each tensor
	Q_block_ptr = tl.make_block_ptr(
		Q_ptr + batch_index * stride_qb,
		shape=(N_QUERIES, DIM),
		strides=(stride_qq, stride_qd),
		offsets=(query_tile_index * Q_TILE_SIZE, 0),
		block_shape=(Q_TILE_SIZE, DIM),
		order=(1, 0),
	)

	K_block_ptr = tl.make_block_ptr(
		K_ptr + batch_index * stride_kb,
		shape=(N_KEYS, DIM),
		strides=(stride_kk, stride_kd),
		offsets=(0, 0),
		block_shape=(K_TILE_SIZE, DIM),
		order=(1, 0),
	)

	V_block_ptr = tl.make_block_ptr(
		V_ptr + batch_index * stride_vb,
		shape=(N_KEYS, DIM),
		strides=(stride_vk, stride_vd),
		offsets=(0, 0),
		block_shape=(K_TILE_SIZE, DIM),
		order=(1, 0),
	)

	L_block_ptr = tl.make_block_ptr(
		L_ptr + batch_index * stride_lb,
		shape=(N_QUERIES,),
		strides=(stride_lq,),
		offsets=(query_tile_index * Q_TILE_SIZE),
		block_shape=(Q_TILE_SIZE,),
		order=(0,),
	)

	O_block_ptr = tl.make_block_ptr(
		O_ptr + batch_index * stride_ob,
		shape=(N_QUERIES, DIM),
		strides=(stride_oq, stride_od),
		offsets=(query_tile_index * Q_TILE_SIZE, 0),
		block_shape=(Q_TILE_SIZE, DIM),
		order=(1, 0),
	)

	# On chip buffer 
	Oi = tl.zeros((Q_TILE_SIZE, DIM), dtype=tl.float32)
	li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
	mi = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

	Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
	if is_causal:
		Qi_idx = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE

	for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
		Ki = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
		Vi = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')
		
		# Compute tile of pre-softmax attention socres
		Si = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
		Si = tl.dot(Qi, tl.trans(Ki), acc=Si) / scale
		rows_max = tl.max(Si, axis=1)
		
		# Compuyte mask if is_causual set to be true
		if is_causal:
			Ki_idx = tl.arange(0, K_TILE_SIZE) + i * K_TILE_SIZE
			casual_mask = Ki_idx[None,:] <= Qi_idx[:, None]
			Si = tl.where(casual_mask, Si, -1e6)
		# Compute tile of attention scores row maximum and store in m_cur 
		m_cur = tl.maximum(mi, rows_max)
		factor = tl.exp(mi - m_cur)
		# Update row max
		mi = m_cur

		Pi = tl.exp(Si - mi[:,None])
		li = factor * li + tl.sum(Pi, axis=1)

		Oi = factor[:, None] * Oi
		# Align data type of Pi to Vi
		Pi = Pi.to(V_block_ptr.type.element_ty)
		Oi = tl.dot(Pi, Vi, acc=Oi)

		K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
		V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

	
	# Do the outer scale
	Oi = Oi / li[:, None]
	# Compue mi + LSE
	li = mi + tl.log(li)

	Oi = Oi.to(O_block_ptr.type.element_ty)
	li = li.to(L_block_ptr.type.element_ty)
	tl.store(O_block_ptr, Oi, boundary_check=(0, 1))
	tl.store(L_block_ptr, li, boundary_check=(0,))

@triton.jit
def flash_bwd_dkdv_kernel(
	Q_ptr, K_ptr, V_ptr,
	L_ptr, D_ptr,
	dO_ptr,
	dK_ptr, dV_ptr,
	stride_qb, stride_qq, stride_qd,
	stride_kb, stride_kk, stride_kd,
	stride_vb, stride_vk, stride_vd,
	stride_lb, stride_lq,
	stride_db, stride_dq,
	stride_dob, stride_doq, stride_dod,
	stride_dkb, stride_dkk, stride_dkd,
	stride_dvb, stride_dvk, stride_dvd, 
	N_QUERIES, N_KEYS,
	scale,
	DIM: tl.constexpr,
	is_causal: tl.constexpr,
	Q_TILE_SIZE: tl.constexpr,
	K_TILE_SIZE: tl.constexpr,
):
	# Program indices
	key_tile_index = tl.program_id(0) # 
	batch_index = tl.program_id(1)

	Q_block_ptr = tl.make_block_ptr(
		Q_ptr + batch_index * stride_qb,
		shape=(N_QUERIES, DIM),
		strides=(stride_qq, stride_qd),
		offsets=(0, 0),
		block_shape=(Q_TILE_SIZE, DIM),
		order=(1,0),
	)

	K_block_ptr = tl.make_block_ptr(
		K_ptr + batch_index * stride_kb,
		shape=(N_KEYS, DIM),
		strides=(stride_kk, stride_kd),
		offsets=(key_tile_index * K_TILE_SIZE, 0),
		block_shape=(K_TILE_SIZE, DIM),
		order=(1, 0),
	)

	V_block_ptr = tl.make_block_ptr(
		V_ptr + batch_index * stride_vb,
		shape=(N_KEYS, DIM),
		strides=(stride_vk, stride_vd),
		offsets=(key_tile_index * K_TILE_SIZE, 0),
		block_shape=(K_TILE_SIZE, DIM),
		order=(1, 0),
	)

	dO_block_ptr = tl.make_block_ptr(
		dO_ptr + batch_index * stride_dob,
		shape=(N_QUERIES, DIM),
		strides=(stride_doq, stride_dod),
		offsets=(0, 0),
		block_shape=(Q_TILE_SIZE, DIM),
		order=(1, 0),
		
	)

	L_block_ptr = tl.make_block_ptr(
		L_ptr + batch_index * stride_lb,
		shape = (N_QUERIES, ),
		strides=(stride_lq, ),
		offsets=(0,),
		block_shape=(Q_TILE_SIZE, ),
		order=(0, ),
	)

	D_block_ptr = tl.make_block_ptr(
		D_ptr + batch_index * stride_db,
		shape=(N_QUERIES, ),
		strides=(stride_dq, ),
		offsets=(0, ),
		block_shape=(Q_TILE_SIZE, ),
		order=(0, ),
	)

	dK_block_ptr = tl.make_block_ptr(
		dK_ptr + batch_index * stride_dkb,
		shape=(N_KEYS, DIM),
		strides=(stride_dkk, stride_dkd),
		offsets=(key_tile_index*K_TILE_SIZE, 0),
		block_shape=(K_TILE_SIZE, DIM),
		order=(1, 0),
	)

	dV_block_ptr = tl.make_block_ptr(
		dV_ptr + batch_index * stride_dvb,
		shape=(N_KEYS, DIM),
		strides=(stride_dvk, stride_dvd),
		offsets=(key_tile_index * K_TILE_SIZE, 0),
		block_shape=(K_TILE_SIZE, DIM),
		order=(1, 0),
	)

	Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
	Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

	dKj = tl.zeros((K_TILE_SIZE, DIM), dtype=tl.float32) # [K_TILE_SIZE, DIM]
	dVj = tl.zeros((K_TILE_SIZE, DIM), dtype=tl.float32) # [K_TILE_SIZE, DIM]

	if is_causal:
		k_idx = tl.arange(0, K_TILE_SIZE) + key_tile_index * K_TILE_SIZE
	
	for i in range((N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE):
		Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
		dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option='zero')
		Li = tl.load(L_block_ptr, boundary_check=(0, ), padding_option='zero') # [Q_TILE_SIZE]
		Di = tl.load(D_block_ptr, boundary_check=(0, ), padding_option='zero')

		Sij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
		Sij = tl.dot(Qi, tl.trans(Kj), acc=Sij) / scale # [Q_TILE_SIZE, K_TILE_SIE]
		
		if is_causal:
			q_idx = tl.arange(0, Q_TILE_SIZE) + i * Q_TILE_SIZE
			causal_mask = k_idx[None, :] <= q_idx[:, None]
			Sij = tl.where(causal_mask, Sij, -1e6)
		# Reconstruct attention probabilities by using saved LSE
		Pij = tl.exp(Sij - Li[:, None]) # [Q_TILE_SIZE, K_TILE_SIZE]
 
		dVj = tl.dot(tl.trans(Pij), dOi, acc=dVj)	

		dPij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
		dPij = tl.dot(dOi, tl.trans(Vj)) # [Q_TILE_SIZE, K_TILE_SIZE]
		
		
		dSij = Pij*(dPij - Di[:, None]) # [Q_TILE_SIZE, K_TILE_SIZE]
		dKj = tl.dot(tl.trans(dSij), Qi, acc=dKj) 

		# Move to next tile
		Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
		dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
		L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
		D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
	
	
	dKj = dKj / scale

	dVj = dVj.to(dV_block_ptr.type.element_ty)
	dKj = dKj.to(dK_block_ptr.type.element_ty)
	tl.store(dV_block_ptr, dVj, boundary_check=(0, 1))
	tl.store(dK_block_ptr, dKj, boundary_check=(0, 1))

	

@triton.jit
def flash_bwd_dq_kernel(
	Q_ptr, K_ptr, V_ptr,
	L_ptr, D_ptr,
	dO_ptr,
	dQ_ptr, 
	stride_qb, stride_qq, stride_qd,
	stride_kb, stride_kk, stride_kd,
	stride_vb, stride_vk, stride_vd,
	stride_lb, stride_lq,
	stride_db, stride_dq,
	stride_dob, stride_doq, stride_dod,
	stride_dqb, stride_dqq, stride_dqd,
	N_QUERIES, N_KEYS,
	scale,
	DIM: tl.constexpr,
	is_causal: tl.constexpr,
	Q_TILE_SIZE: tl.constexpr,
	K_TILE_SIZE: tl.constexpr,
):
	# Program indices
	query_tile_index = tl.program_id(0) # 
	batch_index = tl.program_id(1)

	Q_block_ptr = tl.make_block_ptr(
		Q_ptr + batch_index * stride_qb,
		shape=(N_QUERIES, DIM),
		strides=(stride_qq, stride_qd),
		offsets=(query_tile_index * Q_TILE_SIZE, 0),
		block_shape=(Q_TILE_SIZE, DIM),
		order=(1,0),
	)

	K_block_ptr = tl.make_block_ptr(
		K_ptr + batch_index * stride_kb,
		shape=(N_KEYS, DIM),
		strides=(stride_kk, stride_kd),
		offsets=(0, 0),
		block_shape=(K_TILE_SIZE, DIM),
		order=(1, 0),
	)

	V_block_ptr = tl.make_block_ptr(
		V_ptr + batch_index * stride_vb,
		shape=(N_KEYS, DIM),
		strides=(stride_vk, stride_vd),
		offsets=(0, 0),
		block_shape=(K_TILE_SIZE, DIM),
		order=(1, 0),
	)

	dO_block_ptr = tl.make_block_ptr(
		dO_ptr + batch_index * stride_dob,
		shape=(N_QUERIES, DIM),
		strides=(stride_doq, stride_dod),
		offsets=(query_tile_index * Q_TILE_SIZE, 0),
		block_shape=(Q_TILE_SIZE, DIM),
		order=(1, 0),
		
	)

	L_block_ptr = tl.make_block_ptr(
		L_ptr + batch_index * stride_lb,
		shape = (N_QUERIES, ),
		strides=(stride_lq, ),
		offsets=(query_tile_index * Q_TILE_SIZE,),
		block_shape=(Q_TILE_SIZE, ),
		order=(0, ),
	)

	D_block_ptr = tl.make_block_ptr(
		D_ptr + batch_index * stride_db,
		shape=(N_QUERIES, ),
		strides=(stride_dq, ),
		offsets=(query_tile_index * Q_TILE_SIZE, ),
		block_shape=(Q_TILE_SIZE, ),
		order=(0, ),
	)

	dQ_block_ptr = tl.make_block_ptr(
		dQ_ptr + batch_index * stride_dqb,
		shape=(N_QUERIES, DIM),
		strides=(stride_dqq, stride_dqd),
		offsets=(query_tile_index * Q_TILE_SIZE, 0),
		block_shape=(Q_TILE_SIZE, DIM),
		order=(0, 1),
	)

	Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
	dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option='zero')
	dQi = tl.zeros((Q_TILE_SIZE, DIM), dtype=tl.float32)
	Di = tl.load(D_block_ptr, boundary_check=(0, ), padding_option='zero')
	Li  = tl.load(L_block_ptr, boundary_check=(0, ), padding_option='zero')

	if is_causal:
		q_idx = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE

	for j in range((N_KEYS + K_TILE_SIZE - 1) // K_TILE_SIZE):
		Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
		Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

		Sij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
		Sij = tl.dot(Qi, tl.trans(Kj), acc=Sij) / scale

		if is_causal:
			k_idx = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
			causal_mask = k_idx[None, :] <= q_idx[:, None]
			Sij = tl.where(causal_mask, Sij, -1e6)

		Pij = tl.exp(Sij - Li[:, None]) # [Q_TILE_SIZE, K_TILE_SIZE]
		
		dPij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
		dPij = tl.dot(dOi, tl.trans(Vj)) # [Q_TILE_SIZE, K_TILE_SIZE]

		dSij = Pij * (dPij - Di[:, None]) # [Q_TILE_SIZE, K_TILE_SIZE]
		dQi = tl.dot(dSij, Kj, acc=dQi)
		K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
		V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
	
	dQi = dQi / scale

	dQi = dQi.to(dQ_block_ptr.type.element_ty)
	tl.store(dQ_block_ptr, dQi)

	
		

class FlashAttenTriton(torch.autograd.Function):
	@staticmethod
	def forward(ctx, Q, K, V, is_causal=False):
		NB, NQ, DIM = Q.shape
		_, NK, _ = K.shape
		Q_TILE_SIZE = 16
		K_TILE_SIZE = 16

		O = torch.zeros_like(Q)
		L = torch.zeros((NB, NQ), device=Q.device)
		
		flash_fwd_kernel[((NQ + Q_TILE_SIZE - 1) // Q_TILE_SIZE, NB)](
			Q, K, V, O, L,
			*Q.stride(), *K.stride(), *V.stride(),
			*O.stride(), *L.stride(),
			N_QUERIES=NQ, N_KEYS=NK,
			scale=DIM**0.5,
			is_causal=is_causal,
			DIM=DIM,
			Q_TILE_SIZE=Q_TILE_SIZE,
			K_TILE_SIZE=K_TILE_SIZE
		)
	
		ctx.save_for_backward(Q, K, V, O, L)
		ctx.is_causal = is_causal
		return O
	
	@staticmethod
	def backward(ctx, grad_output):
		Q, K, V, O, L = ctx.saved_tensors
		is_causal = ctx.is_causal

		NB, NQ, DIM = Q.shape
		_, NK, _ = K.shape
		Q_TILE_SIZE = 16
		K_TILE_SIZE = 16

		dQ = torch.zeros_like(Q)
		dK = torch.zeros_like(K)
		dV = torch.zeros_like(V)

		D = torch.sum(O * grad_output, dim=-1)

		flash_bwd_dkdv_kernel[((NK + K_TILE_SIZE - 1) // K_TILE_SIZE, NB)](
			Q, K, V, L, D,
			grad_output, dK, dV,
			*Q.stride(), *K.stride(), *V.stride(), 
			*L.stride(), *D.stride(), *grad_output.stride(), 
			*dK.stride(), *dV.stride(),
			N_QUERIES=NQ, N_KEYS=NK,
			scale=DIM ** 0.5,
			DIM=DIM,
			is_causal=is_causal,
			Q_TILE_SIZE=Q_TILE_SIZE,
			K_TILE_SIZE=K_TILE_SIZE,
		)

		flash_bwd_dq_kernel[((NQ + Q_TILE_SIZE - 1) // Q_TILE_SIZE, NB)](
			Q, K , V, L, D,
			grad_output, dQ,
			*Q.stride(), *K.stride(), *V.stride(),
			*L.stride(), *D.stride(), *grad_output.stride(),
			*dQ.stride(),
			N_QUERIES=NQ, N_KEYS=NK,
			scale=DIM ** 0.5,
			DIM=DIM,
			is_causal=is_causal,
			Q_TILE_SIZE=Q_TILE_SIZE,
			K_TILE_SIZE=K_TILE_SIZE,
		)

		return dQ, dK, dV, None
		

		
	







				
				


			
		


		