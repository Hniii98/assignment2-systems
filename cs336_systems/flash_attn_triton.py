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

	


class FlashAttenTriton(torch.autograd.Function):
	@staticmethod
	def forward(ctx, Q, K, V, is_causal=False):
		NB, NQ, D = Q.shape
		_, NK, _ = K.shape
		Q_TILE_SIZE = 16
		K_TILE_SIZE = 16

		O = torch.zeros_like(Q)
		L = torch.zeros((NB, NQ), device=Q.device)
		
		flash_fwd_kernel[(NQ, NB)](
			Q, K, V, O, L,
			*Q.stride(), *K.stride(), *V.stride(),
			*O.stride(), *L.stride(),
			N_QUERIES=NQ, N_KEYS=NK,
			scale=D**0.5,
			is_causal=is_causal,
			D=D,
			Q_TILE_SIZE=Q_TILE_SIZE,
			K_TILE_SIZE=K_TILE_SIZE
		)
	
		ctx.save_for_backward(L, Q, K, V, O)
		ctx.is_causal = is_causal
		return O
	
	@staticmethod
	def backward(ctx, grad_output):
		raise NotImplementedError
	







				
				


			
		


		