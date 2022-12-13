import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .Model import Model
import time
import torch
from typing import List


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )

def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """ Split a tensor along its last dimension.
        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.
        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def _gather_along_last_dim(input_):
    """Gather tensors and concatinate along the last dimension."""
    world_size = torch.distributed.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    first_dim_size = input_.size(0)
    last_dim_size = input_.size(-1)
    rank = torch.distributed.get_rank()

    output = torch.zeros(first_dim_size*world_size, last_dim_size, device=torch.device("cuda", rank))
    torch.distributed.all_gather_into_tensor(output, input_)
    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat([output[i*first_dim_size: (i+1)*first_dim_size,:] for i in range(world_size)], dim=-1).contiguous()
    return output

def _split_along_last_dim(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = torch.distributed.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = torch.distributed.get_rank()
    output = input_list[rank].contiguous()

    return output

class Gather(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)
    
    @staticmethod
    def forward(ctx, input_):

        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):

        return _split_along_last_dim(grad_output)


class TransE_bmt(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None, local_rank=0, world_size=1, debug=False):
		super(TransE_bmt, self).__init__(ent_tot, rel_tot)
		
		self.debug = debug
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.local_rank = local_rank
		self.world_size = world_size
		self.split_size = int(self.dim / self.world_size)
		assert self.dim % self.world_size == 0
		self.device = torch.device("cuda", self.local_rank)

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False


	def _calc(self, h, t, r, mode):
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		return score

	def forward(self, data):
		if self.debug:
			h = data['batch_h'].to(self.device)
			t = data['batch_t'].to(self.device)
			r = data['batch_r'].to(self.device)
		else:
			batch_h = data['batch_h'].to(self.device)
			batch_t = data['batch_t'].to(self.device)
			batch_r = data['batch_r'].to(self.device)
			h = self.ent_embeddings(batch_h)
			t = self.ent_embeddings(batch_t)
			r = self.rel_embeddings(batch_r)
		mode = data['mode']
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		# split according to local rank
		
		h = h[:, self.split_size*self.local_rank:self.split_size*(self.local_rank + 1)]
		r = r[:, self.split_size*self.local_rank:self.split_size*(self.local_rank + 1)]
		t = t[:, self.split_size*self.local_rank:self.split_size*(self.local_rank + 1)]
		
		score = self._calc(h ,t, r, mode).to(self.device)
		score = score.squeeze()
		# # gather score
		# gather_score = torch.zeros(score.size(0)*self.world_size, self.split_size, device=self.device, requires_grad=True)
		# dist.all_gather_into_tensor(gather_score, score)
		# gather_score = torch.cat([gather_score[i*score.size(0): (i+1)*score.size(0),:] for i in range(self.world_size)], dim=-1)
		
		gather_score = Gather.apply(score)
		# x = input()
		gather_score = torch.norm(gather_score, self.p_norm, -1).flatten()
		if self.margin_flag:
			return self.margin - gather_score
		else:
			return gather_score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()

# import argparse
# import torch
# import torch.distributed as dist

# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", default=-1, type=int)
# args = parser.parse_args()

# torch.cuda.set_device(args.local_rank)
# dist.init_process_group(backend='nccl')
# world_size = torch.distributed.get_world_size()
# transe = TransE_bmt(15000, 300, 200, 1, True, local_rank=args.local_rank, world_size=world_size)

# data = {
# 	"batch_h": torch.zeros(64, 200),
# 	"batch_t": torch.zeros(64, 200),
# 	"batch_r": torch.zeros(64, 200),
# 	'mode': "norma"
# }

# score = transe(data)
# print(score)