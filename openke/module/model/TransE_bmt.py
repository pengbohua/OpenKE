import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import os
import json
import numpy as np

class BaseModule(nn.Module):

	def __init__(self):
		super(BaseModule, self).__init__()
		self.zero_const = nn.Parameter(torch.Tensor([0]))
		self.zero_const.requires_grad = False
		self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
		self.pi_const.requires_grad = False

	def load_checkpoint(self, path):
		self.load_state_dict(torch.load(os.path.join(path)))
		self.eval()

	def save_checkpoint(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		f = open(path, "r")
		parameters = json.loads(f.read())
		f.close()
		for i in parameters:
			parameters[i] = torch.Tensor(parameters[i])
		self.load_state_dict(parameters, strict = False)
		self.eval()

	def save_parameters(self, path):
		f = open(path, "w")
		f.write(json.dumps(self.get_parameters("list")))
		f.close()

	def get_parameters(self, mode = "numpy", param_dict = None):
		all_param_dict = self.state_dict()
		if param_dict == None:
			param_dict = all_param_dict.keys()
		res = {}
		for param in param_dict:
			if mode == "numpy":
				res[param] = all_param_dict[param].cpu().numpy()
			elif mode == "list":
				res[param] = all_param_dict[param].cpu().numpy().tolist()
			else:
				res[param] = all_param_dict[param]
		return res

	def set_parameters(self, parameters):
		for i in parameters:
			parameters[i] = torch.Tensor(parameters[i])
		self.load_state_dict(parameters, strict = False)
		self.eval()

class Model(BaseModule):

	def __init__(self, ent_tot, rel_tot):
		super(Model, self).__init__()
		self.ent_tot = ent_tot
		self.rel_tot = rel_tot

	def forward(self):
		raise NotImplementedError
	
	def predict(self):
		raise NotImplementedError

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
		# 新增3：定义并把模型放置到单独的GPU上
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
		# gather score
		gather_score = torch.zeros(score.size(0)*self.world_size, self.split_size, device=self.device, requires_grad=True)
		dist.all_gather_into_tensor(gather_score, score)
		gather_score = torch.cat([gather_score[i*score.size(0): (i+1)*score.size(0),:] for i in range(self.world_size)], dim=-1)
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