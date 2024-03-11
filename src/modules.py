import torch
from torch import nn

class IMG2TEXT(nn.Module):
	def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
		super().__init__()
		self.i2t_first=IMG2TEXT_MLP(embed_dim, middle_dim, output_dim, n_layer, dropout)
		self.i2t_last=IMG2TEXT_MLP(embed_dim, middle_dim, output_dim, n_layer, dropout)

	def forward(self, x: torch.Tensor):
		first=self.i2t_first(x)
		last=self.i2t_last(x)
		
		return first,last

class IMG2TEXT_MLP(nn.Module):
	def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
		super().__init__()
		self.fc_out = nn.Linear(middle_dim, output_dim)
		layers = []
		dim = embed_dim
		for _ in range(n_layer):
			block = []
			block.append(nn.Linear(dim, middle_dim))
			block.append(nn.Dropout(dropout))
			block.append(nn.LeakyReLU(0.2))
			dim = middle_dim
			layers.append(nn.Sequential(*block))        
		self.layers = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor):
		for layer in self.layers:
			x = layer(x)
		return self.fc_out(x)


class IMG2TEXTwithEXP(nn.Module):
	def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1,exp_dim=64):
		super().__init__()
		self.i2t_first=IMG2TEXT_MLP(embed_dim+exp_dim, middle_dim, output_dim, n_layer, dropout)
		self.i2t_last=IMG2TEXT_MLP(embed_dim+exp_dim, middle_dim, output_dim, n_layer, dropout)
		self.null_exp=torch.nn.Parameter(torch.zeros([exp_dim]))

	def forward(self, x,exp=None,mask=None):
		if exp is None:
			exp = self.null_exp.unsqueeze(0)
		if mask is None:
			mask=torch.ones((len(x),),device=x.device)
		mask=mask.reshape((-1,1))
		exp = exp*mask + self.null_exp.unsqueeze(0)*(1-mask)
		x = torch.cat([x,exp],-1)
		first=self.i2t_first(x)
		last=self.i2t_last(x)
		
		return first,last