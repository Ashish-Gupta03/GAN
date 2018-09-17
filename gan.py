import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Data params
data_mean = 4
data_std = 1.25

# Model params
g_input_size = 1		# Random noise coming into generator
g_hidden_size = 50		# Generator complexity
g_output_size = 1		# size of generated output vector
d_input_size = 100		# minibatch size - cardinality distributions
d_hidden_size = 50		# discriminator complexity
d_output_size = 1		# single dimension 'real' vs 'fake'
minibatch_size = d_input_size

lr = 2e-4
optim_betas = (0.9,0.999)
n_epochs = 30000
print_interval = 200
d_steps = 1		# 'k' steps in original paper. Can put the discriminator on higher training frequency than generator 
g_steps = 1

def decorate_with_diff(data,exponent):
	# data = data.float()
	mean = torch.mean(data.data,1,keepdim=True)
	# print (data)
	mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
	# print (isinstance(mean_broadcast, torch.DoubleTensor))
	# print (torch.tensor(mean_broadcast))
	diffs = torch.pow(data - torch.tensor(mean_broadcast), exponent)
	return torch.cat([data,diffs], 1)

name,preprocess,d_input_func = "Data and variances", lambda d:decorate_with_diff(d,2.0), lambda x:x*2

def extract(v):
	return v.data.storage().tolist()

def stats(d):
	return [np.mean(d),np.std(d)]

def get_generator_input_sampler():
	return lambda m,n:torch.rand(m,n)		# Not Gaussian

def get_distribution_sampler(mu,sigma):
	return lambda n:torch.Tensor(np.random.normal(mu,sigma,(1,n)))		# Gaussian

# Model:Generator and Discriminator

class Generator(nn.Module):
	def __init__(self,input_size,hidden_size,output_size):
		super(Generator, self).__init__()
		self.map1 = nn.Linear(input_size,hidden_size)
		self.map2 = nn.Linear(hidden_size,hidden_size)
		self.map3 = nn.Linear(hidden_size,output_size)

	def forward(self, x):
		x = F.elu(self.map1(x))
		x = F.sigmoid(self.map2(x))
		return self.map3(x)

class Discriminator(nn.Module):
	def __init__(self,input_size,hidden_size,output_size):
		super(Discriminator, self).__init__()
		self.map1 = nn.Linear(input_size,hidden_size)
		self.map2 = nn.Linear(hidden_size,hidden_size)
		self.map3 = nn.Linear(hidden_size,output_size)

	def forward(self, x):
		# print (x.size())
		x = F.elu(self.map1(x))
		x = F.elu(self.map2(x))
		return F.sigmoid(self.map3(x))

d_sampler = get_distribution_sampler(data_mean,data_std)
gi_sampler = get_generator_input_sampler()
G = Generator(g_input_size,g_hidden_size,g_output_size)
D = Discriminator(d_input_func(d_input_size),d_hidden_size,d_output_size)
criterion = nn.BCELoss() # Binary Cross Entropy loss
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas = optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas = optim_betas)

for epoch in range(n_epochs):
	for k in range(d_steps):
		# 1. Train D on real+fake
		D.zero_grad()

		# 1A. Train D on real
		d_real_data = torch.tensor(d_sampler(d_input_size))
		# print("d_real_data "+str(preprocess(d_real_data)))
		d_real_decision = D(preprocess(d_real_data))
		d_real_error = criterion(d_real_decision, torch.tensor(torch.ones(1))) # ones=True
		d_real_error.backward()		# compute gradients but don't change params

		# 1B. Train D on fake
		d_gen_data = torch.tensor(gi_sampler(minibatch_size,g_input_size))
		d_fake_data = G(d_gen_data).detach() # to avoid training G on these lables
		d_fake_decision = D(preprocess(d_fake_data.t()))
		d_fake_error = criterion(d_fake_decision, torch.tensor(torch.zeros(1))) # zeros=False
		d_fake_error.backward()
		d_optimizer.step()		# Onbly optimize D's parameters; changes based on stored gradients from backward()


	for g in range(g_steps):
		# 2. Train G on D's response(but DO NOT train D on these labels)
		G.zero_grad()

		gen_input = gi_sampler(minibatch_size,g_input_size)
		g_fake_data = G(gen_input)
		dg_fake_decision = D(preprocess(g_fake_data.t()))
		g_error = criterion(dg_fake_decision, torch.tensor(torch.ones(1)))	# we want to fool, so pretend it's all genuine
		g_error.backward()
		g_optimizer.step()	# Only optimize G's paramaeter

	if epoch%print_interval==0:
		print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) " % (epoch,extract(d_real_error)[0],extract(d_fake_error)[0],extract(g_error)[0],
															stats(extract(d_real_data)),
                                                            stats(extract(d_fake_data))))