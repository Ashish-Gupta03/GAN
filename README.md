# GAN
Starter code for Generative Adversarial Networks (GANs) in PyTorch

# Requirements
pytorch>=0.4

For pytorch<0.4
* Replace *torch.tensor* with *torch.autograd.Variable*

# Running
Run the code 
* python gan.py

Here, you'll train two nets on a shifted/scaled Gaussian distribution. The 'fake' distribution should match the 'real' one within a reasonable time.
