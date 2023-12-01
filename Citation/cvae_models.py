#Adapted from https://github.com/timbmg/VAE-CVAE-MNIST

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions.transforms import AffineTransform, SigmoidTransform
import torch.distributions.transforms as transform

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, conditional_size=0):

        super().__init__()

        if conditional:
            assert conditional_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, conditional_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, conditional_size)

    def forward(self, x, c=None):

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, means, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return means + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, conditional_size):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += conditional_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, conditional_size):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + conditional_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x

class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, input_dim, num_flows, conditional=False, conditional_size=0):
        super(ConditionalNormalizingFlow, self).__init__()

        # TODO: Rename the arguments to match with those of VAEs 
        self.input_dim = input_dim
        self.latent_size = input_dim
        self.num_flows = num_flows
        self.conditional = conditional
        if conditional: 
            assert conditional_size > 0

        self.conditional_size = conditional_size

        # Define the base distribution for the latent space
        self.base_distribution = MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))

        # Initialize the flows
        self.flows = nn.ModuleList([FlowLayer(input_dim) for _ in range(num_flows)])
        # list of transformations, defines T

        # Conditional vector, will be passed into base top make it condtional

        if conditional:
            self.context_encoder = nn.Sequential(
                nn.Linear(conditional_size, 64),
                nn.ReLU(),
                nn.Linear(64, 2 * input_dim)  # Assuming a diagonal covariance matrix
            )

    def forward(self, c):
        # same as sampling process

        # Encode the context
        if self.conditional:
            context_params = self.context_encoder(c)
            z_distribution = MultivariateNormal(context_params[:, :self.input_dim],torch.diag_embed(torch.exp(context_params[:, self.input_dim:])))
        else:
            z_distribution = MultivariateNormal(torch.zeros(self.input_dim), torch.eye(self.input_dim))

        # Initialize the latent variable using the base distribution
        z = z_distribution.sample()

        total_log_det = 0
        # Apply the normalizing flows
        for flow in self.flows:
            z, log_det = flow(z) 
            total_log_det+=log_det

        # Return the latent variable and the log determinant of the transformation
        return z, total_log_det

    def inference(self,z,c):
        z_samp = torch.zeros_like(c)
        minibatch_size = 128
        miniloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(c), batch_size=minibatch_size, shuffle=False)        
        for i,batch in enumerate(miniloader):
            x = batch[0]
            z_samp_i,_ = self.forward(x)
            z_samp[i:i+len(x)] = z_samp_i 
        return z_samp
    
    def encode(self,x,c=None):
        # apply the inverse of the composed transforms
        z = x*1
        # Apply the inverse normalizing flows
        total_log_det = 0
        for flow in reversed(self.flows):
            z, log_det = flow.inverse(z)
            total_log_det+=log_det
        if self.conditional:
            context_params = self.context_encoder(c)
            z_distribution = MultivariateNormal(context_params[:, :self.input_dim],torch.diag_embed(torch.exp(context_params[:, self.input_dim:])))
        else:
            z_distribution = MultivariateNormal(torch.zeros(self.input_dim), torch.eye(self.input_dim))
        return z, total_log_det, z_distribution


class FlowLayer(nn.Module):
    def __init__(self, latent_size):
        super(FlowLayer, self).__init__()

        self.latent_size = latent_size

        # Define the components of a flow layer
        self.transform = torch.distributions.transforms.ComposeTransform([
            AffineTransform(loc=torch.zeros(latent_size, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")), scale=torch.ones(latent_size, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))),
            SigmoidTransform()
        ])
        # TODO: look for better transformations other than sigmoid, Note: ReLU is not invertible

    def forward(self, x):
        # Apply the flow layer
        z = self.transform(x)
        log_det = self.transform.log_abs_det_jacobian(x, z)
        return z, log_det
    
    def inverse(self, z):
        x = self.transform.inv(z)
        log_det = self.transform.log_abs_det_jacobian(x ,z)
        return x, - log_det
    
class Flow(transform.Transform, nn.Module):
    
    def __init__(self):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)
    
    # Init all parameters
    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
            
    # Hacky hash bypass
    def __hash__(self):
        return nn.Module.__hash__(self)


class PlanarFlow(Flow):
    def __init__(self, in_features, condition_features):
        super(PlanarFlow, self).__init__()

        self.u = nn.Parameter(torch.randn(1, in_features))
        self.w = nn.Parameter(torch.randn(1, in_features))
        self.b = nn.Parameter(torch.randn(1))
        self.h = torch.tanh  # Activation function

        # Additional conditioning for planar flow
        self.condition_layer = nn.Linear(condition_features, in_features)

    def _call(self, z, c):
        u = self.u
        w = self.w + self.condition_layer(c) # w is modified for conditioning
        b = self.b

        zwb = torch.matmul(z, w.t()) + b
        psi = self.h(zwb)
        z = z + u * psi
        return z
    
    def log_abs_det_jacobian(self, z, c):
        u = self.u
        w = self.w + self.condition_layer(c) # w is modified for conditioning
        b = self.b

        zwb = torch.matmul(z, w.t()) + b
        psi = (1 - self.h(zwb)**2) * w
        det_grad = 1 + torch.mm(psi, u.t())
        return torch.log(det_grad.abs() + 1e-9)

class ConditionalPlanarNormalizingFlow(nn.Module):
    def __init__(self, in_features, condition_features, num_flows, density):
        super(ConditionalPlanarNormalizingFlow, self).__init__()
        biject = []
        for _ in range(num_flows):
            biject.append(PlanarFlow(in_features, condition_features))

        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        
        self.base_density = density
        self.final_density = torch.distributions.TransformedDistribution(density, self.transforms)
        self.log_det = []

    def forward(self, z, c):
        # to be interpreted as observed samples to latent space
        self.log_det = []
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z,c))
            z = self.bijectors[b](z,c)
        return z, self.log_det

    def inference(self,z,c):
        return self.transforms.inv()

