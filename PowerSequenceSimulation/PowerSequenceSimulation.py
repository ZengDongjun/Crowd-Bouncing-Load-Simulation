import torch
from torch import nn,optim,autograd
from torch.autograd import Variable
import numpy as np
import os
import scipy.io as sio
import time
import visdom
from matplotlib import pyplot as plt

lam = 10
mb_size = 256
x_dim = 25
y_dim = 21
h_dim = 80
z2_dim = 1
z1_dim = h_dim
n_disc = 5
lr = 1e-4
input_dim_G = 1+y_dim
input_dim_D = 1
output_dim = 1
Seq_len = x_dim
show_size = 1000

data=sio.loadmat('D:\\Project Files\\VSproject\\Project20230601\\Train_1.mat')
Load = data['power']
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def convert_to_one_hot(y,C):
    return np.eye(C)[:,y.reshape(-1)]  # Convert each category to one-hot coding

def get_data(batch_size):
    randidx = np.random.randint(Load.shape[1], size=batch_size)
    datax = Load[0:-1,randidx]
    y = Load[-1,randidx] - 1
    y_int = y.astype(int)
    y_hot = convert_to_one_hot(y_int,y_dim)
    return datax,y_hot

# def xavier_init(size):
#     in_dim = size[0]
#     xavier_mean = 0
#     xavier_stddev = 1./torch.sqrt(in_dim/2.)
#     return torch.normal(mean=xavier_mean,std=xavier_stddev,size=size)

class Generator(nn.Module):

    def __init__(self,input_dim,hidden_dim,output_dim):
        super(Generator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.GRULayer1 = nn.GRUCell(input_dim,hidden_dim)
        self.GRULayer2 = nn.GRUCell(hidden_dim,hidden_dim)
        self.out_FullyConnect = nn.Linear(hidden_dim,output_dim)
        self.relu = nn.ReLU()

    def forward(self,z1,y,x,hidden_1,len):
        GenerOut = []
        state_0 = torch.cat((z1,y),1).to(torch.float32)
        state_1 = hidden_1.to(torch.float32)
        input = torch.cat((x,y),1).to(torch.float32)
        for i in range(len):
            state_0 = self.GRULayer1(input, state_0)
            state_1 = self.GRULayer2(state_0,state_1)
            output = self.out_FullyConnect(state_1)
            output = self.relu(output)
            input = torch.cat((output,y),1).to(torch.float32)
            GenerOut.append(output)
        GenerOut = torch.cat(GenerOut, dim=1).to(torch.float32)
        return GenerOut

class Discriminator(nn.Module):

    def __init__(self,input_dim,hidden_dim,output_dim):
        super(Discriminator,self).__init__()

        self.GRULayer1 = nn.GRUCell(input_dim,hidden_dim)
        self.GRULayer2 = nn.GRUCell(hidden_dim,hidden_dim)
        self.out_FullyConnect = nn.Linear(hidden_dim,output_dim)
        self.Sig = nn.Sigmoid()

    def forward(self,x,hidden0,hidden1):
        len = x.size(1)
        state_0 = hidden0
        state_1 = hidden1
        for i in range(len):
            state_0 = self.GRULayer1(torch.unsqueeze(x[:,i],dim=-1),state_0)
            state_1 = self.GRULayer2(state_0,state_1)
        last = self.out_FullyConnect(state_1)
        return last

def gradient_penalty(D,xr,xf,hidden_0D,hidden_1D):

    eps = torch.empty([mb_size,1],dtype=torch.float32).uniform_(0.,1.).to(device)
    eps = eps.expand_as(xr)
    # interpolation
    x_inter = eps*xr+(1-eps)*xf
    # set it requires gradient
    x_inter.requires_grad_(True)

    pred = D(x_inter,hidden_0D,hidden_1D)
    fake = Variable(Tensor(mb_size,1).fill_(1.0), requires_grad=False)

    grads = autograd.grad(outputs=pred,inputs=x_inter,
                          grad_outputs=fake,
                          create_graph=True,retain_graph=True,only_inputs=True)[0]
    grads = grads.view(grads.size(0),-1)
    gp = ((grads.norm(2,dim=1) - 1)**2).mean()
    return gp

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    G = Generator(input_dim_G,h_dim+y_dim,output_dim).to(device)
    D = Discriminator(input_dim_D,h_dim,output_dim).to(device)
    # PATH01 = '.\\PowerGenerator25_58040.pth'
    # PATH02 = '.\\PowerDiscriminator25_58040.pth'
    # G.load_state_dict(torch.load(PATH01))
    # D.load_state_dict(torch.load(PATH02))
    print(G)
    print(D)
    optim_G = optim.Adam(G.parameters(),lr=lr,betas=(0.5,0.999))
    optim_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

    PATH = '.\\TrainIter_PATH25_224240.pth'
    CheckPoint = torch.load(PATH)
    G.load_state_dict(CheckPoint['model_G'])
    D.load_state_dict(CheckPoint['model_D'])
    optim_G.load_state_dict(CheckPoint['optim_G'])
    optim_D.load_state_dict(CheckPoint['optim_D'])
    checkpoint = CheckPoint['epoch']

    start_time = time.time()

    ObjectValue = []
    ObjectValue_Now = 0

    for len in range(Seq_len, Seq_len + 1):
        for epoch in range(checkpoint+1,1000001):

            # 1. train Discriminator firstly
            for _ in range(5):
                # 1.1 train on real data
                h_0D = torch.zeros([mb_size, h_dim], dtype=torch.float32).to(device)
                h_1D = torch.zeros([mb_size, h_dim], dtype=torch.float32).to(device)
                x_mb, y_mb = get_data(mb_size)
                x_mb = torch.from_numpy(x_mb.T).to(device)
                x_mb = x_mb[:, 0:len]
                y_mb = torch.from_numpy(y_mb.T).to(device)
                xr = torch.cat((y_mb, x_mb), 1)
                xr = xr.to(torch.float32)
                predr = D(xr, h_0D, h_1D)
                lossr = -predr.mean()

                # 1.2 train on fake data
                z1 = torch.normal(mean=0.0, std=10.0, size=[mb_size, z1_dim]).to(device)
                z2 = torch.normal(mean=0.0, std=1.0, size=[mb_size, z2_dim]).to(device)
                h_1G = torch.normal(mean=0.0, std=10.0, size=[mb_size, h_dim + y_dim]).to(device)
                xfG = G(z1, y_mb, z2, h_1G, len)
                xf = torch.cat((y_mb, xfG), 1)
                xf = xf.to(torch.float32)
                predf = D(xf, h_0D, h_1D)
                lossf = predf.mean()

                # 1.3 gradient penalty(GP)
                gp = gradient_penalty(D, xr, xf, h_0D, h_1D)

                # aggregate all
                loss_D = lossr + lossf + lam * gp

                # optimize
                optim_D.zero_grad()
                loss_D.backward()
                optim_D.step()

            # 2. train Generator
            xfG = G(z1, y_mb, z2, h_1G, len)
            xf = torch.cat((y_mb, xfG), 1)
            xf = xf.to(torch.float32)
            predf = D(xf, h_0D, h_1D)
            # max predf.mean()
            loss_G = -predf.mean()

            # optimize
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            # Calculate the objective value
            Delta_ObjectValue_Now = predr.mean() - predf.mean() - lam * gp
            ObjectValue_Now = Delta_ObjectValue_Now.item()
            loss_G_Now = loss_G.item()
            loss_D_Now = loss_D.item()

            if ((len) * epoch) % 100 == 0 or epoch == 1:
                t2 = time.time() - start_time
                print('Len:{}:Iter:{};time:{:.4};D_loss:{:.10};G_loss:{:.10};V(D,G):{:.10}'.format(len,epoch,t2,loss_D.item(),loss_G.item(),ObjectValue_Now))

                if ((len) * epoch) % 1000 == 0 or epoch == 1:
                    if epoch % 1000 == 0 or epoch == 1:
                        ObjectValue.append([epoch,loss_G_Now,loss_D_Now, ObjectValue_Now])
                    # Save the model
                    TrainIter_PATH = '.\\TrainIter_PATH{}_{}.pth'.format(len, epoch)
                    state = {'model_G':G.state_dict(),'model_D':D.state_dict(),'optim_G':optim_G.state_dict(),'optim_D':optim_D.state_dict(),'epoch':epoch}
                    torch.save(state, TrainIter_PATH)
                    # Save the generated figure
                    z1 = torch.normal(mean=0.0, std=10.0, size=[show_size, z1_dim]).to(device)
                    z2 = torch.normal(mean=0.0, std=1.0, size=[show_size, z2_dim]).to(device)
                    h_1G = torch.normal(mean=0.0, std=10.0, size=[show_size, h_dim + y_dim]).to(device)
                    y = np.ones((show_size, 1), dtype=np.int64) - 1
                    y_int = y.astype(int)
                    y_hot = convert_to_one_hot(y_int, y_dim)
                    y_hot = torch.from_numpy(y_hot.T).to(device)
                    xfG = G(z1, y_hot, z2, h_1G, Seq_len)
                    xfG = xfG.cpu()
                    xfG = xfG.detach().numpy()
                    x = np.linspace(1, x_dim, x_dim)
                    fig, ax = plt.subplots()
                    for i in range(show_size):
                        ax.plot(x, xfG[i, :])
                    plt.savefig('.\\Len{}_Iter{}.png'.format((len), epoch))
                    plt.close('all')  # Avoid the leakage of memory

                    if epoch % 10000 == 0 or epoch == 1:
                        np.savetxt('.\\ObjectValue_224240.csv', np.array(ObjectValue))
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # main()
    Fre = [150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350]
    PowerGener = Generator(input_dim_G,h_dim+y_dim,output_dim).to(device)
    PATH = '.\\TrainedWeight_401200.pth'
    PowerGener.load_state_dict(torch.load(PATH)['model_G'])
    for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]:
        xG = []
        for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            z1 = torch.normal(mean=0.0, std=10.0, size=[show_size, z1_dim]).to(device)
            z2 = torch.normal(mean=0.0, std=1.0, size=[show_size, z2_dim]).to(device)
            h_1G = torch.normal(mean=0.0, std=10.0, size=[show_size, h_dim + y_dim]).to(device)
            y = i*np.ones((show_size, 1), dtype=np.int64) - 1
            y_int = y.astype(int)
            y_hot = convert_to_one_hot(y_int, y_dim)
            y_hot = torch.from_numpy(y_hot.T).to(device)
            Seq_len = 200
            xfG = PowerGener(z1, y_hot, z2, h_1G, Seq_len)
            xfG = xfG.cpu()
            xfG = xfG.detach().numpy()
            xG.extend(list(xfG))
            torch.cuda.empty_cache()
        xG = np.array(xG)
        sio.savemat('Generpower{}.mat'.format(Fre[i-1]), {'Generpower{}'.format(Fre[i-1]): xG.T})

