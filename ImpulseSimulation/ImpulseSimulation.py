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
mb_size = 64
x_dim = 100
z_dim = x_dim
y_dim = 21
h_dim = 100
n_disc = 5
lr = 1e-4

data=sio.loadmat('D:\\Project Files\\VSproject\\Project20230601\\Train_1.mat')
Load = data['Impulse']
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def convert_to_one_hot(y,C):
    return np.eye(C)[:,y.reshape(-1)]  # Convert each category to one-hot coding

def get_data(batch_size):
    randidx = np.random.randint(Load.shape[1], size=batch_size)
    datax = Load[0:100, randidx]
    y = Load[-1, randidx] - 1
    y_int = y.astype(int)
    y_hot = convert_to_one_hot(y_int, y_dim)
    return datax, y_hot

def xavier_init(size):
    in_dim = size[0]
    xavier_mean = 0
    xavier_stddev = 1./torch.sqrt(in_dim/2.)
    return torch.normal(mean=xavier_mean,std=xavier_stddev,size=size)

def sample_z(m,n):
    return np.random.uniform(-1.,1.,size=[m,n])

class Reshape(nn.Module):
    def __init__(self,*args):
        super(Reshape,self).__init__()
        self.shape = args
    def forward(self,x):
        return x.view((x.size(0),)+self.shape)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net1 = nn.Sequential(
            nn.Linear(x_dim+y_dim,100),
            nn.Tanh(),
            nn.Linear(100,100),
            nn.Tanh()
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=10,stride=1,padding='same'),
            nn.ReLU()
        )

    def forward(self,input):
        output1 = self.net1(input)
        output1 = torch.unsqueeze(output1,1)
        output2 = self.net2(output1)
        output = torch.squeeze(output2,1)
        return output

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(x_dim+y_dim,100),
            nn.Tanh(),
            nn.Linear(100,1)
        )

    def forward(self,input):
        output = self.net(input)
        return output

def gradient_penalty(D,xr,xf):

    eps = torch.empty([mb_size,1],dtype=torch.float32).uniform_(0.,1.).to(device)
    eps = eps.expand_as(xr)
    # interpolation
    x_inter = eps*xr+(1-eps)*xf
    # set it requires gradient
    x_inter.requires_grad_(True)

    pred = D(x_inter)
    fake = Variable(Tensor(mb_size, 1).fill_(1.0), requires_grad=False)

    grads = autograd.grad(outputs=pred,inputs=x_inter,
                          grad_outputs=fake,
                          create_graph=True,retain_graph=True,only_inputs=True)[0]
    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    G = Generator().to(device)
    D = Discriminator().to(device)
    print(G)
    print(D)
    optim_G = optim.Adam(G.parameters(),lr=lr,betas=(0.5,0.999))
    optim_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

    start_time = time.time()
    epochlen = 2000000
    len = x_dim
    ObjectValue = []
    ObjectValue_Now = 0

    for epoch in range(1,epochlen+1):
        # 1. train Discriminator firstly
        for _ in range(5):
            # 1.1 train on real data
            xr_mb, y_mb = get_data(mb_size)
            xr_mb = torch.from_numpy(xr_mb.T).to(device)
            y_mb = torch.from_numpy(y_mb.T).to(device)
            xr = torch.cat((xr_mb,y_mb),1)
            xr = xr.to(torch.float32)
            predr = D(xr)
            lossr = -predr.mean()

            # 1.2 train on fake data
            z = Variable(Tensor(np.random.uniform(0, 1, (mb_size,z_dim))))
            zy = torch.cat((z, y_mb), 1)
            zy = zy.to(torch.float32)
            xfG = G(zy)
            xf = torch.cat((xfG,y_mb),1)
            xf = xf.to(torch.float32)
            predf = D(xf)
            lossf = predf.mean()

            # 1.3 gradient penalty(GP)
            gp = gradient_penalty(D,xr,xf)

            # aggregate all
            loss_D = lossr + lossf + lam*gp

            # optimize
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # 2. train Generator
        xfG = G(zy)
        xf = torch.cat((xfG, y_mb), 1)
        xf = xf.to(torch.float32)
        predf = D(xf)
        # max predf.mean()
        loss_G = -predf.mean()

        # optimize
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        # Calculate the objective value
        Delta_ObjectValue_Now = predr.mean() - predf.mean() - lam * gp
        ObjectValue_Now = Delta_ObjectValue_Now.item()

        if epoch % 100 == 0 or epoch == 1:
            t2 = time.time()-start_time
            print('Iter:{};time:{:.4};D_loss:{:.10};G_loss:{:.10};V(D,G):{:.10}'.format(epoch,t2,loss_D.item(),loss_G.item(),ObjectValue_Now))
            if epoch % 1000 == 0 or epoch == 1:
                ObjectValue.append([epoch,ObjectValue_Now])
                # Save the model
                PATH = '.\\ImpulseGenerator{}_{}.pth'.format(len,epoch)
                torch.save(G.state_dict(), PATH)
                # Save the generated figure
                z = torch.empty([100, z_dim], dtype=torch.float32).uniform_(0., 1.).to(device)
                y = 1*np.ones((100, 1), dtype=np.int64) - 1
                y_int = y.astype(int)
                y_hot = convert_to_one_hot(y_int, y_dim)
                y_hot = torch.from_numpy(y_hot.T).to(device)
                zy = torch.cat((z, y_hot), 1)
                zy = zy.to(torch.float32)
                xfG = G(zy)
                xfG = xfG.cpu()
                xfG = xfG.detach().numpy()
                x = np.linspace(1, 100, 100)
                fig, ax = plt.subplots()
                for i in range(100):
                    ax.plot(x, xfG[i, :])
                plt.savefig('.\\Len{}_Iter{}.png'.format((len), epoch))
                plt.close('all')  # Avoid the leakage of memory

                if epoch % 10000 == 0 or epoch == 1:
                    np.savetxt('.\\ObjectValue.csv',np.array(ObjectValue))
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # main()
    Fre = [150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350]
    ImpGener = Generator()
    PATH = '.\\TrainedWeight_2589000.pth'
    ImpGener.load_state_dict(torch.load(PATH)['model_G'])
    ImpGener.to(device)
    for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]:
        z = torch.empty([1000000, z_dim], dtype =torch.float32).uniform_(0., 1.).to(device)
        y = i*np.ones((1000000, 1), dtype=np.int64) - 1
        y_int = y.astype(int)
        y_hot = convert_to_one_hot(y_int, y_dim)
        y_hot = torch.from_numpy(y_hot.T).to(device)
        zy = torch.cat((z, y_hot), 1)
        zy = zy.to(torch.float32)
        xfG = ImpGener(zy)
        xfG = xfG.cpu()
        xfG = xfG.detach().numpy()
        sio.savemat('GenerImp{}.mat'.format(Fre[i-1]), {'GenerImp{}'.format(Fre[i-1]): xfG.T})
        torch.cuda.empty_cache()

