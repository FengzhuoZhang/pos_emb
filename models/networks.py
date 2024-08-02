import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#def zero_init(blocks):
#    for module in blocks:



def zero_emb(x):
    return x


class SinPosEmb(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(SinPosEmb, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0).to('cuda')

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)] #.clone().detach()

class RoPosEmb(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(RoPosEmb, self).__init__()

        # Not a parameter
        cos_pos,sin_pos = self._get_rope_table(n_position, d_hid)
        self.register_buffer('cos_table', cos_pos)
        self.register_buffer('sin_table', sin_pos)

    def _get_rope_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = torch.tensor([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        cos_pos = torch.repeat_interleave(sinusoid_table[:, 0::2],2,  1).unsqueeze(0).to('cuda')
        sin_pos = torch.repeat_interleave(sinusoid_table[:, 1::2],2, 1).unsqueeze(0).to('cuda')
        #import ipdb; ipdb.set_trace()
        return cos_pos,sin_pos

    def forward(self, x):
        x2 = torch.stack([-x[...,1::2],x[...,0::2]],-1)
        x2 = torch.reshape(x2,x.shape)
        x_emb = x * self.cos_table[:,:x.size(1)]+ x2*self.sin_table[:,:x.size(1)]

        return x_emb #.clone().detach()


def rope_emb(x):
    return torch.zeros_like(x)

def interweave(x,y):
    return None

class MHA(nn.Module):
    def __init__(self, input_size, heads, dimension):
        super(MHA, self).__init__()
        self.h, self.d = heads, dimension
        self.lq = nn.Linear(input_size, self.h * self.d, bias = False)
        self.lk = nn.Linear(input_size, self.h * self.d, bias = False)#nn.Identity()#
        self.lv = nn.Linear(input_size, self.h * self.d, bias = False)
        self.fc = nn.Linear(self.h * self.d, self.d, bias = False) #nn.Identity()
        # self.lq = nn.Linear(input_size, self.h * self.d)
        # self.lk = nn.Linear(input_size, self.h * self.d)
        # self.lv = nn.Linear(input_size, self.h * self.d)
        # self.fc = nn.Linear(self.h * self.d, self.d)

    def forward(self, q, k, v):
        if len(q.shape)==2:
            q = torch.unsqueeze(q,dim=0)
        if len(k.shape)==2:
            k = torch.unsqueeze(k,dim=0)
        if len(v.shape)==2:
            v = torch.unsqueeze(v,dim=0)
        b, n_q, n_k, h, d = q.size(0), q.size(1), k.size(1), self.h, self.d
        #import ipdb; ipdb.set_trace()
        q, k, v = self.lq(q), self.lk(k), self.lv(v)                    # b, n_*, h*d
        q, k, v = map(lambda x: x.reshape(b, -1, h, d), [q, k, v])      # b, n_*, h, d
        qk = torch.einsum('bnhd,bmhd->bhnm', (q,k))                     # b, h, n_q, n_k
        #import ipdb; ipdb.set_trace()
        att = F.softmax(qk / (self.d ** .5), dim=3)                     # b, h, n_q, n_k
        att_out = torch.einsum('bhnm,bmhd->bnhd', (att,v))              # b, n_q, h, d
        att_out = att_out.reshape(b, -1, h*d)                           # b, n_q, h*d
        out = self.fc(att_out)                                          # b, n_q, d
        return out
    
class CMHA(nn.Module):                          #Causal MHA
    def __init__(self, input_size, heads, dimension):
        super(CMHA, self).__init__()
        self.h, self.d = heads, dimension
        self.lq = nn.Linear(input_size, self.h * self.d, bias = False)
        self.lk = nn.Linear(input_size, self.h * self.d, bias = False)
        self.lv = nn.Linear(input_size, self.h * self.d, bias = False)
        self.fc = nn.Linear(self.h * self.d, self.d, bias = False)
        # self.lq = nn.Linear(input_size, self.h * self.d)
        # self.lk = nn.Linear(input_size, self.h * self.d)
        # self.lv = nn.Linear(input_size, self.h * self.d)
        # self.fc = nn.Linear(self.h * self.d, self.d)
    
    def _mask(self, n, h, d1, d2):
        mask  = torch.tril(torch.ones(d1,d2),diagonal=0)
        mask = torch.stack([ mask for _ in range(h)],dim = 0)
        mask = torch.stack([ mask for _ in range(n)],dim = 0)
        mask = mask == 0
        return mask

    def forward(self, q, k, v):
        if len(q.shape)==2:
            q = torch.unsqueeze(q,dim=0)
        if len(k.shape)==2:
            k = torch.unsqueeze(k,dim=0)
        if len(v.shape)==2:
            v = torch.unsqueeze(v,dim=0)
        b, n_q, n_k, h, d = q.size(0), q.size(1), k.size(1), self.h, self.d
        #import ipdb; ipdb.set_trace()
        q, k, v = self.lq(q), self.lk(k), self.lv(v)                    # b, n_*, h*d
        q, k, v = map(lambda x: x.reshape(b, -1, h, d), [q, k, v])      # b, n_*, h, d
        mask = self._mask(q.shape[0],q.shape[2],q.shape[1],k.shape[1]).to(q.device)
        #import ipdb; ipdb.set_trace()
        qk = torch.einsum('bnhd,bmhd->bhnm', (q,k))                     # b, h, n_q, n_k
        qk = qk.masked_fill(mask,float('-inf'))
        att = F.softmax(qk / (self.d ** .5), dim=3)                     # b, h, n_q, n_k
        att_out = torch.einsum('bhnm,bmhd->bnhd', (att,v))              # b, n_q, h, d
        att_out = att_out.reshape(b, -1, h*d)                           # b, n_q, h*d
        out = self.fc(att_out)                                          # b, n_q, d
        return out

class SingleLayerTransformer(nn.Module):
    def __init__(self,n_dims , pos_emb, causal = False, output_mix=True):
        super(SingleLayerTransformer, self).__init__()
        self.n_dims = n_dims
        self.pos_emb = pos_emb
        self.output_mix = output_mix
        self.causal = causal
        if self.pos_emb == '':
            self.pos_emb_func = zero_emb
        #elif self.pos_emb == 


        self.blocks = nn.ModuleList([])
        if self.causal:
            attn_block = CMHA(input_size = n_dims, heads = 1,dimension = n_dims)
        else:
            attn_block = MHA(input_size = n_dims, heads = 1,dimension = n_dims)
        self.blocks.append(attn_block)
        if self.output_mix == True:
            linear_block = nn.Linear(self.n_dims,1)
            self.blocks.append(linear_block)

    def _readout(self,x,read_y):
        #x = torch.squeeze(x)
        if read_y:
            output = x[:,0::2]
        else:
            output = x
        #import ipdb; ipdb.set_trace()
        return output[:,:,0]
    
    def forward(self,x,read_y = True):
        log = []
        hidden = self.pos_emb_func(x)
        log.append(hidden)
        for module in self.blocks:
            if type(module)== MHA or type(module)== CMHA:
                hidden = module(hidden,hidden,hidden)
            else:
                hidden = module(hidden)
            log.append(hidden)
        output = hidden
        return self._readout(output,read_y), log
    

class MultiLayerTransformer(nn.Module):
    def __init__(self,n_dims , layers, pos_emb, causal = False, output_mix=True):
        super(MultiLayerTransformer, self).__init__()
        self.n_dims = n_dims
        self.pos_emb = pos_emb
        self.output_mix = output_mix
        self.causal = causal
        self.layers = layers
        if self.pos_emb == '':
            self.pos_emb_func = zero_emb
        elif self.pos_emb == 'sin':
            sin_emb = SinPosEmb(d_hid = self.n_dims)
            self.pos_emb_func = sin_emb.forward
        elif self.pos_emb == 'rope':
            assert self.n_dims%2 ==0
            rope_emb = RoPosEmb(d_hid = self.n_dims)
            self.pos_emb_func = rope_emb.forward
        else:
            print("Unknown Embedding")
            raise NotImplementedError

        #elif self.pos_emb == 


        self.blocks = nn.ModuleList([])
        for _ in range(layers):
            if self.causal:
                attn_block = CMHA(input_size = n_dims, heads = 1,dimension = n_dims)
            else:
                attn_block = MHA(input_size = n_dims, heads = 1,dimension = n_dims)
            self.blocks.append(attn_block)
        if self.output_mix == True:
            linear_block = nn.Linear(self.n_dims,1)
            self.blocks.append(linear_block)

    def _readout(self,x,read_y):
        #x = torch.squeeze(x)
        if read_y:
            output = x[:,0::2]
        else:
            output = x
        #import ipdb; ipdb.set_trace()
        return output[:,:,0]
    
    def forward(self,x,read_y = True):
        log = []
        #import ipdb; ipdb.set_trace()
        hidden = self.pos_emb_func(x)
        log.append(hidden)
        for module in self.blocks:
            if type(module)== MHA or type(module)== CMHA:
                hidden = module(hidden,hidden,hidden)
            else:
                hidden = module(hidden)
            log.append(hidden)
        output = hidden
        return self._readout(output,read_y), log

