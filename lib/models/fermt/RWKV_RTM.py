import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
import math
from timm.layers import trunc_normal_

class Weighted(nn.Module):
    def __init__(self, d_model, n_heads=8,norm_layer=partial(nn.LayerNorm, eps=1e-6),  dropout=0.1,
                 init_values=0.):
        super().__init__()
        self.query_norm = norm_layer(d_model)
        self.feat_norm = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,dropout=dropout)
        self.Weighted_Router = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)

    def forward(self, query,feat):
            #query:l,b,d;feat:l,b,d
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query),
                             self.feat_norm(feat),self.feat_norm(feat))[0]
            return query + self.Weighted_Router * attn
        query = _inner_forward(query, feat)
        return query

class Weighted_fusion_Router(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.injector = Weighted(d_model=d_model)

    def forward(self,x,xs):
        x = self.injector(x.permute(1,0,2),xs.permute(1,0,2)).permute(1,0,2)
        return x


class Region_RWKV(nn.Module):
    def __init__(self, dim=192, head=12):
        super().__init__()

        self.n_embd = dim
        self.n_head = head
        self.head_size = self.n_embd // self.n_head
        
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

        self.time_decay = nn.Parameter(torch.zeros(self.n_embd))
        self.time_first = nn.Parameter(torch.zeros(self.n_embd))

        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.output = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, state):
        B, T, C = x.size()
        # Removed 'pass' placeholder from original, state handling is below
        k = self.key(x)    
        v = self.value(x)  
        r_in = self.receptance(x) 
        r = torch.sigmoid(r_in) 
        
        w = torch.exp(self.time_decay)
        u = self.time_first
        
        output_wkv = torch.zeros_like(k)
        
        current_aa, current_bb, current_pp = state if state is not None else (
            torch.zeros(B, T, C, device=x.device, dtype=x.dtype),
            torch.zeros(B, T, C, device=x.device, dtype=x.dtype),
            torch.full((B, T, C), -1e38, device=x.device, dtype=x.dtype) 
        )

        kt = k
        vt = v
        ww = u + kt 
        p = torch.maximum(current_pp, ww) 
        e1 = torch.exp(current_pp - p) 
        e2 = torch.exp(ww - p) 
        wkv_t_step = (e1 * current_aa + e2 * vt) / (e1 * current_bb + e2) 
        output_wkv = wkv_t_step
        ww = current_pp - w 
        p = torch.maximum(ww, kt) 
        e1 = torch.exp(ww - p) 
        e2 = torch.exp(kt - p) 
        current_aa = e1 * current_aa + e2 * vt 
        current_bb = e1 * current_bb + e2      
        current_pp = p                         
            
        rwkv_out = r * output_wkv
        new_wkv_state = (current_aa, current_bb, current_pp)
        return self.output(rwkv_out), new_wkv_state
    

class RWKV_RTM(nn.Module):
    def __init__(self, dim=192, head=12):
        super().__init__()

        self.small_windows = 2

        self.topk = 3
        self.n_embd = dim

        reduction = 4
        self.proj_down_obj = nn.Linear(dim, dim//reduction)
        self.proj_down_ctx = nn.Linear(dim, dim//reduction)
        self.proj_up = nn.Linear(dim//reduction, dim)

        self.reduction = dim // reduction
        self.target_expert = Region_RWKV(dim=self.reduction, head=head)
        self.context_expert = Region_RWKV(dim=self.reduction, head=head)

        self.sr = Semantic_Router(topk=self.topk, win_size=self.small_windows)
        self.wr = Weighted_fusion_Router(d_model=dim)

    def window_partition(self, x, window_size):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C).contiguous()
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C).contiguous()
        return windows

    def window_reverse(self, windows, window_size, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1).contiguous()
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1).contiguous()
        return x

    def forward(self, xz, xs, state_t, state_c):
        B = xs.shape[0]

        target, context = self.select(xz, xs)

        target_out, new_state_t = self.target_expert(self.proj_down_obj(target), state_t)
        target_out = target_out.view(B, -1, target_out.shape[-1])
        len_tp = target_out.shape[1] 

        context_out, new_state_c = self.context_expert(self.proj_down_ctx(context), state_c)
        context_out = context_out.view(B, -1, context_out.shape[-1])

        tc = torch.cat((target_out, context_out), dim=1)

        tc = self.proj_up(tc)
        t_p = tc[:, :len_tp]
        xs = self.interactions(xs, tc)

        return t_p, xs, new_state_t, new_state_c


class Semantic_Router(nn.Module):
    def __init__(self, topk=3, win_size=2):
        super().__init__()
        self.topk = topk
        self.win_size = win_size
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def window_partition(self, x, window_size):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, window_size, window_size, C)
        windows = windows.flatten(2, 3)
        return windows

    def forward(self, z, x):
        B, N_t, C = z.shape
        h_t = int(math.sqrt(N_t))
        z = z.permute(0,2,1).reshape(B,C,h_t,h_t)
        z_max = self.avgpool(z).permute(0,2,3,1).reshape(B,1,C)

        B, N_s, C = x.shape
        h_s = int(math.sqrt(N_s)) 

        response_map = ((F.normalize(z_max,dim=-1,p=2) @ F.normalize(x,dim=-1,p=2).transpose(-2,-1))).permute(0,2,1).reshape(B,1,h_s,h_s)
        x = x.permute(0,2,1).reshape(B,C,h_s,h_s)
        x_windows = self.window_partition(x, self.win_size)  # b num_win n_win c

        windows = self.window_partition(response_map, self.win_size)  # b num_win n_win 1
        windows_mean = torch.mean(windows, dim=2, keepdim=True)  # b num_win 1 1
        indices = torch.sort(windows_mean, dim=1, descending=True)[1]

        index_windows = indices[:, :self.topk]
        non_index_windows = indices[:, self.topk:]

        index_windows = index_windows.expand(-1,-1,x_windows.size(2),x_windows.size(3))
        non_index_windows = non_index_windows.expand(-1,-1,x_windows.size(2),x_windows.size(3))

        x_selected = torch.gather(x_windows, dim=1, index=index_windows)
        x_no_selected = torch.gather(x_windows, dim=1, index=non_index_windows)

        x_selected = x_selected.view(-1, self.win_size * self.win_size, C)
        x_no_selected = x_no_selected.view(-1, self.win_size * self.win_size, C)

        return x_selected, x_no_selected
    

if __name__ == '__main__':
    ipt = torch.rand(32, 256, 192)
    model = RWKV_RTM(dim=192, head=12)
    spatial_state, channel_state = None, None
    opt, spatial_state, channel_state = model(ipt, spatial_state, channel_state)
    print(opt.shape)