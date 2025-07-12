import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.bool), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1) # b x ls x ls
    
    return subsequent_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head*d_k)
        self.w_k = nn.Linear(d_model, n_head*d_k)
        self.w_v = nn.Linear(d_model, n_head*d_v)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0/(d_model+d_v)))

        self.temperature = np.power(d_k, 0.5)
        self.attn_dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(n_head*d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, q, k, v, mask=None):
        # q: BS X T X D, k: BS X T X D, v: BS X T X D, mask: BS X T X T 
        bs, n_q, _ = q.shape
        bs, n_k, _ = k.shape
        bs, n_v, _ = v.shape

        assert n_k == n_v

        residual = q

        q = self.w_q(q).view(bs, n_q, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_q, self.d_k)
        k = self.w_k(k).view(bs, n_k, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, self.d_k)
        v = self.w_v(v).view(bs, n_v, self.n_head, self.d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, self.d_v)

        attn = torch.bmm(q, k.transpose(1, 2)) # (n_head*bs) X n_q X n_k
        attn = attn / self.temperature

        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1) # (n_head*bs) x n_q x n_k 
            attn = attn.masked_fill(mask, -np.inf)

        attn = F.softmax(attn, dim=2) # (n_head*bs) X n_q X n_k
        
        attn = self.attn_dropout(attn)
        output = torch.bmm(attn, v) # (n_head*bs) X n_q X d_v

        output = output.view(self.n_head, bs, n_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(bs, n_q, -1)
        # BS X n_q X (n_head*D)

        # output = self.fc(output) # BS X n_q X D
        output = self.dropout(self.fc(output)) # BS X n_q X D
        output = self.layer_norm(output + residual) # BS X n_q X D

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: BS X N X D
        residual = x
        output = x.transpose(1, 2) # BS X D X N
        output = self.w_2(F.relu(self.w_1(output))) # BS X D X N
        output = output.transpose(1, 2) # BS X N X D
        output = self.dropout(output)
        output = self.layer_norm(output + residual) # BS X N X D

        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model)

    def forward(self, decoder_input, self_attn_time_mask, self_attn_padding_mask):
        # decode_input: BS X T X D
        # time_mask: BS X T X T (padding postion are ones)
        # padding_mask: BS X T (padding position are zeros, diff usage from above)
        bs, dec_len, dec_hidden = decoder_input.shape
        
        decoder_out, dec_self_attn = self.self_attn(decoder_input, decoder_input, decoder_input, \
                                mask=self_attn_time_mask)
        # BS X T X D, BS X T X T
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()
        # BS X T X D

        decoder_out = self.pos_ffn(decoder_out) # BS X T X D
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()

        return decoder_out, dec_self_attn
        # BS X T X D, BS X T X T

class CrossHOIDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v):
        super(CrossHOIDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model)

    def forward(self, decoder_input_1, decoder_input_2, self_attn_padding_mask):
        # decoder_input_1/2: BS X T X D_o, BS X T X D_h, decoder_1: k, v; decoder_2: q. 
        # time_mask: BS X T X T (padding postion are ones)
        # padding_mask: BS X T (padding position are zeros, diff usage from above)
        
        bs, dec_len, _ = decoder_input_1.shape
        
        decoder_input_1 = decoder_input_1.reshape(bs*dec_len, -1, 1) # (BS*T) X D_1 X 1 
        decoder_input_2 = decoder_input_2.reshape(bs*dec_len, -1, 1) # (BS*T) X D_2 X 1 

        decoder_out, dec_self_attn = self.self_attn(decoder_input_2, decoder_input_1, decoder_input_1, mask=None)
        # (BS*T) X D_2 X 1, (BS*T) X D_2 X D_1

        self_attn_padding_mask = self_attn_padding_mask.reshape(bs*dec_len, 1) 

        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()
        # (BS*T) X D_2 X 1 

        decoder_out = self.pos_ffn(decoder_out) # (BS*T) X D_2 X 1 
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()
        # (BS*T) X D_2 X 1 

        decoder_out = decoder_out.reshape(bs, seq_len, -1) # BS X T X D_2 
        dec_self_attn = dec_self_attn.reshape(bs, seq_len, dec_self_attn.shape[1], dec_self_attn.shape[2])

        return decoder_out, dec_self_attn
        # BS X T X D_2, BS X T X D_2 X D_1 

class CrossHumanObjectDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v):
        super(CrossHumanObjectDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model)

    def forward(self, decoder_input_1, decoder_input_2, self_attn_time_mask, self_attn_padding_mask):
        # decoder_input_1/2: BS X T X D_o, BS X T X D_h
        # time_mask: BS X T X T (padding postion are ones)
        # padding_mask: BS X T (padding position are zeros, diff usage from above)
        
        # bs, dec_len, _ = decoder_input_1.shape
        
        decoder_out, dec_self_attn = self.self_attn(decoder_input_2, decoder_input_1, decoder_input_1, \
                                mask=self_attn_time_mask)
        # BS X T X D, BS X T X T
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()
        # BS X T X D

        decoder_out = self.pos_ffn(decoder_out) # BS X T X D
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()

        return decoder_out, dec_self_attn
        # BS X T X D, BS X T X T

class CrossDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v):
        super(CrossDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model)

    def forward(self, decoder_input, k_input, self_attn_time_mask, self_attn_padding_mask):
        # decode_input: BS X T X D
        # k_input: BS X K X D 
        # time_mask: BS X T X T (padding postion are ones)
        # padding_mask: BS X T (padding position are zeros, diff usage from above)
        bs, dec_len, dec_hidden = decoder_input.shape
        
        decoder_out, dec_self_attn = self.self_attn(decoder_input, k_input, k_input, \
                                mask=self_attn_time_mask)
        # BS X T X D, BS X T X T
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()
        # BS X T X D

        decoder_out = self.pos_ffn(decoder_out) # BS X T X D
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()

        return decoder_out, dec_self_attn
        # BS X T X D, BS X T X T


class Decoder(nn.Module):
    def __init__(
            self,
            d_feats, d_model,
            n_layers, n_head, d_k, d_v, max_timesteps, use_full_attention=False):
        super(Decoder, self).__init__()

        self.start_conv = nn.Conv1d(d_feats, d_model, 1) # (input: 17*3)
        self.position_vec = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_timesteps+1, d_model, padding_idx=0),
            freeze=True)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, n_head, d_k, d_v)
            for _ in range(n_layers)])

        self.use_full_attention = use_full_attention 

    def forward(self, decoder_input, padding_mask, decoder_pos_vec, obj_embedding=None):
        # decoder_input: BS X D X T 
        # padding_mask: BS X 1 X T
        # decoder_pos_vec: BS X 1 X T
        # obj_embedding: BS X 1 X D

        dec_self_attn_list = []

        padding_mask = padding_mask.squeeze(1) # BS X T
        decoder_pos_vec = decoder_pos_vec.squeeze(1) # BS X T

        input_embedding = self.start_conv(decoder_input)  # BS X D X T
        input_embedding = input_embedding.transpose(1, 2) # BS X T X D
        if obj_embedding is not None:
            new_input_embedding = torch.cat((obj_embedding, input_embedding), dim=1) # BS X (T+1) X D 
        else:
            new_input_embedding = input_embedding

        # self.position_vec = self.position_vec.cuda()
        pos_embedding = self.position_vec(decoder_pos_vec) # BS X T X D
        
        # Time mask is same for all blocks, while padding mask differ according to the position of block
        if self.use_full_attention:
            time_mask = None
        else:
            time_mask = get_subsequent_mask(decoder_pos_vec) 
        # BS X T X T (Prev steps are 0, later 1)
       
        dec_output = new_input_embedding + pos_embedding # BS X T X D
        for dec_layer in self.layer_stack:
            dec_output, dec_self_attn = dec_layer(
                dec_output, # BS X T X D
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T

            dec_self_attn_list += [dec_self_attn]

        return dec_output, dec_self_attn_list
        # BS X T X D, list


class CrossDecoder(nn.Module):
    def __init__(
            self,
            d_feats, d_input_key, d_model,
            n_layers, n_head, d_k, d_v, max_timesteps, use_full_attention=False):
        super(CrossDecoder, self).__init__()

        self.start_conv = nn.Conv1d(d_feats, d_model, 1) # (input: 17*3)
        self.input_k_conv = nn.Conv1d(d_input_key, d_model, 1) # (input: 17*3)
        self.position_vec = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_timesteps+1, d_model, padding_idx=0),
            freeze=True)
        self.layer_stack = nn.ModuleList([CrossDecoderLayer(d_model, n_head, d_k, d_v)
            for _ in range(n_layers)])

        self.use_full_attention = use_full_attention 

    def forward(self, decoder_input, k_input, padding_mask, decoder_pos_vec, k_pos_vec):
        # decoder_input: BS X D X T 
        # k_input: BS X D X K 
        # padding_mask: BS X 1 X T
        # decoder_pos_vec: BS X 1 X T

        dec_self_attn_list = []

        padding_mask = padding_mask.squeeze(1) # BS X T
        decoder_pos_vec = decoder_pos_vec.squeeze(1) # BS X T
        k_pos_vec = k_pos_vec.squeeze(1) # BS X T 

        input_embedding = self.start_conv(decoder_input)  # BS X D X T
        input_embedding = input_embedding.transpose(1, 2) # BS X T X D
        # self.position_vec = self.position_vec.cuda()
        pos_embedding = self.position_vec(decoder_pos_vec) # BS X T X D

        k_input = self.input_k_conv(k_input) # BS X D X K 
        k_input = k_input.transpose(1, 2) # BS X K X D 
        k_pos_embedding = self.position_vec(k_pos_vec) # BS X T X D
        k_input = k_input + k_pos_embedding

        # Time mask is same for all blocks, while padding mask differ according to the position of block
        if self.use_full_attention:
            time_mask = None
        else:
            time_mask = get_subsequent_mask(decoder_pos_vec) 
        # BS X T X T (Prev steps are 0, later 1)
        dec_output = input_embedding + pos_embedding # BS X T X D
 
        for dec_layer in self.layer_stack:
            dec_output, dec_self_attn = dec_layer(
                dec_output, # BS X T X D
                k_input,
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T

            dec_self_attn_list += [dec_self_attn]

        return dec_output, dec_self_attn_list
        # BS X T X D, list


class PerStepDecoder(nn.Module):
    def __init__(
            self,
            d_feats, d_model,
            n_layers, n_head, d_k, d_v, 
            max_timesteps, context_window_size):
        super(PerStepDecoder, self).__init__()

        self.start_conv = nn.Conv1d(d_feats, d_model, 1) # (input: 17*3)
        self.position_vec = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_timesteps+1, d_model, padding_idx=0),
            freeze=True)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, n_head, d_k, d_v)
            for _ in range(n_layers)])

        self.decoder_input = None # BS X D X T 

        self.max_timesteps = max_timesteps 

        self.context_window_size = context_window_size 

    def forward(self, curr_step_input, mode="train", use_full_attention=False):
        # curr_step_input: BS X D 

        # Update decoder_input, padding_mask, decoder_pos_vec 
        if self.decoder_input is None:
            self.decoder_input = curr_step_input[:, :, None] # BS X D X T(1)
        else:
            if curr_step_input is not None:
                self.decoder_input = torch.cat((self.decoder_input, curr_step_input[:, :, None]), dim=-1) # B X D X T  
                if mode == "test" and self.decoder_input.shape[-1] > self.context_window_size:
                    self.decoder_input = self.decoder_input[:, :, -self.context_window_size:] 
            else:
                self.decoder_input = self.decoder_input.detach() # B X D X T 

        batch_size, _, seq_len = self.decoder_input.shape 

        # In training, no need for masking 
        padding_mask = torch.ones(batch_size, seq_len).to(self.decoder_input.device).bool() # BS X timesteps
        # Get position vec for position-wise embedding
        pos_vec = torch.arange(seq_len)+1 # timesteps
        pos_vec = pos_vec[None, :].to(self.decoder_input.device).repeat(batch_size, 1) # BS X timesteps

        dec_self_attn_list = []

        input_embedding = self.start_conv(self.decoder_input)  # BS X D X T
        input_embedding = input_embedding.transpose(1, 2) # BS X T X D
        # self.position_vec = self.position_vec.cuda()
        pos_embedding = self.position_vec(pos_vec) # BS X T X D
        
        # Time mask is same for all blocks, while padding mask differ according to the position of block
        if use_full_attention:
            time_mask = None 
        else:
            time_mask = get_subsequent_mask(pos_vec) 
        # BS X T X T (Prev steps are 0, later 1)
        
        dec_output = input_embedding + pos_embedding # BS X T X D
        for dec_layer in self.layer_stack:
            dec_output, dec_self_attn = dec_layer(
                dec_output, # BS X T X D
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T

            dec_self_attn_list += [dec_self_attn]

        if curr_step_input is None:
            return dec_output # B X T X D 
        else:
            return dec_output[:, -1, :]
        # return dec_output[:, -1, :], dec_self_attn_list
        # BS X D, list
   
class InteractionDecoder(nn.Module):
    def __init__(
            self,
            d_feats, d_model,
            n_layers, n_head, d_k, d_v, max_timesteps, use_full_attention=False):
        super(InteractionDecoder, self).__init__()

        self.start_conv_obj = nn.Conv1d(d_feats, d_model, 1) # (input: 17*3)
        self.start_conv_human = nn.Conv1d(d_feats, d_model, 1) # (input: 17*3)
        self.position_vec = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_timesteps+1, d_model, padding_idx=0),
            freeze=True)
        self.layer_stack_obj = nn.ModuleList([DecoderLayer(d_model, n_head, d_k, d_v)
            for _ in range(n_layers)])
        self.layer_stack_human = nn.ModuleList([DecoderLayer(d_model, n_head, d_k, d_v)
            for _ in range(n_layers)])
        self.layer_stack_cross_obj = nn.ModuleList([CrossHumanObjectDecoderLayer(d_model, n_head, d_k, d_v)
            for _ in range(n_layers)])
        self.layer_stack_cross_human = nn.ModuleList([CrossHumanObjectDecoderLayer(d_model, n_head, d_k, d_v)
            for _ in range(n_layers)])

        self.use_full_attention = use_full_attention 

    def forward(self, decoder_input, padding_mask, decoder_pos_vec, obj_embedding=None):
        # decoder_input: BS X D(3+9+24*3+22*6+D_cond) X T, D_cond: 256 + (3+9+24*3+22*6)  
        # decoder_input_obj: BS X D(3+9) X T 
        # decoder_input_human: BS X D(24*3+22*6) X T 
        # padding_mask: BS X 1 X T
        # decoder_pos_vec: BS X 1 X T
        # obj_embedding(actually noise embedding): BS X 1 X D

        # Prepare object motion input
        input_embedding_obj = self.start_conv_obj(decoder_input)  # BS X D_o X T
        input_embedding_obj = input_embedding_obj.transpose(1, 2) # BS X T X D_o
        if obj_embedding is not None:
            new_input_embedding_obj = torch.cat((obj_embedding, input_embedding_obj), dim=1) # BS X (T+1) X D_o 
        else:
            new_input_embedding_obj = input_embedding_obj

        # Prepare human motion input
        input_embedding_human = self.start_conv_human(decoder_input)  # BS X D_h X T
        input_embedding_human = input_embedding_human.transpose(1, 2) # BS X T X D_h
        if obj_embedding is not None:
            new_input_embedding_human = torch.cat((obj_embedding, input_embedding_human), dim=1) # BS X (T+1) X D_o 
        else:
            new_input_embedding_human = input_embedding_obj

        # Prepare temporal position embedding 
        padding_mask = padding_mask.squeeze(1) # BS X T
        decoder_pos_vec = decoder_pos_vec.squeeze(1) # BS X T
        pos_embedding = self.position_vec(decoder_pos_vec) # BS X T X D

        dec_self_attn_obj_list = []
        dec_self_attn_human_list = []
        dec_self_attn_obj_cross_list = []
        dec_self_attn_human_cross_list = []
        
        # Time mask is same for all blocks, while padding mask differ according to the position of block
        if self.use_full_attention:
            time_mask = None
        else:
            time_mask = get_subsequent_mask(decoder_pos_vec) 
        # BS X T X T (Prev steps are 0, later 1)
       
        dec_output_obj = new_input_embedding_obj + pos_embedding # BS X T X D_o 
        dec_output_human = new_input_embedding_human + pos_embedding # BS X T X D_h
        for idx in range(len(self.layer_stack_obj)):
            # Self-attention for object motion
            tmp_dec_output_obj, dec_self_attn_obj = self.layer_stack_obj[idx](
                dec_output_obj, # BS X T X D
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T

            # Self-attention for human motion 
            tmp_dec_output_human, dec_self_attn_human = self.layer_stack_human[idx](
                dec_output_human, # BS X T X D
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T

            dec_output_obj, dec_cross_attn_obj = self.layer_stack_cross_obj[idx](
                tmp_dec_output_human, 
                tmp_dec_output_obj, 
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T

            # Add cross attention so object motion and human motion can be related 
            dec_output_human, dec_cross_attn_human = self.layer_stack_cross_human[idx](
                tmp_dec_output_obj, 
                tmp_dec_output_human, 
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T

            dec_self_attn_obj_list += [dec_self_attn_obj]
            dec_self_attn_human_list += [dec_self_attn_human] 
            dec_self_attn_obj_cross_list += [dec_cross_attn_obj] 
            dec_self_attn_human_cross_list += [dec_cross_attn_human] 

        return dec_output_obj, dec_output_human 
        # BS X T X D, list

class InteractionSharedDecoder(nn.Module):
    def __init__(
            self,
            d_feats, d_model,
            n_layers, n_head, d_k, d_v, max_timesteps, use_full_attention=False):
        super(InteractionDecoder, self).__init__()

        d_feats_obj = 3 + 9 
        d_feats_human = 24 * 3 + 22 * 6 
        self.start_conv_obj = nn.Conv1d(d_feats_obj, d_model, 1) # (input: 17*3)
        self.start_conv_human = nn.Conv1d(d_feats_human, d_model, 1) # (input: 17*3)
        self.position_vec = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_timesteps+1, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack_self = nn.ModuleList([DecoderLayer(d_model, n_head, d_k, d_v)
            for _ in range(n_layers)])
        self.layer_stack_cross = nn.ModuleList([CrossHOIDecoderLayer(d_model, n_head, d_k, d_v)
            for _ in range(n_layers)])

        self.use_full_attention = use_full_attention 

    def forward(self, decoder_input, padding_mask, decoder_pos_vec, noise_embedding=None):
        # decoder_input: BS X D(3+9+24*3+22*6) X T
        # decoder_input_obj: BS X D(3+9) X T
        # decoder_input_human: BS X D(24*3+22*6) X T
        # padding_mask: BS X 1 X T
        # decoder_pos_vec: BS X 1 X T
        # noise_embedding(actually noise embedding): BS X 1 X D

        decoder_input_obj = decoder_input[:, :12, :]
        decoder_input_human = decoder_input[:, 12:, :]

        # Prepare object motion input
        input_embedding_obj = self.start_conv_obj(decoder_input_obj)  # BS X D_o X T
        input_embedding_obj = input_embedding_obj.transpose(1, 2) # BS X T X D_o
        if noise_embedding is not None:
            new_input_embedding_obj = torch.cat((noise_embedding, input_embedding_obj), dim=1) # BS X (T+1) X D_o 
        else:
            new_input_embedding_obj = input_embedding_obj

        # Prepare human motion input
        input_embedding_human = self.start_conv_human(decoder_input_human)  # BS X D_h X T
        input_embedding_human = input_embedding_human.transpose(1, 2) # BS X T X D_h
        if noise_embedding is not None:
            new_input_embedding_human = torch.cat((noise_embedding, input_embedding_human), dim=1) # BS X (T+1) X D_o 
        else:
            new_input_embedding_human = input_embedding_obj

        # Prepare temporal position embedding 
        padding_mask = padding_mask.squeeze(1) # BS X T
        decoder_pos_vec = decoder_pos_vec.squeeze(1) # BS X T
        pos_embedding = self.position_vec(decoder_pos_vec) # BS X T X D

        # Prepare positional embedding for object and human motion dimension 
        cross_pos_vec_obj = torch.arange(num_steps)+1 # timesteps
        cross_pos_vec_obj = cross_pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1) # BS X 1 X timesteps
        cross_pos_embedding_obj = self.position_vec(cross_pos_vec_obj) # BS X T X D

        # Time mask is same for all blocks, while padding mask differ according to the position of block
        if self.use_full_attention:
            time_mask = None
        else:
            time_mask = get_subsequent_mask(decoder_pos_vec) 
        # BS X T X T (Prev steps are 0, later 1)

        # Need to do temporal self-attention, also cross-attention betwen human and object dimension. 
        dec_self_attn_obj_list = []
        dec_self_attn_human_list = []
        dec_self_attn_obj_cross_list = []
        dec_self_attn_human_cross_list = []

        dec_output_obj = new_input_embedding_obj + pos_embedding # BS X T X D 
        dec_output_human = new_input_embedding_human + pos_embedding # BS X T X D
        for idx in range(len(self.layer_stack_self)):
            # Self-attention for object motion
            dec_output_obj, dec_self_attn_obj = self.layer_stack_self[idx](
                dec_output_obj, # BS X T X D
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T
            # BS X T X D

            # Self-attention for human motion
            dec_output_human, dec_self_attn_human = self.layer_stack_self[idx](
                dec_output_human, # BS X T X D
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T
            # BS X T X D 

            # Add cross attention so object motion and human motion can be related 
            dec_output_human, dec_cross_attn_human = self.layer_stack_cross[idx](
                dec_output_obj, # k, v
                dec_output_human, # q
                self_attn_padding_mask=padding_mask) # BS X T
            # BS X T X D_human, BS X T X D_human X D_obj 

            # Add cross attention so object motion and human motion can be related 
            dec_output_obj, dec_cross_attn_obj = self.layer_stack_cross[idx](
                dec_output_human, 
                dec_output_obj,  
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T
            # BS X T X D_obj, BS X T X D_obj X D_human   

            dec_self_attn_obj_list += [dec_self_attn_obj]
            dec_self_attn_human_list += [dec_self_attn_human] 
            dec_self_attn_obj_cross_list += [dec_cross_attn_obj] 
            dec_self_attn_human_cross_list += [dec_cross_attn_human] 

        return dec_output_obj, dec_output_human 
        # BS X T X D, list