import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from models.context_encoders import MapEncoderPtsMA


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: must be (T, B, H)
        :return:
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OutputModel(nn.Module):
    '''
    This class operates on the output of AutoBot-Joint's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution and possibly predicts the yaw.
    '''
    def __init__(self, d_k=64, predict_yaw=False):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        self.predict_yaw = predict_yaw
        out_len = 5
        if predict_yaw:
            out_len = 6

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.observation_model = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k)), nn.ReLU(),
            init_(nn.Linear(self.d_k, self.d_k)), nn.ReLU(),
            init_(nn.Linear(self.d_k, out_len))
        )
        self.min_stdev = 0.01

    def forward(self, agent_latent_state):
        T = agent_latent_state.shape[0]
        BK = agent_latent_state.shape[1]
        pred_obs = self.observation_model(agent_latent_state.reshape(-1, self.d_k)).reshape(T, BK, -1)
        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        rho = torch.tanh(pred_obs[:, :, 4]) * 0.9  # for stability
        if self.predict_yaw:
            yaws = pred_obs[:, :, 5]  # for stability
            return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho, yaws], dim=2)
        else:
            return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=2)

class AutoBot(nn.Module):
    def __init__(self, config):
        super(AutoBot, self).__init__()
        self.config = config
        self.encoder = AutoBotEncoder(self.config)
        decoder_net = getattr(sys.modules[__name__], config.decoder)
        self.decoder = decoder_net(self.config)
        self.train()
    
    def forward(self, ego_in, agents_in, roads, agent_types):
        feat_x_encoded = self.encoder(ego_in, agents_in, roads, agent_types) # B * 64
        out_dists, mode_probs = self.decoder(feat_x_encoded)
        return (out_dists, mode_probs)


class AutoBotEncoder(nn.Module):
    def __init__(self, config):
        super(AutoBotEncoder, self).__init__()

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.config = config
        self.k_attr = self.config.k_attr
        self.map_attr = self.config.map_attr
        self.d_k = self.config.hidden_size
        self._M = self.config.num_other_agents  # num agents other then the main agent.
        self.c = self.config.num_modes
        self.T = self.config.pred_horizon
        self.L_enc = self.config.num_encoder_layers
        self.dropout = self.config.dropout
        self.num_heads = self.config.tx_num_heads
        self.L_dec = self.config.num_decoder_layers
        self.tx_hidden_size = self.config.tx_hidden_size
        self.use_map_lanes = self.config.use_map_lanes
        self.predict_yaw = self.config.predict_yaw
        
        # added later
        self.num_agent_types = self.config.num_agent_types

        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))

        # ============================== Positional encoder ==============================
        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0, max_len=self.config.pos_emb_max_len)

        # ============================== AutoBot-Joint ENCODER ==============================
        self.social_attn_layers = []
        self.temporal_attn_layers = []
        if not self.config.full_attention:
            for _ in range(self.L_enc):
                tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                            dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
                self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=2))

                tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                            dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
                self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

            self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
            self.social_attn_layers = nn.ModuleList(self.social_attn_layers)
        else:
            # ============================== new try ==============================
            self.uni_encoder_attn_layers = []
            for _ in range(self.L_enc):
                uni_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                            dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
                self.uni_encoder_attn_layers.append(nn.TransformerEncoder(uni_encoder_layer, num_layers=2))
            self.uni_encoder_attn_layers = nn.ModuleList(self.uni_encoder_attn_layers)

        # ============================== MAP ENCODER ==========================
        if self.use_map_lanes:
            self.map_encoder = MapEncoderPtsMA(d_k=self.d_k, map_attr=self.map_attr, dropout=self.dropout)

        # ============================== AGENT TYPES Encoders ==============================
        # if num_agent_types is 0, mean that the dataset doesn't provide types info, then skip it.
        if self.num_agent_types != 0:

            self.emb_agent_types = nn.Sequential(init_(nn.Linear(self.num_agent_types, self.d_k)))
        
        
        self.train()
    
    def forward(self, ego_in, agents_in, roads, agent_types):
        '''
        :param ego_in: one agent called ego, shape [B, T_obs, k_attr+1] with last values being the existence mask.
        :param agents_in: other scene agents, shape [B, T_obs, M-1, k_attr+1] with last values being the existence mask.
        :param roads: [B, M, S, P, map_attr+1] representing the road network or
                      [B, 1, 1] if self.use_map_lanes is False.
        :param agent_types: [B, M, num_agent_types] one-hot encoding of agent types, with the first agent idx being ego.
        '''
        B = ego_in.size(0)

        # Encode all input observations
        ego_tensor, _agents_tensor, opps_masks, env_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)
        agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)

        if not self.config.full_attention:
            # Process through AutoBot's encoder
            for i in range(self.L_enc):
                agents_emb = self.temporal_attn_encoder_fn(agents_emb, opps_masks, layer=self.temporal_attn_layers[i])
                agents_emb = self.social_attn_encoder_fn(agents_emb, opps_masks, layer=self.social_attn_layers[i])
        else:
            # ============================== new try ==============================
            for i in range(self.L_enc):
                agents_emb = self.uni_attn_encoder_fn(agents_emb, opps_masks, layer=self.uni_encoder_attn_layers[i])


        # Process map information
        if self.use_map_lanes:
            orig_map_features, orig_road_segs_masks = self.map_encoder(roads, agents_emb)
            map_features = orig_map_features.unsqueeze(2).repeat(1, 1, self.c, 1, 1).view(-1, B * self.c * (self._M+1), self.d_k)
            road_segs_masks = orig_road_segs_masks.unsqueeze(2).repeat(1, self.c, 1, 1).view(B * self.c * (self._M+1), -1)

        # Repeat the tensors for the number of modes.
        opps_masks_modes = opps_masks.unsqueeze(1).repeat(1, self.c, 1, 1).view(B*self.c, ego_in.shape[1], -1)
        context = agents_emb.unsqueeze(2).repeat(1, 1, self.c, 1, 1)
        context = context.view(ego_in.shape[1], B*self.c, self._M+1, self.d_k)

        # embed agent types
        if self.num_agent_types != 0:
            agent_types_features = self.emb_agent_types(agent_types).unsqueeze(1).\
                repeat(1, self.c, 1, 1).view(-1, self._M+1, self.d_k)
            agent_types_features = agent_types_features.unsqueeze(0).repeat(self.T, 1, 1, 1)
        
            latent = [agent_types_features, map_features, road_segs_masks, context, opps_masks_modes, agents_emb, orig_map_features, orig_road_segs_masks]

        else:
            latent = [map_features, road_segs_masks, context, opps_masks_modes, agents_emb, orig_map_features, orig_road_segs_masks]
        return latent

    def process_observations(self, ego, agents):
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks = ego[:, :, -1]

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        # temp_masks = torch.cat((env_masks.unsqueeze(-1), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).type(torch.BoolTensor).to(agents.device)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks
    
    def temporal_attn_encoder_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        agent_masks[:, -1][agent_masks.sum(-1) == T_obs] = False  # Ensure agent's that don't exist don't throw NaNs.
        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_obs, B * (self._M + 1), -1)),
                                src_key_padding_mask=agent_masks)
        return agents_temp_emb.view(T_obs, B, self._M+1, -1)

    def uni_attn_encoder_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs = agents_emb.size(0)
        num_agents = self._M + 1
        B = agent_masks.size(0)
        H = agents_emb.size(3)
        

        agent_masks = agent_masks.permute(0, 2, 1) # [B, N, T]
        agent_masks[:,:,-1][agent_masks.sum(-1) == T_obs] = False  # Ensure agent's that don't exist don't throw NaNs.
        agent_masks = agent_masks.reshape(-1, num_agents * T_obs)
        agents_temp_emb = layer(self.pos_encoder(agents_emb.permute(2,0,1,3).reshape(-1, B, H)),
                                src_key_padding_mask=agent_masks) # (NT, B, H)
        return agents_temp_emb.view(num_agents, T_obs, B, -1).permute(1,2,0,3) # (T, B, N, H)

    def social_attn_encoder_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(self._M + 1, B * T_obs, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, self._M+1))
        agents_soc_emb = agents_soc_emb.view(self._M+1, B, T_obs, -1).permute(2, 1, 0, 3)
        return agents_soc_emb

class AutoBotDecoder(nn.Module):
    def __init__(self, config):
        
        super(AutoBotDecoder, self).__init__()

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        self.config = config
        self.k_attr = self.config.k_attr
        self.map_attr = self.config.map_attr
        self.d_k = self.config.hidden_size
        self._M = self.config.num_other_agents  # num agents other then the main agent.
        self.c = self.config.num_modes
        self.T = self.config.pred_horizon
        self.L_enc = self.config.num_encoder_layers
        self.dropout = self.config.dropout
        self.num_heads = self.config.tx_num_heads
        self.L_dec = self.config.num_decoder_layers
        self.tx_hidden_size = self.config.tx_hidden_size
        self.use_map_lanes = self.config.use_map_lanes
        self.predict_yaw = self.config.predict_yaw
        
        # added later
        self.num_agent_types = self.config.num_agent_types
        
        # ============================== MAP ENCODER ===================================
        if self.use_map_lanes:
            self.map_attn_layers = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
        
        # ============================== AutoBot-Joint DECODER ==============================
        if self.num_agent_types != 0:
            self.dec_agenttypes_encoder = nn.Sequential(
                init_(nn.Linear(2 * self.d_k, self.d_k)), nn.ReLU(),
                init_(nn.Linear(self.d_k, self.d_k))
            )


        self.Q = nn.Parameter(torch.Tensor(self.T, 1, self.c, 1, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)

        if not self.config.full_attention:
            self.social_attn_decoder_layers = []
            self.temporal_attn_decoder_layers = []
            for _ in range(self.L_dec):
                tx_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                            dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
                self.temporal_attn_decoder_layers.append(nn.TransformerDecoder(tx_decoder_layer, num_layers=2))
                tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                            dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
                self.social_attn_decoder_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

            self.temporal_attn_decoder_layers = nn.ModuleList(self.temporal_attn_decoder_layers)
            self.social_attn_decoder_layers = nn.ModuleList(self.social_attn_decoder_layers)
        else:
            # ============================== new try ==============================
            self.uni_attn_decoder_layers = []
            for _ in range(self.L_dec):
                uni_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                            dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
                self.uni_attn_decoder_layers.append(nn.TransformerDecoder(uni_decoder_layer, num_layers=2))
            self.uni_attn_decoder_layers = nn.ModuleList(self.uni_attn_decoder_layers)

        # ============================== OUTPUT MODEL ==============================
        self.output_model = OutputModel(d_k=self.d_k, predict_yaw=self.predict_yaw)

        # ============================== Mode Prob prediction (P(z|X_1:t)) ==============================
        self.P = nn.Parameter(torch.Tensor(self.c, 1, 1, self.d_k), requires_grad=True)  # Appendix C.2.
        nn.init.xavier_uniform_(self.P)

        if self.use_map_lanes:
            self.mode_map_attn = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)

        self.prob_decoder = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
        self.prob_predictor = init_(nn.Linear(self.d_k, 1))

        self.train()

    def forward(self, latent):
        '''
        :return:
            pred_obs: shape [c, T, B, M, 5(6)] c trajectories for all agents with every point being the params of
                                        Bivariate Gaussian distribution (and the yaw prediction if self.predict_yaw).
            mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        '''
        if self.num_agent_types != 0:
            agent_types_features, map_features, road_segs_masks, context, opps_masks_modes, agents_emb, orig_map_features, orig_road_segs_masks = latent
        else:
            map_features, road_segs_masks, context, opps_masks_modes, agents_emb, orig_map_features, orig_road_segs_masks = latent
        
        B = self.config.batch_size

        # AutoBot-Joint Decoding
        dec_parameters = self.Q.repeat(1, B, 1, self._M+1, 1).view(self.T, B*self.c, self._M+1, -1)
        if self.num_agent_types != 0:
            dec_parameters = torch.cat((dec_parameters, agent_types_features), dim=-1)
            dec_parameters = self.dec_agenttypes_encoder(dec_parameters)
        agents_dec_emb = dec_parameters

        for d in range(self.L_dec):
            if self.use_map_lanes and d == 1:
                agents_dec_emb = agents_dec_emb.reshape(self.T, -1, self.d_k)
                agents_dec_emb_map = self.map_attn_layers(query=agents_dec_emb, key=map_features, value=map_features,
                                                          key_padding_mask=road_segs_masks)[0]
                agents_dec_emb = agents_dec_emb + agents_dec_emb_map
                agents_dec_emb = agents_dec_emb.reshape(self.T, B*self.c, self._M+1, -1)

            if not self.config.full_attention:
                agents_dec_emb = self.temporal_attn_decoder_fn(agents_dec_emb, context, opps_masks_modes, layer=self.temporal_attn_decoder_layers[d])
                agents_dec_emb = self.social_attn_decoder_fn(agents_dec_emb, opps_masks_modes, layer=self.social_attn_decoder_layers[d])
            else:
                # ============================== new try ==============================
                agents_dec_emb = self.uni_attn_decoder_fn(agents_dec_emb, context, opps_masks_modes, layer=self.uni_attn_decoder_layers[d])

        out_dists = self.output_model(agents_dec_emb.reshape(self.T, -1, self.d_k))
        out_dists = out_dists.reshape(self.T, B, self.c, self._M+1, -1).permute(2, 0, 1, 3, 4)

        # Mode prediction
        mode_params_emb = self.P.repeat(1, B, self._M+1, 1).view(self.c, -1, self.d_k)
        mode_params_emb = self.prob_decoder(query=mode_params_emb, key=agents_emb.reshape(-1, B*(self._M+1), self.d_k),
                                            value=agents_emb.reshape(-1, B*(self._M+1), self.d_k))[0]
        if self.use_map_lanes:
            orig_map_features = orig_map_features.view(-1, B*(self._M+1), self.d_k)
            orig_road_segs_masks = orig_road_segs_masks.view(B*(self._M+1), -1)
            mode_params_emb = self.mode_map_attn(query=mode_params_emb, key=orig_map_features, value=orig_map_features,
                                                 key_padding_mask=orig_road_segs_masks)[0] + mode_params_emb

        mode_probs = self.prob_predictor(mode_params_emb).squeeze(-1).view(self.c, B, self._M+1).sum(2).transpose(0, 1)
        mode_probs = F.softmax(mode_probs, dim=1)

        # return  # [c, T, B, M, 5], [B, c]
        return out_dists, mode_probs

    def temporal_attn_decoder_fn(self, agents_emb, context, agent_masks, layer):
        '''
        :param agents_emb: (T, BK, N, H)
        :param context: (T_in, BK, N, H)
        :param agent_masks: (BK, T, N)
        :return: (T, BK, N, H)
        '''
        T_obs = context.size(0)
        BK = agent_masks.size(0)
        time_masks = self.generate_decoder_mask(seq_len=self.T, device=agents_emb.device)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        agent_masks[:, -1][agent_masks.sum(-1) == T_obs] = False  # Ensure that agent's that don't exist don't make NaN.
        agents_emb = agents_emb.reshape(self.T, -1, self.d_k)  # [T, BxKxN, H]
        context = context.view(-1, BK*(self._M+1), self.d_k)

        agents_temp_emb = layer(agents_emb, context, tgt_mask=time_masks, memory_key_padding_mask=agent_masks)
        agents_temp_emb = agents_temp_emb.view(self.T, BK, self._M+1, -1)

        return agents_temp_emb

    def uni_attn_decoder_fn(self, agents_emb, context, agent_masks, layer):
        '''
        :param agents_emb: (T, BK, N, H)
        :param context: (T_in, BK, N, H)
        :param agent_masks: (BK, T_in, N)
        :return: (T, BK, N, H)
        '''
        T_obs = context.size(0)
        BK = agent_masks.size(0)
        num_agents = self._M+1
        H = context.size(3)
        time_masks = self.generate_decoder_mask(seq_len=self.T, device=agents_emb.device) # [T,T]
        time_masks = time_masks.repeat(num_agents, num_agents) # [NT, NT]
        
        agent_masks = agent_masks.permute(0, 2, 1) # [BK, N, T_in]
        agent_masks[:,:,-1][agent_masks.sum(-1) == T_obs] = False  # Ensure agent's that don't exist don't throw NaNs.
        agent_masks = agent_masks.reshape(-1, num_agents * T_obs)           #[BK, NT_in]

        agents_emb = agents_emb.permute(2,0,1,3).reshape(num_agents * self.T, -1, self.d_k)     # [NT, BxK, H]
        context = context.permute(2,0,1,3).reshape(-1, BK, self.d_k)                               # [NT_in, BxK, H]

        agents_temp_emb = layer(agents_emb, context, tgt_mask=time_masks, memory_key_padding_mask=agent_masks) # [NT, BxK, H]
        agents_temp_emb = agents_temp_emb.view(num_agents, self.T, BK, -1).permute(1,2,0,3)  # (T, BK, N, H)

        return agents_temp_emb

    def social_attn_decoder_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, BK, N, H)
        :param agent_masks: (BK, T, N)
        :return: (T, BK, N, H)
        '''
        B = agent_masks.size(0)
        agent_masks = agent_masks[:, -1:].repeat(1, self.T, 1).view(-1, self._M + 1)  # take last timestep of all agents.
        agent_masks[:, -1][agent_masks.sum(-1) == self._M+1] = False  # Ensure the time don't exist valid agents don't throw NaNs.
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(self._M + 1, B * self.T, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks)
        agents_soc_emb = agents_soc_emb.view(self._M + 1, B, self.T, -1).permute(2, 1, 0, 3)
        return agents_soc_emb

    def generate_decoder_mask(self, seq_len, device):
        ''' For masking out the subsequent info. '''
        subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
        return subsequent_mask
