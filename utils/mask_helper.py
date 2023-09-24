import torch


def conditional_padding(ego_in, ego_out, agents_in, agents_out):
    time_dim = 1
    observation_time = ego_in.shape[time_dim]

    # ego_in, ego_out, agents_in, agents_out
    ego_cat = torch.cat((ego_in, ego_out) , dim=time_dim) # cat in the time dimension
    agent_cat = torch.cat((agents_in, agents_out), dim=time_dim)

    agents_out = agent_cat.clone()
    agent_cat[:,observation_time:,:,:] = 0
    agents_in = agent_cat

    gt_agents = torch.cat((ego_cat.unsqueeze(2), agent_cat), dim=2)
    mask = torch.rand((gt_agents[:,:, :,-1].shape)).to(gt_agents.device)
    mask[:, observation_time:, 1:] = 0 # other agent's future time is masked
    return ego_cat, ego_cat, agents_in, agents_out, mask.type('torch.BoolTensor').to(gt_agents.device)

def get_random_masked(ego_in, ego_out, agents_in, agents_out, config):
    if config.mask_strategy == "pointwise":
        return get_random_masked_pointwise(ego_in, ego_out, agents_in, agents_out, config.mask_ratio)
    elif config.mask_strategy == "patchwise":
        return get_random_masked_patchwise(ego_in, ego_out, agents_in, agents_out, config.mask_ratio, config.min_patch_size, config.max_patch_size)
    elif config.mask_strategy == "time-only":
        return get_random_masked_time_only(ego_in, ego_out, agents_in, agents_out, config.mask_ratio)
    else:
        raise NotImplementedError

        
def get_random_masked_pointwise(ego_in, ego_out, agents_in, agents_out, mask_percentage):
    time_dim = 1
    # ego_in, ego_out, agents_in, agents_out
    ego_data = torch.cat((ego_in, ego_out) , dim=time_dim) # cat in the time dimension
    agents_data = torch.cat((agents_in, agents_out), dim=time_dim)
    gt_agents = torch.cat((ego_data.unsqueeze(2), agents_data), dim=2)
    mask = torch.rand((gt_agents[:,:, :,-1].shape)).to(gt_agents.device)
    mask = (mask > mask_percentage).unsqueeze(-1) # [B, T, M, 1]

    gt_masked = gt_agents * mask
    ego_masked = gt_masked[:,:,0,:]
    agents_masked = gt_masked[:,:,1:,:]
    return ego_masked, ego_data, agents_masked, agents_data, mask

def get_random_masked_patchwise(ego_in, ego_out, agents_in, agents_out, mask_percentage, min_patch_size, max_patch_size):
    time_dim = 1
    # ego_in, ego_out, agents_in, agents_out
    ego_data = torch.cat((ego_in, ego_out) , dim=time_dim) # cat in the time dimension
    agents_data = torch.cat((agents_in, agents_out), dim=time_dim)
    gt_agents = torch.cat((ego_data.unsqueeze(2), agents_data), dim=2)

    # get patch indices
    batch_size, seq_len, num_agents, feature_dim = gt_agents.shape
    
    patch_size = torch.randint(low=min_patch_size, high=max_patch_size+1, size=(1,))
    num_patches = seq_len // patch_size  # Calculate the number of patches

    # create mask for each patch
    patch_mask = torch.rand((batch_size, num_patches, num_agents, 1)).to(gt_agents.device)
    patch_mask = (patch_mask > mask_percentage) # [B, P, M, 1]
    mask = patch_mask.unsqueeze(2).repeat(1,1,patch_size, 1, 1).reshape(batch_size, num_patches*patch_size, num_agents, 1)
    # cat the rest one
    if num_patches*patch_size != seq_len:
        mask = torch.cat((mask, torch.zeros((batch_size, seq_len - num_patches*patch_size, num_agents, 1)).to(gt_agents.device)), dim= time_dim)
    gt_masked = gt_agents * mask
    ego_masked = gt_masked[:,:,0,:]
    agents_masked = gt_masked[:,:,1:,:]
    return ego_masked, ego_data, agents_masked, agents_data, mask.bool()


def get_random_masked_time_only(ego_in, ego_out, agents_in, agents_out, mask_percentage):
    time_dim = 1
    # ego_in, ego_out, agents_in, agents_out
    ego_data = torch.cat((ego_in, ego_out) , dim=time_dim) # cat in the time dimension
    agents_data = torch.cat((agents_in, agents_out), dim=time_dim)
    gt_agents = torch.cat((ego_data.unsqueeze(2), agents_data), dim=2)

    # get patch indices
    batch_size, seq_len, num_agents, feature_dim = gt_agents.shape

    mask = torch.rand((batch_size, seq_len, 1)).to(gt_agents.device) # [B, T, 1]
    mask = (mask > mask_percentage).unsqueeze(2).repeat(1, 1, num_agents, 1) # [B, T, M, 1]

    gt_masked = gt_agents * mask
    ego_masked = gt_masked[:,:,0,:]
    agents_masked = gt_masked[:,:,1:,:]
    return ego_masked, ego_data, agents_masked, agents_data, mask


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore