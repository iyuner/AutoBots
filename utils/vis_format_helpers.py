import numpy as np
import torch
import matplotlib.pyplot as plt

def reformat(ego_in, ego_out, agents_in, agents_out):
    time_dim = 1
    observation_time = ego_in.shape[time_dim]

    # ego_in, ego_out, agents_in, agents_out
    ego_cat = torch.cat((ego_in, ego_out) , dim=time_dim) # cat in the time dimension
    agent_cat = torch.cat((agents_in, agents_out), dim=time_dim)
    
    ego_out = ego_cat.clone()
    ego_cat[:,observation_time:,:] = 0
    ego_in = ego_cat

    agents_out = agent_cat.clone()
    agent_cat[:,observation_time:,:,:] = 0
    agents_in = agent_cat

    return ego_in, ego_out, agents_in, agents_out



def make_2d_rotation_matrix(angle_in_radians: float) -> np.ndarray:
    """
    Makes rotation matrix to rotate point in x-y plane counterclockwise
    by angle_in_radians.
    """
    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                        [np.sin(angle_in_radians), np.cos(angle_in_radians)]])

def convert_local_to_global(agents_in, agents_out, pred_agent):
    # agents_in: [T_obs, M-1, 5]
    # agents_out: [T_future, M-1, 5]
    # agents_all = np.concatenate((agents_in, agents_out), axis=0)
    global_pred_agent= np.zeros((pred_agent.shape))
    for n in range(agents_in.shape[1]):
        
        if agents_in[:, n, -1].sum() >= 2:
            # then we can use the past to compute the angle to +y-axis
            diff = agents_in[-1, n, :2] - agents_in[-2, n, :2]
            yaw = np.arctan2(diff[1], diff[0])
            angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
            translation = agents_in[-1, n, :2]
        elif agents_in[:, n, -1].sum() == 1:
            # then we need to use the future to compute angle to +y-axis
            diff = agents_out[0, n, :2] - agents_in[-1, n, :2]
            yaw = np.arctan2(diff[1], diff[0])
            translation = agents_in[-1, n, :2]
            angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
        else:
            # the agent does not exist...
            angle_of_rotation = None
            translation = None

        if angle_of_rotation is not None:
            global_pred_agent[:, :, n, :2] = convert_global_coords_to_local(coordinates=pred_agent[:, :, n, :2], yaw=-angle_of_rotation) + translation
    return global_pred_agent


def convert_global_coords_to_local(coordinates: np.ndarray, yaw: float) -> np.ndarray:
    """
    Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    :param coordinates: x,y locations. array of shape [n_steps, 2].
    :param translation: Tuple of (x, y, z) location that is the center of the new frame.
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations in frame stored in array of share [n_times, 2].
    """
    transform = make_2d_rotation_matrix(angle_in_radians=yaw)
    if len(coordinates.shape) > 2:
        coord_shape = coordinates.shape
        return np.dot(transform, coordinates.reshape((-1, 2)).T).T.reshape(*coord_shape)
    return np.dot(transform, coordinates.T).T[:, :2]

def visualize_joint_reformat_16_interaction(ego_in, agents_in, context_img, pred_obs, ego_out, agents_out, obs_T, predict_yaw, all_occluded, original_roads):
    """
    ego_in: [B, T, 11] where 11 is [dx,dy,dtheta,x,y,vx,vy,theta,L,W,occ,valid]
    agents_in: [B, T, M-1, 11] where 8 is [dx,dy,dtheta,x,y,vx,vy,theta,L,W,valid]
    context_img: [B, M, S, P, 8] # only use [0,1,-1] dim out of 8
    pred_obs: [K, T, B, M, 6] # the last one of 6 is yaw
    all_occluded : [B, M]
    """
    # print("obs_T, agents_in.shape", obs_T, agents_in.shape)
    # print("original_roads.shape", original_roads.shape) # 1, 78, 80, 8
    random_show_ith_image = torch.randint(0,ego_in.shape[0], (1,))[0]
    ego_in = ego_in[random_show_ith_image][:, [0,1,3,4,11]] # [T, 5]
    agents_in = agents_in[random_show_ith_image][:,:,[0,1,3,4,11]] # [T, M-1, 5]
    context_img = context_img[random_show_ith_image] # [M, S, P, 4]
    original_roads = original_roads[random_show_ith_image] # [S,P,8]
    pred_obs = pred_obs[:,:,random_show_ith_image,:,:5] # [K, T, M, 5] # ignore yaw
    ego_out = ego_out[random_show_ith_image][:,[0,1,3,4,11]] # [T, 5]
    agents_out = agents_out[random_show_ith_image][:,:,[0,1,3,4,11]] # [T, M-1, 5]
    all_occluded = all_occluded[random_show_ith_image] # [M]
    print(all_occluded)
    all_in = torch.cat((ego_in.unsqueeze(1), agents_in), dim=1) # [T_obs, M, 5] 
    all_out = torch.cat((ego_out.unsqueeze(1), agents_out), dim=1) # [T_future, M, 5] 
    figure = plt.figure(figsize=(8,7))
    
    valid_input_agent_x = torch.masked_select(all_in[:,:,2], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    valid_input_agent_y = torch.masked_select(all_in[:,:,3], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    x_min = torch.min(valid_input_agent_x) - 30 #min(torch.min(valid_input_agent_x), torch.min(valid_input_gs_x))-10
    x_max = torch.max(valid_input_agent_x) + 30 #max(torch.max(valid_input_agent_x), torch.max(valid_input_gs_x))+10
    y_min = torch.min(valid_input_agent_y) - 30 #min(torch.min(valid_input_agent_y), torch.min(valid_input_gs_y))-10
    y_max = torch.max(valid_input_agent_y) + 30 #max(torch.max(valid_input_agent_y), torch.max(valid_input_gs_y))+10

    """" figure 1: the input image """
    
    # plot roads
    num_agent = context_img.shape[0]
    num_segment = context_img.shape[1]
    for j in range(original_roads.shape[0]):
        valid_x = torch.masked_select(original_roads[j,:,0], original_roads[j,:,-1].type(torch.BoolTensor).to(context_img.device))
        valid_y = torch.masked_select(original_roads[j,:,1], original_roads[j,:,-1].type(torch.BoolTensor).to(context_img.device))
        plt.scatter(
            valid_x.cpu(),
            valid_y.cpu(),
            color = "grey",
            s = 0.01,
            alpha=0.5,
            zorder=0,)

    # plot all agent trajectory, including ego
    for j in range(num_agent):
        # plot given observed input 
        input_valid_x = torch.masked_select(all_in[:obs_T,j,2], all_in[:obs_T,j,-1].type(torch.BoolTensor).to(all_in.device))
        input_valid_y = torch.masked_select(all_in[:obs_T,j,3], all_in[:obs_T,j,-1].type(torch.BoolTensor).to(all_in.device))
        if all_occluded[j]== 0:
            color = "darkgoldenrod"
        else:
            color = plt.cm.Pastel1(j)

        if input_valid_x.shape[0] == 0:
            continue
        else:
            if j == 0:
                plt.scatter(
                    input_valid_x.cpu(),
                    input_valid_y.cpu(),
                    color = 'darkblue',
                    s = 5,
                    alpha=1)
                
                # Plot the end marker for the end trajectory
                plt.arrow(
                    input_valid_x[-2].cpu(), 
                    input_valid_y[-2].cpu(),
                    input_valid_x[-1].cpu() - input_valid_x[-2].cpu(),
                    input_valid_y[-1].cpu() - input_valid_y[-2].cpu(),
                    color= "darkblue",
                    alpha=1,
                    linewidth=1,
                    head_width=1.1,
                    zorder=2,
                )
            else:
                plt.scatter(
                    input_valid_x.cpu(),
                    input_valid_y.cpu(),
                    color = color,
                    s = 5,
                    alpha=1,
                    zorder=1,
                )
                # Plot the end marker for the end trajectory
                if len(input_valid_x) >1 :
                    plt.arrow(
                        input_valid_x[-2].cpu(), 
                        input_valid_y[-2].cpu(),
                        input_valid_x[-1].cpu() - input_valid_x[-2].cpu(),
                        input_valid_y[-1].cpu() - input_valid_y[-2].cpu(),
                        color= color,
                        alpha=1,
                        linewidth=1,
                        head_width=1.1,
                        zorder=2,
                    )
    
    # plot predicted trajectory for occluded agent. (hidden part for the inputs) [K, T, M, 5]
    pred_agent = pred_obs[:,:,1:,:]
    print(agents_out.shape)
    print(agents_out[:obs_T,3,:])
    print(agents_out[obs_T:,3,:])
    print(pred_agent[0,:,3,:2])
    global_pred_agent = convert_local_to_global(agents_out[:obs_T,:,2:].cpu().numpy(), agents_out[obs_T:,:,2:].cpu().numpy(), pred_agent.cpu().numpy())
    global_pred_agent = torch.from_numpy(global_pred_agent).float().to(agents_out.device)
    print(global_pred_agent[0,:,2,:2])
    num_modes = pred_obs.shape[0]
    for k in range(num_modes):
        for j in range(global_pred_agent.shape[2]):
            color = plt.cm.Pastel1(j+1)
            plt.plot(
                global_pred_agent[k,obs_T:,j,0].cpu(),
                global_pred_agent[k,obs_T:,j,1].cpu(),
                "--",
                color=color,
                alpha=1,
                linewidth=0.8,
                zorder=5,
            )

    plt.gca().set_aspect('equal', adjustable='box')
    # set the grey boundary
    for spine in plt.gca().spines.values():
        spine.set_color((0.5, 0.5, 0.5, 0.5))
    plt.close(figure)
    return figure





def visualize_joint_reformat_16(ego_in, agents_in, context_img, pred_obs, ego_out, agents_out, obs_T, predict_yaw=False, all_occluded=None, original_roads=None):
    # input size is full size T
    # just visualize one image
    random_show_ith_image = torch.randint(0,ego_in.shape[0], (1,))[0]
    # argoverse and nuscences dataset
    if not predict_yaw: 
        ego_in = ego_in[random_show_ith_image] # [T, 5]
        agents_in = agents_in[random_show_ith_image] # [T, M-1, 5]
        context_img = context_img[random_show_ith_image] # [M, S, P, 4]
        pred_obs = pred_obs[:,:,random_show_ith_image,:,:] # [K, T, M, 5]
        ego_out = ego_out[random_show_ith_image] # [T, 5]
    # interaction dataset
    else:
        return visualize_joint_reformat_16_interaction(ego_in, agents_in, context_img, pred_obs, ego_out, agents_out, obs_T, predict_yaw, all_occluded, original_roads)
        """
        ego_in: [B, T, 11] where 11 is [dx,dy,dtheta,x,y,vx,vy,theta,L,W,valid]
        agents_in: [B, T, M-1, 11] where 8 is [dx,dy,dtheta,x,y,vx,vy,theta,L,W,valid]
        context_img: [B, M, S, P, 8] # only use [0,1,-1] dim out of 8
        pred_obs: [K, T, B, M, 6] # the last one of 6 is yaw
        all_occluded : [B, M]
        """
        ego_in = ego_in[random_show_ith_image][:, [0,1,3,4,10]] # [T, 5]
        agents_in = agents_in[random_show_ith_image][:,:,[0,1,3,4,10]] # [T, M-1, 5]
        context_img = context_img[random_show_ith_image] # [M, S, P, 4]
        pred_obs = pred_obs[:,:,random_show_ith_image,:,:5] # [K, T, M, 5] # ignore yaw
        ego_out = ego_out[random_show_ith_image][:,[0,1,3,4,10]] # [T, 5]
        agents_out = agents_out[random_show_ith_image][:,:,[0,1,3,4,10]] # [T, M-1, 5]
        all_occluded = all_occluded[random_show_ith_image] # [M]
    
    all_in = torch.cat((ego_in.unsqueeze(1), agents_in), dim=1) # [T_obs, M, 5] 
    figure = plt.figure(figsize=(2, 2))
    
    valid_input_agent_x = torch.masked_select(all_in[:,:,0], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    valid_input_agent_y = torch.masked_select(all_in[:,:,1], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    x_min = torch.min(valid_input_agent_x) - 30 #min(torch.min(valid_input_agent_x), torch.min(valid_input_gs_x))
    x_max = torch.max(valid_input_agent_x) + 30 #max(torch.max(valid_input_agent_x), torch.max(valid_input_gs_x))
    y_min = torch.min(valid_input_agent_y) - 30 #min(torch.min(valid_input_agent_y), torch.min(valid_input_gs_y))
    y_max = torch.max(valid_input_agent_y) + 30 #max(torch.max(valid_input_agent_y), torch.max(valid_input_gs_y))
    
    plt.xlim(-35, 25)
    plt.ylim(-20, 40)
    
    # plot roads
    num_agent = context_img.shape[0]
    num_segment = context_img.shape[1]
    for i in range(num_segment):
        j = 0
        valid_x = torch.masked_select(context_img[j,i,:,0], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        valid_y = torch.masked_select(context_img[j,i,:,1], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        plt.plot(
            valid_x.cpu(),
            valid_y.cpu(),
            "-",
            color="grey",
            alpha=0.5,
            linewidth=0.5,
            zorder=0,
        )

    # plot predicted trajectory for ego agent. (hidden part for the inputs) [K, T, M, 5]
    num_modes = pred_obs.shape[0]
    for k in range(num_modes):
        j = 0
        plt.plot(
            pred_obs[k,obs_T:,j,0].cpu(),
            pred_obs[k,obs_T:,j,1].cpu(),
            "--",
            color="seagreen",
            alpha=0.8,
            linewidth=1,
            zorder=4,
            )
    
    # plot all agent trajectory
    for j in range(num_agent-1, -1, -1):
        # plot given observed input 
        input_valid_x = torch.masked_select(all_in[:obs_T,j,2], all_in[:obs_T,j,-1].type(torch.BoolTensor).to(all_in.device))
        input_valid_y = torch.masked_select(all_in[:obs_T,j,3], all_in[:obs_T,j,-1].type(torch.BoolTensor).to(all_in.device))
    
        if j == 0:
            color = "darkblue"
        else:
            color = "darkgoldenrod"

        if input_valid_x.shape[0] != 0:
            plt.plot(
                input_valid_x.cpu(),
                input_valid_y.cpu(),
                "-",
                c = color,
                alpha=1,
                linewidth=1,
                zorder=2,
            )
            # Plot the end marker for the end trajectory
            if len(input_valid_x) >1 :
                plt.arrow(
                    input_valid_x[-2].cpu(), 
                    input_valid_y[-2].cpu(),
                    input_valid_x[-1].cpu() - input_valid_x[-2].cpu(),
                    input_valid_y[-1].cpu() - input_valid_y[-2].cpu(),
                    color= color,
                    alpha=1,
                    linewidth=1,
                    head_width=1.1,
                    zorder=2,
                )
    
    plt.plot(
        ego_out[obs_T:,0].cpu(),
        ego_out[obs_T:,1].cpu(),
        "-",
        color="tomato",
        alpha=0.8,
        linewidth=1.5,
        zorder=3,
    )
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', adjustable='box')
    # set the grey boundary
    for spine in plt.gca().spines.values():
        spine.set_color((0.5, 0.5, 0.5, 0.5))
    plt.close(figure)
    return figure

def visualize_joint(ego_in, agents_in, context_img, pred_obs, ego_out, agents_out, predict_yaw=False):
    if predict_yaw:
        return visualize_joint_interaction(ego_in, agents_in, context_img, pred_obs, ego_out, agents_out, predict_yaw)
    
    # just visualize one image
    random_show_ith_image = torch.randint(0,ego_in.shape[0], (1,))[0]
    ego_in = ego_in[random_show_ith_image] # [T_obs, 5]
    agents_in = agents_in[random_show_ith_image] # [T_obs, M-1, 5]
    context_img = context_img[random_show_ith_image] # [M, S, P, k_arr+1], for nuscene last dim is 4, argo is 3 
    pred_obs = pred_obs[:,:,random_show_ith_image,:,:] # [K, T, M, 5]
    ego_out = ego_out[random_show_ith_image] # [T_future, 5]
    agents_out = agents_out[random_show_ith_image] # [T_future, M-1, 5]
    
    all_in = torch.cat((ego_in.unsqueeze(1), agents_in), dim=1) # [T_obs, M, 5] 
    all_out = torch.cat((ego_out.unsqueeze(1), agents_out), dim=1) # [T_future, M, 5] 
    figure = plt.figure(figsize=(3, 3))
    
    valid_input_agent_x = torch.masked_select(all_in[:,:,0], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    valid_input_agent_y = torch.masked_select(all_in[:,:,1], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    valid_input_gs_x = torch.masked_select(context_img[:,:,:,0], context_img[:,:,:,-1].type(torch.BoolTensor).to(context_img.device))
    valid_input_gs_y = torch.masked_select(context_img[:,:,:,1], context_img[:,:,:,-1].type(torch.BoolTensor).to(context_img.device))
    x_min = min(torch.min(valid_input_agent_x), torch.min(valid_input_gs_x))
    x_max = max(torch.max(valid_input_agent_x), torch.max(valid_input_gs_x))
    y_min = min(torch.min(valid_input_agent_y), torch.min(valid_input_gs_y))
    y_max = max(torch.max(valid_input_agent_y), torch.max(valid_input_gs_y))
    
    
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)

    # plot roads
    num_agent = context_img.shape[0]
    num_segment = context_img.shape[1]
    for i in range(num_segment):
        j = 0
        valid_x = torch.masked_select(context_img[j,i,:,0], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        valid_y = torch.masked_select(context_img[j,i,:,1], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        plt.plot(
            valid_x.cpu(),
            valid_y.cpu(),
            "--",
            color="grey",
            alpha=1,
            linewidth=0.5,
            zorder=0,
        )
        

    # plot predicted trajectory for ego agent. (hidden part for the inputs) [K, T, M, 5]
    num_modes = pred_obs.shape[0]
    num_agent = pred_obs.shape[2]
    for k in range(num_modes):
        j = 0
        plt.plot(
            np.append(0, pred_obs[k,:,j,0].detach().cpu().numpy()),
            np.append(0, pred_obs[k,:,j,1].detach().cpu().numpy()),
            "--",
            color="seagreen",
            alpha=1,
            linewidth=0.5,
            zorder=4,
        )
    
    if not predict_yaw:
        global_x_idx = 2
        global_y_idx = 3
    else:
        global_x_idx = 0
        global_y_idx = 1

    # plot agent trajectory
    for j in range(num_agent):
        # plot given observed input 
        input_valid_x = torch.masked_select(all_in[:,j,global_x_idx], all_in[:,j,-1].type(torch.BoolTensor).to(all_in.device))
        input_valid_y = torch.masked_select(all_in[:,j,global_y_idx], all_in[:,j,-1].type(torch.BoolTensor).to(all_in.device))
    
        if j == 0:
            color = "darkblue"
        else:
            color = "darkgoldenrod"

        if input_valid_x.shape[0] != 0:
            plt.plot(
                input_valid_x.cpu(),
                input_valid_y.cpu(),
                "-",
                c = color,
                alpha=1,
                linewidth=1,
                zorder=2,
            )
            # Plot the end marker for the end trajectory
            plt.arrow(
                input_valid_x[-2].cpu(), 
                input_valid_y[-2].cpu(),
                input_valid_x[-1].cpu() - input_valid_x[-2].cpu(),
                input_valid_y[-1].cpu() - input_valid_y[-2].cpu(),
                color= "darkblue",
                alpha=1,
                linewidth=1,
                head_width=1.1,
                zorder=2,
            )

    # plot the ground truth trajectory
    plt.plot(
        np.append(0,ego_out[:,0].cpu()),
        np.append(0,ego_out[:,1].cpu()),
        "-",
        color="tomato",
        alpha=1,
        linewidth=1.5,
        zorder=3,
    )
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks([])
    plt.yticks([])
    # Remove the black boundary
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.close(figure)
    return figure


def visualize_joint_interaction(ego_in, agents_in, context_img, pred_obs, ego_out, agents_out, predict_yaw):
    # just visualize one image
    random_show_ith_image = torch.randint(0,ego_in.shape[0], (1,))[0]
    """
    ego_in: [B, T, 11] where 11 is [dx,dy,dtheta,x,y,vx,vy,theta,L,W,occ,valid]
    agents_in: [B, T, M-1, 11] where 8 is [dx,dy,dtheta,x,y,vx,vy,theta,L,W,valid]
    context_img: [B, M, S, P, 8] # only use [0,1,-1] dim out of 8
    pred_obs: [K, T, B, M, 6] # the last one of 6 is yaw
    all_occluded : [B, M]
    """
    ego_in = ego_in[random_show_ith_image][:, [0,1,3,4,11]] # [T, 5]
    agents_in = agents_in[random_show_ith_image][:,:,[0,1,3,4,11]] # [T, M-1, 5]
    context_img = context_img[random_show_ith_image] # [M, S, P, 4]
    pred_obs = pred_obs[:,:,random_show_ith_image,:,:5] # [K, T, M, 5] # ignore yaw
    ego_out = ego_out[random_show_ith_image][:,[0,1,3,4,11]] # [T, 5]
    agents_out = agents_out[random_show_ith_image][:,:,[0,1,3,4,11]] # [T, M-1, 5]
    all_in = torch.cat((ego_in.unsqueeze(1), agents_in), dim=1) # [T_obs, M, 5] 
    all_out = torch.cat((ego_out.unsqueeze(1), agents_out), dim=1) # [T_future, M, 5] 
    figure = plt.figure(figsize=(8,7))

    valid_input_agent_x = torch.masked_select(all_in[:,:,0], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    valid_input_agent_y = torch.masked_select(all_in[:,:,1], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    valid_input_gs_x = torch.masked_select(context_img[:,:,:,0], context_img[:,:,:,-1].type(torch.BoolTensor).to(context_img.device))
    valid_input_gs_y = torch.masked_select(context_img[:,:,:,1], context_img[:,:,:,-1].type(torch.BoolTensor).to(context_img.device))
    x_min = min(torch.min(valid_input_agent_x), torch.min(valid_input_gs_x))
    x_max = max(torch.max(valid_input_agent_x), torch.max(valid_input_gs_x))
    y_min = min(torch.min(valid_input_agent_y), torch.min(valid_input_gs_y))
    y_max = max(torch.max(valid_input_agent_y), torch.max(valid_input_gs_y))
    
    plt.xlim(x_min.cpu(), x_max.cpu())
    plt.ylim(y_min.cpu(), y_max.cpu())

    # plot roads
    num_agent = context_img.shape[0]
    num_segment = context_img.shape[1]
    for i in range(num_segment):
        j = 0
        valid_x = torch.masked_select(context_img[j,i,:,0], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        valid_y = torch.masked_select(context_img[j,i,:,1], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        plt.plot(
            valid_x.cpu(),
            valid_y.cpu(),
            "--",
            color="grey",
            alpha=1,
            linewidth=0.5,
            zorder=0,
        )

    # plot predicted trajectory for ego agent. (hidden part for the inputs) [K, T, M, 5]
    num_modes = pred_obs.shape[0]
    num_agent = pred_obs.shape[2]
    for k in range(num_modes):
        for j in range(num_agent):
            if j ==0:
                color="seagreen"
            else:    
                color = plt.cm.Pastel1(j+1)
            plt.plot(
                np.append(0, pred_obs[k,:,j,0].detach().cpu().numpy()),
                np.append(0, pred_obs[k,:,j,1].detach().cpu().numpy()),
                "--",
                color=color,
                alpha=1,
                linewidth=0.5,
                zorder=4,
            )
    
    if not predict_yaw:
        global_x_idx = 2
        global_y_idx = 3
    else:
        global_x_idx = 0
        global_y_idx = 1


    
    # plot agent trajectory
    for j in range(num_agent):
        # plot given observed input 
        input_valid_x = torch.masked_select(all_in[:,j,global_x_idx], all_in[:,j,-1].type(torch.BoolTensor).to(all_in.device))
        input_valid_y = torch.masked_select(all_in[:,j,global_y_idx], all_in[:,j,-1].type(torch.BoolTensor).to(all_in.device))
    
        if j == 0:
            color = "darkblue"
        else:
            color = "darkgoldenrod"

        if input_valid_x.shape[0] != 0:
            plt.plot(
                input_valid_x.cpu(),
                input_valid_y.cpu(),
                "-",
                c = color,
                alpha=1,
                linewidth=1,
                zorder=2,
            )
            # Plot the end marker for the end trajectory
            plt.arrow(
                input_valid_x[-2].cpu(), 
                input_valid_y[-2].cpu(),
                input_valid_x[-1].cpu() - input_valid_x[-2].cpu(),
                input_valid_y[-1].cpu() - input_valid_y[-2].cpu(),
                color= "darkblue",
                alpha=1,
                linewidth=1,
                head_width=1.1,
                zorder=2,
            )
    plt.plot(
        np.append(0,ego_out[:,0].cpu()),
        np.append(0,ego_out[:,1].cpu()),
        "-",
        color="tomato",
        alpha=1,
        linewidth=1.5,
        zorder=3,
    )
    
    plt.gca().set_aspect('equal', adjustable='box')
    # Remove the black boundary
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.close(figure)
    return figure

def visualize_joint_mask(ego_in, agents_in, context_img, pred_obs, ego_out, agents_out, obs_T, predict_yaw= False):
    # input size is full size T
    # just visualize one image
    random_show_ith_image = torch.randint(0,ego_in.shape[0], (1,))[0]
    # argoverse and nuscences dataset
    if not predict_yaw: 
        ego_in = ego_in[random_show_ith_image] # [T, 5]
        agents_in = agents_in[random_show_ith_image] # [T, M-1, 5]
        context_img = context_img[random_show_ith_image] # [M, S, P, 4]
        pred_obs = pred_obs[:,:,random_show_ith_image,:,:] # [K, T, M, 5]
        ego_out = ego_out[random_show_ith_image] # [T, 5]
        agents_out = agents_out[random_show_ith_image] # [T, M-1, 5]
    # interaction dataset
    else:
        """
        ego_in: [B, T, 11] where 11 is [dx,dy,dtheta,x,y,vx,vy,theta,L,W,valid]
        agents_in: [B, T, M-1, 11] where 8 is [dx,dy,dtheta,x,y,vx,vy,theta,L,W,valid]
        context_img: [B, M, S, P, 8] # only use [0,1,-1] dim out of 8
        pred_obs: [K, T, B, M, 6] # the last one of 6 is yaw
        """
        ego_in = ego_in[random_show_ith_image][:, [0,1,3,4,10]] # [T, 5]
        agents_in = agents_in[random_show_ith_image][:,:,[0,1,3,4,10]] # [T, M-1, 5]
        context_img = context_img[random_show_ith_image] # [M, S, P, 4]
        pred_obs = pred_obs[:,:,random_show_ith_image,:,:5] # [K, T, M, 5] # ignore yaw
        ego_out = ego_out[random_show_ith_image][:,[0,1,3,4,10]] # [T, 5]
        agents_out = agents_out[random_show_ith_image][:,:,[0,1,3,4,10]] # [T, M-1, 5]
    

    all_in = torch.cat((ego_in.unsqueeze(1), agents_in), dim=1) # [T_obs, M, 5] 
    all_out = torch.cat((ego_out.unsqueeze(1), agents_out), dim=1) # [T_future, M, 5] 
    figure = plt.figure(figsize=(10,10))
    
    valid_input_agent_x = torch.masked_select(all_in[:,:,2], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    valid_input_agent_y = torch.masked_select(all_in[:,:,3], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    valid_input_gs_x = torch.masked_select(context_img[0,:,:,0], context_img[0,:,:,-1].type(torch.BoolTensor).to(context_img.device))
    valid_input_gs_y = torch.masked_select(context_img[0,:,:,1], context_img[0,:,:,-1].type(torch.BoolTensor).to(context_img.device))
    x_min = min(torch.min(valid_input_agent_x), torch.min(valid_input_gs_x))-10
    x_max = max(torch.max(valid_input_agent_x), torch.max(valid_input_gs_x))+10
    y_min = min(torch.min(valid_input_agent_y), torch.min(valid_input_gs_y))-10
    y_max = max(torch.max(valid_input_agent_y), torch.max(valid_input_gs_y))+10

    """" figure 1: the input image """
    # ax.imshow(z, aspect="auto")
    plt.subplot(1,3,1)
    plt.xlim(x_min.cpu(), x_max.cpu())
    plt.ylim(y_min.cpu(), y_max.cpu())
    
    # plot roads
    num_agent = context_img.shape[0]
    num_segment = context_img.shape[1]
    for i in range(num_segment):
        j = 0
        valid_x = torch.masked_select(context_img[j,i,:,0], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        valid_y = torch.masked_select(context_img[j,i,:,1], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        plt.scatter(
            valid_x.cpu(),
            valid_y.cpu(),
            c = "grey",
            s = 1,
            alpha=1,
            zorder=0,)
    
    if not predict_yaw:
        global_x_idx = 2
        global_y_idx = 3
    else:
        global_x_idx = 0
        global_y_idx = 1
    # plot agent trajectory, except ego (we keep ego as all visible, so not masked)
    for j in range(num_agent-1):
        # plot given observed input 
        input_valid_x = torch.masked_select(agents_in[:,j,global_x_idx], agents_in[:,j,-1].type(torch.BoolTensor).to(all_in.device))
        input_valid_y = torch.masked_select(agents_in[:,j,global_y_idx], agents_in[:,j,-1].type(torch.BoolTensor).to(all_in.device))
    
        color = plt.cm.Set1(j)

        if input_valid_x.shape[0] != 0:
            plt.scatter(
                input_valid_x.cpu(),
                input_valid_y.cpu(),
                c = color,
                s = 5,
                alpha=1)
            plt.scatter(
                input_valid_x[0].cpu(),
                input_valid_y[0].cpu(),
                c = color,
                alpha=1)

    """" figure 2: the predicted image """
    plt.subplot(1,3,2)
    plt.xlim(x_min.cpu(), x_max.cpu())
    plt.ylim(y_min.cpu(), y_max.cpu())
    for i in range(num_segment):
        j = 0
        valid_x = torch.masked_select(context_img[j,i,:,0], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        valid_y = torch.masked_select(context_img[j,i,:,1], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        plt.scatter(
            valid_x.cpu(),
            valid_y.cpu(),
            c = "grey",
            s = 0.2,
            alpha=1)

    # plot predicted trajectory for ego agent. (hidden part for the inputs) [K, T, M, 5]
    pred_agent = pred_obs[:,:,1:,:]
    global_pred_agent = convert_local_to_global(agents_out[:obs_T,:,2:].cpu().numpy(), agents_out[obs_T:,:,2:].cpu().numpy(), pred_agent.cpu().numpy())
    global_pred_agent = torch.from_numpy(global_pred_agent).float().to(agents_out.device)
    
    num_modes = pred_obs.shape[0]
    for k in range(num_modes):
        for j in range(global_pred_agent.shape[2]):
            color = plt.cm.Set1(j)
            plt.plot(
                global_pred_agent[k,:,j,0].cpu(),
                global_pred_agent[k,:,j,1].cpu(),
                "-",
                color=color,
                alpha=1,
                linewidth=0.5,
            )
    
    

    
    """" figure 3: the ground truth image """
    plt.subplot(1,3,3)
    plt.xlim(x_min.cpu(), x_max.cpu())
    plt.ylim(y_min.cpu(), y_max.cpu())
    for i in range(num_segment):
        j = 0
        valid_x = torch.masked_select(context_img[j,i,:,0], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        valid_y = torch.masked_select(context_img[j,i,:,1], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        plt.scatter(
            valid_x.cpu(),
            valid_y.cpu(),
            c = "grey",
            s = 0.2,
            alpha=1)
    for j in range(num_agent-1):
        # plot given observed input 
        input_valid_x = torch.masked_select(agents_out[:,j,2], agents_out[:,j,-1].type(torch.BoolTensor).to(all_in.device))
        input_valid_y = torch.masked_select(agents_out[:,j,3], agents_out[:,j,-1].type(torch.BoolTensor).to(all_in.device))
    
        color = plt.cm.Set1(j)

        if input_valid_x.shape[0] != 0:
            plt.scatter(
                input_valid_x.cpu(),
                input_valid_y.cpu(),
                c = color,
                s = 5,
                alpha=1)
            plt.scatter(
                input_valid_x[0].cpu(),
                input_valid_y[0].cpu(),
                c = color,
                alpha=1)
    
    plt.close(figure)
    return figure


def visualize_joint_conditional_mask(ego_in, agents_in, context_img, pred_obs, ego_out, agents_out, obs_T):
    # input size is full size T
    # just visualize one image
    random_show_ith_image = torch.randint(0,ego_in.shape[0], (1,))[0]
    random_show_ith_image= 0
    ego_in = ego_in[random_show_ith_image] # [T, 5]
    agents_in = agents_in[random_show_ith_image] # [T, M-1, 5]
    context_img = context_img[random_show_ith_image] # [M, S, P, 4]
    pred_obs = pred_obs[:,:,random_show_ith_image,:,:] # [K, T, M, 5]
    ego_out = ego_out[random_show_ith_image] # [T, 5]
    agents_out = agents_out[random_show_ith_image] # [T, M-1, 5]
    

    all_in = torch.cat((ego_in.unsqueeze(1), agents_in), dim=1) # [T_obs, M, 5] 
    all_out = torch.cat((ego_out.unsqueeze(1), agents_out), dim=1) # [T_future, M, 5] 
    figure = plt.figure(figsize=(2,2))
    
    valid_input_agent_x = torch.masked_select(all_in[:,:,2], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    valid_input_agent_y = torch.masked_select(all_in[:,:,3], all_in[:,:,-1].type(torch.BoolTensor).to(all_in.device))
    x_min = torch.min(valid_input_agent_x) - 30 #min(torch.min(valid_input_agent_x), torch.min(valid_input_gs_x))-10
    x_max = torch.max(valid_input_agent_x) + 30 #max(torch.max(valid_input_agent_x), torch.max(valid_input_gs_x))+10
    y_min = torch.min(valid_input_agent_y) - 30 #min(torch.min(valid_input_agent_y), torch.min(valid_input_gs_y))-10
    y_max = torch.max(valid_input_agent_y) + 30 #max(torch.max(valid_input_agent_y), torch.max(valid_input_gs_y))+10
    
    plt.xlim(-35, 25)
    plt.ylim(-20, 40)
    
    # plot roads
    num_agent = context_img.shape[0]
    num_segment = context_img.shape[1]
    for i in range(num_segment):
        j = 0
        valid_x = torch.masked_select(context_img[j,i,:,0], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        valid_y = torch.masked_select(context_img[j,i,:,1], context_img[j,i,:,-1].type(torch.BoolTensor).to(context_img.device))
        plt.plot(
            valid_x.cpu(),
            valid_y.cpu(),
            "-",
            color="grey",
            alpha=0.5,
            linewidth=0.5,
            zorder=0,
        )
    
    # plot all agent trajectory, including ego
    for j in range(num_agent):
        # plot given observed input 
        input_valid_x = torch.masked_select(all_in[:obs_T,j,2], all_in[:obs_T,j,-1].type(torch.BoolTensor).to(all_in.device))
        input_valid_y = torch.masked_select(all_in[:obs_T,j,3], all_in[:obs_T,j,-1].type(torch.BoolTensor).to(all_in.device))
        color = plt.cm.Pastel1(j)

        if input_valid_x.shape[0] != 0:
            if j == 0:
                plt.plot(
                    input_valid_x.cpu(),
                    input_valid_y.cpu(),
                    "-",
                    color = "darkblue",
                    alpha=1,
                    linewidth=1,
                    zorder=2,
                )
                # Plot the end marker for the end trajectory
                plt.arrow(
                    input_valid_x[-2].cpu(), 
                    input_valid_y[-2].cpu(),
                    input_valid_x[-1].cpu() - input_valid_x[-2].cpu(),
                    input_valid_y[-1].cpu() - input_valid_y[-2].cpu(),
                    color= "darkblue",
                    alpha=1,
                    linewidth=1,
                    head_width=1.1,
                    zorder=2,
                )
            else:
                plt.plot(
                    input_valid_x.cpu(),
                    input_valid_y.cpu(),
                    "-",
                    color = color,
                    alpha=1,
                    linewidth=1,
                    zorder=2,
                )
                # Plot the end marker for the end trajectory
                if len(input_valid_x) >1 :
                    plt.arrow(
                        input_valid_x[-2].cpu(), 
                        input_valid_y[-2].cpu(),
                        input_valid_x[-1].cpu() - input_valid_x[-2].cpu(),
                        input_valid_y[-1].cpu() - input_valid_y[-2].cpu(),
                        color= color,
                        alpha=1,
                        linewidth=1,
                        head_width=1.1,
                        zorder=2,
                    )
                
    
    # plot predicted trajectory for ego agent. (hidden part for the inputs) [K, T, M, 5]
    pred_agent = pred_obs[:,:,1:,:]
    global_pred_agent = convert_local_to_global(agents_out[:obs_T,:,2:].cpu().numpy(), agents_out[obs_T:,:,2:].cpu().numpy(), pred_agent.cpu().numpy())
    global_pred_agent = torch.from_numpy(global_pred_agent).float().to(agents_out.device)
    
    num_modes = pred_obs.shape[0]
    for k in range(num_modes):
        # j = 0
        for j in range(global_pred_agent.shape[2]):
       
            color = plt.cm.Pastel1(j+1)

            plt.plot(
                global_pred_agent[k,obs_T:,j,0].cpu(),
                global_pred_agent[k,obs_T:,j,1].cpu(),
                "--",
                color=color,
                alpha=1,
                linewidth=0.8,
                zorder=1,
            )
            

    plt.plot(
        ego_out[obs_T:,0].cpu(),
        ego_out[obs_T:,1].cpu(),
        "-",
        color="tomato",
        alpha=0.8,
        linewidth=0.5,
        zorder=1,
    )
    # Remove x and y ticks
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', adjustable='box')
    # set the grey boundary
    for spine in plt.gca().spines.values():
        spine.set_color((0.5, 0.5, 0.5, 0.5))
    plt.close(figure)
    return figure

