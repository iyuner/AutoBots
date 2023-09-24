import os
import h5py
from torch.utils.data import Dataset
import numpy as np
from utils.vis_format_helpers import convert_global_coords_to_local
import warnings
warnings.filterwarnings("ignore")

class ArgoH5Dataset(Dataset):
    def __init__(self, dset_path, split_name="train", orig_ego=False, use_map_lanes=True, model_type="Autobot-Ego"):
        self.data_root = dset_path
        self.split_name = split_name
        self.orig_ego = orig_ego
        self.pred_horizon = 30
        self.num_others = 5
        self.map_attr = 2
        self.k_attr = 2
        self.use_map_lanes = use_map_lanes
        self.scene_context_img = True
        self.predict_yaw = False

        # add later
        self.obs_horizon = 20
        self.num_agent_types = 0 # since argoverse1 only has Agent or others types
        if "Joint" in model_type:
            self.k_attr = 4
            self.use_joint_version = True
        else:
            self.k_attr = 2
            self.use_joint_version = False

        dataset = h5py.File(os.path.join(self.data_root, split_name+'_dataset.hdf5'), 'r')
        self.dset_len = len(dataset["ego_trajectories"])

    def get_input_output_seqs(self, ego_data, agents_data):
        in_len = 20

        # Ego
        ego_in = ego_data[:in_len]
        ego_out = ego_data[in_len:]

        # Other agents
        agents_in = agents_data[:in_len, :self.num_others]
        agents_out= agents_data[in_len:, :self.num_others]
        return ego_in, ego_out, agents_in, agents_out

    def __getitem__(self, idx: int):
        dataset = h5py.File(os.path.join(self.data_root, self.split_name + '_dataset.hdf5'), 'r')
        ego_data = dataset['ego_trajectories'][idx]
        agents_data = dataset['agents_trajectories'][idx]
        ego_in, ego_out, agents_in, agent_out = self.get_input_output_seqs(ego_data, agents_data)

        if self.use_map_lanes:
            roads = dataset['road_pts'][idx]
        else:
            roads = np.zeros((1, 1))  # dummy
        
        # Normalize all other agents futures
        # make agents have 2 sets of x,y positions (one centered @(0,0) and pointing up, and the other being raw
        if self.use_joint_version:
            roads = self.get_agent_roads(roads, agents_in)
            in_ego, out_ego, agents_in, out_agents, roads = \
                self.rotate_agent_datas(ego_in, ego_out, agents_in, agent_out, roads)

            if "test" in self.split_name:
                extra = dataset['extras'][idx]
                return in_ego, agents_in, roads, extra
            agent_types = np.zeros((self.num_others+1, 1))
            return in_ego, out_ego, agents_in, out_agents, roads, agent_types
        
        else:
            # orginal version
            if "test" in self.split_name:
                extra = dataset['extras'][idx]
                return ego_in, agents_in, roads, extra
            elif self.orig_ego:  # for validation with re-rotation to global coordinates
                extra = dataset['extras'][idx]
                ego_data = dataset['orig_egos'][idx]
                ego_out = ego_data[20:]
                return ego_in, ego_out, agents_in, roads, extra

            return ego_in, ego_out, agents_in, roads
    
    def rotate_agent_datas(self, ego_in, ego_out, agents_in, agents_out, roads):
        new_ego_in = np.zeros((len(ego_in), ego_in.shape[1]+2))
        new_ego_out = np.zeros((len(ego_out), ego_out.shape[1] + 2))
        new_agents_in = np.zeros((len(agents_in), self.num_others, agents_in.shape[2]+2))
        new_agents_out = np.zeros((len(agents_out), self.num_others, agents_out.shape[2] + 2))
        new_roads = roads.copy()

        # Ego trajectories
        new_ego_in[:, :2] = ego_in[:, :2]
        new_ego_in[:, 2:] = ego_in
        new_ego_out[:, :2] = ego_out[:, :2]
        new_ego_out[:, 2:] = ego_out

        for n in range(self.num_others):
            new_agents_in[:, n, 2:] = agents_in[:, n]
            new_agents_out[:, n, 2:] = agents_out[:, n]
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
                new_agents_in[:, n, :2] = convert_global_coords_to_local(coordinates=agents_in[:, n, :2] - translation, yaw=angle_of_rotation)
                new_agents_out[:, n, :2] = convert_global_coords_to_local(coordinates=agents_out[:, n, :2] - translation, yaw=angle_of_rotation)
                new_agents_in[:, n, :2][np.where(new_agents_in[:, n, -1] == 0)] = 0.0
                new_agents_out[:, n, :2][np.where(new_agents_out[:, n, -1] == 0)] = 0.0
                if self.use_map_lanes:
                    new_roads[n+1, :, :, :2] = convert_global_coords_to_local(coordinates=new_roads[n+1, :, :, :2] - translation, yaw=angle_of_rotation)
                    new_roads[n+1, :, :, 2] -= angle_of_rotation
                    new_roads[n+1][np.where(new_roads[n+1, :, :, -1] == 0)] = 0.0

        return new_ego_in, new_ego_out, new_agents_in, new_agents_out, new_roads

    def get_agent_roads(self, roads, agents_in):
        N = 150
        curr_roads = roads.copy()
        curr_roads[np.where(curr_roads[:, :, -1] == 0)] = np.nan
        mean_roads = np.nanmean(curr_roads, axis=1)[:, :2]

        # Ego Agent
        args_closest_roads = np.argsort(np.linalg.norm(np.array([[0.0, 0.0]]) - mean_roads, axis=-1))
        if len(args_closest_roads) >= N:
            per_agent_roads = [roads[args_closest_roads[:N]]]
        else:
            ego_roads = np.zeros((N, 10, 3))
            ego_roads[:len(args_closest_roads)] = roads[args_closest_roads]
            per_agent_roads = [ego_roads]

        # Other Agents
        for n in range(self.num_others):
                if agents_in[-1, n, 2]:
                    args_closest_roads = np.argsort(np.linalg.norm(agents_in[-1:, n, :2] - mean_roads, axis=-1))
                    if len(args_closest_roads) >= N:
                        per_agent_roads.append(roads[args_closest_roads[:N]])
                    else:
                        agent_roads = np.zeros((N, 10, 3))
                        agent_roads[:len(args_closest_roads)] = roads[args_closest_roads]
                        per_agent_roads.append(agent_roads)
                else:
                    per_agent_roads.append(np.zeros((N, 10, 3)))

        roads = np.array(per_agent_roads)
        roads[:, :, :, 2][np.where(roads[:, :, :, 2] < 0.0)] += 2 * np.pi  # making all orientations between 0 and 2pi

        # ensure pt closest to ego has an angle of pi/2
        temp_ego_roads = roads[0].copy()
        temp_ego_roads[np.where(temp_ego_roads[:, :, -1] == 0)] = np.nan
        dist = np.linalg.norm(temp_ego_roads[:, :, :2] - np.array([[[0.0, 0.0]]]), axis=-1)
        closest_pt = temp_ego_roads[np.where(dist == np.nanmin(dist))]
        angle_diff = closest_pt[0, 2] - (np.pi/2)
        roads[:, :, :, 2] -= angle_diff
        return roads

    def __len__(self):
        return self.dset_len


if __name__ == '__main__':
    dst = ArgoH5Dataset(dset_path="/hdd2/argoverse", split_name="train", use_map_lanes=True)
