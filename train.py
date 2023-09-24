import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import yaml
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim, nn
import torch.distributions as D
import os.path as osp
from easydict import EasyDict
import wandb
import random
import shutil
import numpy as np
from utils.train_helper import nll_loss_multimodes_joint, min_xde_K, nll_loss_multimodes
from utils.vis_format_helpers import reformat, visualize_joint_reformat_16, visualize_joint, visualize_joint_mask, visualize_joint_conditional_mask
from utils.mask_helper import get_random_masked, conditional_padding
from models.autobot_joint import AutoBot
from datasets.argoverse.dataset import ArgoH5Dataset
from datasets.nuscenes.dataset import NuscenesH5Dataset
from datasets.interaction_dataset.dataset import InteractionDataset

class Trainer:
    def __init__(self, config):
        self.config = config
        self.start_epoch = 1
        self.smallest_minade_k = 5.0  # for computing best models
        self.smallest_minfde_k = 5.0  # for computing best models

        # distributed training
        self.device = torch.device('cuda:{}'.format(self.config.local_rank))
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend="nccl", init_method="env://",)
        self.dist = self.config.local_rank >= 0
        self.is_main_process = ((self.config.local_rank ==0) or (self.config.local_rank == -1))

        # seed all
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # build dir, dataloader, model, optimizer
        self._build()

        # if use mask, and must use joint loss instead of only ego future loss
        if self.config.stage == "pretrain" or self.config.stage == "conditional_prediction": 
            assert self.config.only_cal_ego_future_loss_in_join_model == False
        if self.config.stage == "future_prediction":
            if "Interaction" in self.config.dataset:
                assert self.config.only_cal_ego_future_loss_in_join_model == False
            else:
                assert self.config.only_cal_ego_future_loss_in_join_model == True


    def train(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.start_epoch + self.config.epochs + 1):
            if self.is_main_process:
                print("Epoch:", epoch)
            epoch_marg_ade_losses = []
            epoch_marg_fde_losses = []
            epoch_marg_mode_probs = []
            epoch_scene_ade_losses = []
            epoch_scene_fde_losses = []
            epoch_mode_probs = []
            # """
            for i, batch in enumerate(self.train_loader):
                # Forward pass
                ego_in, ego_out, agents_in, agents_out, map_lanes, agent_types = self._data_to_device(batch)
                
                mask = None
                if self.config.stage == "pretrain":
                    ego_in, ego_out, agents_in, agents_out, mask = get_random_masked(ego_in, ego_out, agents_in, agents_out, self.config)
                    mask = None
                elif self.config.stage == "conditional_prediction":
                    # padding zeros to the agent_in
                    ego_in, ego_out, agents_in, agents_out, mask = conditional_padding(ego_in, ego_out, agents_in, agents_out)
                    mask = None # to avoid nan
                elif self.config.reformat: # padding zeros
                    ego_in, ego_out, agents_in, agents_out = reformat(ego_in, ego_out, agents_in, agents_out)
                
                pred_obs, mode_probs = self.model.module(ego_in, agents_in, map_lanes, agent_types)

                if not self.config.only_cal_ego_future_loss_in_join_model:
                    # original loss. If use masking, also use the loss for all agents
                    nll_loss, kl_loss, post_entropy, adefde_loss = \
                            nll_loss_multimodes_joint(pred_obs, ego_out, agents_out, mode_probs,
                                                    entropy_weight=self.config.entropy_weight,
                                                    kl_weight=self.config.kl_weight,
                                                    use_FDEADE_aux_loss=self.config.use_FDEADE_aux_loss,
                                                    agent_types=agent_types,
                                                    predict_yaw=self.config.predict_yaw,
                                                    mask=mask)
                else:
                    # only care about the ego for (pred_obs.sahpe[2]=12)
                    if not self.config.reformat:
                        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs[:,:,:,0,:], ego_out[:, :, :2], mode_probs,
                                                                                        entropy_weight=self.config.entropy_weight,
                                                                                        kl_weight=self.config.kl_weight,
                                                                                            use_FDEADE_aux_loss=self.config.use_FDEADE_aux_loss)
                    else:
                        # only care about the ego future for (pred_obs.sahpe[2]=16)
                        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs[:,self.config.obs_horizon:,:,0,:], ego_out[:, self.config.obs_horizon:, :2], mode_probs,
                                                                                        entropy_weight=self.config.entropy_weight,
                                                                                        kl_weight=self.config.kl_weight,
                                                                                        use_FDEADE_aux_loss=self.config.use_FDEADE_aux_loss)



                self.optimizer.zero_grad()
                (nll_loss + adefde_loss + kl_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()

                if self.is_main_process:
                    wandb.log({"Loss/nll": nll_loss.item()})
                    wandb.log({"Loss/adefde": adefde_loss.item()})
                    wandb.log({"Loss/kl": kl_loss.item()})
                    wandb.log({"Loss/total": (nll_loss + adefde_loss + kl_loss).item()})

                    
                with torch.no_grad():
                    if not self.config.only_cal_ego_future_loss_in_join_model:
                        ade_losses, fde_losses = self._compute_marginal_errors(pred_obs, ego_out, agents_out, agents_in, mask)
                        epoch_marg_ade_losses.append(ade_losses.reshape(-1, self.config.num_modes))
                        epoch_marg_fde_losses.append(fde_losses.reshape(-1, self.config.num_modes))
                        epoch_marg_mode_probs.append(
                            mode_probs.unsqueeze(1).repeat(1, self.config.num_other_agents + 1, 1).detach().cpu().numpy().reshape(
                                -1, self.config.num_modes))

                        scene_ade_losses, scene_fde_losses = self._compute_joint_errors(pred_obs, ego_out, agents_out, mask)
                        epoch_scene_ade_losses.append(scene_ade_losses)
                        epoch_scene_fde_losses.append(scene_fde_losses)
                        epoch_mode_probs.append(mode_probs.detach().cpu().numpy())
                    else:
                        if not self.config.reformat:
                            ade_losses, fde_losses = self._compute_ego_errors(pred_obs[:,:,:,0,:], ego_out)
                        else:
                            # only care about the ego future for (pred_obs.sahpe[2]=16)
                            ade_losses, fde_losses = self._compute_ego_errors(pred_obs[:,self.config.obs_horizon:,:,0,:], ego_out[:,self.config.obs_horizon:,:])
                        epoch_marg_ade_losses.append(ade_losses)
                        epoch_marg_fde_losses.append(fde_losses)
                        epoch_marg_mode_probs.append(mode_probs.detach().cpu().numpy())

                if i % 20 == 0 and self.is_main_process:
                    print(i, "/", len(self.train_loader.dataset)//self.config.batch_size,
                          "NLL loss", round(nll_loss.item(), 2), "KL loss", round(kl_loss.item(), 2),
                          "Prior Entropy", round(torch.mean(D.Categorical(mode_probs).entropy()).item(), 2),
                          "Post Entropy", round(post_entropy, 2), "ADE+FDE loss", round(adefde_loss.item(), 2))

            
            epoch_marg_ade_losses = np.concatenate(epoch_marg_ade_losses)
            epoch_marg_fde_losses = np.concatenate(epoch_marg_fde_losses)
            epoch_marg_mode_probs = np.concatenate(epoch_marg_mode_probs)
            train_minade_c = min_xde_K(epoch_marg_ade_losses, epoch_marg_mode_probs, K=self.config.num_modes)
            train_minfde_c = min_xde_K(epoch_marg_fde_losses, epoch_marg_mode_probs, K=self.config.num_modes)

            # Log train metrics

            if self.is_main_process:
                print("Train Marg. minADE_{}".format(self.config.num_modes), train_minade_c[0], "Train Marg. minFDE_{}".format(self.config.num_modes), train_minfde_c[0])
                wandb.log({"metrics/Train minADE_{}".format(self.config.num_modes): train_minade_c[0], "epoch": epoch})
                wandb.log({"metrics/Train minFDE_{}".format(self.config.num_modes): train_minfde_c[0], "epoch": epoch})

            if not self.config.only_cal_ego_future_loss_in_join_model:
                epoch_scene_ade_losses = np.concatenate(epoch_scene_ade_losses)
                epoch_scene_fde_losses = np.concatenate(epoch_scene_fde_losses)
                mode_probs = np.concatenate(epoch_mode_probs)

                train_sminade_c = min_xde_K(epoch_scene_ade_losses, mode_probs, K=self.config.num_modes)
                train_sminfde_c = min_xde_K(epoch_scene_fde_losses, mode_probs, K=self.config.num_modes)
                if self.is_main_process:
                    print("Train Scene minADE_{}".format(self.config.num_modes), train_sminade_c[0], "Train Scene minFDE_{}".format(self.config.num_modes), train_sminfde_c[0])

                    wandb.log({"metrics/Train Scene minADE_{}".format(self.config.num_modes): train_sminade_c[0], "epoch": epoch})
                    wandb.log({"metrics/Train Scene minFDE_{}".format(self.config.num_modes): train_sminfde_c[0], "epoch": epoch})

            # update learning rate
            self.optimizer_scheduler.step()
            if epoch % self.config.eval_every == 0:
                if self.is_main_process:
                    self.eval(epoch)
                    # self.save_model(epoch)
                    self.model.train()
                    print("Best minADE c", self.smallest_minade_k, "Best minFDE c", self.smallest_minfde_k)
                    wandb.log({"metrics/Best_minADE_{}".format(self.config.num_modes): self.smallest_minade_k, "epoch": epoch})
                    wandb.log({"metrics/Best_minFDE_{}".format(self.config.num_modes): self.smallest_minfde_k, "epoch": epoch})
                    dist.barrier()
                else:
                    dist.barrier()
        
        self.save_model(epoch=epoch)
        wandb.finish()

    def eval(self, epoch=None):
        self.model.eval()
        with torch.no_grad():
            val_marg_ade_losses = []
            val_marg_fde_losses = []
            val_marg_mode_probs = []
            val_scene_ade_losses = []
            val_scene_fde_losses = []
            val_mode_probs = []
            random_show_ith_image = torch.randint(0,len(self.val_loader), (1,))
            for i, data in enumerate(self.val_loader):
                ego_in, ego_out, agents_in, agents_out, context_img, agent_types = self._data_to_device(data)
                
                mask = None
                if self.config.stage == "pretrain":
                    ego_in, ego_out, agents_in, agents_out, mask = get_random_masked(ego_in, ego_out, agents_in, agents_out, self.config)
                elif self.config.stage == "conditional_prediction":
                    # padding zeros to the agent_in
                    ego_in, ego_out, agents_in, agents_out, mask = conditional_padding(ego_in, ego_out, agents_in, agents_out)
                    mask = None
                elif self.config.reformat: # padding zeros
                    ego_in, ego_out, agents_in, agents_out = reformat(ego_in, ego_out, agents_in, agents_out)

                

                # for interaction dataset, mask on the occluded, and only calculate the occluded object
                all_occluded = None
                if "Interaction" in self.config.dataset and self.config.task == "eval" and self.config.inter_occ == True:# and self.config.reformat == False:
                    time_agent_occluded = (agents_in[:,:, :, 10] == 1)      # [B, T_obs, A-1], will only calculate the history part
                    agent_occluded = torch.any(time_agent_occluded, dim=1)  # [B, A-1]
                    agents_in = agents_in *  ~time_agent_occluded.unsqueeze(-1)                 # hide the occluded moment by padding
                    # cat with ego
                    ego_occluded_mask = torch.zeros((ego_in.shape[0], 1)).to(self.device).bool() # set as 0, ego not occluded
                    all_occluded = torch.cat((ego_occluded_mask, agent_occluded), dim=1)
                    mask = ~all_occluded.unsqueeze(1)      # [B, 1, A]      # only calculate the hidden occluded agent's error, T->valid

                pred_obs, mode_probs = self.model.module(ego_in, agents_in, context_img, agent_types)

                if not self.config.only_cal_ego_future_loss_in_join_model:
                    if "Interaction" in self.config.dataset and self.config.task == "eval" and self.config.inter_occ == True and self.config.reformat:
                        # Marginal metrics
                        ade_losses, fde_losses = self._compute_marginal_errors(pred_obs, ego_out, agents_out, agents_in, mask, inter_cal_future=True, horizon=self.config.obs_horizon)
                        val_marg_ade_losses.append(ade_losses.reshape(-1, self.config.num_modes))
                        val_marg_fde_losses.append(fde_losses.reshape(-1, self.config.num_modes))
                        val_marg_mode_probs.append(
                            mode_probs.unsqueeze(1).repeat(1, self.config.num_other_agents + 1, 1).detach().cpu().numpy().reshape(
                                -1, self.config.num_modes))

                        # Joint metrics
                        scene_ade_losses, scene_fde_losses = self._compute_joint_errors(pred_obs, ego_out, agents_out, mask)#, inter_cal_future=True, horizon=self.config.obs_horizon)
                        val_scene_ade_losses.append(scene_ade_losses)
                        val_scene_fde_losses.append(scene_fde_losses)
                        val_mode_probs.append(mode_probs.detach().cpu().numpy())
                    else:

                        # Marginal metrics
                        ade_losses, fde_losses = self._compute_marginal_errors(pred_obs, ego_out, agents_out, agents_in, mask)
                        val_marg_ade_losses.append(ade_losses.reshape(-1, self.config.num_modes))
                        val_marg_fde_losses.append(fde_losses.reshape(-1, self.config.num_modes))
                        val_marg_mode_probs.append(
                            mode_probs.unsqueeze(1).repeat(1, self.config.num_other_agents + 1, 1).detach().cpu().numpy().reshape(
                                -1, self.config.num_modes))

                        # Joint metrics
                        scene_ade_losses, scene_fde_losses = self._compute_joint_errors(pred_obs, ego_out, agents_out, mask)
                        val_scene_ade_losses.append(scene_ade_losses)
                        val_scene_fde_losses.append(scene_fde_losses)
                        val_mode_probs.append(mode_probs.detach().cpu().numpy())
                else:
                    # only care about the ego for (pred_obs.sahpe[2]=12)
                    if not self.config.reformat:
                        ade_losses, fde_losses = self._compute_ego_errors(pred_obs[:,:,:,0,:], ego_out)
                    else:# only care about the ego future for (pred_obs.sahpe[2]=16)
                        ade_losses, fde_losses = self._compute_ego_errors(pred_obs[:,self.config.obs_horizon:,:,0,:], ego_out[:,self.config.obs_horizon:,:])
                    val_marg_ade_losses.append(ade_losses)
                    val_marg_fde_losses.append(fde_losses)
                    val_marg_mode_probs.append(mode_probs.detach().cpu().numpy())

                

            val_marg_ade_losses = np.concatenate(val_marg_ade_losses)
            val_marg_fde_losses = np.concatenate(val_marg_fde_losses)
            val_marg_mode_probs = np.concatenate(val_marg_mode_probs)
            val_minade_c = min_xde_K(val_marg_ade_losses, val_marg_mode_probs, K=self.config.num_modes)
            val_minfde_c = min_xde_K(val_marg_fde_losses, val_marg_mode_probs, K=self.config.num_modes)
            # Log train metrics
            wandb.log({"metrics/Val minADE_{}".format(self.config.num_modes): val_minade_c[0], "epoch": epoch})
            wandb.log({"metrics/Val minFDE_{}".format(self.config.num_modes): val_minfde_c[0], "epoch": epoch})
            print("Marg. minADE_{}".format(self.config.num_modes), val_minade_c[0], "Marg. minFDE_{}".format(self.config.num_modes), val_minfde_c[0])

            if not self.config.only_cal_ego_future_loss_in_join_model:
                val_scene_ade_losses = np.concatenate(val_scene_ade_losses)
                val_scene_fde_losses = np.concatenate(val_scene_fde_losses)
                val_mode_probs = np.concatenate(val_mode_probs)
                val_sminade_c = min_xde_K(val_scene_ade_losses, val_mode_probs, K=self.config.num_modes)
                val_sminfde_c = min_xde_K(val_scene_fde_losses, val_mode_probs, K=self.config.num_modes)
                # Log train metrics
                wandb.log({"metrics/Val Scene minADE_{}".format(self.config.num_modes): val_sminade_c[0], "epoch": epoch})
                wandb.log({"metrics/Val Scene minFDE_{}".format(self.config.num_modes): val_sminfde_c[0], "epoch": epoch})
                print("Scene minADE_{}".format(self.config.num_modes), val_sminade_c[0], "Scene minFDE_{}".format(self.config.num_modes), val_sminfde_c[0])

            self.model.train()
            if self.config.task == "train":
                if not self.config.only_cal_ego_future_loss_in_join_model:
                    self.save_model(minade_k=val_sminade_c[0], minfde_k=val_sminfde_c[0], epoch=epoch)
                else:
                    self.save_model(minade_k=val_minade_c[0], minfde_k=val_minfde_c[0], epoch=epoch)



    def save_model(self, epoch=0, minade_k=None, minfde_k=None):
        checkpoint = {
                "AutoBot": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
            }
        if minade_k is not None and minfde_k is not None:
            if minade_k < self.smallest_minade_k:
                self.smallest_minade_k = minade_k
                torch.save(checkpoint, osp.join(self.model_dir, f"{self.config.dataset}_best_models_ade{epoch}.pt"))

            if minfde_k < self.smallest_minfde_k:
                self.smallest_minfde_k = minfde_k
                torch.save(checkpoint, osp.join(self.model_dir, f"{self.config.dataset}_best_models_fde{epoch}.pt"))
        elif epoch is not 0:
            torch.save(checkpoint, osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"))
        else:
            raise NotImplementedError

    def _build(self):
        self._build_dir()
        self._build_data_loader()
        self._build_model()
        print("> Everything built. Have fun :)")
    
    def _build_dir(self):
        if self.is_main_process:
            self.model_dir = osp.join("./experiments", self.config.exp_name, self.config.slurm)
            print(self.model_dir)
            os.makedirs(self.model_dir, exist_ok=True)

            # Copy the configuration file to the result directory
            shutil.copy(self.config.config_file, self.model_dir)

            # initalize wandb board
            run = wandb.init(project="MAE", entity="iyuner", name=(self.config.slurm+"_"+self.config.exp_name), config=self.config, mode=self.config.wandb_mode)
            self.config.learning_rate = run.config.learning_rate
            print("> Directory built!")
    
    def _build_data_loader(self):
        if "Argoverse" in self.config.dataset:
            train_dset = ArgoH5Dataset(dset_path=self.config.dataset_path, split_name="train",
                                       use_map_lanes=self.config.use_map_lanes, model_type=self.config.model_type)
            val_dset = ArgoH5Dataset(dset_path=self.config.dataset_path, split_name="val",
                                     use_map_lanes=self.config.use_map_lanes, model_type=self.config.model_type)
        
        elif "Nuscenes" in self.config.dataset:
            train_dset = NuscenesH5Dataset(dset_path=self.config.dataset_path, split_name="train",
                                           model_type=self.config.model_type, use_map_img=self.config.use_map_image,
                                           use_map_lanes=self.config.use_map_lanes)
            val_dset = NuscenesH5Dataset(dset_path=self.config.dataset_path, split_name="val",
                                         model_type=self.config.model_type, use_map_img=self.config.use_map_image,
                                         use_map_lanes=self.config.use_map_lanes)
        elif "Interaction" in self.config.dataset:
            if self.config.task == "train":
                assert self.config.inter_occ == False
            if self.config.task == "eval":
                assert self.config.inter_occ == True

            train_dset = InteractionDataset(dset_path=self.config.dataset_path, split_name="train",
                                            use_map_lanes=self.config.use_map_lanes, evaluation=False, same_scale=self.config.same_scale)   # don't downsampling
            val_dset = InteractionDataset(dset_path=self.config.dataset_path, split_name="val",
                                          use_map_lanes=self.config.use_map_lanes, evaluation=False, same_scale=self.config.same_scale, occ=self.config.inter_occ)    # don't downsampling
            if self.config.same_scale:
                train_dset.obs_horizon = 10
                train_dset.pred_horizon = 30
            else:
                train_dset.obs_horizon = 5
                train_dset.pred_horizon = 15
        else:
            raise NotImplementedError
        
        
        self.config.num_other_agents = train_dset.num_others
        self.config.obs_horizon = train_dset.obs_horizon

        if self.config.stage == "pretrain" or self.config.stage == "conditional_prediction" or self.config.reformat:
            self.config.pred_horizon = train_dset.pred_horizon + train_dset.obs_horizon
        else:
            self.config.pred_horizon = train_dset.pred_horizon
        
        self.config.k_attr = train_dset.k_attr
        self.config.map_attr = train_dset.map_attr
        self.config.predict_yaw = train_dset.predict_yaw
        if "Joint" in self.config.model_type:
            self.config.num_agent_types = train_dset.num_agent_types

        train_sampler = DistributedSampler(train_dset, shuffle=True)
        self.train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=self.config.batch_size, num_workers=12, drop_last=True, pin_memory=True, sampler=train_sampler,
        )
        val_sampler = DistributedSampler(val_dset, shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(
            val_dset, batch_size=self.config.batch_size, num_workers=12, drop_last=True, pin_memory=True, sampler=val_sampler,
        )

        print("Train dataset loaded with length", len(train_dset))
        print("Val dataset loaded with length", len(val_dset))

    def _build_model(self):
        model = AutoBot(self.config).to(self.device)
        model = DDP(model, device_ids=[self.config.local_rank], output_device=self.config.local_rank, find_unused_parameters=True)
        self.model = model#.cuda()

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=float(self.config.learning_rate), eps=self.config.adam_epsilon)
        self.optimizer_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.learning_rate_sched, gamma=0.5,
                                               verbose=True)

        # check whehter need to load saved checkpoint
        if self.config.task == "eval" or self.config.resume:
            ckpt = torch.load(self.config.ckpt_path, map_location=lambda storage, loc: storage)
            if self.config.frozen_encoder:
                self.model.load_state_dict(ckpt['AutoBot']) #, strict = False)
                for param in self.model.module.encoder.parameters():
                    param.requires_grad = False
            else:
                self.model.load_state_dict(ckpt['AutoBot']) #, strict = False)

            # load the optimizer and epoch status if it's keeping training
            if self.config.task == "train" and self.config.load_opz_epoch:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.start_epoch = ckpt["epoch"]
        
        if self.is_main_process and self.config.task == "train":
            wandb.watch(self.model)
        print("> Model and optimizer built!")
    
    def _data_to_device(self, data):
        ego_in, ego_out, agents_in, agents_out, context_img, agent_types = data
        ego_in = ego_in.float().to(self.device)
        ego_out = ego_out.float().to(self.device)
        agents_in = agents_in.float().to(self.device)
        agents_out = agents_out.float().to(self.device)
        context_img = context_img.float().to(self.device)
        agent_types = agent_types.float().to(self.device)
        return ego_in, ego_out, agents_in, agents_out, context_img, agent_types #, original_roads

    def _compute_marginal_errors(self, preds, ego_gt, agents_gt, agents_in, mask=None, inter_cal_future=False, horizon=None):
        agents_gt = torch.cat((ego_gt.unsqueeze(2), agents_gt), dim=2)
        agent_masks = agents_gt[:, :, :, -1]
        if mask is not None:
            agent_masks = agent_masks * ~(mask.squeeze(-1)) # calculate the hidden parts
        agent_masks[agent_masks == 0] = float('nan')
        agents_gt = agents_gt.unsqueeze(0).permute(0, 2, 1, 3, 4)
        if inter_cal_future:
            error = torch.norm(preds[:, horizon:, :, :, :2] - agents_gt[:, horizon:, :, :, :2], 2, dim=-1) * agent_masks[:,horizon:,:].permute(1,0,2)
        else:
            error = torch.norm(preds[:, :, :, :, :2] - agents_gt[:, :, :, :, :2], 2, dim=-1) * agent_masks.permute(1,0,2)
        ade_losses = np.nanmean(error.cpu().numpy(), axis=1).transpose(1, 2, 0)
        fde_losses = error[:, -1].cpu().numpy().transpose(1, 2, 0)
        return ade_losses, fde_losses
    
    def _compute_joint_errors(self, preds, ego_gt, agents_gt, mask=None):
        agents_gt = torch.cat((ego_gt.unsqueeze(2), agents_gt), dim=2)
        agents_masks = agents_gt[:, :, :, -1]
        if mask is not None:
            agents_masks = agents_masks * ~(mask.squeeze(-1)) # calculate the hidden parts
        agents_masks[agents_masks == 0] = float('nan')
        ade_losses = []
        for k in range(self.config.num_modes):
            ade_error = (torch.norm(preds[k, :, :, :, :2].transpose(0, 1) - agents_gt[:, :, :, :2], 2, dim=-1)
                         * agents_masks).cpu().numpy()
            ade_error = np.nanmean(ade_error, axis=(1, 2))
            ade_losses.append(ade_error)
        ade_losses = np.array(ade_losses).transpose()

        fde_losses = []
        for k in range(self.config.num_modes):
            fde_error = (torch.norm(preds[k, -1, :, :, :2] - agents_gt[:, -1, :, :2], 2, dim=-1) * agents_masks[:, -1]).cpu().numpy()
            fde_error = np.nanmean(fde_error, axis=1)
            fde_losses.append(fde_error)
        fde_losses = np.array(fde_losses).transpose()

        return ade_losses, fde_losses
    
    def _compute_ego_errors(self, ego_preds, ego_gt):
        ego_gt = ego_gt.transpose(0, 1).unsqueeze(0)
        ade_losses = torch.mean(torch.norm(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2], 2, dim=-1), dim=1).transpose(0, 1).cpu().numpy()
        fde_losses = torch.norm(ego_preds[:, -1, :, :2] - ego_gt[:, -1, :, :2], 2, dim=-1).transpose(0, 1).cpu().numpy()
        return ade_losses, fde_losses

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='')
    parser.add_argument("--slurm", type=str, default="", help="Experiment identifier")
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    # add args parameter into config
    for k, v in vars(args).items():
       config[k] = v
    config["exp_name"] = args.config_file.split("/")[-1].split(".")[0]
    config = EasyDict(config)

    agent = Trainer(config)
    
    if config.task == "train":
        agent.train()
    elif config.task == "eval":
        agent.eval()
    else:
        raise NotImplementedError

