"""
@author: Arman

This script evaluates the RL baselines algorithms PPO_mini. 

"""

import torch
import torch.nn as nn
import math
import numpy as np
import time

from util import GaussianProcessSafetyEstimatorGPyTorch, SafeStateSetManager
from ActorCritic import RolloutBuffer, ActorCritic
from typing import List, Tuple, Optional, Dict, Any
import gpytorch


################################## set device ##################################
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))


# --- SPPO Class with Safety Integration ---
class SPPO:
    def __init__(self, env, 
                 lr_actor: float = 3e-4, 
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99, 
                 K_epochs: int = 10,  
                 eps_clip: float = 0.2,
                 has_continuous_action_space: bool = False, 
                 action_std_init: float =0.6,
                 action_std_decay_rate: float = 0.05,
                 min_action_std: float = 0.1,
                 action_std_decay_freq: int = int(2.5e5), # Corrected int conversion
                 safety_threshold_h: float = -0.8,
                 neighborhood_radius_v: float = 0.5,
                 beta_t_sqrt_val: float = 2.0, 
                 gp_length_scale: float = 1.0,
                 gp_signal_variance: float = 1.0,
                 gp_noise_level: float = 1e-2, 
                 steps_per_epoch_for_buffer: int = 500, 
                 verbose: int = 0,
                 config_dict: Optional[Dict] = None 
                 ):

        self.is_sppo = config_dict.get("SPPO", False)
        self.env = env
        self.config_dict = config_dict if config_dict is not None else {}

        self.has_continuous_action_space = has_continuous_action_space
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.steps_per_epoch_for_buffer = steps_per_epoch_for_buffer
        self.verbose = verbose

        self.buffer = RolloutBuffer()

        self.state_dim = self.env.observation_space.shape[0]
        if self.has_continuous_action_space:
            self.action_dim = self.env.action_space.shape[0]
            self.action_std = action_std_init
            self.action_std_init = action_std_init
            self.action_std_decay_rate = action_std_decay_rate
            self.min_action_std = min_action_std
            self.action_std_decay_freq = action_std_decay_freq
        else:
            self.action_dim = self.env.action_space.n

        self.policy = ActorCritic(self.state_dim, self.action_dim, self.has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
                    ])
        self.policy_old = ActorCritic(self.state_dim, self.action_dim, self.has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())       
        self.MseLoss = nn.MSELoss()

        if self.is_sppo:

            self.gp_safety_estimator = GaussianProcessSafetyEstimatorGPyTorch(
                    state_dim=self.state_dim,
                    num_inducing_points=config_dict.get("gp_num_inducing_points", 100), # New config
                    initial_lengthscale=gp_length_scale,
                    initial_signal_variance=gp_signal_variance,
                    initial_noise_level=gp_noise_level,
                    learning_rate_gp=config_dict.get("gp_lr", 0.01),
                    num_gp_opt_iters=config_dict.get("gp_iters", 20),
                    inducing_points_init_method = config_dict.get("inducing_points_init_method", "random_subset"), # or "kmeans" or provide tensor
                    gp_training_batch_size = config_dict.get("gp_training_batch_size", 128),
                    verbose= (self.verbose == -1) # Pass verbose flag
            )
            self.gp_training_losses: List[float] = []
            self.safety_threshold_h = safety_threshold_h
            self.safe_state_manager = SafeStateSetManager(
                safety_threshold_h=safety_threshold_h,
                neighborhood_radius_v=neighborhood_radius_v,
                state_dim=self.state_dim
            )
            self.beta_t_sqrt_val = beta_t_sqrt_val
            self.global_gp_update_count = 0

    def initialize_safety_from_H0(self, H0_states: List[np.ndarray], H0_safety_values: List[float]):
        if not H0_states:
            if self.verbose > 0: print("Warning: H0 is empty. GP cannot be initialized. SPPO safety might not work correctly.")
            return
        if self.verbose > 0: print(f"Initializing GP with {len(H0_states)} points from H0...")
        
        # Convert all H0 data to tensors first
        self.gp_safety_estimator.train_x_tensor = torch.tensor(np.array(H0_states), dtype=torch.float32, device=device)
        self.gp_safety_estimator.train_y_tensor = torch.tensor(np.array(H0_safety_values), dtype=torch.float32, device=device)

        # Now initialize model and likelihood (this will call _initialize_inducing_points)
        self.gp_safety_estimator._initialize_model_and_likelihood() # This will use all H0_states for inducing point selection if needed
               
        # Then, perform an initial training on this H0 data
        if self.gp_safety_estimator.model is not None and self.gp_safety_estimator.train_x_tensor.shape[0] > 0:
            self.gp_safety_estimator.model.train()
            self.gp_safety_estimator.likelihood.train()
            optimizer = torch.optim.Adam(self.gp_safety_estimator.model.parameters(), lr=self.gp_safety_estimator.lr_gp)
            elbo = gpytorch.mlls.VariationalELBO(self.gp_safety_estimator.likelihood, self.gp_safety_estimator.model, num_data=self.gp_safety_estimator.train_x_tensor.size(0))
            initial_losses = []
            for _ in range(self.config_dict.get("gp_init_iters")): # Use more iters for initial fit
                optimizer.zero_grad()
                output = self.gp_safety_estimator.model(self.gp_safety_estimator.train_x_tensor)
                loss = -elbo(output, self.gp_safety_estimator.train_y_tensor)
                if torch.isnan(loss): break
                loss.backward()
                optimizer.step()
                initial_losses.append(loss.item())
            print(f"GP Initial H0 fit avg loss: {np.mean(initial_losses):.3f} over {len(initial_losses)} iters")
            self.gp_safety_estimator._is_initialized_and_trained = True
        
        self.global_gp_update_count = 1 
        print("GP initialized with H0. Global GP update count set to 1.")


    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        elif self.verbose > 0:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling SPPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self):
        if self.has_continuous_action_space:
            if self.verbose > 0: print("--------------------------------------------------------------------------------------------")
            self.action_std = self.action_std - self.action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= self.min_action_std):
                self.action_std = self.min_action_std
                if self.verbose > 0: print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                if self.verbose > 0: print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)
            if self.verbose > 0: print("--------------------------------------------------------------------------------------------")
        elif self.verbose > 0:
            print("WARNING : Calling SPPO::decay_action_std() on discrete action space policy")
    
    def select_action(self, state: np.ndarray) -> Any:
        
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        state_tensor = torch.FloatTensor(state.copy()).to(device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            action_tensor, action_logprob_tensor, state_val_tensor = self.policy_old.act(state_tensor)

        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action_tensor)
        self.buffer.logprobs.append(action_logprob_tensor)
        self.buffer.state_values.append(state_val_tensor)

        if self.has_continuous_action_space:
            return action_tensor.detach().cpu().numpy().flatten()
        else:
            return action_tensor.item()


    def update(self):
        
        rewards_mc = []
        discounted_reward = 0
        
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_mc.insert(0, discounted_reward)
            
        rewards_mc_tensor = torch.tensor(np.array(rewards_mc).flatten(), dtype=torch.float32).to(device)
        if len(rewards_mc_tensor) > 1: 
            rewards_mc_tensor = (rewards_mc_tensor - rewards_mc_tensor.mean()) / (rewards_mc_tensor.std() + 1e-7)
        elif len(rewards_mc_tensor) == 1 and rewards_mc_tensor.std().item() < 1e-7 : # single item, std is 0
             rewards_mc_tensor = (rewards_mc_tensor - rewards_mc_tensor.mean()) / (1e-7)
        elif len(rewards_mc_tensor) == 1 : # single item, std might not be 0 if it's already a tensor from somewhere
             rewards_mc_tensor = (rewards_mc_tensor - rewards_mc_tensor.mean()) / (rewards_mc_tensor.std() + 1e-7)


        old_states = torch.cat(self.buffer.states, dim=0).detach()
        old_actions = torch.cat(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.cat(self.buffer.logprobs, dim=0).detach()
        
        old_state_values_squeezed = torch.cat(self.buffer.state_values, dim=0).detach().squeeze()
        if old_state_values_squeezed.ndim == 0: old_state_values_squeezed = old_state_values_squeezed.unsqueeze(0)


        advantages = rewards_mc_tensor.detach() - old_state_values_squeezed.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        elif len(advantages) == 1 and advantages.std().item() < 1e-7:
            advantages = (advantages - advantages.mean()) / (1e-8)
        elif len(advantages) == 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards_mc_tensor) - 0.001 * dist_entropy.mean() # Ensure entropy is scalar
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    
    def learn(self, total_timesteps: int = 10000):
        if self.verbose > 0: print(f"Starting SPPO training for {total_timesteps} timesteps...")
        
        current_total_timesteps_elapsed = 0
        if self.is_sppo:
            state = self.env.reset() 
            if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
            if state.ndim > self.state_dim and state.shape[0] == 1: # e.g. shape (1, 15) instead of (15,)
                state = state[0]
            
            _env_to_check = self.env
            if hasattr(_env_to_check, 'envs') and isinstance(_env_to_check.envs, list) and len(_env_to_check.envs) > 0:
            # It's a VecEnv, get the first (and likely only) underlying environment
                _env_to_check = _env_to_check.envs[0]
                if self.verbose > 0:
                    print(f"SPPO: Unwrapped VecEnv. Current env type: {type(_env_to_check)}")

            # Now check if this is a Monitor wrapper (common with stable-baselines3)
            if hasattr(_env_to_check, 'env') and not callable(getattr(_env_to_check, 'env')): # Check it's an attribute, not a method
                potential_inner_env = getattr(_env_to_check, 'env')
                if potential_inner_env is not _env_to_check: # Make sure it's actually an inner env
                    _env_for_H0_check = potential_inner_env
                    if self.verbose > 0:
                        print(f"SPPO: Unwrapped Monitor (or similar). Using env: {type(_env_for_H0_check)}")
                else:
                    _env_for_H0_check = _env_to_check # It was some other object with an 'env' attr pointing to self
            else:
                _env_for_H0_check = _env_to_check # Not a VecEnv, and not a Monitor-like wrapper with .env

            # Now check and call on the potentially deeply unwrapped environment
            if hasattr(_env_for_H0_check, 'get_initial_safe_set_H0'):
                if self.verbose > 0:
                    print(f"SPPO: Calling get_initial_safe_set_H0() on {_env_for_H0_check}")
                H0_states_env, H0_safety_values_env = _env_for_H0_check.get_initial_safe_set_H0(self.safety_threshold_h)
                self.initialize_safety_from_H0(H0_states_env, H0_safety_values_env)
            elif self.verbose > 0:
                print(f"Warning: Environment {type(_env_for_H0_check)} does not have 'get_initial_safe_set_H0' method. SPPO safety relies on initial GP training.")

        state = self.env.reset() 
        if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
        if state.ndim > self.state_dim and state.shape[0] == 1: # e.g. shape (1, 15) instead of (15,)
            state = state[0]

        while current_total_timesteps_elapsed < total_timesteps:
            reset_count = 0
            violation_count = 0
            cum_reward = 0
            start = time.perf_counter()
            for i_step_in_buffer_collection in range(self.steps_per_epoch_for_buffer):
                
                current_total_timesteps_elapsed += 1
                
                action = self.select_action(state) 
                env_action = action
                env_action = np.array([env_action])

                next_state_raw, reward, terminated, info = self.env.step(env_action)
                
                if not isinstance(next_state_raw, np.ndarray): next_state_raw = np.array(next_state_raw, dtype=np.float32)
                if next_state_raw.ndim > self.state_dim and next_state_raw.shape[0] == 1: # e.g. shape (1, 15) instead of (15,)
                    next_state_raw = next_state_raw[0]

                if isinstance(reward, (list, np.ndarray)): reward = reward[0]
                cum_reward += reward
                if isinstance(terminated, (list, np.ndarray)): terminated = terminated[0]
                actual_info_dict = info[0] if isinstance(info, list) and len(info)>0 else (info if isinstance(info, dict) else {})
                if self.is_sppo:
                    m_s_prime = actual_info_dict.get('safety_value_m_s_prime', 0.0)
                violation_count += actual_info_dict.get('total_violations')

                if not self.is_sppo and actual_info_dict.get('total_violations') > 0:
                    terminated = True
                
                if self.verbose > 1 and current_total_timesteps_elapsed % 1 == 0 :
                    if self.is_sppo:
                        print(f"Step {current_total_timesteps_elapsed}: s={state.round(2)}, a={action}, s'={next_state_raw.round(2)}, r={reward:.2f}, m(s')={m_s_prime:.2f}, term={terminated}")
                    else:
                        print(f"Step {current_total_timesteps_elapsed}: s={state.round(2)}, a={action}, s'={next_state_raw.round(2)}, r={reward:.2f}, term={terminated}")
                
                if self.is_sppo:
                    actual_task_done = terminated 
                    effective_done = actual_task_done
                    self.global_gp_update_count += 1                 
                    
                    avg_gp_loss = self.gp_safety_estimator.update(next_state_raw, m_s_prime)
                    
                    if avg_gp_loss is not None and not math.isnan(avg_gp_loss): # Check if loss is valid
                        self.gp_training_losses.append(avg_gp_loss)
                        if self.verbose == -1 and self.global_gp_update_count % self.steps_per_epoch_for_buffer == 0: # Log every 50 GP updates
                            print(f"GP Update {self.global_gp_update_count}: Avg. Loss = {avg_gp_loss:.4f}")
                    elif avg_gp_loss is not None and math.isnan(avg_gp_loss) and self.verbose == -1:
                        print(f"Warning: GP Update {self.global_gp_update_count} resulted in NaN loss.")
                    
                    is_next_state_considered_safe = self.safe_state_manager.is_state_safe_to_enter(
                        next_state_raw,
                        self.gp_safety_estimator,
                        self.beta_t_sqrt_val, 
                        self.global_gp_update_count
                    )
                    
                    if not is_next_state_considered_safe:
                        effective_done = True 
                        if self.verbose > 0: print(f"SAFETY INTERVENTION at step {current_total_timesteps_elapsed}: Next state s' {next_state_raw.round(2)} deemed unsafe. Effective_done=True.")
                
                    state = next_state_raw
                    if effective_done:
                        #state = self.env.reset() 
                        if hasattr(_env_for_H0_check, 'reset_to_safe_set_H0'):
                            state, _ = _env_for_H0_check.reset_to_safe_set_H0()
                        if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
                        state = state[0] if state.ndim > self.state_dim and state.shape[0] == 1 else state
                        if self.verbose > 0 : print(f"Environment reset. New state: {state.round(2)}")
                        reset_count += 1
                    self.buffer.safety_values_m_s_prime.append(m_s_prime)
                    self.buffer.rewards.append(float(reward))
                    self.buffer.is_terminals.append(effective_done)
                else:
                    self.buffer.rewards.append(reward)
                    self.buffer.is_terminals.append(terminated)
                
                if self.has_continuous_action_space and current_total_timesteps_elapsed % self.action_std_decay_freq == 0:
                    self.decay_action_std()
            

            self.update()

            end = time.perf_counter()   
            duration_ms = (end - start) * 1000

            if self.is_sppo:
                print(f"Step: {current_total_timesteps_elapsed:>5}, Cummulative Reward = {cum_reward:>6}, Reset times = {reset_count:>3}, Total violation = {violation_count:>3}, GP Loss = {avg_gp_loss:>.4f}, Duration = {duration_ms:>5.1f}")
            else:
                print(f"Step: {current_total_timesteps_elapsed:>6}, Cummulative Reward = {cum_reward:>6}, Total violation = {violation_count:>3}")
            if self.verbose > 0 and (current_total_timesteps_elapsed % self.steps_per_epoch_for_buffer) == 0 :
                print(f"PPO Update performed after {current_total_timesteps_elapsed} timesteps. Buffer size: {len(self.buffer.rewards)}")
        
        if self.verbose > 0: print(f"SPPO training finished after {current_total_timesteps_elapsed} timesteps.")
        if hasattr(self.env, 'close'):
            self.env.close()


