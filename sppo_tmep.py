import numpy as np
import math
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from typing import List, Tuple, Dict, Any, Optional

# For environment spaces, typically from a library like Gymnasium
# For this standalone example, we'll define simple space-like objects
class SimpleSpace:
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

class SimpleDiscreteSpace:
    def __init__(self, n: int):
        self.n = n
        self.shape = () # For discrete, shape is often empty tuple

# --- Placeholder for device ---
device = torch.device('cpu')
try:
    device
except NameError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device not defined, using {device}. Define 'device' globally for specific GPU/CPU choice.")

# --- Gaussian Process and Safe State Manager (from previous discussions) ---
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C_kernel, WhiteKernel
# To suppress specific warnings if necessary
import warnings
from sklearn.exceptions import ConvergenceWarning


class GaussianProcessSafetyEstimator:
    def __init__(self,
                 state_dim: int,
                 length_scale: float = 1.0,
                 signal_variance: float = 1.0,
                 noise_level_gp: float = 1e-2, # Renamed from noise_level to avoid conflict
                 random_state: Optional[int] = None):
        kernel = C_kernel(constant_value=signal_variance, constant_value_bounds="fixed") \
                 * RBF(length_scale=length_scale, length_scale_bounds="fixed") \
                 + WhiteKernel(noise_level=noise_level_gp, noise_level_bounds="fixed")
        
        self.gp_model = GaussianProcessRegressor(kernel=kernel,
                                                 normalize_y=True,
                                                 random_state=random_state,
                                                 optimizer=None, # Explicitly disable optimizer
                                                 n_restarts_optimizer=0) # Redundant but harmless
        self.observed_states: List[np.ndarray] = []
        self.observed_safety_values: List[float] = []
        self._is_fitted: bool = False
        self.state_dim = state_dim # Store state_dim for reshaping if necessary

    def update(self, state: np.ndarray, safety_value_m_s: float):
        self.observed_states.append(state.flatten()) # Ensure state is 1D for list of states
        self.observed_safety_values.append(safety_value_m_s)

        if len(self.observed_states) > 0:
            X = np.array(self.observed_states)
            if X.ndim == 1 and self.state_dim == 1: # Ensure X is 2D if state_dim is 1
                 X = X.reshape(-1, 1)
            elif X.ndim == 1 and self.state_dim > 1:
                 # This case implies states were not correctly shaped before appending if state_dim > 1
                 # Assuming states are already correctly flattened to 1D before appending,
                 # np.array(self.observed_states) should produce a 2D array.
                 print(f"Warning: GP received flattened 1D array for X with state_dim {self.state_dim}, reshaping. Check state shapes.")
                 X = X.reshape(len(self.observed_states), self.state_dim)


            y = np.array(self.observed_safety_values)
            try:
                # Suppress convergence warnings specifically during fitting if they persist and are understood
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.gaussian_process")
                    self.gp_model.fit(X, y)
                self._is_fitted = True
            except Exception as e:
                print(f"Warning: GP fitting failed. Error: {e}. Number of samples: {len(X) if isinstance(X, np.ndarray) else 0}")
                self._is_fitted = False
        else:
            self._is_fitted = False

    def predict(self, state: np.ndarray) -> Tuple[float, float]:
        if not self._is_fitted:
            prior_mean = 0.0
            try:
                # Attempt to access nested kernel attributes for signal variance
                if hasattr(self.gp_model.kernel_, 'k1') and hasattr(self.gp_model.kernel_.k1, 'k1') and hasattr(self.gp_model.kernel_.k1.k1, 'constant_value'): # C() * RBF()
                    prior_variance = self.gp_model.kernel_.k1.k1.constant_value
                elif hasattr(self.gp_model.kernel_, 'k1') and hasattr(self.gp_model.kernel_.k1, 'constant_value'): # C()
                     prior_variance = self.gp_model.kernel_.k1.constant_value
                else: # Fallback for other kernel structures or uninitialized
                    prior_variance = 1.0
            except AttributeError:
                 prior_variance = 1.0 # General fallback
            return prior_mean, prior_variance

        s_reshaped = state.flatten().reshape(1, -1)
        try:
            mean, std_dev = self.gp_model.predict(s_reshaped, return_std=True)
            return mean[0], std_dev[0]**2
        except Exception as e:
            print(f"Warning: GP prediction failed for state {state}. Error: {e}")
            prior_mean = 0.0
            try:
                if hasattr(self.gp_model.kernel_, 'k1') and hasattr(self.gp_model.kernel_.k1, 'k1') and hasattr(self.gp_model.kernel_.k1.k1, 'constant_value'):
                    prior_variance = self.gp_model.kernel_.k1.k1.constant_value
                elif hasattr(self.gp_model.kernel_, 'k1') and hasattr(self.gp_model.kernel_.k1, 'constant_value'):
                     prior_variance = self.gp_model.kernel_.k1.constant_value
                else:
                    prior_variance = 1.0
            except AttributeError:
                 prior_variance = 1.0
            return prior_mean, prior_variance


    def calculate_lower_bound(self, state: np.ndarray, beta_t_sqrt: float, t_step: int,
                              previous_l_value_for_state: Optional[float] = None) -> float:
        predicted_mean_safety, predicted_variance = self.predict(state)
        sigma_s = math.sqrt(max(1e-9, predicted_variance))

        current_lower_estimate = predicted_mean_safety - beta_t_sqrt * sigma_s

        if t_step == 1 or previous_l_value_for_state is None:
            return current_lower_estimate
        else:
            return max(previous_l_value_for_state, current_lower_estimate)

class SafeStateSetManager:
    def __init__(self,
                 safety_threshold_h: float,
                 neighborhood_radius_v: float,
                 state_dim: int,
                 num_samples_per_axis_direction: int = 1,
                 max_neighborhood_points: int = 65): # Paper uses v for radius
        self.safety_threshold_h = safety_threshold_h
        self.neighborhood_radius_v = neighborhood_radius_v
        self.state_dim = state_dim
        self.num_samples_per_axis_direction = max(1, num_samples_per_axis_direction) # Ensure at least 1
        self.max_neighborhood_points = max_neighborhood_points
        self._l_value_tracker: Dict[Tuple[float, ...], float] = {}
        self._t_step_tracker: Dict[Tuple[float, ...], int] = {} # Tracks 't' when l_value was computed for a state

    def _sample_neighborhood_I_hat(self, state: np.ndarray) -> List[np.ndarray]:
        state_flat = state.flatten()
        # Always include the state itself
        neighborhood_tuples = {tuple(state_flat.tolist())}

        if self.neighborhood_radius_v > 1e-6 and self.num_samples_per_axis_direction > 0:
            for i in range(self.state_dim):
                for j in range(1, self.num_samples_per_axis_direction + 1):
                    # Sample points at different fractions of the radius along each axis
                    step_size = (self.neighborhood_radius_v * j) / self.num_samples_per_axis_direction
                    
                    s_plus = state_flat.copy()
                    s_plus[i] += step_size
                    neighborhood_tuples.add(tuple(s_plus.tolist()))

                    s_minus = state_flat.copy()
                    s_minus[i] -= step_size
                    neighborhood_tuples.add(tuple(s_minus.tolist()))
        
        sampled_points = [np.array(list(s_tuple)) for s_tuple in neighborhood_tuples]
        
        if len(sampled_points) > self.max_neighborhood_points:
            # If capped, ensure the original state_to_check is part of the sampled points
            original_state_tuple = tuple(state_flat.tolist())
            # Convert list of arrays to set of tuples for efficient checking
            current_sampled_tuples_for_check = {tuple(p.tolist()) for p in sampled_points}


            if original_state_tuple not in current_sampled_tuples_for_check:
                # This should ideally not happen if original state is added first to set
                # But as a fallback if random sampling is used:
                # Create a temporary list for np.random.choice
                temp_list_for_sampling = [p for p in sampled_points if not np.array_equal(p, state_flat)]
                
                num_to_sample = self.max_neighborhood_points -1
                if num_to_sample < 0: num_to_sample = 0 # Should not happen if max_neighborhood_points >=1

                if len(temp_list_for_sampling) > num_to_sample :
                    indices = np.random.choice(len(temp_list_for_sampling), num_to_sample , replace=False)
                    sampled_points_subset = [temp_list_for_sampling[i] for i in indices]
                else: # take all available if less than num_to_sample
                    sampled_points_subset = temp_list_for_sampling

                sampled_points_subset.append(state_flat.copy()) # Ensure original is in
                return sampled_points_subset
            else: # Original is already there, just sample from the full list
                indices = np.random.choice(len(sampled_points), self.max_neighborhood_points, replace=False)
                return [sampled_points[i] for i in indices]
        
        return sampled_points


    def is_state_safe_to_enter(self,
                               state_to_check: np.ndarray,
                               gp_estimator: GaussianProcessSafetyEstimator,
                               beta_t_sqrt: float,
                               current_gp_update_count: int # This is the global 't' for l_t
                              ) -> bool:
        neighborhood_states = self._sample_neighborhood_I_hat(state_to_check)
        if not neighborhood_states: return False # Should include at least state_to_check

        for s_neighbor in neighborhood_states:
            s_neighbor_tuple = tuple(s_neighbor.flatten().tolist())
            
            previous_l_value = self._l_value_tracker.get(s_neighbor_tuple, None)
            
            l_s_neighbor = gp_estimator.calculate_lower_bound(
                s_neighbor,
                beta_t_sqrt,
                t_step=current_gp_update_count, 
                previous_l_value_for_state=previous_l_value
            )
            
            self._l_value_tracker[s_neighbor_tuple] = l_s_neighbor
            self._t_step_tracker[s_neighbor_tuple] = current_gp_update_count

            if l_s_neighbor < self.safety_threshold_h:
                return False 
        return True 

    def update_state_safety_knowledge(self, state: np.ndarray, l_value: float, t_step: int):
        state_tuple = tuple(state.flatten().tolist())
        self._l_value_tracker[state_tuple] = l_value
        self._t_step_tracker[state_tuple] = t_step

# --- PPO Actor-Critic and Buffer ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        actor_layers = [
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        ]
        if has_continuous_action_space:
            actor_layers.append(nn.Linear(64, action_dim))
        else:
            actor_layers.append(nn.Linear(64, action_dim)) 
        self.actor = nn.Sequential(*actor_layers)
            
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("Warning: set_action_std called on discrete action space policy")

    def act(self, state): 
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            # Ensure action_var has same batch size as action_mean if state is batched
            current_action_var = self.action_var
            if action_mean.ndim > 1 and self.action_var.ndim == 1 :
                current_action_var = self.action_var.expand_as(action_mean)
            
            cov_mat = torch.diag_embed(current_action_var)
            if action_mean.ndim == 1: action_mean = action_mean.unsqueeze(0)
            if cov_mat.ndim == 2 and action_mean.ndim == 2 : cov_mat = cov_mat.unsqueeze(0)


            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_logits = self.actor(state)
            dist = Categorical(logits=action_logits) 

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action, action_logprob, state_val

    def evaluate(self, state, action): 
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var_expanded = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var_expanded)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            if action.shape != action_mean.shape:
                 if action.ndim == 1 and self.action_dim ==1: 
                     action = action.unsqueeze(-1)
        else:
            action_logits = self.actor(state)
            dist = Categorical(logits=action_logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

class RolloutBuffer:
    def __init__(self):
        self.actions: List[torch.Tensor] = []
        self.states: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.state_values: List[torch.Tensor] = []
        self.is_terminals: List[bool] = []
        self.safety_values_m_s_prime: List[float] = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.safety_values_m_s_prime[:]

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

        self.gp_safety_estimator = GaussianProcessSafetyEstimator(
            state_dim=self.state_dim,
            length_scale=gp_length_scale,
            signal_variance=gp_signal_variance,
            noise_level_gp=gp_noise_level 
        )
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
        for s, m_s in zip(H0_states, H0_safety_values):
            self.gp_safety_estimator.update(s, m_s)
        
        self.global_gp_update_count = 1 
        if self.verbose > 0: print("GP initialized with H0. Global GP update count set to 1.")


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
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards_mc_tensor) - 0.01 * dist_entropy.mean() # Ensure entropy is scalar
            
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
        
        if hasattr(self.env, 'get_initial_safe_set_H0'):
            H0_states_env, H0_safety_values_env = self.env.get_initial_safe_set_H0() 
            self.initialize_safety_from_H0(H0_states_env, H0_safety_values_env)
        elif self.verbose > 0:
            print("Warning: Environment does not have 'get_initial_safe_set_H0' method. SPPO safety relies on initial GP training.")

        state, info = self.env.reset() 
        if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)


        while current_total_timesteps_elapsed < total_timesteps:
            for i_step_in_buffer_collection in range(self.steps_per_epoch_for_buffer):
                current_total_timesteps_elapsed += 1

                action = self.select_action(state) 
                
                env_action = action
                if not self.has_continuous_action_space: 
                    env_action = action # .item() if action is a tensor, select_action returns item for discrete

                next_state_raw, reward, terminated, truncated, info = self.env.step(env_action)
                if not isinstance(next_state_raw, np.ndarray): next_state_raw = np.array(next_state_raw, dtype=np.float32)

                m_s_prime = info.get('safety_value_m_s_prime', 0.0) 

                if self.verbose > 1 and current_total_timesteps_elapsed % 1 == 0 :
                    print(f"Step {current_total_timesteps_elapsed}: s={state.round(2)}, a={action}, s'={next_state_raw.round(2)}, r={reward:.2f}, m(s')={m_s_prime:.2f}, term={terminated}, trunc={truncated}")

                self.global_gp_update_count += 1
                self.gp_safety_estimator.update(next_state_raw, m_s_prime)
                
                is_next_state_considered_safe = self.safe_state_manager.is_state_safe_to_enter(
                    next_state_raw,
                    self.gp_safety_estimator,
                    self.beta_t_sqrt_val, 
                    self.global_gp_update_count
                )

                actual_task_done = terminated or truncated 
                effective_done = actual_task_done
                
                if not is_next_state_considered_safe:
                    effective_done = True 
                    if self.verbose > 0: print(f"SAFETY INTERVENTION at step {current_total_timesteps_elapsed}: Next state s' {next_state_raw.round(2)} deemed unsafe. Effective_done=True.")
                
                self.buffer.rewards.append(float(reward))
                self.buffer.is_terminals.append(effective_done)
                self.buffer.safety_values_m_s_prime.append(m_s_prime)

                state = next_state_raw
                if effective_done:
                    state, info = self.env.reset() 
                    if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
                    if self.verbose > 1 : print(f"Environment reset. New state: {state.round(2)}")
                
                if self.has_continuous_action_space and current_total_timesteps_elapsed % self.action_std_decay_freq == 0:
                    self.decay_action_std()
                
                if current_total_timesteps_elapsed >= total_timesteps:
                    break
            
            self.update()
            if self.verbose > 0 and (current_total_timesteps_elapsed // self.steps_per_epoch_for_buffer) % 10 == 0 :
                 print(f"PPO Update performed after {current_total_timesteps_elapsed} timesteps. Buffer size: {len(self.buffer.rewards)}")

            if current_total_timesteps_elapsed >= total_timesteps:
                break
        
        if self.verbose > 0: print(f"SPPO training finished after {current_total_timesteps_elapsed} timesteps.")
        if hasattr(self.env, 'close'):
            self.env.close()

# --- Main function (modified from user's query) ---
def main():
    # Suppress ConvergenceWarnings from sklearn GP if they are persistent and understood
    # warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.gaussian_process")

    # Configuration
    config = {
        # Environment specific
        "state_dim": 2, 
        "action_dim": 2,
        "has_continuous_action_space": True, 
        # SPPO Safety parameters
        "safety_threshold_h": -0.8,       
        "neighborhood_radius_v": 0.5,     
        "beta_t_sqrt_val": 2.0,           
        "gp_length_scale": 4,           
        "gp_signal_variance": 1.0,        
        "gp_noise_level": 0.001,           
        # PPO specific params
        "action_std_init": 0.6,           
        "action_std_decay_rate": 0.005, # Slower decay for longer training   
        "min_action_std": 0.1,            
        "action_std_decay_freq": int(5e3),# Decay more frequently
        "lr_actor": 3e-4,                 
        "lr_critic": 1e-3,                
        "gamma_discount": 0.99,           
        "K_epochs": 10,                   
        "eps_clip": 0.2,                  
        # Training loop params
        "total_timesteps": 500 * 50,      # epochs * steps_per_epoch for a shorter test run
                                          # Original: 500 epochs * 500 steps/epoch = 250,000
        "steps_per_epoch_for_buffer": 500,
        "verbose_level": 2                
    }

    # Dummy Environment (Gymnasium-like API)
    class DummySafeReachEnv:
        def __init__(self, state_dim, action_dim, H0_states_list, H0_safety_list, max_episode_len=50):
            self.observation_space = SimpleSpace(shape=(state_dim,))
            self.action_space = SimpleSpace(shape=(action_dim,)) 
            
            self.initial_safe_states_H0 = [np.array(s, dtype=np.float32) for s in H0_states_list]
            # Ensure safety values are float
            self.initial_safe_safety_values_H0 = [float(m) for m in H0_safety_list] 
            
            self.current_state: np.ndarray = self.initial_safe_states_H0[0].copy()
            self.goal_pos = np.array([2.5, 5.0], dtype=np.float32)
            self.center_unsafe = np.array([10.0, 10.0], dtype=np.float32) # Center of unsafe region
            self.unsafe_radius = 3.0 # Radius of primary unsafe region
            self.max_steps_per_episode = max_episode_len 
            self.current_episode_steps = 0
            if config["verbose_level"] > 0: print("DummySafeReachEnv: Initialized.")

        def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
            if seed is not None:
                np.random.seed(seed) # For reproducibility of reset if needed
            
            idx = np.random.randint(len(self.initial_safe_states_H0))
            self.current_state = self.initial_safe_states_H0[idx].copy()
            self.current_episode_steps = 0
            return self.current_state, {} 

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
            # Assuming action from Tanh is in [-1, 1], scale it to a reasonable step size
            # Max step size of 1 as per paper's description for Safe-Reach
            # If action_dim is 2, action is [dx, dy]
            scaled_action = np.array(action, dtype=np.float32) * 1.0 

            self.current_state = np.clip(self.current_state + scaled_action, 0.0, 20.0)
            self.current_episode_steps += 1

            dist_to_goal_prev = np.linalg.norm(self.current_state - scaled_action - self.goal_pos)
            dist_to_goal_curr = np.linalg.norm(self.current_state - self.goal_pos)
            
            # Reward: moving to the goal point (distance difference) + punishment for movement
            reward = (dist_to_goal_prev - dist_to_goal_curr) # Positive if moved closer
            reward -= 0.05 * np.linalg.norm(scaled_action) # Punishment for movement length

            # Safety value m(s') for the current_state (which is s')
            # From paper Fig 3a:
            # Goal (2.5, 5.0) is yellow/green (safe, value ~1 to 2)
            # Start (17,11) is green (safe, value ~1 to 2)
            # Unsafe region is purple, roughly centered around (12.5, 7.5) or (10,10) with values ~-3
            # Let's define a more structured safety landscape based on these observations
            m_s_prime = 0.0
            # Unsafe region 1 (large purple)
            if 7.5 < self.current_state[0] < 17.5 and 0.0 < self.current_state[1] < 7.5:
                m_s_prime = -2.5 + (self.current_state[1]/7.5) # Gradient towards -2.5
            # Unsafe region 2 (smaller dark purple spot)
            elif 12.5 < self.current_state[0] < 17.5 and 12.5 < self.current_state[1] < 17.5:
                 m_s_prime = -1.5
            # Safe region near goal
            elif dist_to_goal_curr < 4.0:
                m_s_prime = 1.5 - 0.2 * dist_to_goal_curr # Higher closer to goal
            # Safe region near start
            elif np.linalg.norm(self.current_state - np.array([17.0, 11.0])) < 4.0:
                m_s_prime = 1.0
            else: # General area, make it slightly positive or neutral
                m_s_prime = 0.2
            m_s_prime += np.random.normal(0, 0.05) # Small noise as in paper's GP assumption

            terminated = dist_to_goal_curr < 0.5 
            if terminated:
                reward += 1000 

            truncated = self.current_episode_steps >= self.max_steps_per_episode
            
            info = {'safety_value_m_s_prime': float(m_s_prime)}
            
            return self.current_state.copy(), float(reward), terminated, truncated, info
            
        def get_initial_safe_set_H0(self) -> Tuple[List[np.ndarray], List[float]]:
             # For H0, calculate true safety value for the initial state(s)
             H0_true_safety_values = []
             for s0 in self.initial_safe_states_H0:
                m_s0 = 0.0
                dist_to_goal_s0 = np.linalg.norm(s0 - self.goal_pos)
                if 7.5 < s0[0] < 17.5 and 0.0 < s0[1] < 7.5: m_s0 = -2.5 + (s0[1]/7.5)
                elif 12.5 < s0[0] < 17.5 and 12.5 < s0[1] < 17.5: m_s0 = -1.5
                elif dist_to_goal_s0 < 4.0: m_s0 = 1.5 - 0.2 * dist_to_goal_s0
                elif np.linalg.norm(s0 - np.array([17.0, 11.0])) < 4.0: m_s0 = 1.0
                else: m_s0 = 0.2
                H0_true_safety_values.append(m_s0)
             return self.initial_safe_states_H0, H0_true_safety_values

        def close(self):
            if config["verbose_level"] > 0: print("DummySafeReachEnv closed.")

    # --- Initialize Environment ---
    initial_H0_states_list = [np.array([17.0, 11.0])] 
    # The safety value for H0 points should be their true m(s), env will calculate this
    env = DummySafeReachEnv(
        state_dim=config["state_dim"], 
        action_dim=config["action_dim"],
        H0_states_list=initial_H0_states_list,
        H0_safety_list=[], # Dummy, get_initial_safe_set_H0 will provide true values
        max_episode_len=config.get("max_episode_length", 50) # Get from paper Table 2
    )

    # --- Instantiate SPPO Agent ---
    agent = SPPO(
        env=env,
        lr_actor=config["lr_actor"],
        lr_critic=config["lr_critic"],
        gamma=config["gamma_discount"],
        K_epochs=config["K_epochs"],
        eps_clip=config["eps_clip"],
        has_continuous_action_space=config["has_continuous_action_space"],
        action_std_init=config["action_std_init"],
        action_std_decay_rate=config["action_std_decay_rate"],
        min_action_std=config["min_action_std"],
        action_std_decay_freq=config["action_std_decay_freq"],
        safety_threshold_h=config["safety_threshold_h"],
        neighborhood_radius_v=config["neighborhood_radius_v"],
        beta_t_sqrt_val=config["beta_t_sqrt_val"],
        gp_length_scale=config["gp_length_scale"],
        gp_signal_variance=config["gp_signal_variance"],
        gp_noise_level=config["gp_noise_level"],
        steps_per_epoch_for_buffer=config["steps_per_epoch_for_buffer"],
        verbose=config["verbose_level"],
        config_dict=config 
    )

    # --- Start Learning ---
    agent.learn(total_timesteps=config["total_timesteps"])

    print("Training finished.")

if __name__ == "__main__":
    main()
