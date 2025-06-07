import numpy as np
import math
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from typing import List, Tuple, Dict, Any, Optional

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

import gpytorch

# 1. Define the GPyTorch VariationalGP Model
class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self,
                 inducing_points: torch.Tensor, # e.g., shape [num_inducing, state_dim]
                 state_dim: int,
                 initial_lengthscale: float = 1.0,
                 mean_module: Optional[gpytorch.means.Mean] = None,
                 covar_module: Optional[gpytorch.kernels.Kernel] = None):
        
        num_inducing_points = inducing_points.size(0)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(VariationalGPModel, self).__init__(variational_strategy)

        self.mean_module = mean_module if mean_module is not None else gpytorch.means.ConstantMean()
        
        if covar_module is not None:
            self.covar_module = covar_module
        else:
            rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=None if state_dim > 0 else 1)
            if state_dim > 0:
                rbf_kernel.lengthscale = torch.tensor([initial_lengthscale], device=inducing_points.device)
            self.covar_module = gpytorch.kernels.ScaleKernel(rbf_kernel)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessSafetyEstimatorGPyTorch: # Now using VariationalGP
    def __init__(self,
                 state_dim: int,
                 num_inducing_points: int = 100, # Number of inducing points for sparse GP
                 initial_lengthscale: float = 1.0,
                 initial_signal_variance: float = 1.0,
                 initial_noise_level: float = 1e-2,
                 learning_rate_gp: float = 0.01,
                 num_gp_opt_iters: int = 50,
                 inducing_points_init_method: str = "random_subset", # or "kmeans" or provide tensor
                 verbose: bool = False):
        
        self.state_dim = state_dim
        self.num_inducing_points = num_inducing_points
        self.initial_lengthscale = initial_lengthscale
        self.initial_signal_variance = initial_signal_variance
        self.initial_noise_level = initial_noise_level
        self.lr_gp = learning_rate_gp
        self.num_gp_opt_iters = num_gp_opt_iters
        self.inducing_points_init_method = inducing_points_init_method
        self.verbose = verbose

        self.likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self.model: Optional[VariationalGPModel] = None # Changed from ExactGPModel
        
        self.train_x_tensor: Optional[torch.Tensor] = None
        self.train_y_tensor: Optional[torch.Tensor] = None
        
        self._is_initialized_and_trained: bool = False
        self.inducing_points_tensor: Optional[torch.Tensor] = None


    def _initialize_inducing_points(self):
        """Initializes inducing points based on current training data."""
        if self.train_x_tensor is None or self.train_x_tensor.shape[0] == 0:
            # If no data, initialize randomly in some plausible range (e.g. based on state bounds if known)
            # This is a fallback; ideally, init with a subset of early data.
            # For now, we'll require some data before initializing them properly.
            print("Warning: Cannot initialize inducing points without some training data. Will attempt on first update.")
            return False

        num_data = self.train_x_tensor.shape[0]
        num_to_select = min(num_data, self.num_inducing_points)

        if self.inducing_points_init_method == "random_subset" or num_data <= self.num_inducing_points:
            indices = torch.randperm(num_data)[:num_to_select]
            self.inducing_points_tensor = self.train_x_tensor[indices].clone().to(device)
        elif self.inducing_points_init_method == "kmeans":
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=num_to_select, random_state=0, n_init='auto').fit(self.train_x_tensor.cpu().numpy())
                self.inducing_points_tensor = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
            except ImportError:
                print("Warning: sklearn.cluster.KMeans not found for inducing point init. Falling back to random_subset.")
                indices = torch.randperm(num_data)[:num_to_select]
                self.inducing_points_tensor = self.train_x_tensor[indices].clone().to(device)
        else: # Assume it's a pre-defined tensor if not a known method string
            if isinstance(self.inducing_points_init_method, torch.Tensor):
                self.inducing_points_tensor = self.inducing_points_init_method.clone().to(device)
            else:
                raise ValueError("Invalid inducing_points_init_method or provided tensor.")
        
        if self.verbose:
            print(f"Initialized {self.inducing_points_tensor.shape[0]} inducing points.")
        return True

    def _initialize_model_and_likelihood(self):
        if self.inducing_points_tensor is None:
            if not self._initialize_inducing_points():
                return # Cannot initialize model without inducing points

        assert self.inducing_points_tensor is not None, "Inducing points must be initialized."

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6) # Small positive noise
        ).to(device)
        self.likelihood.noise = torch.tensor(self.initial_noise_level**2, device=device)

        self.model = VariationalGPModel(
            inducing_points=self.inducing_points_tensor,
            state_dim=self.state_dim,
            initial_lengthscale=self.initial_lengthscale,
        ).to(device)
        
        if hasattr(self.model.covar_module, 'outputscale'):
             self.model.covar_module.outputscale = torch.tensor(self.initial_signal_variance, device=device)


    def update(self, state: np.ndarray, safety_value_m_s: float):
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        safety_value_tensor = torch.tensor([safety_value_m_s], dtype=torch.float32, device=device)

        if self.train_x_tensor is None or self.train_y_tensor is None:
            self.train_x_tensor = state_tensor
            self.train_y_tensor = safety_value_tensor
        else:
            self.train_x_tensor = torch.cat([self.train_x_tensor, state_tensor], dim=0)
            self.train_y_tensor = torch.cat([self.train_y_tensor, safety_value_tensor], dim=0)

        # Initialize or re-initialize inducing points if model is not yet created
        # or if we want to re-select inducing points periodically (more advanced)
        if self.model is None or self.likelihood is None:
            self._initialize_model_and_likelihood()
            if self.model is None: # Still couldn't initialize (e.g. no data for inducing points)
                return

        assert self.model is not None and self.likelihood is not None, "Model/Likelihood init failed."
        
        # For VariationalGP, we don't use set_train_data typically. We train on the full dataset.
        # The model internally knows about its inducing points.

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_gp)
        # Use VariationalELBO for Approximate GPs
        elbo = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.train_x_tensor.size(0))

        if self.train_x_tensor.shape[0] > 0:
            # To handle potentially large datasets with SVGP, one might use mini-batches here.
            # For SPPO accumulating data and re-training, using the full dataset is common.
            for i in range(self.num_gp_opt_iters):
                optimizer.zero_grad()
                output = self.model(self.train_x_tensor) # Model forward pass
                loss = -elbo(output, self.train_y_tensor)
                if torch.isnan(loss):
                    if self.verbose: print(f"Warning: NaN loss in GP training (iter {i}). Skipping. Data points: {self.train_x_tensor.shape[0]}")
                    break
                loss.backward()
                optimizer.step()
            self._is_initialized_and_trained = True
            if self.verbose and self.num_gp_opt_iters > 0 :
                print(f"GP training loss after {self.num_gp_opt_iters} iters: {loss.item():.3f}")

        else:
            self._is_initialized_and_trained = False


    def predict(self, state: np.ndarray) -> Tuple[float, float]:
        if not self._is_initialized_and_trained or self.model is None or self.likelihood is None:
            return 0.0, self.initial_signal_variance 

        self.model.eval()
        self.likelihood.eval()
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # The likelihood is applied to the output of the GP model (the latent function)
            # For ApproximateGP, model(state_tensor) gives the distribution f(x)
            # likelihood(model(state_tensor)) gives the predictive distribution y(x)
            predictive_distribution = self.likelihood(self.model(state_tensor))
        
        mean = predictive_distribution.mean.cpu().item()
        variance = predictive_distribution.variance.cpu().item()
        return mean, variance

    def calculate_lower_bound(self, state: np.ndarray, beta_t_sqrt: float, t_step: int,
                              previous_l_value_for_state: Optional[float] = None) -> float:
        predicted_mean_safety, predicted_variance = self.predict(state)
        sigma_s = math.sqrt(max(1e-9, predicted_variance))

        current_lower_estimate = predicted_mean_safety - beta_t_sqrt * sigma_s

        if t_step == 1 or previous_l_value_for_state is None:
            return current_lower_estimate
        else:
            return max(previous_l_value_for_state, current_lower_estimate)

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

        #self.gp_safety_estimator = GaussianProcessSafetyEstimator(
        #    state_dim=self.state_dim,
        #    length_scale=gp_length_scale,
        #    signal_variance=gp_signal_variance,
        #    noise_level_gp=gp_noise_level 
        #)

        # Inside SPPO.__init__
        # Inside SPPO.__init__
        self.gp_safety_estimator = GaussianProcessSafetyEstimatorGPyTorch(
                state_dim=self.state_dim,
                num_inducing_points=config_dict.get("gp_num_inducing_points", 100), # New config
                initial_lengthscale=gp_length_scale,
                initial_signal_variance=gp_signal_variance,
                initial_noise_level=gp_noise_level,
                learning_rate_gp=config_dict.get("gp_lr", 0.01),
                num_gp_opt_iters=config_dict.get("gp_iters", 20),
                verbose= (self.verbose > 1) # Pass verbose flag
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
                    if self.verbose > 0 : print(f"Environment reset. New state: {state.round(2)}")
                
                if self.has_continuous_action_space and current_total_timesteps_elapsed % self.action_std_decay_freq == 0:
                    self.decay_action_std()
                
                if current_total_timesteps_elapsed >= total_timesteps:
                    break
            
            self.update()
            if self.verbose > 0 and (current_total_timesteps_elapsed % self.steps_per_epoch_for_buffer) == 0 :
                 print(f"PPO Update performed after {current_total_timesteps_elapsed} timesteps. Buffer size: {len(self.buffer.rewards)}")

            if current_total_timesteps_elapsed >= total_timesteps:
                break
        
        if self.verbose > 0: print(f"SPPO training finished after {current_total_timesteps_elapsed} timesteps.")
        if hasattr(self.env, 'close'):
            self.env.close()

# --- Simple Space Placeholders (if not using Gymnasium or similar) ---
class SimpleSpace:
    def __init__(self, shape: tuple):
        self.shape = shape

class SimpleDiscreteSpace:
    def __init__(self, n: int):
        self.n = n
        self.shape = () # For discrete, shape is often an empty tuple

# --- Dummy Environment for PRB Allocation ---
class DummyPRBEnv:
    def __init__(self, state_dim=6, num_actions=1000, num_slices=3, total_prbs_available=50, max_episode_len=50):
        self.observation_space = SimpleSpace(shape=(state_dim,))
        self.action_space = SimpleDiscreteSpace(n=num_actions) # Discrete action space
        
        self.num_slices = num_slices
        self.total_prbs_available = total_prbs_available
        self.state_dim = state_dim
        self.num_actions = num_actions

        self.current_state: np.ndarray = np.zeros(state_dim, dtype=np.float32)
        self.current_prb_allocs: np.ndarray = np.zeros(num_slices, dtype=np.int32)
        self.slice_gbr_prbs: np.ndarray = np.array([3, 4, 2], dtype=np.int32) # Example GBRs in PRBs for 3 slices
        
        # Pre-generate a mapping from discrete action index to PRB allocations
        self._action_to_prb_map: List[np.ndarray] = self._generate_action_map()

        self.max_steps_per_episode = max_episode_len
        self.current_episode_steps = 0
        
        # H0: Initial safe states and their safety values
        # Example: low demand, GBRs met comfortably
        #s0_allocs = np.array([7, 10, 5], dtype=np.int32) # Allocations meeting GBRs
        #s0_demands = np.array([4, 5, 3], dtype=np.int32)   # Low demands
        #s0_state = self._form_state(s0_allocs, s0_demands)
        #s0_safety = self._calculate_safety_value(s0_allocs)

        #self.initial_safe_states_H0 = [s0_state]
        #self.initial_safe_safety_values_H0 = [s0_safety]
        self.initial_safe_states_H0, self.initial_safe_safety_values_H0 = self.get_initial_safe_set_H0()

        print(f"DummyPRBEnv: Initialized. Total PRBs: {self.total_prbs_available}, Num Actions: {len(self._action_to_prb_map)}")
        if len(self._action_to_prb_map) != num_actions:
            print(f"Warning: Generated action map size ({len(self._action_to_prb_map)}) doesn't match requested num_actions ({num_actions}). Using generated size.")
            self.action_space = SimpleDiscreteSpace(n=len(self._action_to_prb_map))


    def _generate_action_map(self) -> List[np.ndarray]:
        """Generates a list of possible PRB allocations for 3 slices."""
        allocations = []
        # This is a simplified way to get a variety of allocations.
        # For exactly N actions, a more structured generation or lookup table is needed.
        # Here, we aim for roughly N distinct allocations.
        # This method will likely not produce exactly self.num_actions, adjust self.action_space.n accordingly.
        count = 0
        max_actions_to_generate = self.num_actions * 2 # Generate more to pick from if needed
        
        # Try to make somewhat distinct allocations up to num_actions
        # Iterating over possible PRBs for slice 1 and 2
        for p1 in range(self.total_prbs_available + 1):
            for p2 in range(self.total_prbs_available - p1 + 1):
                if count >= max_actions_to_generate and len(allocations) >= self.num_actions : break
                p3 = self.total_prbs_available - p1 - p2
                allocations.append(np.array([p1, p2, p3], dtype=np.int32))
                count += 1
            if count >= max_actions_to_generate and len(allocations) >= self.num_actions : break
        
        # If not enough, add random valid allocations (though the above should be enough)
        while len(allocations) < self.num_actions and len(allocations) < 1326: # 1326 is theoretical max for 50 PRBs, 3 slices
            a = np.random.randint(0, self.total_prbs_available + 1, size=self.num_slices)
            if np.sum(a) <= self.total_prbs_available : # allow for underutilization too as an action type
                 # or force sum if required by making one slice residual.
                 # For simplicity, let's ensure actions sum to total_prbs_available for now.
                 if np.sum(a) > 0 : # Avoid all zeros if possible to normalize
                    a = np.round(a / np.sum(a) * self.total_prbs_available).astype(np.int32)
                    # Correct sum due to rounding
                    diff = self.total_prbs_available - np.sum(a)
                    a[np.random.choice(self.num_slices)] += diff # Add diff to a random slice
                    if not any(np.array_equal(x,a) for x in allocations):
                        allocations.append(a)

        # If we have more than num_actions, sample down to num_actions
        if len(allocations) > self.num_actions:
            indices = np.random.choice(len(allocations), self.num_actions, replace=False)
            allocations = [allocations[i] for i in indices]
        elif not allocations: # Ensure at least one action if total_prbs_available is very small
            allocations.append(np.array([self.total_prbs_available // self.num_slices] * (self.num_slices -1) + \
                               [self.total_prbs_available - (self.total_prbs_available // self.num_slices) * (self.num_slices-1)], dtype=np.int32) )


        return allocations if allocations else [np.zeros(self.num_slices, dtype=np.int32)]


    def _form_state(self, current_allocs: np.ndarray, current_demands: np.ndarray) -> np.ndarray:
        """ Forms the 5-dimensional state. Normalize if necessary for NN. """
        # Example state: [alloc_s1, alloc_s2, demand_s1, demand_s2, demand_s3]
        # Normalize by total_prbs_available for ratios if preferred
        s = np.zeros(self.state_dim, dtype=np.float32)
        s[0] = current_allocs[0] / self.total_prbs_available # Slice 1 alloc ratio
        s[1] = current_allocs[1] / self.total_prbs_available # Slice 2 alloc ratio
        s[2] = current_allocs[2] / self.total_prbs_available # Slice 2 alloc ratio
        # Slice 3 alloc ratio can be inferred: 1 - s[0] - s[1]
        s[3] = current_demands[0] / (self.total_prbs_available + 1e-6) # Slice 1 demand ratio
        s[4] = current_demands[1] / (self.total_prbs_available + 1e-6) # Slice 2 demand ratio
        s[5] = current_demands[2] / (self.total_prbs_available + 1e-6) # Slice 3 demand ratio
        return s

    def _calculate_safety_value(self, current_allocs: np.ndarray) -> float:
        """ Calculates safety value m(s'). Positive if GBRs met, negative otherwise. """
        gbr_satisfaction = current_allocs - self.slice_gbr_prbs
        # Safety is the minimum margin by which GBRs are met (or how badly the worst one is violated)
        # Normalize by GBR to make it relative? Or use raw PRB difference.
        # For example, min satisfaction margin.
        min_margin = np.min(gbr_satisfaction) 
        # Scale it to be in a reasonable range, e.g. if GBRs are ~5-10 PRBs
        return float(min_margin / 5.0) # Example scaling to get values around -1 to 1

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
        
        # Reset to a somewhat random state (demands) but ensure allocations are reasonable initially
        self.current_prb_allocs = self._action_to_prb_map[np.random.randint(len(self._action_to_prb_map))]
        # Simulate some random demands (e.g., up to GBR * 1.5)
        current_demands = np.random.randint(1, self.slice_gbr_prbs * 1.5 + 1, size=self.num_slices)
        
        self.current_state = self._form_state(self.current_prb_allocs, current_demands)
        self.current_episode_steps = 0
        return self.current_state.copy(), {}

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if not (0 <= action_index < len(self._action_to_prb_map)):
            raise ValueError(f"Action index {action_index} out of bounds for map size {len(self._action_to_prb_map)}")
        
        self.current_prb_allocs = self._action_to_prb_map[action_index]
        self.current_episode_steps += 1

        # Simulate demands (can be stochastic or based on a model)
        # For simplicity, let demands fluctuate around GBRs
        current_demands = np.maximum(1, np.random.normal(loc=self.slice_gbr_prbs, scale=3.0, size=self.num_slices).astype(np.int32))
        current_demands = np.clip(current_demands, 1, self.total_prbs_available)

        # Calculate achieved throughput (e.g., min(demand, allocation))
        achieved_prbs = np.minimum(current_demands, self.current_prb_allocs)
        
        # Reward: Sum of achieved PRBs (as proxy for throughput) + bonus for meeting GBRs
        reward = float(self.total_prbs_available - np.sum(achieved_prbs))
        gbr_met_bonus = np.sum((achieved_prbs >= self.slice_gbr_prbs) * 2.0) # Bonus for each GBR met
        reward += gbr_met_bonus
        
        # Penalty if total allocation > total_prbs_available (should not happen with action map)
        if (np.sum(self.current_prb_allocs) < current_demands).all() :
            penalty = np.sum((achieved_prbs <= self.slice_gbr_prbs) * 1.0)
            reward -= 100 * penalty # Severe penalty

        # Safety value m(s') for the new state (based on new allocations)
        m_s_prime = self._calculate_safety_value(self.current_prb_allocs)
        
        # Form next state based on new allocations and new demands
        self.current_state = self._form_state(self.current_prb_allocs, current_demands)
        
        terminated = False # e.g., if a global objective is met, or always False for continuous task
        truncated = self.current_episode_steps >= self.max_steps_per_episode
        #truncated = False
            
        info = {'safety_value_m_s_prime': m_s_prime, 'allocations': self.current_prb_allocs.copy()}
            
        return self.current_state.copy(), reward, terminated, truncated, info
            
    def get_initial_safe_set_H0(self) -> Tuple[List[np.ndarray], List[float]]:
        # Provide some known safe initial states and their safety values
        # Example: allocation meets GBRs, demands are low
        s0_allocs1 = np.array([7, 10, 5], dtype=np.int32) # Sum 22 <= 50
        s0_demands1 = np.array([3, 4, 2], dtype=np.int32)
        state1 = self._form_state(s0_allocs1, s0_demands1)
        safety1 = self._calculate_safety_value(s0_allocs1) # Should be positive

        s0_allocs2 = np.array([10, 15, 10], dtype=np.int32) # Sum 35 <= 50
        s0_demands2 = np.array([5, 8, 3], dtype=np.int32) # Demands match GBRs
        state2 = self._form_state(s0_allocs2, s0_demands2)
        safety2 = self._calculate_safety_value(s0_allocs2) # Should be positive/zero

        # Ensure H0 states are different enough for GP
        return [state1, state2], [safety1, safety2]

    def close(self):
        print("DummyPRBEnv closed.")

def main():
    # Suppress ConvergenceWarnings from sklearn GP if they are persistent and understood
    # import warnings
    # from sklearn.exceptions import ConvergenceWarning
    # warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.gaussian_process")

    config = {
        # Environment specific
        "state_dim": 6, 
        "action_dim": 1000, # This will be adjusted by DummyPRBEnv._generate_action_map
        "has_continuous_action_space": False, 
        "total_prbs_env": 50,
        "num_slices_env": 3,
        "max_episode_length_env": 100, # Shorter episodes for faster iterations
        # SPPO Safety parameters
        "safety_threshold_h": 0.0, # Safety if min GBR margin is >= 0
        "neighborhood_radius_v": 0.05, # Smaller radius for normalized state space features
        "beta_t_sqrt_val": 1.5,       # Needs careful tuning based on B, U, delta
        "gp_length_scale": 1006,       # Tuned for normalized state space
        "gp_signal_variance": 1.0,    
        "gp_noise_level": 0.05,       
        # PPO specific params
        "action_std_init": 0.6, # Not used for discrete
        "lr_actor": 5e-4,       # Learning rates might need adjustment       
        "lr_critic": 1e-3,      
        "gamma_discount": 0.99,           
        "K_epochs": 4, # Fewer PPO epochs for faster iteration         
        "eps_clip": 0.2,                  
        # Training loop params
        "total_timesteps": 500 * 100, # epochs * steps_per_epoch; 50k for a quicker test
        "steps_per_epoch_for_buffer": 100, # Number of steps to collect before PPO update
        "verbose_level": 1                
    }

    # --- Initialize Environment ---
    env = DummyPRBEnv(
        state_dim=config["state_dim"],
        num_actions=config["action_dim"], # Env will adjust its action_space.n if map is different
        num_slices=config["num_slices_env"],
        total_prbs_available=config["total_prbs_env"],
        max_episode_len=config["max_episode_length_env"]
    )
    
    # Update config's action_dim if env adjusted it
    actual_action_dim = env.action_space.n
    if config["action_dim"] != actual_action_dim:
        print(f"Note: Environment adjusted action dimension to {actual_action_dim}")
        config["action_dim"] = actual_action_dim


    # --- Instantiate SPPO Agent ---
    # Ensure SPPO class is defined in the scope (from sppo_integrated_code artifact)
    # This main function assumes SPPO and its dependencies are available.
    try:
        agent = SPPO(
            env=env,
            lr_actor=config["lr_actor"],
            lr_critic=config["lr_critic"],
            gamma=config["gamma_discount"],
            K_epochs=config["K_epochs"],
            eps_clip=config["eps_clip"],
            has_continuous_action_space=config["has_continuous_action_space"],
            action_std_init=config["action_std_init"], # Not used by discrete
            # action_std_decay_rate, min_action_std, action_std_decay_freq not needed for discrete
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
    except NameError:
        print("ERROR: The SPPO class (and its dependencies like GaussianProcessSafetyEstimator, etc.)")
        print("needs to be defined in the same script or imported.")
        print("Please ensure the code from the 'sppo_integrated_code' artifact is available.")
        return # Exit if SPPO class is not found

    # --- Start Learning ---
    print(f"Initializing SPPO for PRB Allocation with State Dim: {env.observation_space.shape[0]}, Action Dim: {env.action_space.n}")
    agent.learn(total_timesteps=config["total_timesteps"])

    print("Training finished.")
    # Example: Save the model
    # agent.save("sppo_prb_alloc_model.pth")

if __name__ == "__main__":
    # Define necessary classes like SPPO, GaussianProcessSafetyEstimator, etc., here or import them
    # For this to run, you need the full SPPO class definition from the previous artifact.
    # Example:
    # from sppo_integrated_code import SPPO, GaussianProcessSafetyEstimator, ... (if they were in a module)
    
    # Since they are expected to be in the same scope for the artifact, just calling main()
    # assumes they are defined above this main function.
    main()