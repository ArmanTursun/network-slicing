"""
@author: Arman

This script provides the utils for safe-ppo 

"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional
import torch
# To suppress specific warnings if necessary
import warnings
from sklearn.exceptions import ConvergenceWarning

import gpytorch

# --- Placeholder for device ---
device = torch.device('cpu')
try:
    device
except NameError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device not defined, using {device}. Define 'device' globally for specific GPU/CPU choice.")

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
                 gp_training_batch_size = 128,
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

        self.gp_training_batch_size = gp_training_batch_size

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
        
        if True: #self.verbose:
            print(f"Initialized {self.inducing_points_tensor.shape[0]} inducing points with {self.inducing_points_init_method} method.")
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
                return None

        assert self.model is not None and self.likelihood is not None, "Model/Likelihood init failed."
        
        # For VariationalGP, we don't use set_train_data typically. We train on the full dataset.
        # The model internally knows about its inducing points.

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_gp)
        # Use VariationalELBO for Approximate GPs
        elbo = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.train_x_tensor.size(0))

        avg_loss_this_update = None
        if self.train_x_tensor.shape[0] > 0 and self.num_gp_opt_iters > 0:
            # To handle potentially large datasets with SVGP, use mini-batches here.
            # For SPPO accumulating data and re-training, using the full dataset is common.
            losses_in_iters = []

            # Define a batch size for GP training
            # This batch_size is for the GP's own optimization loop, separate from PPO's buffer.
            # It should be small enough for efficiency but large enough for stable ELBO estimates.
            gp_training_batch_size = min(self.gp_training_batch_size, self.train_x_tensor.shape[0])

            for i in range(self.num_gp_opt_iters):
                # Sample a mini-batch from the full training data
                perm = torch.randperm(self.train_x_tensor.size(0), device=device)
                idx = perm[:gp_training_batch_size]
                batch_x, batch_y = self.train_x_tensor[idx], self.train_y_tensor[idx]

                optimizer.zero_grad()
                #output = self.model(self.train_x_tensor) # Model forward pass
                #loss = -elbo(output, self.train_y_tensor)
                output = self.model(batch_x) # Model forward pass
                loss = -elbo(output, batch_y)
                if torch.isnan(loss):
                    if self.verbose: print(f"Warning: NaN loss in GP training (iter {i}). Skipping. Data points: {self.train_x_tensor.shape[0]}")
                    avg_loss_this_update = float('nan') # Mark as NaN
                    break
                loss.backward()
                optimizer.step()
                losses_in_iters.append(loss.item())
            self._is_initialized_and_trained = True
            if losses_in_iters: # if not broken by NaN on first iter
                avg_loss_this_update = np.mean(losses_in_iters)
                if self.verbose and self.num_gp_opt_iters > 0 and not math.isnan(avg_loss_this_update): # Check if avg_loss is not NaN
                    # This print was originally here, it's fine to keep if verbose
                    print(f"GP training avg loss after {len(losses_in_iters)} iters: {avg_loss_this_update:.3f}")
                    pass

        else:
            self._is_initialized_and_trained = False
        
        return avg_loss_this_update # Return the average loss


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
                               gp_estimator: GaussianProcessSafetyEstimatorGPyTorch,
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