from PPO_Safe import SPPO
import numpy as np
from typing import List, Tuple, Optional, Dict

# For environment spaces, typically from a library like Gymnasium
# For this standalone example, we'll define simple space-like objects
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
        "gp_num_inducing_points": 100,
        "gp_lr": 0.01,
        "gp_iters": 20,       
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