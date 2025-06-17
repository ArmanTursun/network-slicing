
import torch
import gpytorch
import numpy as np
import random

# ===================================================================
# 1. PRIORITIZED EXPERIENCE REPLAY (PER) IMPLEMENTATION
# ===================================================================

class SumTree:
    """A binary tree data structure where parent nodes are sums of their children."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    """A fixed-size replay buffer that samples experiences based on their priority."""
    def __init__(self, capacity, prob_alpha=0.6):
        self.tree = SumTree(capacity)
        self.prob_alpha = prob_alpha  # Controls how much prioritization is used
        self.epsilon = 0.01          # Small value to ensure no experience has zero priority

    def add(self, experience):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, experience)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total_priority / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get_leaf(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        return batch, idxs, np.array(priorities)

    def update_priorities(self, tree_idxs, errors):
        priorities = (np.abs(errors) + self.epsilon) ** self.prob_alpha
        for i, p in zip(tree_idxs, priorities):
            self.tree.update(i, p)

    def __len__(self):
        return self.tree.n_entries

# ===================================================================
# 2. GPYTORCH MODEL AND AGENT
# ===================================================================

class VariationalGPModel(gpytorch.models.ApproximateGP):
    """The SVGP Model using a Matérn kernel."""
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SVGPAgent:
    """The main agent class wrapping the GP model and the PER buffer."""
    def __init__(self, input_dim, num_inducing_points=100, buffer_capacity=10000, learning_rate=0.01):
        # Initialize inducing points randomly in a reasonable range (e.g., 0-1)
        inducing_points = torch.rand(num_inducing_points, input_dim)
        
        self.model = VariationalGPModel(inducing_points=inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # The loss function for SVGP
        self.loss_fn = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=buffer_capacity)
        
        # The Prioritized Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

    def predict_with_uncertainty(self, x_numpy):
        x_tensor = torch.from_numpy(x_numpy.astype(np.float32)).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.likelihood(self.model(x_tensor))
            mean = posterior.mean.cpu().numpy()
            stddev = posterior.stddev.cpu().numpy()
        return mean, stddev

    def train_step(self, batch_size):
        """Performs one full learning step on a batch of data from the PER buffer."""
        if len(self.replay_buffer) < batch_size:
            return # Not enough memories to train yet

        # 1. Sample a prioritized batch from the buffer
        experiences, idxs, _ = self.replay_buffer.sample(batch_size)
        
        # Unpack experiences into batches of x and y
        batch_x = np.array([exp[0] for exp in experiences])
        batch_y = np.array([exp[1] for exp in experiences])
        
        x_tensor = torch.from_numpy(batch_x.astype(np.float32)).to(self.device)
        y_tensor = torch.from_numpy(batch_y.astype(np.float32)).to(self.device)

        # 2. Perform the model update (training)
        self.model.train()
        self.likelihood.train()
        self.optimizer.zero_grad()
        output = self.model(x_tensor)
        loss = -self.loss_fn(output, y_tensor)
        loss.backward()
        self.optimizer.step()
        
        # 3. Update priorities in the buffer
        # Get the errors (difference between prediction and true values) for the batch
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            preds = self.likelihood(self.model(x_tensor)).mean
            errors = (preds - y_tensor).abs().cpu().numpy()
        
        self.replay_buffer.update_priorities(idxs, errors)

### How You Would Use This in Your `run` Loop

# Main code would add experiences to the agent's buffer and then tell the agent to train itself.

'''
# Conceptual example of the new run loop

# Create ONE agent instance for each KPI you want to model
# e.g., one for delay, one for throughput
delay_agent = SVGPAgent(input_dim=...)
throughput_agent = SVGPAgent(input_dim=...)

# ... inside the run loop for each step ...
# 1. Get state and choose an action using the agents' predict methods
state, info = env.get_state()
action = select_action(state, delay_agent, throughput_agent)

# 2. Take the step
next_state, reward, _, _, info = env.step(action)

# 3. Add the new experience to the replay buffers
# The experience is the input vector 'x' and the true outcome 'y'
x_vector = np.append(state, action / n_prbs)
true_delay = info['delay']
true_throughput = info['throughput']

delay_agent.replay_buffer.add((x_vector, true_delay))
throughput_agent.replay_buffer.add((x_vector, true_throughput))

# 4. Perform a training step on both agents
delay_agent.train_step(batch_size=64)
throughput_agent.train_step(batch_size=64)
'''