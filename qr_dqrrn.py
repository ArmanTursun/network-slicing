import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -----------------------------------------------------------------------------
# 1. THE NEURAL NETWORK MODEL
# -----------------------------------------------------------------------------
class DQRNN(nn.Module):
    """
    A simple Feed-Forward Neural Network to predict a quantile.
    """
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.2):
        super(DQRNN, self).__init__()
        # Define the network layers
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, 1) # Outputs a single quantile value
        
        self.relu = nn.ReLU()
        # Dropout is used for regularization and for estimating uncertainty
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        # No activation on the output layer for a regression task
        x = self.output_layer(x)
        return x

# -----------------------------------------------------------------------------
# 2. THE PINBALL LOSS FUNCTION
# -----------------------------------------------------------------------------
def pinball_loss(prediction, target, quantile):
    """
    Calculates the pinball loss function.
    """
    error = target - prediction
    # Loss for under-prediction
    loss_under = quantile * error
    # Loss for over-prediction
    loss_over = (1 - quantile) * -error
    
    # Combine the losses
    loss = torch.where(error >= 0, loss_under, loss_over)
    return loss.mean()

# -----------------------------------------------------------------------------
# 3. THE NEW LEARNER CLASS
# -----------------------------------------------------------------------------
class DeepQuantileRegressor:
    """
    Manages the DQRNN model, its training, and predictions.
    This class replaces your KernelizedOnlineQuantileRegressor.
    """
    def __init__(self, input_dim, quantile=0.05, learning_rate=1e-4, mc_dropout_samples=10):
        self.input_dim = input_dim
        self.quantile = quantile
        self.learning_rate = learning_rate
        self.mc_dropout_samples = mc_dropout_samples # For uncertainty estimation

        # Initialize the model and optimizer
        self.model = DQRNN(input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # We no longer need the SV (Support Vector) memory class

    def set_quantile(self, new_quantile):
        """ Allows the meta-learner to adjust the quantile. """
        self.quantile = new_quantile

    def predict_with_uncertainty(self, x):
        """
        Makes a prediction and estimates uncertainty using Monte Carlo Dropout.
        """
        # Convert numpy array to a PyTorch tensor
        x_tensor = torch.from_numpy(x).float().unsqueeze(0) # Add batch dimension
        
        # Set the model to evaluation mode, but keep dropout enabled for uncertainty
        self.model.train() 
        with torch.no_grad():
            # Get multiple predictions by running the forward pass several times
            predictions = [self.model(x_tensor).item() for _ in range(self.mc_dropout_samples)]
        
        # Prediction is the mean of the samples
        prediction = np.mean(predictions)
        # Uncertainty is the standard deviation of the samples
        uncertainty = np.std(predictions)
        
        return prediction, uncertainty

    def update(self, x, y_true, sla_threshold):
        """
        Performs a single training step on the neural network.
        This replaces the old update method that added support vectors.
        """
        # Set the model to training mode
        self.model.train()
        
        # Convert inputs to PyTorch tensors
        x_tensor = torch.from_numpy(x).float().unsqueeze(0)
        y_true_tensor = torch.from_numpy(np.array([y_true])).float().unsqueeze(0)
        
        # --- The Training Step ---
        # 1. Reset gradients
        self.optimizer.zero_grad()
        
        # 2. Make a prediction
        prediction = self.model(x_tensor)
        
        # 3. Calculate the pinball loss
        loss = pinball_loss(prediction, y_true_tensor, self.quantile)
        
        # 4. Perform backpropagation to calculate gradients
        loss.backward()
        
        # 5. Update the network's weights
        self.optimizer.step()