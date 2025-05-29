import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from google.colab import files # For downloading files in Colab

# --- Configuration ---
N_SAMPLES = 100
X_RANGE_MIN = 0
X_RANGE_MAX = 2 * np.pi
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 1000
LEARNING_RATE = 0.01
SEQ_LENGTH_RNN = 10 # Sequence length for RNN input
HIDDEN_SIZE_FFN = 64
HIDDEN_SIZE_RNN = 32
# Add an interval for checking validation loss during training
VALIDATION_INTERVAL = 50 # Check validation loss every 50 epochs

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Data Generation ---
def generate_sine_wave_data(n_samples, x_min, x_max):
    """Generates sine wave data."""
    x = np.linspace(x_min, x_max, n_samples)
    y = np.sin(x)
    return x, y

x_np, y_np = generate_sine_wave_data(N_SAMPLES, X_RANGE_MIN, X_RANGE_MAX)

# Reshape for PyTorch (input feature dimension is 1)
X_data_ffn = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32).to(device)
y_data_ffn = torch.tensor(y_np.reshape(-1, 1), dtype=torch.float32).to(device)

# For RNN, we need to create sequences
def create_sequences(data, seq_length):
    """Creates sequences for RNN input."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    # Return sequences and targets as numpy arrays initially
    return np.array(xs), np.array(ys)

# Use y_np (sine values) as the base for sequence generation
# create_sequences returns numpy arrays now
X_data_rnn_seq_np, y_data_rnn_target_np = create_sequences(y_np, SEQ_LENGTH_RNN)
# The 'x_axis' for RNN plots will need to correspond to the target points
x_axis_rnn_original_indices = x_np[SEQ_LENGTH_RNN:]


# --- 2. Data Splitting ---
# FFN Data
X_train_ffn, X_test_ffn, y_train_ffn, y_test_ffn = train_test_split(
    X_data_ffn, y_data_ffn, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)
# We need to sort X_test_ffn for plotting if shuffle=True (which is good for training)
# Store original x-values for plotting test set
sorted_indices_ffn = torch.argsort(X_test_ffn.squeeze())
X_test_plot_ffn = X_test_ffn[sorted_indices_ffn]
y_test_plot_ffn = y_test_ffn[sorted_indices_ffn]


# RNN Data
# Note: For time series, it's often better to split chronologically,
# but for a stationary sine wave, random split after sequence creation is acceptable.
# If data had trends, a chronological split would be essential.
# We split X_data_rnn_seq_np, y_data_rnn_target_np, AND the corresponding x_axis values together
(X_train_rnn_np, X_test_rnn_np,
 y_train_rnn_np, y_test_rnn_np,
 x_axis_train_rnn, x_axis_test_rnn) = train_test_split(
    X_data_rnn_seq_np, y_data_rnn_target_np, x_axis_rnn_original_indices,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)

# Convert numpy arrays to PyTorch tensors after splitting
X_train_rnn = torch.tensor(X_train_rnn_np, dtype=torch.float32).to(device)
X_test_rnn = torch.tensor(X_test_rnn_np, dtype=torch.float32).to(device)
y_train_rnn = torch.tensor(y_train_rnn_np, dtype=torch.float32).unsqueeze(1).to(device) # Ensure y has shape (batch, 1)
y_test_rnn = torch.tensor(y_test_rnn_np, dtype=torch.float32).unsqueeze(1).to(device) # Ensure y has shape (batch, 1)


# For plotting RNN, we need to sort the test data based on the x-axis values
if X_test_rnn.shape[0] > 0: # Check if test set is not empty
    # Sort the test x-axis values and get the indices
    # Use torch.argsort for consistency if X_test_rnn is a tensor, but x_axis_test_rnn is numpy
    # Let's convert x_axis_test_rnn to a tensor for consistent sorting
    x_axis_test_rnn_tensor = torch.tensor(x_axis_test_rnn, dtype=torch.float32)
    sorted_indices_rnn = torch.argsort(x_axis_test_rnn_tensor)

    # Apply the sorting indices to all test sets
    X_test_plot_rnn = X_test_rnn[sorted_indices_rnn]
    y_test_plot_rnn = y_test_rnn[sorted_indices_rnn]
    # The x-axis values themselves should also be sorted for plotting
    x_axis_plot_rnn = x_axis_test_rnn[sorted_indices_rnn.cpu().numpy()] # Apply to numpy array

else:
    print("Warning: RNN test set is empty. Adjust N_SAMPLES or SEQ_LENGTH_RNN.")
    X_test_plot_rnn = X_test_rnn # Will be empty
    y_test_plot_rnn = y_test_rnn # Will be empty
    x_axis_plot_rnn = np.array([]) # Will be empty


# --- 3. Model Implementations ---

# a. Feedforward Neural Network (FFN)
class FFNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# b. Elman-Type Recurrent Neural Network
class ElmanRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(ElmanRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True means input/output tensors are (batch, seq, feature)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0) # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :]) # We only care about the output of the last sequence element
        return out

# --- 4. Training Function (Modified to track losses) ---
def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs, validation_interval, model_name="Model"):
    print(f"\n--- Training {model_name} ---")
    train_losses = []
    val_losses = []
    epochs_run = []

    for epoch in range(epochs):
        model.train() # Set model to training mode

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        epochs_run.append(epoch + 1)

        # Evaluate on validation set at intervals
        if (epoch + 1) % validation_interval == 0 or epoch == 0 or (epoch + 1) == epochs:
             model.eval() # Set model to evaluation mode
             with torch.no_grad():
                 val_outputs = model(X_val)
                 val_loss = criterion(val_outputs, y_val)
                 val_losses.append(val_loss.item())
                 print(f'{model_name} - Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
        elif (epoch + 1) % (epochs // 10) == 0: # Print training loss at other intervals
             print(f'{model_name} - Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}')


    print(f"--- {model_name} Training Finished ---")
    # Return losses along with the model
    # Return epochs_run corresponding to train_losses
    # Return epochs_run_val corresponding to val_losses (only at intervals)
    # We need to track which epochs the validation loss was recorded at
    val_epochs = [(i + 1) for i in range(epochs) if (i + 1) % validation_interval == 0 or i == 0 or (i + 1) == epochs]

    return model, train_losses, epochs_run, val_losses, val_epochs


# --- 5. Plotting and 6. MSE Calculation (Evaluate function remains the same) ---
def evaluate_and_plot(model, X_test_sorted, y_test_sorted, x_axis_for_plot, model_name, is_rnn=False):
    print(f"\n--- Evaluating {model_name} ---")
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # Check if the tensor is empty using .numel() for PyTorch tensors
        if X_test_sorted.numel() == 0:
            print(f"Test set for {model_name} is empty. Skipping evaluation.")
            # Return empty arrays/tensors for plotting later
            # Ensure consistent return types (np.nan for mse, empty np arrays for plots)
            return float('nan'), np.array([]), np.array([])

        predictions = model(X_test_sorted)

    # Move predictions to CPU for numpy and plotting
    predictions_np = predictions.cpu().numpy()
    y_test_np = y_test_sorted.cpu().numpy()

    # Calculate MSE
    mse = mean_squared_error(y_test_np, predictions_np)
    print(f'{model_name} - Test MSE: {mse:.6f}')

    # Plotting (individual model plot)
    plt.figure(figsize=(12, 6))
    plt.plot(x_np, y_np, label='True Sine Wave (Full Data)', color='gray', linestyle='--') # Plot full original wave
    # Use squeeze() for 1D plotting
    plt.plot(x_axis_for_plot, y_test_np.squeeze(), label=f'Actual Test Values ({model_name})', marker='o', linestyle='None', color='blue')
    plt.plot(x_axis_for_plot, predictions_np.squeeze(), label=f'Predicted Values ({model_name})', marker='x', linestyle='-', color='red')

    plt.title(f'{model_name}: True vs. Predicted Sine Wave Values')
    plt.xlabel('X (Input)')
    plt.ylabel('sin(X) (Output)')
    # Ensure y-axis is proportional
    # Need to handle cases where predictions_np might be empty
    min_val = min(y_np.min(), predictions_np.min() if predictions_np.size > 0 else y_np.min())
    max_val = max(y_np.max(), predictions_np.max() if predictions_np.size > 0 else y_np.max())
    padding = (max_val - min_val) * 0.1
    plt.ylim(min_val - padding -0.1, max_val + padding+0.1) # Add a bit more padding for clarity
    plt.legend()
    plt.grid(True)
    plt.show()

    return mse, y_test_np.squeeze(), predictions_np.squeeze()

def plot_combined_results(x_orig, y_orig,
                          x_train_ffn, y_train_ffn,
                          x_test_ffn_plot, y_test_ffn_actual, y_test_ffn_predicted,
                          x_test_rnn_plot, y_test_rnn_actual, y_test_rnn_predicted):
    """Plots the original data, training data, and test predictions from both models."""
    plt.figure(figsize=(14, 8))

    # Plot original full sine wave
    plt.plot(x_orig, y_orig, label='True Sine Wave (Full Data)', color='gray', linestyle='--')

    # Plot FFN training data
    plt.plot(x_train_ffn.cpu().numpy().squeeze(), y_train_ffn.cpu().numpy().squeeze(),
             'bo', label='FFN Training Data', alpha=0.5, markersize=5)

    # Plot RNN training data (targets). Need corresponding x-axis values for training targets.
    # x_axis_train_rnn was generated during the split.
    # The y_train_rnn target values correspond to these x_axis_train_rnn values.
    # Need to move to CPU and squeeze
    # Check if x_axis_train_rnn is not empty before plotting
    if x_axis_train_rnn.size > 0:
         plt.plot(x_axis_train_rnn, y_train_rnn.cpu().numpy().squeeze(),
                  'go', label='RNN Training Targets', alpha=0.5, markersize=5)


    # Plot FFN test predictions
    # Use .numel() to check if the tensor is empty
    if x_test_ffn_plot.numel() > 0:
        plt.plot(x_test_ffn_plot.cpu().numpy().squeeze(), y_test_ffn_predicted,
                 'r-', label='FFN Test Predictions', linewidth=2)
        # Optionally plot FFN test actual points for comparison
        plt.plot(x_test_ffn_plot.cpu().numpy().squeeze(), y_test_ffn_actual,
                 'ro', label='FFN Test Actual', alpha=0.7, markersize=6)


    # Plot RNN test predictions
    # Use .size > 0 for the numpy array check
    if x_test_rnn_plot.size > 0:
        plt.plot(x_test_rnn_plot, y_test_rnn_predicted,
                 'm-', label='RNN Test Predictions', linewidth=2)
        # Optionally plot RNN test actual points for comparison
        plt.plot(x_test_rnn_plot, y_test_rnn_actual,
                 'mo', label='RNN Test Actual', alpha=0.7, markersize=6)


    plt.title('Sine Wave Regression: FFN vs. Elman RNN')
    plt.xlabel('X (Input)')
    plt.ylabel('sin(X) (Output)')

    # Ensure y-axis is proportional
    # Need to handle cases where test sets are empty (nan values might appear)
    # Collect all numpy arrays that might contain y-values
    y_values_list = [y_orig,
                     y_train_ffn.cpu().numpy().squeeze(),
                     y_train_rnn.cpu().numpy().squeeze()]

    # Append test actual/predicted if they are not empty
    if y_test_ffn_actual.size > 0: y_values_list.append(y_test_ffn_actual)
    if y_test_ffn_predicted.size > 0: y_values_list.append(y_test_ffn_predicted)
    if y_test_rnn_actual.size > 0: y_values_list.append(y_test_rnn_actual)
    if y_test_rnn_predicted.size > 0: y_values_list.append(y_test_rnn_predicted)

    # Concatenate all y-values and filter NaNs
    all_y_values = np.concatenate(y_values_list)
    all_y_values = all_y_values[~np.isnan(all_y_values)]

    if all_y_values.size > 0:
        min_val = np.min(all_y_values)
        max_val = np.max(all_y_values)
        padding = (max_val - min_val) * 0.1
        plt.ylim(min_val - padding -0.1, max_val + padding+0.1) # Add a bit more padding
    # else, plt.ylim will use default automatic scaling if all_y_values is empty


    plt.legend()
    plt.grid(True)
    plt.show()

# --- Function to plot loss curves ---
def plot_losses(ffn_train_losses, ffn_train_epochs, ffn_val_losses, ffn_val_epochs,
                rnn_train_losses, rnn_train_epochs, rnn_val_losses, rnn_val_epochs):
    """Plots the training and validation losses for both models."""
    plt.figure(figsize=(12, 6))

    # FFN Losses
    if ffn_train_epochs: # Check if lists are not empty
      plt.plot(ffn_train_epochs, ffn_train_losses, label='FFN Training Loss', color='blue')
    if ffn_val_epochs:
      plt.plot(ffn_val_epochs, ffn_val_losses, label='FFN Validation Loss', color='cyan', linestyle='--')

    # RNN Losses
    if rnn_train_epochs:
      plt.plot(rnn_train_epochs, rnn_train_losses, label='RNN Training Loss', color='red')
    if rnn_val_epochs:
       plt.plot(rnn_val_epochs, rnn_val_losses, label='RNN Validation Loss', color='magenta', linestyle='--')


    plt.title('Training and Validation Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log') # Use log scale for better visualization of potentially small losses
    plt.legend()
    plt.grid(True)
    plt.show()


# --- Main Execution ---

# FFN Model
print("\n--- FFN Model ---")
ffn_model = FFNModel(input_size=1, hidden_size=HIDDEN_SIZE_FFN, output_size=1).to(device)
criterion_ffn = nn.MSELoss()
optimizer_ffn = optim.Adam(ffn_model.parameters(), lr=LEARNING_RATE)

# Train FFN model and get losses
ffn_model, ffn_train_losses, ffn_train_epochs, ffn_val_losses, ffn_val_epochs = train_model(
    ffn_model, criterion_ffn, optimizer_ffn,
    X_train_ffn, y_train_ffn, X_test_plot_ffn, y_test_plot_ffn, # Use sorted test set as validation
    EPOCHS, VALIDATION_INTERVAL, "FFN"
)
# Capture actual and predicted values for combined plot
# Pass the numpy array x_axis for plotting FFN test results
mse_ffn, y_test_actual_ffn_plot, y_test_predicted_ffn_plot = evaluate_and_plot(
    ffn_model, X_test_plot_ffn, y_test_plot_ffn, X_test_plot_ffn.cpu().numpy().squeeze(), "FFN"
)


# Elman RNN Model
print("\n--- Elman RNN Model ---")
# For RNN, input_size is 1 because each element in the sequence is a single sine value
# Initialize variables to store RNN losses even if training is skipped
elman_rnn_model = None
rnn_train_losses = []
rnn_train_epochs = []
rnn_val_losses = []
rnn_val_epochs = []
mse_rnn = float('nan')
y_test_actual_rnn_plot = np.array([])
y_test_predicted_rnn_plot = np.array([])


if X_train_rnn.shape[0] > 0 and X_test_plot_rnn.shape[0] > 0: # Check if train and test sets are not empty
    elman_rnn_model = ElmanRNNModel(input_size=1, hidden_size=HIDDEN_SIZE_RNN, output_size=1).to(device)
    criterion_rnn = nn.MSELoss()
    optimizer_rnn = optim.Adam(elman_rnn_model.parameters(), lr=LEARNING_RATE)

    # RNN inputs need to be (batch_size, seq_length, input_size=1)
    # X_train_rnn already has shape (batch, seq_len) from create_sequences, add last dim
    # Train RNN model and get losses
    elman_rnn_model, rnn_train_losses, rnn_train_epochs, rnn_val_losses, rnn_val_epochs = train_model(
        elman_rnn_model, criterion_rnn, optimizer_rnn,
        X_train_rnn.unsqueeze(-1), y_train_rnn, X_test_plot_rnn.unsqueeze(-1), y_test_plot_rnn, # Use sorted test set as validation
        EPOCHS, VALIDATION_INTERVAL, "Elman RNN"
    )

    # X_test_plot_rnn needs to be unsqueezed as well for evaluation
    # Capture actual and predicted values for combined plot
    mse_rnn, y_test_actual_rnn_plot, y_test_predicted_rnn_plot = evaluate_and_plot(
        elman_rnn_model, X_test_plot_rnn.unsqueeze(-1), y_test_plot_rnn,
        x_axis_plot_rnn, "Elman RNN", is_rnn=True
    )
else:
    print("RNN training or test set is empty. Skipping RNN model training and evaluation.")


print("\n--- Final MSE Comparison ---")
# Check if mse_ffn is not nan before printing
if not np.isnan(mse_ffn):
    print(f"FFN Test MSE: {mse_ffn:.6f}")
else:
    print("FFN Test MSE: N/A (Test set empty)")

# Check if mse_rnn is not nan before printing
if not np.isnan(mse_rnn):
    print(f"Elman RNN Test MSE: {mse_rnn:.6f}")
else:
     print("Elman RNN Test MSE: N/A (Train or Test set empty)")


# --- Plot Loss Curves ---
plot_losses(ffn_train_losses, ffn_train_epochs, ffn_val_losses, ffn_val_epochs,
            rnn_train_losses, rnn_train_epochs, rnn_val_losses, rnn_val_epochs)


# --- Combined Plot ---
# Need to ensure x_test_plot_ffn is the correct x-axis for the FFN test plot results
# It already contains the sorted x values from the FFN test set.
plot_combined_results(x_np, y_np,
                      X_train_ffn, y_train_ffn, # FFN training data
                      X_test_plot_ffn, y_test_actual_ffn_plot, y_test_predicted_ffn_plot, # FFN test results
                      x_axis_plot_rnn, y_test_actual_rnn_plot, y_test_predicted_rnn_plot) # RNN test results


# --- 7. Export Dataset ---
print("\n--- Exporting Dataset ---")
dataset_df = pd.DataFrame({'x_value': x_np, 'sin_x_value': y_np})
csv_filename = '/content/drive/MyDrive/wave_data.csv'
dataset_df.to_csv(csv_filename, index=False)
print(f"Dataset saved to {csv_filename}")

# Provide download link for Colab
try:
    files.download(csv_filename)
    print(f"'{csv_filename}' download initiated.")
except Exception as e:
    print(f"Could not initiate download in this environment: {e}")
    print(f"You can find the file '{csv_filename}' in the Colab file explorer.")
