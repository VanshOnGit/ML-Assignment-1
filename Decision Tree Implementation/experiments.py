import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from Base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)

def create_fake_data(N, P, input_type, output_type):
    if input_type == 'real':
        X = pd.DataFrame(np.random.randn(N, P))
    elif input_type == 'discrete':
        X = pd.DataFrame({i: np.random.randint(0, 2, size=N) for i in range(P)})
    
    if output_type == 'real':
        y = pd.Series(np.random.randn(N))
    elif output_type == 'discrete':
        y = pd.Series(np.random.randint(0, 2, size=N), dtype='category')
    
    return X, y

# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs

def measure_time(N, P, input_type, output_type, criterion):
    X, y = create_fake_data(N, P, input_type, output_type)
    tree = DecisionTree(criterion=criterion)
    
    # Measure time for fitting
    start_time = time.time()
    tree.fit(X, y)
    fit_time = time.time() - start_time
    
    # Measure time for prediction
    start_time = time.time()
    tree.predict(X)
    predict_time = time.time() - start_time
    
    return fit_time, predict_time

# ...
# Function to plot the results

def plot_results(fit_times, predict_times, N_values, P_values):
    for criterion, times in fit_times.items():
        plt.plot(P_values, times, label=f'{criterion} - Fit')
    plt.xlabel('Number of Features (P)')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title('Fit Time vs. Number of Features')
    plt.show()
    
    for criterion, times in predict_times.items():
        plt.plot(P_values, times, label=f'{criterion} - Predict')
    plt.xlabel('Number of Features (P)')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title('Predict Time vs. Number of Features')
    plt.show()

# ...
# Other functions

def run_experiments(N_values, P_values, criteria):
    fit_times = {criterion: [] for criterion in criteria}
    predict_times = {criterion: [] for criterion in criteria}
    
    for N in N_values:
        for P in P_values:
            for criterion in criteria:
                fit_time, predict_time = measure_time(N, P, 'real', 'real', criterion)
                fit_times[criterion].append(fit_time)
                predict_times[criterion].append(predict_time)
    
    return fit_times, predict_times

# ...
# Run the functions, Learn the DTs and Show the results/plots

