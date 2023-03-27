import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def monte_carlo_simulation(num_simulations, num_games, net_rewards, probabilities):
    gain_history = []

    for _ in range(num_simulations):
        total_gain = 0
        for _ in range(num_games):
            gain = random.choices(net_rewards, probabilities)[0]
            total_gain += gain

        gain_history.append(total_gain)

    return gain_history

def calculate_expected_values(rewards, net_rewards, probabilities):
    expected_values = [net_reward * probability for net_reward, probability in zip(net_rewards, probabilities)]
    return expected_values

def calculate_wins_losses(num_simulations, num_games, rewards, probabilities):
    # Placeholder for future implementation
    pass

def run_simulation():
    # Run the Monte Carlo simulation
    gain_history = monte_carlo_simulation(num_simulations, num_games, net_rewards, probabilities)

    # Calculate and display the average gain per game from the simulation
    average_gain_per_game = np.mean(gain_history) / num_games
    average_expected_value_simulation = average_gain_per_game

    # Calculate the expected values
    expected_values = calculate_expected_values(rewards, net_rewards, probabilities)

    # Display results
    if display_expected_values:
        for reward, net_reward, expected_value in zip(rewards, net_rewards, expected_values):
            print(f"Expected Value for Net Reward ({reward} - {cost_per_game}): {net_reward} = {expected_value}")

    if display_theoretical_expected_value:
        print("Theoretical Expected Value: ", theoretical_expected_value)

    if display_simulation_expected_value:
        print("Average Expected Value from Simulation: ", average_expected_value_simulation)

    if display_histogram:
        # Visualize the results using a histogram
        sns.histplot(gain_history, kde=True)
        plt.xlabel('Total Net Gain')
        plt.ylabel('Frequency')
        plt.title('Monte Carlo Simulation Results')
        plt.show()

# Set your variables here
cost_per_game = 1000
num_simulations = 1000
num_games = 100

rewards = [0, 1000, 5000, 10000, 25000, 50000, 250000]
probabilities = [0.48445, 0.45, 0.05, 0.01, 0.005, 0.0005, 0.00005]

# Calculate net rewards by subtracting the cost per game
net_rewards = [reward - cost_per_game for reward in rewards]

# Add the probability of not winning any reward
net_rewards[0] = 0
probabilities[0] = 1 - sum(probabilities[1:])

# Calculate the expected value for each reward
expected_values = calculate_expected_values(rewards, net_rewards, probabilities)

# Calculate the theoretical expected value
theoretical_expected_value = sum(expected_values)

# Set display options for results
display_expected_values = True
display_theoretical_expected_value = True
display_simulation_expected_value = True
display_histogram = True

# Run the simulation and display the results
run_simulation()
