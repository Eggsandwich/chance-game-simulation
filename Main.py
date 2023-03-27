import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def monte_carlo_simulation_with_bankruptcy(num_simulations, num_games, net_rewards, probabilities, starting_balance):
    gain_history = []
    bankruptcy_count = 0

    for _ in range(num_simulations):
        total_gain = 0
        balance = starting_balance
        for _ in range(num_games):
            gain = random.choices(net_rewards, probabilities)[0]
            total_gain += gain
            balance += gain
            if balance <= 0:
                bankruptcy_count += 1
                break

        gain_history.append(total_gain)

    return gain_history, bankruptcy_count

def calculate_bankruptcy_probability(num_simulations, bankruptcy_count):
    return bankruptcy_count / num_simulations

def calculate_expected_values(rewards, net_rewards, probabilities):
    expected_values = [net_reward * probability for net_reward, probability in zip(net_rewards, probabilities)]
    return expected_values

def calculate_wins_losses(num_simulations, num_games, rewards, probabilities):
    win_history = []
    loss_history = []
    max_win_streak_history = []
    max_loss_streak_history = []

    for _ in range(num_simulations):
        wins = 0
        losses = 0
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for _ in range(num_games):
            reward = random.choices(rewards, probabilities)[0]
            if reward == 0:
                losses += 1
                loss_streak += 1
                win_streak = 0
            else:
                wins += 1
                win_streak += 1
                loss_streak = 0

            max_win_streak = max(max_win_streak, win_streak)
            max_loss_streak = max(max_loss_streak, loss_streak)

        win_history.append(wins)
        loss_history.append(losses)
        max_win_streak_history.append(max_win_streak)
        max_loss_streak_history.append(max_loss_streak)

    avg_wins = np.mean(win_history)
    avg_losses = np.mean(loss_history)
    avg_max_win_streak = np.mean(max_win_streak_history)
    avg_max_loss_streak = np.mean(max_loss_streak_history)

    return avg_wins, avg_losses, avg_max_win_streak, avg_max_loss_streak

def calculate_theoretical_expected_wins_losses(probability_of_win, num_games):
    expected_wins = probability_of_win * num_games
    expected_losses = (1 - probability_of_win) * num_games
    return expected_wins, expected_losses

def calculate_theoretical_max_streak(probability_of_win, num_games):
    expected_max_win_streak = np.log(num_games) / np.log(1 / probability_of_win)
    expected_max_loss_streak = np.log(num_games) / np.log(1 / (1 - probability_of_win))
    return expected_max_win_streak, expected_max_loss_streak
  
def run_simulation():
    # Run the Monte Carlo simulation
    gain_history, bankruptcy_count = monte_carlo_simulation_with_bankruptcy(num_simulations, num_games, net_rewards, probabilities, starting_balance)

    # Calculate and display the average gain per game from the simulation
    average_gain_per_game = np.mean(gain_history) / num_games
    average_expected_value_simulation = average_gain_per_game

    # Calculate the expected values
    expected_values = calculate_expected_values(rewards, net_rewards, probabilities)

    # Calculate wins, losses, and streaks
    avg_wins, avg_losses, avg_max_win_streak, avg_max_loss_streak = calculate_wins_losses(num_simulations, num_games, rewards, probabilities)

    # Display results
    if display_expected_values:
        for reward, net_reward, expected_value in zip(rewards, net_rewards, expected_values):
            print(f"Expected Value for Net Reward ({reward} - {cost_per_game}): {net_reward} = {expected_value}")

    if display_theoretical_expected_value:
        print("Theoretical Expected Value: ", theoretical_expected_value)

    if display_simulation_expected_value:
        print("Average Expected Value from Simulation: ", average_expected_value_simulation)

    if display_total_wins_losses:
        print("Average Wins: ", avg_wins)
        print("Average Losses: ", avg_losses)
 
    if display_max_wins_losses:
        print("Average Maximum Win Streak: ", avg_max_win_streak)
        print("Average Maximum Loss Streak: ", avg_max_loss_streak)

    if display_bankruptcy_probability:
        bankruptcy_probability = calculate_bankruptcy_probability(num_simulations, bankruptcy_count)
        print("Bankruptcy Probability: ", bankruptcy_probability)

    if display_histogram:
        # Visualize the results using a histogram
        sns.histplot(gain_history, kde=True)
        plt.xlabel('Total Net Gain')
        plt.ylabel('Frequency')
        plt.title('Monte Carlo Simulation Results')
        plt.show()
        
    if display_theoretical_wins_losses:
        print("Theoretical Expected Wins: ", theoretical_expected_wins)
        print("Theoretical Expected Losses: ", theoretical_expected_losses)

    if display_theoretical_max_wins_losses:
        print("Theoretical Maximum Win Streak: ", theoretical_max_win_streak)
        print("Theoretical Maximum Loss Streak: ", theoretical_max_loss_streak)

# Set your variables here
cost_per_game = 1000
num_simulations = 1000
num_games = 100
starting_balance = 50000

rewards = [0, 1000, 5000, 10000, 25000, 50000, 250000]
probabilities = [0.48445, 0.45, 0.05, 0.01, 0.005, 0.0005, 0.00005]

# Calculate net rewards by subtracting the cost per game
net_rewards = [reward - cost_per_game for reward in rewards]

# Add the probability of not winning any reward
net_rewards[0] = 0
probabilities[0] = 1 - sum(probabilities[1:])

# Calculate the expected value for each reward
expected_values = [net_reward * probability for net_reward, probability in zip(net_rewards, probabilities)]

# Calculate the theoretical expected value
theoretical_expected_value = sum(expected_values)

# Calculate theoretical values
probability_of_win = sum(probabilities[1:])
theoretical_expected_wins, theoretical_expected_losses = calculate_theoretical_expected_wins_losses(probability_of_win, num_games)
theoretical_max_win_streak, theoretical_max_loss_streak = calculate_theoretical_max_streak(probability_of_win, num_games)

# Set display options for results
display_expected_values = True
display_theoretical_expected_value = True
display_simulation_expected_value = True
display_total_wins_losses = True
display_max_wins_losses = True
display_theoretical_wins_losses = True
display_theoretical_max_wins_losses = True
display_bankruptcy_probability = True
display_histogram = True

# Run the simulation and display the results
run_simulation()

    
