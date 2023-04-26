import sys

from sympy import plot_backends; sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress, permutations, product
from scipy.ndimage import shift
from matplotlib import cm
plt.rcParams.update({'text.usetex': True})


def simulate(move, loc, trap, acce):
    ''' Simulate the ranking at the end of the leg scoring round
    Args:
        - move <list of boolean> [5]: a list of boolean values indicating whether the camel has moved
            Ture - not moved, False - moved
        - loc <list of str> [5]: a list of str values indicating the location of the camel with the
            format of "tile_index-camel_index", where for the camle index: 0 means upper, 4 means lower
            be aware that here the index of camel is alined by LEFT,
            e.g. [[1, 2, 0, 0, 0], [3, 4, 5, 0, 0]...]
        - trap <list of str> [5]: a list of str values indicating the location of the trap with the
            format of "tile_index-0"
        - acce <list of str> [5]: a list of str values indicating the location of the acceleration tile with the
            format of "tile_index-0"
    Returns:
        - ranking <numpy.array> [5, 5]: a 2D array representing the ranking of the camels with the first
            index indicating the camel and the second index indicating the probability of ranking
        - reward <numpy.array> [5]: a 1D array representing the reward of the camels
    Notes:
        - Camel index: 0 - blue, 1 - green, 2 - red, 3 - yellow, 4 - white
    '''
    # Setup basic parameters
    simulate_num = 1000
    init_local = np.zeros(5, dtype=int)
    init_order = np.zeros(5, dtype=int)
    trap_list = []
    acce_list = []
    final_winner = np.zeros(5, dtype=int)
    final_loser = np.zeros(5, dtype=int)
    final_winner_reward = np.zeros(5, dtype=int)
    final_loser_reward = np.zeros(5, dtype=int)

    # Preprocess the input
    move_order = np.array(list(compress(range(5), [j for j in move])))
    for camel_idx, loc_string in enumerate(loc):
        tile_idx, order_idx = loc_string.split('-')
        init_local[camel_idx] = int(tile_idx) - 1
        init_order[camel_idx] = int(order_idx)
    for _, loc_string in enumerate(trap):
        trap_local, _ = loc_string.split('-')
        trap_list.append(int(trap_local) - 1)
    for _, loc_string in enumerate(acce):
        acce_local, _ = loc_string.split('-')
        acce_list.append(int(acce_local) - 1)

    # Simulate the ranking
    ranking = np.zeros((5, 5))
    for i in range(simulate_num):
        local = init_local.copy()
        order = init_order.copy()

        # Randomize the move order and step
        move_step = np.random.randint(1, 4, len(move_order))
        np.random.shuffle(move_order)
        for camel, step in zip(move_order, move_step):
            if local[camel] + step in trap_list: # Check if the camel will move to the trap
                step -= 1

            if local[camel] + step in acce_list: # Check if the camel will move to the trap
                step += 1

            if step == 0: # One step move to the trap
                continue

            same_tile_camels = local == local[camel] # Camels in the same tile
            target_tile_camels = local == (local[camel] + step) # Camels in the target tile
            upper_camels = np.logical_and(order <= order[camel], same_tile_camels) # Camels on the top of current moving camel

            move_camel = np.logical_and(same_tile_camels, upper_camels) # Find the camel to move
            change_order = np.logical_xor(same_tile_camels, upper_camels) # Find the camel to change the order

            local[move_camel] += step # Move the camel
            order[target_tile_camels] += move_camel.sum() # Update the order
            order[change_order] -= move_camel.sum() # Update the order

            if local[camel] > 15: # One camel reach the final line
                rank_value = local * 10 - order
                rank_idx = np.argsort(-rank_value)
                winner = rank_idx[0]
                loser = rank_idx[-1]
                final_winner[winner] += 1
                final_loser[loser] += 1
                break

        # Statistic the ranking
        rank_value = local * 10 - order
        rank_idx = np.argsort(-rank_value)
        for idx, camel_idx in enumerate(rank_idx):
            ranking[camel_idx, idx] += 1

    # Calculate the probability of ranking
    ranking = ranking / simulate_num

    # Calculate the reward
    rewards_5 = ranking[:, 0] * 5 + ranking[:, 1] * 1 + (ranking[:, 2] + ranking[:, 3] + ranking[:, 4]) * (-1)
    rewards_3 = ranking[:, 0] * 3 + ranking[:, 1] * 1 + (ranking[:, 2] + ranking[:, 3] + ranking[:, 4]) * (-1)
    rewards_2 = ranking[:, 0] * 2 + ranking[:, 1] * 1 + (ranking[:, 2] + ranking[:, 3] + ranking[:, 4]) * (-1)
    rewards = np.stack((rewards_5, rewards_3, rewards_2), axis=0)

    # Calculate the final winner and loser
    if final_winner.sum() != 0:
        # final_winner_rate = final_winner / simulate_num
        # final_loser_rate = final_loser / simulate_num
        ending_rate = np.sum(final_winner) / simulate_num
        final_winner = final_winner / final_winner.sum()
        final_loser = final_loser / final_loser.sum()
        final_winner_reward = final_winner * 8 - (1 - final_winner)
        final_loser_reward = final_loser * 8 - (1 - final_loser)
        # final_winner_reward = np.multiply(final_winner_reward, final_winner_rate)
        # final_loser_reward = np.multiply(final_loser_reward, final_loser_rate)
        final_winner_reward = final_winner_reward * ending_rate
        final_loser_reward = final_loser_reward * ending_rate

        print(f'Ending rate: {ending_rate:.2f}')

    plot_result(ranking, rewards, final_winner_reward, final_loser_reward)
    return ranking, rewards, final_winner, final_loser, final_winner_reward, final_loser_reward


def theory(move, loc):
    ''' Calculate the theoritical ranking at the end of the leg scoring round
    Args:
        - move <list of boolean> [5]: a list of boolean values indicating whether the camel has moved
            Ture - not moved, False - moved
        - loc <list of str> [5]: a list of str values indicating the location of the camel with the
            format of "tile_index-camel_index"
    Returns:
        - ranking <numpy.array> [5, 5]: a 2D array representing the ranking of the camels with the first
            index indicating the camel and the second index indicating the probability of ranking
        - reward <numpy.array> [5]: a 1D array representing the reward of the camels
    Notes:
        - Camel index: 0 - blue, 1 - green, 2 - red, 3 - yellow, 4 - white
    '''
    # Setup basic parameters
    init_tile = np.zeros((16, 5), dtype=int) - 1
    init_camel_loc = np.zeros((5, 2), dtype=int)

    # Preprocess the input
    move_camel_list = list(compress(range(5), [j for j in move]))
    move_camel_num = len(move_camel_list)
    for camel_idx, camel_loc_str in enumerate(loc):
        init_tile[int(camel_loc_str.split('-')[0]), int(camel_loc_str.split('-')[1])] = camel_idx
        init_camel_loc[camel_idx] = [int(camel_loc_str.split('-')[0]), int(camel_loc_str.split('-')[1])]

    # Setup the status and probability
    order = list(permutations(move_camel_list, move_camel_num))
    move_step_list = list(product(range(1, 4), repeat=move_camel_num))
    order_num = np.math.factorial(move_camel_num)
    step_num = 3 ** move_camel_num
    ranking = np.zeros((5, 5))

    # Go through all the possible order of the camels
    for i, j in product(range(order_num), range(step_num)):
        # Simulate the movement
        move_order = order[i]
        move_step = move_step_list[j]
        tile = np.copy(init_tile)
        camel_loc = np.copy(init_camel_loc)
        if len(move_order) == 0: break
        for camel_idx, step in zip(move_order, move_step):
            # Create space in the target place 
            tile[camel_loc[camel_idx][0] + step] = \
                shift(tile[camel_loc[camel_idx][0] + step], camel_loc[camel_idx][1] + 1, cval=-1)

            # Move the camel
            tile[camel_loc[camel_idx][0] + step, :camel_loc[camel_idx][1] + 1] = \
                tile[camel_loc[camel_idx][0], :camel_loc[camel_idx][1] + 1]

            # Remove the camel from the original place
            tile[camel_loc[camel_idx][0]] = \
                shift(tile[camel_loc[camel_idx][0]], -camel_loc[camel_idx][1] - 1, cval=-1)

            # Update the location of the camel
            camel_loc_new = np.zeros((5, 2), dtype=int)
            for camel_idx in range(5):
                camel_loc_new[camel_idx] = np.argwhere(tile == camel_idx)[0]
            camel_loc = camel_loc_new

        # Statistic the ranking
        rank_value = camel_loc_new[:, 0] * 10 - camel_loc_new[:, 1]
        rank_idx = np.argsort(-rank_value)
        for idx, camel_idx in enumerate(rank_idx):
            ranking[camel_idx, idx] += 1

    # Calculate the probability of ranking
    ranking = ranking / (order_num * step_num)

    # Calculate the reward
    rewards_5 = ranking[:, 0] * 5 + ranking[:, 1] * 1 + (ranking[:, 2] + ranking[:, 3] + ranking[:, 4]) * (-1)
    rewards_3 = ranking[:, 0] * 3 + ranking[:, 1] * 1 + (ranking[:, 2] + ranking[:, 3] + ranking[:, 4]) * (-1)
    rewards_2 = ranking[:, 0] * 2 + ranking[:, 1] * 1 + (ranking[:, 2] + ranking[:, 3] + ranking[:, 4]) * (-1)
    rewards = np.stack((rewards_5, rewards_3, rewards_2), axis=0)

    plot_result(ranking, rewards)

    return ranking, rewards
        
def plot_result(ranking, rewards, final_winner_reward=None, final_loser_reward=None):
    title_font = {'family': 'Arial Black', 'fontsize': 18, 'fontweight': 'bold'}
    label_font = {'family': 'Arial Black', 'fontsize': 18}

    camel_list = ['Blue', 'Green', 'Red', 'Yellow', 'White']
    camel_color = ['mediumblue', 'forestgreen', 'firebrick', 'gold', 'gainsboro']
    bet_list = ['5', '3', '2']

    for i in range(5): # Plot for each ranking
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.bar(camel_list, ranking[:, i], color=camel_color)

        ax.set_ylim(0, 1)
        ax.set_xlabel('Camel', fontdict=label_font)
        ax.set_ylabel('Probability', fontdict=label_font)
        ax.set_title(f'{i+1}st Ranking Probability', fontdict=title_font)

        ax.grid(axis='both', color='black', alpha=0.1)
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        plt.savefig(f'./static/fig/{i+1}_ranking_probability.png')

    for i in range(5): # Plot for each rewards
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.bar(bet_list, rewards[:, i], color=camel_color[i])

        ax.set_ylim(-1, 5)
        ax.set_xlabel('Bet', fontdict=label_font)
        ax.set_ylabel('Rewards', fontdict=label_font)
        ax.set_title(f'{camel_list[i]} Camel Bet Rewards Analysis', fontdict=title_font)

        ax.grid(axis='both', color='black', alpha=0.1)
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        plt.savefig(f'./static/fig/{camel_list[i]}_rewards.png')

    # Plot for final winner rewards
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.bar(camel_list, final_winner_reward, color=camel_color)

    ax.set_ylim(-1, 8)
    ax.set_xlabel('Bet', fontdict=label_font)
    ax.set_ylabel('Rewards', fontdict=label_font)
    ax.set_title(f'Final Winner Rewards', fontdict=title_font)

    ax.grid(axis='both', color='black', alpha=0.1)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig(f'./static/fig/winner_rewards.png')

    # Plot for final loser rewards
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.bar(camel_list, final_loser_reward, color=camel_color)

    ax.set_ylim(-1, 8)
    ax.set_xlabel('Bet', fontdict=label_font)
    ax.set_ylabel('Rewards', fontdict=label_font)
    ax.set_title(f'Final Loser Rewards', fontdict=title_font)

    ax.grid(axis='both', color='black', alpha=0.1)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig(f'./static/fig/loser_rewards.png')


if __name__ == '__main__':
    # # Test case 1: all moved
    # move = [False, False, False, False, False]
    # loc = ['1-0', '1-1', '1-2', '1-3', '1-4']
    # trap = ['2-0']
    # acce = ['4-0']
    # ranking, rewards = simulate(move, loc, trap, acce)
    # print(f'\nTest Case 1: all moved\n{ranking}\n{rewards}')

    # Test case 2: partile moved
    move = [True, True, True, True, True]
    # loc = ['1-0', '1-1', '1-2', '1-3', '1-4']
    loc = ['16-0', '5-0', '5-1', '5-2', '5-3']
    trap = ['4-0']
    acce = ['6-0']
    ranking, rewards, final_winner, final_loser, final_winner_reward, final_loser_reward = simulate(move, loc, trap, acce)
    print(f'\nTest Case 2: partial moved\n{ranking}\n{rewards}\n{final_winner}\n{final_loser}\n{final_winner_reward}\n{final_loser_reward}')

    # # Test case 3: all not moved
    # move = [True, True, True, True, True]
    # loc = ['1-0', '1-1', '1-2', '1-3', '1-4']
    # trap = ['2-0']
    # acce = ['4-0']
    # ranking, rewards = simulate(move, loc, trap, acce)
    # print(f'\nTest Case 3: all not moved\n{ranking}\n{rewards}')

    # # Test case 4: all moved (theoretically)
    # move = [False, False, False, False, False]
    # loc = ['0-0', '0-1', '0-2', '0-3', '0-4']
    # ranking, rewards = theory(move, loc)
    # print(f'\nTest Case 4: all moved\n{ranking}\n{rewards}')

    # # Test case 5: partile moved (theoretically)
    # move = [False, False, False, True, True]
    # loc = ['0-0', '0-1', '0-2', '0-3', '0-4']
    # ranking, rewards = theory(move, loc)
    # print(f'\nTest Case 5: partial moved\n{ranking}\n{rewards}')

    # # Test case 6: all not moved (theoretically)
    # move = [True, True, True, True, True]
    # loc = ['0-0', '0-1', '0-2', '0-3', '0-4']
    # ranking, rewards = theory(move, loc)
    # print(f'\nTest Case 6: all not moved\n{ranking}\n{rewards}')
