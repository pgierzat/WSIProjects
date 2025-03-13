import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

SELL = 0
BUY = 1


def next_state_and_reward(s, a):
    if s == 10:
        return (10, 0)
    if a == SELL:
        if s > 0:
            return (s - 1, 1)
        else:
            return (0, 0)
    elif a == BUY:
        if s < 9:
            return (s + 1, 0)
        else:
            return (10, 100)


def epsilon_greedy(Q_values, epsilon):
    if random.random() < epsilon:
        return random.choice([SELL, BUY])
    else:
        return np.argmax(Q_values)


def q_learning_inventory(
    H=4, num_episodes=5000, alpha=0.1, gamma=1.0, epsilon=0.1, seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    Q = np.zeros((H, 11, 2))

    rewards_history = []

    for episode in range(num_episodes):
        s = 3
        t = 0
        done = False
        ep_reward = 0

        while not done:
            if s == 10 or t == H:
                done = True
                continue

            a = epsilon_greedy(Q[t, s, :], epsilon)
            s_next, r = next_state_and_reward(s, a)
            ep_reward += r

            if t + 1 < H and s_next != 10:
                best_next = np.max(Q[t + 1, s_next, :])
                Q[t, s, a] += alpha * (r + gamma * best_next - Q[t, s, a])
            else:
                Q[t, s, a] += alpha * (r - Q[t, s, a])

            s = s_next
            t += 1

        rewards_history.append(ep_reward)
    policy = np.zeros((H, 11), dtype=int)
    for tt in range(H):
        for ss in range(11):
            if ss == 10:
                policy[tt, ss] = -1
            else:
                policy[tt, ss] = np.argmax(Q[tt, ss, :])

    return Q, policy, rewards_history


def run_experiments(
    H=4,
    episodes=5000,
    alpha_list=[0.01, 0.05, 0.1, 0.3],
    epsilon_list=[0.0, 0.05, 0.1, 0.3],
    gamma_list=[1.0],
    seed=42,
):
    results = {}

    for gamma in gamma_list:
        for alpha in alpha_list:
            for epsilon in epsilon_list:
                _, policty, rewards = q_learning_inventory(
                    H=H,
                    num_episodes=episodes,
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    seed=seed,
                )
                last_n = episodes // 10
                if episodes < last_n:
                    last_n = episodes
                avg_reward = np.mean(rewards[-last_n:])
                std_reward = np.std(rewards[-last_n:])

                results[(alpha, epsilon, gamma)] = (avg_reward, std_reward)
    episode_best_rewards = [r for r in rewards]

    plt.figure(figsize=(10, 6))
    plt.plot(
        episode_best_rewards,
        marker="o",
        linestyle="None",
        color="blue",
        label="Najlepsza nagroda w epizodzie",
    )
    plt.xlabel("Epizod")
    plt.ylabel("Nagroda w epizodzie")
    plt.title("Najlepsza nagroda w każdym epizodzie")
    plt.legend()
    plt.show()
    if 1.0 in gamma_list:
        gamma_for_plot = 1.0
        mat = np.zeros((len(alpha_list), len(epsilon_list)))
        for i_a, alpha in enumerate(alpha_list):
            for i_e, epsilon in enumerate(epsilon_list):
                if (alpha, epsilon, gamma_for_plot) in results:
                    mat[i_a, i_e] = results[(alpha, epsilon, gamma_for_plot)][0]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            mat,
            annot=True,
            xticklabels=epsilon_list,
            yticklabels=alpha_list,
            cmap="YlGnBu",
        )
        plt.xlabel("epsilon")
        plt.ylabel("alpha")
        plt.title(
            f"Średnia nagroda ostatnie {last_n} epizodów dla gamma={gamma_for_plot}"
        )
        plt.show()

    return results


def interpret_results(results):
    best_key = None
    best_val = -9999
    for key, val in results.items():
        avg_r, std_r = val
        if avg_r > best_val:
            best_val = avg_r
            best_key = key
    print(
        f"Najlepsza średnia nagroda = {best_val:.2f} uzyskana dla parametrów (alpha, epsilon, gamma) = {best_key}"
    )


if __name__ == "__main__":
    alpha_list = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    epsilon_list = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    gamma_list = [1.0]

    results = run_experiments(
        H=6,
        episodes=100,
        alpha_list=alpha_list,
        epsilon_list=epsilon_list,
        gamma_list=gamma_list,
        seed=42,
    )
    print(f"{max(results)}")

    interpret_results(results)
