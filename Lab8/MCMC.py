import random
import time
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools

P_C = {"tak": 0.2, "nie": 0.8}
P_G_given_C = {"tak": {"tak": 0.8, "nie": 0.2}, "nie": {"tak": 0.1, "nie": 0.9}}
P_B_given_C = {
    "tak": {"silny": 0.6, "umiarkowany": 0.3, "brak": 0.1},
    "nie": {"silny": 0.1, "umiarkowany": 0.3, "brak": 0.6},
}
P_L_given_G_B = {
    ("tak", "silny"): {"tak": 0.95, "nie": 0.05},
    ("tak", "umiarkowany"): {"tak": 0.8, "nie": 0.2},
    ("tak", "brak"): {"tak": 0.6, "nie": 0.4},
    ("nie", "silny"): {"tak": 0.3, "nie": 0.7},
    ("nie", "umiarkowany"): {"tak": 0.5, "nie": 0.5},
    ("nie", "brak"): {"tak": 0.1, "nie": 0.9},
}


def initialize_state():
    return {
        "C": random.choice(["tak", "nie"]),
        "G": random.choice(["tak", "nie"]),
        "B": random.choice(["silny", "umiarkowany", "brak"]),
        "L": random.choice(["tak", "nie"]),
    }


def sample_C(state):
    probs = {
        c: P_C[c] * P_G_given_C[c][state["G"]] * P_B_given_C[c][state["B"]]
        for c in ["tak", "nie"]
    }
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    return random.choices(list(probs.keys()), weights=probs.values())[0]


def sample_G(state):
    probs = {
        g: P_G_given_C[state["C"]][g] * P_L_given_G_B[(g, state["B"])][state["L"]]
        for g in ["tak", "nie"]
    }
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    return random.choices(list(probs.keys()), weights=probs.values())[0]


def sample_B(state):
    probs = {
        b: P_B_given_C[state["C"]][b] * P_L_given_G_B[(state["G"], b)][state["L"]]
        for b in ["silny", "umiarkowany", "brak"]
    }
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    return random.choices(list(probs.keys()), weights=probs.values())[0]


def sample_L(state):
    probs = {l: P_L_given_G_B[(state["G"], state["B"])][l] for l in ["tak", "nie"]}
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    return random.choices(list(probs.keys()), weights=probs.values())[0]


def gibbs_sampling(evidence, query, iterations):
    state = initialize_state()
    for var, value in evidence.items():
        state[var] = value

    query_counts = Counter()

    for _ in range(iterations):
        if "C" not in evidence:
            state["C"] = sample_C(state)
        if "G" not in evidence:
            state["G"] = sample_G(state)
        if "B" not in evidence:
            state["B"] = sample_B(state)
        if "L" not in evidence:
            state["L"] = sample_L(state)

        query_counts[state[query]] += 1

    total = sum(query_counts.values())
    return {k: v / total for k, v in query_counts.items()}


def plot_heatmap_for_c():
    queries = ["C", "G", "B", "L"]
    iterations_list = [100, 500, 1000, 5000, 10000]
    evidence_combinations = [
        {"G": g, "B": b, "L": l}
        for g in ["tak", "nie"]
        for b in ["silny", "umiarkowany", "brak"]
        for l in ["tak", "nie"]
    ]

    results = {query: {} for query in queries}
    execution_times = []

    for iters in iterations_list:
        start_time = time.time()
        for query in queries:
            probabilities = []
            for evidence in evidence_combinations:
                result = gibbs_sampling(evidence, query=query, iterations=iters)
                probabilities.append(
                    {
                        "G": evidence.get("G"),
                        "B": evidence.get("B"),
                        "L": evidence.get("L"),
                        f"{query}_prob": result.get("tak", 0),
                    }
                )
            results[query][iters] = pd.DataFrame(probabilities)
        execution_times.append(time.time() - start_time)

    for query in queries:
        for iters in iterations_list:
            df = results[query][iters]
            df["B_L"] = df["B"] + ", " + df["L"]
            heatmap_data = df.pivot(index="G", columns="B_L", values=f"{query}_prob")

            plt.figure(figsize=(12, 6))
            sns.heatmap(
                heatmap_data,
                annot=True,
                cmap="YlGnBu",
                cbar_kws={"label": f"P({query}=tak)"},
            )
            plt.title(f"Heatmapa P({query}=tak) dla {iters} iteracji")
            plt.xlabel("B, L")
            plt.ylabel("G")
            plt.xticks(rotation=45)
            plt.show()
    # Wykres czasu wykonania
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_list, execution_times, marker="o", linestyle="--", color="b")
    plt.title("Czas wykonania algorytmu Gibbs Sampling w zależności od liczby iteracji")
    plt.xlabel("Liczba iteracji")
    plt.ylabel("Czas wykonania (s)")
    plt.grid()
    plt.show()


def plot_heatmap_for_B_combinations(gibbs_sampling_func, query, iterations):
    variables = {
        "C": ["tak", "nie"],
        "L": ["tak", "nie"],
        "G": ["tak", "nie"],
    }
    evidence_combinations = list(itertools.product(*variables.values()))

    heatmap_data = []
    for comb in evidence_combinations:
        evidence = dict(zip(variables.keys(), comb))
        result = gibbs_sampling_func(evidence, query, iterations)
        heatmap_data.append(
            {
                "C": evidence["C"],
                "L": evidence["L"],
                "G": evidence["G"],
                "P(B=silny)": result["silny"],
            }
        )
    df = pd.DataFrame(heatmap_data)

    heatmap_pivot = df.pivot_table(index="C", columns=["L", "G"], values="P(B=silny)")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar=True,
    )
    plt.title(
        f"Rozkład P(B=silny) dla różnych kombinacji dowodów ({iterations} iteracji)"
    )
    plt.xlabel("Dowody (L, G)")
    plt.ylabel("C")
    plt.tight_layout()
    plt.show()


evidence = {"L": "tak", "G": "nie", "B": "silny"}
print(gibbs_sampling(evidence, "C", 1000))
