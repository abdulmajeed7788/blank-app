uncrossable-rush-sim
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# Crash multiplier generator
# ==============================
def generate_crash_multipliers(n, alpha=2.0, xm=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    pareto = rng.pareto(alpha, size=n)
    return xm * (1.0 + pareto)

# ==============================
# Strategy rules
# ==============================
def fixed_target_fn(target):
    return lambda i, last_profit, last_target, bankroll: target

def mixed_plan_fn(low=2.0, high=6.0, high_every=6):
    def fn(i, last_profit, last_target, bankroll):
        if (i + 1) % high_every == 0:
            return high
        return low
    return fn

def make_dynamic_auto_rule(starting_bankroll, base=2.0, conservative=1.5, threshold=0.8):
    def fn(i, last_profit, last_target, bankroll):
        if bankroll < starting_bankroll * threshold:
            return conservative
        return base
    return fn

# ==============================
# Simulation
# ==============================
def simulate_one_trial(rounds, base_bet, starting_bankroll, cashout_fn,
                       alpha=2.0, xm=1.0, house_edge=0.01, rng=None):
    bankroll = starting_bankroll
    last_profit = 0.0
    last_target = 2.0
    if rng is None:
        rng = np.random.default_rng()
    crashes = generate_crash_multipliers(rounds, alpha=alpha, xm=xm, rng=rng)
    for i in range(rounds):
        target = float(cashout_fn(i, last_profit, last_target, bankroll))
        bet = base_bet if bankroll >= base_bet else bankroll
        if bet <= 0:
            return 0.0
        crash = crashes[i]
        if crash >= target:
            payout_mult = target * (1.0 - house_edge)
            profit = bet * payout_mult - bet
            bankroll += profit
        else:
            profit = -bet
            bankroll += profit
        last_profit = profit
        last_target = target
        if bankroll <= 0:
            return 0.0
    return bankroll

def monte_carlo_trials(trials, rounds, base_bet, starting_bankroll, cashout_fn,
                       alpha=2.0, xm=1.0, house_edge=0.01):
    rng = np.random.default_rng()
    finals = np.empty(trials)
    for t in range(trials):
        trial_rng = np.random.default_rng(rng.integers(1, 10**9))
        finals[t] = simulate_one_trial(rounds, base_bet, starting_bankroll,
                                       cashout_fn, alpha, xm, house_edge, rng=trial_rng)
    return finals

# ==============================
# Streamlit UI
# ==============================
st.title("🎲 Uncrossable Rush Strategy Simulator")

# User inputs
bankroll = st.number_input("Starting bankroll ($)", 100, 10000, 500)
bet = st.number_input("Base bet ($)", 1, 100, 5)
rounds = st.slider("Rounds per session", 50, 1000, 250)
trials = st.slider("Monte Carlo trials", 100, 5000, 1000)
alpha = st.selectbox("Crash distribution tail (α)", [1.5, 2.0, 3.0])
strategy_name = st.selectbox("Strategy", [
    "Fixed 1.8x", "Fixed 2.0x", "Fixed 3.0x",
    "Mixed (2x + 1/6 @ 6x)",
    "Dynamic Auto (2x → 1.5x if bankroll <80%)"
])

# Strategy selection
strategies = {
    "Fixed 1.8x": fixed_target_fn(1.8),
    "Fixed 2.0x": fixed_target_fn(2.0),
    "Fixed 3.0x": fixed_target_fn(3.0),
    "Mixed (2x + 1/6 @ 6x)": mixed_plan_fn(2.0, 6.0, 6),
    "Dynamic Auto (2x → 1.5x if bankroll <80%)": make_dynamic_auto_rule(bankroll, 2.0, 1.5, 0.8)
}

if st.button("Run Simulation"):
    finals = monte_carlo_trials(trials, rounds, bet, bankroll,
                                strategies[strategy_name], alpha=alpha)

    # Results
    st.subheader("📊 Results")
    st.write(f"**Mean final bankroll:** ${np.mean(finals):.2f}")
    st.write(f"**Median final bankroll:** ${np.median(finals):.2f}")
    st.write(f"**Std deviation:** ${np.std(finals):.2f}")
    st.write(f"**Risk of ruin (≤ 0):** {100*np.mean(finals <= 0):.2f}%")
    st.write(f"**Chance bankroll ≤ 50%:** {100*np.mean(finals <= bankroll*0.5):.2f}%")
    st.write(f"**Chance profitable (> start):** {100*np.mean(finals > bankroll):.2f}%")

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(finals, bins=40, color="skyblue", edgecolor="black")
    ax.axvline(bankroll, color="red", linestyle="--", label="Starting bankroll")
    ax.set_title("Final Bankroll Distribution")
    ax.set_xlabel("Final Bankroll")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    # Download CSV
    df = pd.DataFrame({"final_bankroll": finals})
    st.download_button("Download results CSV",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name="rush_simulation_results.csv",
                       mime="text/csv")streamlit
numpy
pandas
matplotlib
