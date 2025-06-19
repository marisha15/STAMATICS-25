# Question 1

A Wiener Process $W(t)$ is a continuous-time stochastic process that satisfies:
- $W(0) = 0$
- Independent increments
- Normally distributed increments with $W(t) - W(s) \sim \mathcal{N}(0, t - s)$ for $t > s$
- Continuous paths

We simulate a single realization (path) of $W(t)$ over the interval $[0, 1]$ using time discretization.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_wiener_path(T=1, N=1000):
    dt = T / N
    t = np.linspace(0, T, N+1)
    dW = np.random.normal(0, np.sqrt(dt), N)
    W = np.insert(np.cumsum(dW), 0, 0)
    return t, W

# Simulation
T = 1
N = 1000
t, W = simulate_wiener_path(T, N)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t, W, label='Wiener Process')
plt.title('Question 1: Wiener Process Path')
plt.xlabel('Time')
plt.ylabel('W(t)')
plt.grid(True)
plt.legend()
plt.show()
```

---

# Question 2

A **Geometric Brownian Motion (GBM)** is given by:
\[ S(t) = S(0) \cdot \exp\left(\sigma W(t) + \left(\alpha - \frac{1}{2}\sigma^2\right)t\right) \]

We simulate 5 paths of $S(t)$ starting from $S(0) = 100$, with parameters $\alpha = 0.1$, $\sigma = 0.2$.

```python
# Parameters
S0 = 100
alpha = 0.1
sigma = 0.2
T = 1
N = 1000
paths = 5
dt = T / N
t = np.linspace(0, T, N+1)

plt.figure(figsize=(10, 5))

for i in range(paths):
    dW = np.random.normal(0, np.sqrt(dt), N)
    W = np.insert(np.cumsum(dW), 0, 0)
    S = S0 * np.exp(sigma * W + (alpha - 0.5 * sigma**2) * t)
    plt.plot(t, S, label=f"Path {i+1}")

plt.title("Question 2: 5 Paths of Geometric Brownian Motion")
plt.xlabel("Time")
plt.ylabel("S(t)")
plt.grid(True)
plt.legend()
plt.show()
```

---

# Question 3

We aim to show:
\[ \mathbb{E}[W_s W_t] = \min(s, t) \]

### Proof:
Assume $s \leq t$. Then,
\[ W_t = W_s + (W_t - W_s) \]
Since $W_s$ and $(W_t - W_s)$ are independent and $\mathbb{E}[W_t - W_s] = 0$:
\[
\mathbb{E}[W_s W_t] = \mathbb{E}[W_s^2] + \mathbb{E}[W_s (W_t - W_s)] = s + 0 = s = \min(s, t)
\]

Thus, proven.

---

# Question 4

We show:
- $W_t - W_s \sim \mathcal{N}(0, t - s)$
- Increments over disjoint intervals are independent

### Verification by simulation:

```python
samples = 10000
s = 0.3
t = 0.7
increments = []

for _ in range(samples):
    dW = np.random.normal(0, np.sqrt(dt), N)
    W = np.insert(np.cumsum(dW), 0, 0)
    increments.append(W[int(t*N)] - W[int(s*N)])

plt.hist(increments, bins=60, density=True, alpha=0.7)
plt.title(f"Question 4: Histogram of W({t}) - W({s})")
plt.xlabel("Increment")
plt.ylabel("Density")
plt.grid(True)
plt.show()
```

This matches the shape of a normal distribution with mean 0 and variance $t - s = 0.4$.

---

# Question 5

We prove:
\[ \mathbb{E}[W_t | \mathcal{F}_s] = W_s \quad \text{for } 0 \leq s \leq t \]

### Explanation:
From Brownian motion properties:
\[ W_t = W_s + (W_t - W_s) \]
Given $\mathcal{F}_s$ (the information up to time $s$), $W_s$ is known and $W_t - W_s$ is independent of $\mathcal{F}_s$ with mean 0.
\[ \mathbb{E}[W_t | \mathcal{F}_s] = W_s + \mathbb{E}[W_t - W_s | \mathcal{F}_s] = W_s + 0 = W_s \]

This confirms that Brownian motion is a **martingale**.

---
