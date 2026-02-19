
"""
AI Mathematical Tools – Probability & Random Variables

Instructions:
- Implement ALL functions.
- Do NOT change function names or signatures.
- Do NOT print inside functions.
- You may use: math, numpy, matplotlib.
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Part 1 — Probability Foundations
# ============================================================

def probability_union(PA, PB, PAB):
    """
    P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
    """
    return PA + PB - PAB


def conditional_probability(PAB, PB):
    """
    P(A|B) = P(A ∩ B) / P(B)
    """
    if PB == 0:
        raise ValueError("P(B) must be non-zero")
    return PAB / PB


def are_independent(PA, PB, PAB, tol=1e-9):
    """
    True if:
        |P(A ∩ B) - P(A)P(B)| < tol
    """
    return abs(PAB - (PA * PB)) < tol


def bayes_rule(PBA, PA, PB):
    """
    P(A|B) = P(B|A)P(A) / P(B)
    """
    if PB == 0:
        raise ValueError("P(B) must be non-zero")
    return (PBA * PA) / PB


# ============================================================
# Part 2 — Bernoulli Distribution
# ============================================================

def bernoulli_pmf(x, theta):
    """
    f(x, theta) = theta^x (1-theta)^(1-x)
    """
    if x not in [0, 1]:
        return 0
    return (theta ** x) * ((1 - theta) ** (1 - x))


def bernoulli_theta_analysis(theta_values):
    """
    Returns:
        (theta, P0, P1, is_symmetric)
    """
    results = []
    for theta in theta_values:
        P0 = bernoulli_pmf(0, theta)
        P1 = bernoulli_pmf(1, theta)
        is_symmetric = abs(P0 - P1) < 1e-9
        results.append((theta, P0, P1, is_symmetric))
    return results


# ============================================================
# Part 3 — Normal Distribution
# ============================================================

def normal_pdf(x, mu, sigma):
    """
    Normal PDF:
        1/(sqrt(2π)σ) * exp(-(x-μ)^2 / (2σ^2))
    """
    if sigma <= 0:
        raise ValueError("Standard deviation must be positive")
    coefficient = 1 / (math.sqrt(2 * math.pi) * sigma)
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coefficient * math.exp(exponent)


def normal_histogram_analysis(mu_values,
                              sigma_values,
                              n_samples=10000,
                              bins=30):
    """
    For each (mu, sigma):

    Return:
        (
            mu,
            sigma,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        )
    """
    results = []
    
    for mu in mu_values:
        for sigma in sigma_values:
            # Generate random samples from normal distribution
            samples = np.random.normal(loc=mu, scale=sigma, size=n_samples)
            
            # Calculate sample statistics
            sample_mean = np.mean(samples)
            sample_variance = np.var(samples)
            
            # Theoretical values
            theoretical_mean = mu
            theoretical_variance = sigma ** 2
            
            # Calculate errors
            mean_error = abs(sample_mean - theoretical_mean)
            variance_error = abs(sample_variance - theoretical_variance)
            
            # Optional: create a histogram for visualization
            plt.figure(figsize=(10, 6))
            plt.hist(samples, bins=bins, density=True, alpha=0.7)
            
            # Plot the theoretical PDF
            x = np.linspace(min(samples), max(samples), 1000)
            y = np.array([normal_pdf(val, mu, sigma) for val in x])
            plt.plot(x, y, 'r-', linewidth=2)
            
            plt.title(f'Normal Distribution (μ={mu}, σ={sigma})')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.close()  # Close the figure to avoid displaying it
            
            results.append((
                mu,
                sigma,
                sample_mean,
                theoretical_mean,
                mean_error,
                sample_variance,
                theoretical_variance,
                variance_error
            ))
    
    return results


# ============================================================
# Part 4 — Uniform Distribution
# ============================================================

def uniform_mean(a, b):
    """
    (a + b) / 2
    """
    return (a + b) / 2


def uniform_variance(a, b):
    """
    (b - a)^2 / 12
    """
    return ((b - a) ** 2) / 12


def uniform_histogram_analysis(a_values,
                               b_values,
                               n_samples=10000,
                               bins=30):
    """
    For each (a, b):

    Return:
        (
            a,
            b,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        )
    """
    results = []
    
    for a in a_values:
        for b in b_values:
            if a >= b:
                continue  # Skip invalid parameter combinations
                
            # Generate random samples from uniform distribution
            samples = np.random.uniform(low=a, high=b, size=n_samples)
            
            # Calculate sample statistics
            sample_mean = np.mean(samples)
            sample_variance = np.var(samples)
            
            # Theoretical values
            theoretical_mean = uniform_mean(a, b)
            theoretical_variance = uniform_variance(a, b)
            
            # Calculate errors
            mean_error = abs(sample_mean - theoretical_mean)
            variance_error = abs(sample_variance - theoretical_variance)
            
            # Optional: create a histogram for visualization
            plt.figure(figsize=(10, 6))
            plt.hist(samples, bins=bins, density=True, alpha=0.7)
            
            # Plot the theoretical PDF (which is constant)
            plt.axhline(y=1/(b-a), color='r', linestyle='-', linewidth=2)
            
            plt.title(f'Uniform Distribution (a={a}, b={b})')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.close()  # Close the figure to avoid displaying it
            
            results.append((
                a,
                b,
                sample_mean,
                theoretical_mean,
                mean_error,
                sample_variance,
                theoretical_variance,
                variance_error
            ))
    
    return results


if __name__ == "__main__":
    print("All required functions implemented.")
