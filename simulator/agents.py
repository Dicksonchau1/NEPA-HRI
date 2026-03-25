"""
NEPA-HRI agent reply functions.
Three agents for comparison:
  1. Sentiment-Only    — classify emotion state, pick canned reply
  2. Static-Valence    — fixed positive reply, no trajectory awareness
  3. NEPA-HRI          — stability-envelope aware, Regime A vs B switching
"""
import numpy as np

KAPPA = 1.8  # stability envelope tolerance factor

def reply_sentiment(E_t, S_t, delta_t, theta):
    """
    Sentiment-Only baseline.
    Labels current emotion, picks canned response.
    No trajectory or stability awareness.
    """
    if E_t > 0.7:    return -0.25 + 0.05*np.random.randn()
    elif E_t > 0.4:  return -0.10 + 0.05*np.random.randn()
    else:            return  0.05 + 0.05*np.random.randn()


def reply_static_valence(E_t, S_t, delta_t, theta):
    """
    Static-Valence baseline.
    Fixed positive-valence reply every frame.
    No adaptation to interaction trajectory.
    """
    return -0.15 + 0.08*np.random.randn()


def reply_nepa(E_t, S_t, delta_t, theta):
    """
    NEPA-HRI Trajectory Agent.

    Regime A — inside Omega (|delta_t| <= kappa * S_t):
        Gentle proportional nudge: -theta[0] * delta_t

    Regime B — outside Omega (|delta_t| > kappa * S_t):
        Stronger bounded de-escalation: -theta[1] * sign(delta_t) * min(|delta_t|, 0.8)

    theta[0] = inside_gain  (init 0.4, learned via TD update in simulator)
    theta[1] = outside_gain (init 0.8, learned via TD update in simulator)
    """
    if abs(delta_t) <= KAPPA * S_t:
        return -theta[0] * delta_t + 0.04*np.random.randn()
    else:
        return -theta[1]*np.sign(delta_t)*min(abs(delta_t), 0.8) + 0.04*np.random.randn()
