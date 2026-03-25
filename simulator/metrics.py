"""
NEPA-HRI standalone metric functions.
All importable independently.
"""
import numpy as np

def p_escalation(delta, S, kappa=1.8):
    """P(escalation) = fraction of frames where |delta_t| > kappa * S_t"""
    T = len(delta)
    return sum(1 for t in range(1,T) if abs(delta[t]) > kappa*S[t]) / (T-1)

def t_recover(delta, S, kappa=1.8):
    """
    T_recover = mean frames to re-enter Omega after each escalation event.
    T_recover = min{ k : S_{t+k} > |delta_{t+k}| / kappa }
    """
    T = len(delta)
    escalations = [t for t in range(1,T) if abs(delta[t]) > kappa*S[t]]
    recoveries, i = [], 0
    while i < len(escalations):
        esc_t, k = escalations[i], escalations[i]+1
        while k < T and abs(delta[k]) > kappa*S[k]:
            k += 1
        if k < T:
            recoveries.append(k - esc_t)
        while i < len(escalations) and escalations[i] < k:
            i += 1
    return float(np.mean(recoveries)) if recoveries else 0.0

def feeling_load(E, decay=0.75):
    """
    F_t = decay * F_{t-1} + (1-decay) * E_t
    Returns (F_trajectory, total_sum)
    """
    T, F = len(E), np.zeros(len(E))
    F[0] = E[0]
    for t in range(1,T):
        F[t] = decay*F[t-1] + (1-decay)*E[t]
    return F, float(F.sum())

def volatility_index(E, window=4):
    """VI_t = Var(E_{t-window:t})"""
    T, VI = len(E), np.zeros(len(E))
    for t in range(1,T):
        w = E[max(0,t-window):t+1]
        VI[t] = np.var(w) if len(w) > 1 else 0.0
    return VI

def stability_index(E, window=6):
    """S_t = 1 / (1 + Var(E_{t-window:t}))  bounded in (0,1]"""
    return 1.0 / (1.0 + volatility_index(E, window))

def augmentation_score(F):
    """AS_t = F_t - F_{t-1}"""
    return np.diff(F, prepend=F[0])
