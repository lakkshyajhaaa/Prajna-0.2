import numpy as np

def test_typical_values():
    # Say we have a good match:
    S1 = 0.95
    S2 = 0.75
    M = S1 - S2 # 0.20
    U = 0.95
    
    R_old = 0.4 * S1 + 0.3 * M + 0.3 * U
    print("Old R:", R_old)
    
    Q_norm = 200 / 1000 # typical small crop variance 
    penalty = 0.1 * 0.05 + 0.15 * (1 - Q_norm)
    T_acc = 0.70 + penalty
    print("Old T_acc:", T_acc)
    
    R_new = 0.6 * S1 + 0.2 * M + 0.2 * U
    print("New R:", R_new)
    
    Q_norm_new = min(200 / 300.0, 1.0)
    penalty_new = 0.1 * 0.05 + 0.15 * (1 - Q_norm_new)
    T_acc_new = 0.65 + penalty_new
    print("New T_acc:", T_acc_new)

test_typical_values()
