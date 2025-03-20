import numpy as np

class EZDiffusionModel:
    def __init__(self):
        return None

    def simulate_predicted(self, a, v, t):
        if not isinstance(a, (int, float)) or not (0.5 <= a <= 2):
            raise ValueError("Boundary separation 'a' must be a number between 0.5 and 2.")
        if not isinstance(v, (int, float)) or not (0.5 <= v <= 2):
            raise ValueError("Drift rate 'v' must be a number between 0.5 and 2.")
        if not isinstance(t, (int, float)) or not (0.1 <= t <= 0.5):
            raise ValueError("Nondecision time 't' must be a number between 0.1 and 0.5.")
        y = np.exp(-a * v)
        R_pred = 1 / (1 + y)
        M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
        V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / (y + 1)**2)
        return R_pred, M_pred, V_pred

    def simulate_observed(self, R_pred, M_pred, V_pred, N):
        T_obs = np.random.binomial(N, R_pred)
        R_obs = T_obs / N
        M_obs = np.random.normal(M_pred, V_pred / N)
        V_obs = np.random.gamma((N - 1) / 2, 2 * V_pred / (N - 1))
        return T_obs, M_obs, V_obs, R_obs
    
    def simulate_estimated(self, R_obs, M_obs, V_obs):
        if R_obs == 1: #safeguard suggested by gpt so L does not break
            R_obs = 0.999
        elif R_obs < 0.1:
            R_obs = 0.001
        L = np.log(R_obs / (1 - R_obs))
        v_est = np.sign(R_obs - 0.5) * np.power(L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs, 1/4) #sign function provided by gpt
        if v_est == 0:
            v_est = 0.001
        a_est = L / v_est
        t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))
        return v_est, a_est, t_est

    def recover_parameters(self, a, v, t, N): 
        b = []
        for i in range(1000):
            R_pred, M_pred, V_pred = self.simulate_predicted(a, v, t)
            T_obs, M_obs, V_obs, R_obs = self.simulate_observed(R_pred, M_pred, V_pred, N)
            v_est, a_est, t_est = self.simulate_estimated(R_obs, M_obs, V_obs)
            interString = round(float(v - v_est), 6), round(float(a - a_est), 6), round(float(t - t_est), 6) #b and b^2 is suggested by gpt
            b.append(sum(interString))
        EstimatedBias = sum(b) / 1000
        b_Sq = EstimatedBias ** 2
        return EstimatedBias, b_Sq

model = EZDiffusionModel()
a = np.random.uniform(0.5, 2)
v = np.random.uniform(0.5, 2)
t = np.random.uniform(0.1, 0.5)

b = []
bsq = []
N_values = [10, 40, 4000]
for N in N_values:
    b_int, bsq_int = model.recover_parameters(a, v, t, N)
    b.append(b_int)
    bsq.append(bsq_int)
    print(f"When N is {N}, b = {b_int}, b^2 = {bsq_int}")
