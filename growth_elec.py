import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

# ==============================================================================
# 1. パラメータ定義 (factory.py ベース)
# ==============================================================================
class Parameters:
    def __init__(self):
        # --- 部屋のサイズ定義 (245m2 x 高さ4m) ---
        self.k_V_chm = 980.0       # 体積 [m3]
        self.k_A = 740.0           # 表面積 [m2]

        # --- 熱容量 (空気 + 構造物) ---
        self.k_C_chm = 2.0e6       # [J/K] 

        # --- 空調能力 (40kW級) ---
        self.k_tec_power = 400.0   
        
        # --- 熱損失係数 ---
        self.k_U = 6.0             
        
        # --- 換気能力 ---
        self.k_c = 1007            
        self.k_rho = 1.2           
        self.k_u_v = 7.5           
        self.k_leak = 1.0e-4       
        self.k_q = 0.0             

        # --- LED モデル (200m2分) ---
        power_per_channel = 7500.0 
        self.k_Q_mi = np.array([1.0, 1.0, 1.0, 1.0]) * power_per_channel
        self.eta_LUi = np.array([0.3, 0.3, 0.3, 0.3])
        
        # k_I_mi: 光量子束密度(PPFD)。
        self.k_I_mi = np.array([1.0, 1.0, 1.0, 1.0]) * 167.0
        
        # --- 植物パラメータ ---
        self.k_a_med = 200.0       # 栽培面積 [m2]
        self.k_LAI = 53
        self.k_I_p = 3.55e-9
        self.k_p_1 = 5.11e-6
        self.k_p_2 = 2.30e-4
        self.k_p_3 = 6.29e-4
        self.k_Gamma_p = 5.2e-5
        self.k_resp = 4.87e-7
        
        # ★指定パラメータ
        self.k_alpha_beta = 0.544  
        self.k_B_resp = 2.65e-7

# ==============================================================================
# 2. 物理モデル (factory.py ベース)
# ==============================================================================
def plant_factory_model_TCB(t, x, u_MPC, u_I_vec, d_vec, p):
    T, C, B = x
    u_T, u_V = u_MPC
    T_out, C_out = d_vec
    
    u_Ii = np.array(u_I_vec) / 100.0
    
    # I: 光量子束密度 (PPFD)
    I = np.sum(p.eta_LUi * p.k_I_mi * u_Ii)
    # LED発熱: 総量 (W)
    phi_Q_LED = np.sum(p.k_Q_mi * u_Ii)

    temp_effect = (-p.k_p_1 * T**2 + p.k_p_2 * T - p.k_p_3)
    C_comp = C - p.k_Gamma_p
    
    # --- 1. 単位面積あたりの光合成・呼吸 (Density) ---
    if I > 0 and temp_effect > 0 and C_comp > 0:
        phot_num = p.k_I_p * I * temp_effect * C_comp
        phot_den = p.k_I_p * I + temp_effect * C_comp
        # k_a_med (200.0) を掛けない
        phi_C_phot_density = (1 - np.exp(-p.k_LAI * B)) * (phot_num / (phot_den + 1e-9))
    else:
        phi_C_phot_density = 0
        
    phi_C_resp_density = p.k_resp * B * (2**(0.1 * T - 2.5))
    
    # --- 2. 植物の成長 (kg/m2) ---
    dBdt = p.k_alpha_beta * phi_C_phot_density - p.k_B_resp * phi_C_resp_density

    # --- 3. 部屋全体への影響 (Total Flux) ---
    # ここで面積(200.0)を掛けて総量にする
    phi_C_sub_total = p.k_a_med * (phi_C_phot_density - phi_C_resp_density)

    # 温度変化
    phi_Q_TEC = p.k_tec_power * u_T + p.k_q * (T_out - T)
    phi_Q_lo = p.k_A * p.k_U * (T_out - T)
    phi_Q_vx = p.k_c * p.k_rho * (T_out - T) * p.k_u_v * u_V
    dTdt = (phi_Q_vx + phi_Q_lo + phi_Q_TEC + phi_Q_LED) / p.k_C_chm

    # CO2濃度変化 (総量 phi_C_sub_total を使用)
    phi_C_exch = (C_out - C) * p.k_u_v * u_V
    phi_C_leak = (C_out - C) * p.k_leak
    dCdt = (phi_C_exch + phi_C_leak - phi_C_sub_total) / p.k_V_chm

    return [dTdt, dCdt, dBdt]

# ==============================================================================
# 3. 最適化用モデル (CasADi / factory.py ベース)
# ==============================================================================
def plant_factory_model_TCB_ca(x, u, d, p, I_k, phi_Q_LED_k):
    T, C, B = [x[i] for i in range(3)]
    u_T, u_V = [u[i] for i in range(2)] 
    T_out, C_out = d[0], d[1]
    
    temp_effect = (-p.k_p_1 * T**2 + p.k_p_2 * T - p.k_p_3)
    C_comp = C - p.k_Gamma_p
    phot_num = p.k_I_p * I_k * temp_effect * C_comp
    phot_den = p.k_I_p * I_k + temp_effect * C_comp
    
    # 単位面積あたり
    phi_C_phot_density = (1 - ca.exp(-p.k_LAI * B)) * (phot_num / (phot_den + 1e-9))
    phi_C_resp_density = p.k_resp * B * ca.exp((0.1 * T - 2.5) * np.log(2))
    
    # 成長 (密度)
    dBdt = p.k_alpha_beta * phi_C_phot_density - p.k_B_resp * phi_C_resp_density
    
    # CO2総量 (面積を掛ける)
    phi_C_sub_total = p.k_a_med * (phi_C_phot_density - phi_C_resp_density)

    phi_Q_TEC = p.k_tec_power * u_T + p.k_q * (T_out - T)
    phi_Q_lo = p.k_A * p.k_U * (T_out - T)
    phi_Q_vx = p.k_c * p.k_rho * (T_out - T) * p.k_u_v * u_V
    dTdt = (phi_Q_vx + phi_Q_lo + phi_Q_TEC + phi_Q_LED_k) / p.k_C_chm

    phi_C_exch = (C_out - C) * p.k_u_v * u_V
    phi_C_leak = (C_out - C) * p.k_leak
    dCdt = (phi_C_exch + phi_C_leak - phi_C_sub_total) / p.k_V_chm

    return ca.vertcat(dTdt, dCdt, dBdt)

# ==============================================================================
# 4. MPCコントローラ (factory.py ベース: N=72, dt=300)
# ==============================================================================
class MPCController_CasADi_TCB:
    def __init__(self, params, N=72, dt=300):
        self.p = params
        self.N = N
        self.dt = dt
        self.num_states = 3 
        self.num_inputs = 3 
        
        self.x_bounds = [(5, 40), (1.96e-6, 1.7e-2), (1e-6, 0.5)] 
        self.u_bounds_static = [(-100, 100), (0, 1)]
        
        self.R = ca.diagcat(1.0, 1.0, 10) 
        self.w_B = 1e12
        self.P = ca.diagcat(100, 1e8, 0)
        self.W_du = ca.diagcat(1.0, 50.0, 200.0)
        
        self._build_solver()

    def _build_solver(self):
        p = self.p
        x = ca.MX.sym('x', self.num_states)
        u = ca.MX.sym('u', self.num_inputs)
        d = ca.MX.sym('d', 2)
        I_k_sym = ca.MX.sym('I_k_sym')
        phi_Q_LED_k_sym = ca.MX.sym('phi_Q_LED_k_sym')
        
        f = ca.Function('f', [x, u, d, I_k_sym, phi_Q_LED_k_sym], 
                         [plant_factory_model_TCB_ca(x, u, d, p, I_k_sym, phi_Q_LED_k_sym)])
        
        self.opti = ca.Opti()
        self.X = self.opti.variable(self.num_states, self.N + 1)
        self.U_T = self.opti.variable(1, self.N)
        self.U_V = self.opti.variable(1, self.N)
        self.U_I_scalar = self.opti.variable(1, self.N)
        
        self.x0_param = self.opti.parameter(self.num_states)
        self.X_ref_param = self.opti.parameter(self.num_states, self.N + 1)
        self.U_ref_param = self.opti.parameter(self.num_inputs, self.N)
        self.D_pred_param = self.opti.parameter(2, self.N)
        self.U_I_max_param = self.opti.parameter(1, self.N)
        self.u_prev_param = self.opti.parameter(self.num_inputs)
        
        obj = 0
        for k in range(self.N):
            u_k = ca.vertcat(self.U_T[k], self.U_V[k], self.U_I_scalar[k])
            u_Ii_k = self.U_I_scalar[k] / 100.0 
            I_k = ca.sum1(p.eta_LUi * p.k_I_mi * u_Ii_k)
            phi_Q_LED_k = ca.sum1(p.k_Q_mi * u_Ii_k)

            xdot_k = f(self.X[:, k], u_k, self.D_pred_param[:, k], I_k, phi_Q_LED_k)
            dBdt_k = xdot_k[2]
            
            cost_biomass = -self.w_B * dBdt_k 
            e_u = u_k - self.U_ref_param[:, k] 
            cost_input = ca.mtimes([e_u.T, self.R, e_u])
            
            if k == 0:
                du = u_k - self.u_prev_param
            else:
                u_k_prev = ca.vertcat(self.U_T[k-1], self.U_V[k-1], self.U_I_scalar[k-1])
                du = u_k - u_k_prev
            
            cost_slew = ca.mtimes([du.T, self.W_du, du])
            obj += cost_biomass + cost_input + cost_slew
        
        e_term = self.X[:, self.N] - self.X_ref_param[:, self.N]
        obj += ca.mtimes([e_term.T, self.P, e_term]) 
        
        self.opti.minimize(obj)
        
        for k in range(self.N):
            u_k = ca.vertcat(self.U_T[k], self.U_V[k], self.U_I_scalar[k])
            u_Ii_k = self.U_I_scalar[k] / 100.0
            I_k_dyn = ca.sum1(p.eta_LUi * p.k_I_mi * u_Ii_k)
            phi_Q_LED_k_dyn = ca.sum1(p.k_Q_mi * u_Ii_k)
            x_next = self.X[:, k] + self.dt * f(self.X[:, k], u_k, self.D_pred_param[:, k], I_k_dyn, phi_Q_LED_k_dyn)
            self.opti.subject_to(self.X[:, k+1] == x_next)

        for j in range(self.num_states):
            self.opti.subject_to(self.opti.bounded(self.x_bounds[j][0], self.X[j, :], self.x_bounds[j][1]))
        
        self.opti.subject_to(self.opti.bounded(self.u_bounds_static[0][0], self.U_T, self.u_bounds_static[0][1]))
        self.opti.subject_to(self.opti.bounded(self.u_bounds_static[1][0], self.U_V, self.u_bounds_static[1][1]))
        self.opti.subject_to(0 <= self.U_I_scalar)
        self.opti.subject_to(self.U_I_scalar <= self.U_I_max_param) 
        self.opti.subject_to(self.X[:, 0] == self.x0_param)
        
        p_opts = {'expand': True}
        s_opts = {'max_iter': 500, 'print_level': 0, 'tol': 1e-4}
        self.opti.solver('ipopt', p_opts, s_opts)

    def get_control_action(self, x0, t_current, u_prev_val, d_func, ref_func):
        t_pred = np.arange(self.N + 1) * self.dt + t_current
        x_ref_full = ref_func(t_pred, self.num_states)
        x_ref_pred = np.vstack([x_ref_full[:, 0], x_ref_full[:, 1], np.full(self.N + 1, x0[2])])
        u_ref_pred = np.zeros((self.num_inputs, self.N)) 
        d_pred = d_func(t_pred[:-1]).T
        
        u_i_max_vals = []
        u_i_guess_vals = []
        for k in range(self.N):
            t_k = t_current + k * self.dt
            hour = (t_k / 3600.0) % 24.0
            if 4.0 <= hour < 20.0:
                u_i_max_vals.append(100.0)
                u_i_guess_vals.append(50.0)
            else:
                u_i_max_vals.append(0.1)    
                u_i_guess_vals.append(0.0)
        
        self.opti.set_value(self.U_I_max_param, u_i_max_vals)
        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.X_ref_param, x_ref_pred)
        self.opti.set_value(self.U_ref_param, u_ref_pred)
        self.opti.set_value(self.D_pred_param, d_pred)
        self.opti.set_value(self.u_prev_param, u_prev_val)
        
        self.opti.set_initial(self.U_I_scalar, u_i_guess_vals)
        self.opti.set_initial(self.U_T, u_prev_val[0])
        self.opti.set_initial(self.U_V, u_prev_val[1])
        self.opti.set_initial(self.U_I_scalar, u_prev_val[2])
        
        try:
            sol = self.opti.solve()
            return np.array([sol.value(self.U_T)[0], sol.value(self.U_V)[0], sol.value(self.U_I_scalar)[0]])
        except:
            return u_prev_val

# ==============================================================================
# 5. 補助関数
# ==============================================================================
def get_reference(t, num_states):
    f_hz = 1.0 / (24 * 3600)
    t_sec = np.array(t)
    T_ref = 20 - 3 * np.cos(2 * np.pi * f_hz * t_sec)
    C_ref = 9.05e-4 - 1.8e-4 * np.cos(2 * np.pi * f_hz * t_sec)
    
    if t_sec.ndim == 0: return np.array([T_ref, C_ref] + [0]*(num_states-2))
    x_ref = np.zeros((len(t), num_states))
    x_ref[:, 0] = T_ref
    x_ref[:, 1] = C_ref
    return x_ref

def disturbance_function(t):
    f_hz = 1 / (24 * 3600)
    t_sec = np.array(t)
    T_out = 30 - 5 * np.cos(2 * np.pi * f_hz * t_sec)
    C_out = np.full_like(t_sec, 9e-4 ,dtype=float) 
    if t_sec.ndim == 0: return np.array([T_out, C_out])
    return np.vstack([T_out, C_out]).T

# ==============================================================================
# 6. 電気代計算用 関数 (新規追加)
# ==============================================================================
def calculate_electricity_cost(u_hist, dt, p, final_fresh_weight):
    """
    シミュレーション結果(u_hist)から電気代を計算して表示する
    u_hist: [AC_input(-100~100), Vent_input(0~1), Light_scalar(0~100)] の配列 (Steps x 3)
    dt: 制御周期(秒)
    p: パラメータオブジェクト
    final_fresh_weight: 最終収穫量 [kg/m2]
    """
    
    # --- 設定値 ---
    COP_AC = 4.0               # 空調の成績係数
    ELEC_PRICE = 17.0          # 電気代単価 [円/kWh]
    FAN_POWER_KW = 0.5         # 換気ファンの消費電力 [kW] (仮定)
    
    print("\n" + "="*50)
    print(f"=== Electricity Cost Analysis (COP={COP_AC}, Price={ELEC_PRICE} JPY/kWh) ===")
    
    total_energy_kWh = 0.0
    energy_led_kWh = 0.0
    energy_ac_kWh = 0.0
    energy_fan_kWh = 0.0
    
    # u_hist はメインループ終了時にダミーデータが1つ追加されている可能性があるため
    # 長さをチェックしてループする
    steps = len(u_hist)
    
    for k in range(steps):
        # 操作量の取得
        u_ac_val = u_hist[k, 0]    # -100 ~ 100
        u_vent_val = u_hist[k, 1]  # 0 ~ 1
        u_light_val = u_hist[k, 2] # 0 ~ 100
        
        # 1. LED 消費電力 (kW)
        # モデル定義: power_per_channel = 7500W, 4 channels
        # u_light_val は % なので /100 する
        # k_Q_mi はパラメータ定義で既に power_per_channel が入っている
        # 合計W = sum(k_Q_mi * (u_light_val/100))
        # kW変換 = W / 1000
        u_Ii = np.array([u_light_val/100.0]*4)
        power_led_kw = np.sum(p.k_Q_mi * u_Ii) / 1000.0
        
        # 2. 空調 (AC) 消費電力 (kW)
        # k_tec_power は係数(400.0)。 u_ac_val(100) -> 40,000W (熱量)
        # 消費電力 = 熱量 / COP
        thermal_power_w = abs(u_ac_val) * p.k_tec_power
        power_ac_kw = (thermal_power_w / COP_AC) / 1000.0
        
        # 3. 換気ファン 消費電力 (kW)
        power_fan_kw = u_vent_val * FAN_POWER_KW
        
        # 積算 (kWh) = kW * (dt / 3600 h)
        dt_hour = dt / 3600.0
        energy_led_kWh += power_led_kw * dt_hour
        energy_ac_kWh += power_ac_kw * dt_hour
        energy_fan_kWh += power_fan_kw * dt_hour
        
    total_energy_kWh = energy_led_kWh + energy_ac_kWh + energy_fan_kWh
    total_cost = total_energy_kWh * ELEC_PRICE
    
    # 結果表示
    print(f"Total Energy Consumption : {total_energy_kWh:.2f} kWh")
    print(f"Total Electricity Cost   : {int(total_cost):,} JPY")
    
    if final_fresh_weight > 0:
        # 200m2全体での収穫量
        total_yield_kg = final_fresh_weight * p.k_a_med 
        cost_per_kg = total_cost / total_yield_kg
        print(f"Cost per kg (Yield)      : {cost_per_kg:.1f} JPY/kg")
        print(f"  (Assumed Area: {p.k_a_med} m2, Yield: {total_yield_kg:.1f} kg)")
    
    print("-" * 30)
    print("Breakdown:")
    print(f"  - LED Light : {energy_led_kWh:.1f} kWh ({energy_led_kWh/total_energy_kWh*100:.1f}%)")
    print(f"  - A/C (HVAC): {energy_ac_kWh:.1f} kWh ({energy_ac_kWh/total_energy_kWh*100:.1f}%)")
    print(f"  - Vent Fan  : {energy_fan_kWh:.1f} kWh ({energy_fan_kWh/total_energy_kWh*100:.1f}%)")
    print("="*50 + "\n")

# ==================================================================
# 7. メイン処理 (Long-term Simulation)
# ==================================================================
def main():
    print("=== Warehouse Scale Long-term Simulation (Factory Params) ===")
    
    # 目標設定
    fresh_start = 0.03
    fresh_target = 3.0
    dry_start = fresh_start / 20.0  # 0.0015
    dry_target = fresh_target / 20.0 # 0.15
    
    print(f"Goal: {fresh_start:.2f} kg -> {fresh_target:.2f} kg (Fresh Weight)")
    
    p = Parameters()
    mpc = MPCController_CasADi_TCB(p, N=36, dt=600) 
    
    # 初期状態: [Temp, CO2, Biomass]
    # factory.pyに合わせ、ある程度制御しやすい初期値からスタート
    x_current = np.array([20.0, 0.0008, dry_start])
    t_current = 0.0
    
    history_x = [x_current]
    history_t = [t_current]
    history_u = [] 
    
    # 前回制御入力（初期値）
    u_applied = np.array([0.0, 0.5, 0.0])
    
    max_days = 100 # 無限ループ防止用の最大日数
    steps_max = int(max_days * 24 * 3600 / mpc.dt)
    
    start_time = time.time()
    
    print("Simulating... (dt=600s, N=36)")
    
    for step in range(steps_max):
        # 進捗表示 (1日ごと)
        if step % (24 * 3600 // mpc.dt) == 0:
            day = step / (24 * 3600 // mpc.dt)
            fw = x_current[2] * 20.0
            print(f"  Day {day:>2.0f}: Fresh Weight = {fw:.3f} kg/m2 (Elapsed: {time.time()-start_time:.1f}s)")

        # 目標達成判定
        if x_current[2] >= dry_target:
            print(f"  [Target Reached!] at Step {step}")
            break
            
        # MPC計算
        u_next = mpc.get_control_action(x_current, t_current, u_applied, disturbance_function, get_reference)
        u_applied = u_next 
        history_u.append(u_next)
        
        # 物理シミュレーション
        u_TC = u_next[0:2]
        u_I_4d = np.full(4, u_next[2])
        d = disturbance_function(t_current)
        
        sol = solve_ivp(plant_factory_model_TCB, [t_current, t_current+mpc.dt], x_current, 
                        args=(u_TC, u_I_4d, d, p), method='RK45')
        
        x_current = sol.y[:, -1]
        t_current += mpc.dt
        
        # クリップ
        for j in range(3):
             x_current[j] = np.clip(x_current[j], mpc.x_bounds[j][0], mpc.x_bounds[j][1])
        
        history_x.append(x_current)
        history_t.append(t_current)

    # 配列化
    history_x = np.array(history_x)
    history_t = np.array(history_t)
    
    # history_u は長さが1つ足りないので最後を繰り返すか、history_xを1つ削る
    # 電気代計算用に「加工前」のリストを保持しておく（最後の1ステップの操作量も計算に入れるため）
    u_hist_for_cost = np.array(history_u)

    # ここではグラフ用に history_u の最後の値をダミー追加して長さを合わせる
    if len(history_u) > 0:
        history_u.append(history_u[-1])
    history_u = np.array(history_u)

    total_days = history_t[-1] / (24 * 3600)
    final_fresh_weight = history_x[-1, 2] * 20.0
    
    print("="*40)
    print(f"Simulation Finished.")
    print(f"  Total Days   : {total_days:.2f} Days")
    print(f"  Final Weight : {final_fresh_weight:.4f} kg/m2")
    print("="*40)
    
    # --- ★追加機能: 電気代計算 ---
    # 加工前の u_hist_for_cost を渡すことで、シミュレーションステップと整合させる
    calculate_electricity_cost(u_hist_for_cost, mpc.dt, p, final_fresh_weight)
    
    # --- グラフ1: 成長比較 (初日 vs 最終日0:00-24:00) ---
    plot_growth_comparison(history_t, history_x, fresh_target)
    
    # --- グラフ2: 全期間の詳細推移 (Long-term Environment) ---
    plot_long_term_env(history_t, history_x, history_u, p)

def plot_growth_comparison(t_arr, x_arr, target_val):
    w_fresh = x_arr[:, 2] * 20.0
    
    # 1. 初日 (Day 0)
    # 0 <= t < 24h
    day1_mask = (t_arr < 24 * 3600)
    t_day1 = t_arr[day1_mask]
    w_day1 = w_fresh[day1_mask]
    
    # 2. 最終日 (Last Full Day or Partial Day)
    # 最終時刻から「何日目か」を計算
    t_end = t_arr[-1]
    final_day_idx = int(t_end // (24 * 3600))
    
    # 最終日の 0:00 (start) から 24:00 (end) までの範囲を作成
    t_start_last = final_day_idx * 24 * 3600
    t_end_last   = (final_day_idx + 1) * 24 * 3600
    
    last_day_mask = (t_arr >= t_start_last) & (t_arr <= t_end_last)
    t_last = t_arr[last_day_mask]
    w_last = w_fresh[last_day_mask]
    
    # 正規化 (グラフのx軸を0~24時間にするため)
    t_day1_norm = t_day1 / 3600.0
    t_last_norm = (t_last - t_start_last) / 3600.0
    
    # 重みの増分 (0スタートに合わせる)
    w_day1_norm = w_day1 - w_day1[0]
    w_last_norm = w_last - w_last[0]

    # プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 上段: 全体成長
    ax1.plot(t_arr / (24*3600), w_fresh, label='Biomass (Fresh)')
    ax1.axhline(target_val, color='r', linestyle='--', label='Target')
    ax1.set_title(f'Growth Trajectory (Target: {target_val}kg) - Total {t_arr[-1]/(24*3600):.2f} Days')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Weight [kg/m2]')
    ax1.grid(True)
    ax1.legend()
    
    # 下段: 速度比較
    ax2.plot(t_day1_norm, w_day1_norm, 'g-', linewidth=2, label='Day 0 Growth')
    if len(t_last_norm) > 0:
        ax2.plot(t_last_norm, w_last_norm, 'b-', linewidth=2, label=f'Day {final_day_idx} Growth')
    
    ax2.set_title(f'Daily Growth Comparison (Day 0 vs Day {final_day_idx})')
    ax2.set_xlabel('Time of Day [hours]')
    ax2.set_ylabel('Weight Increase in 24h [kg/m2]')
    ax2.set_xlim([0, 24])
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('growth_long_comparison.pdf')
    print("Graph saved as 'growth_long_comparison.pdf'")
    plt.show() # 必要に応じてコメントアウト解除

def plot_long_term_env(t, x_hist, u_hist, p):
    # tは秒単位、x軸を「日」にする
    t_days = t / (24 * 3600)
    
    # u_hist は [Input, TimeStep] なので分解
    # u_hist[:, 0] = AC, [:, 1] = Vent, [:, 2] = LightScalar
    
    # 外乱取得
    d_vals = disturbance_function(t) # (N, 2) or (2, N) check
    if d_vals.shape[0] != len(t):
        d_vals = d_vals.T
    
    # 光強度の計算 (W/m2)
    u_I_scalar = u_hist[:, 2]
    # 全チャンネル同じ値と仮定
    u_I_4d = np.tile(u_I_scalar[:, np.newaxis], (1, 4))
    I_hist = np.sum(p.eta_LUi * p.k_I_mi * (u_I_4d / 100.0), axis=1)

    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # (a) Light
    axs[0].plot(t_days, u_I_scalar, 'm-', linewidth=0.5, label='Light Input (%)')
    ax0r = axs[0].twinx()
    ax0r.plot(t_days, I_hist, 'k--', linewidth=0.5, alpha=0.7, label='Intensity (W/m2)')
    axs[0].set_ylabel('Input [%]')
    ax0r.set_ylabel('Intensity [W/m2]')
    axs[0].legend(loc='upper left')
    ax0r.legend(loc='upper right')
    axs[0].set_title('(a) Light Environment (Full Duration)')
    axs[0].grid(True)
    
    # (b) Temperature
    axs[1].plot(t_days, x_hist[:, 0], 'b-', linewidth=1.0, label='T_in')
    axs[1].plot(t_days, d_vals[:, 0], 'g--', linewidth=0.5, alpha=0.5, label='T_out')
    axs[1].set_ylabel('Temp [°C]')
    axs[1].legend(loc='upper left')
    
    ax1r = axs[1].twinx()
    ax1r.plot(t_days, u_hist[:, 0], 'r', linewidth=0.5, alpha=0.4, label='AC Input')
    ax1r.set_ylabel('AC Input [%]')
    ax1r.set_ylim([-110, 110])
    ax1r.legend(loc='upper right')
    axs[1].set_title('(b) Temperature & AC')
    axs[1].grid(True)
    
    # (c) CO2
    # ppm換算: kg/m3 / 1.83 * 1e6 (概算) または factory.py に従うならそのまま
    # factory.pyのプロットでは *1e6/1.83 している
    co2_ppm = x_hist[:, 1] * 1e6 / 1.83
    axs[2].plot(t_days, co2_ppm, 'b-', linewidth=1.0, label='CO2 (ppm)')
    axs[2].set_ylabel('CO2 [ppm]')
    
    ax2r = axs[2].twinx()
    ax2r.plot(t_days, u_hist[:, 1], 'orange', linewidth=0.5, alpha=0.6, label='Vent Input')
    ax2r.set_ylabel('Vent Input [0-1]')
    axs[2].set_title('(c) CO2 & Ventilation')
    axs[2].grid(True)
    
    # (d) Biomass Growth
    axs[3].plot(t_days, x_hist[:, 2] * 20.0, 'k-', linewidth=2.0, label='Biomass (Fresh)')
    axs[3].set_ylabel('Fresh Weight [kg/m2]')
    axs[3].set_xlabel('Time [Days]')
    axs[3].set_title('(d) Biomass Growth')
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.savefig("long_term_environment_factory_params_elec.pdf")
    print("Graph saved as 'long_term_environment_factory_params_elec.pdf'")
    plt.show()

if __name__ == '__main__':
    main()