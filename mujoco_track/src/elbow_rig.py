# -*- coding: utf-8 -*-
"""팔꿈치 1자유도 리그 sim-to-real 준비 — 가상 실물(T3.1 3상태 SMA 모델, 모듈 수준) vs MuJoCo 1차 열지연 모델.
① 리그 MJCF(길항 케이블 2개, 풀리 r=25 mm, 말단 2 kg) ② 식별 프로토콜(P4 등척 스텝) → 1차 모델 파라미터
③ 개루프 검증(식별에 안 쓴 파형) ④ 폐루프 데모 지표(스텝·정현파 대역·열 마진)
사용: python3 -I elbow_rig.py <outdir>
"""
import sys, json, math
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np, mujoco
from scipy.optimize import least_squares

outdir = sys.argv[1]
# ------------------------------------------------------------------ T3.1 v1.0 파라미터 ([문헌]·[가정]) + 모듈 수준 가정
P = dict(As=55.0, Af=72.0, Ms=52.0, Mf=38.0, C=8.0e6,        # 변태 문턱 °C, 응력-문턱 기울기 Pa/K [문헌]
         epsL=0.40, EA=65e9, EM=28e9,                         # 모듈 수준 최대 수축 변형률(EPS_MAX 0.40 — 코일/번들 증폭, 와이어 4.2 % 아님), 탄성계수 [문헌]
         Ca=1.8, hA=3.6, Tamb=25.0, Tmax=120.0, th_min=0.12,  # 열용량 J/K, 등가 열전달(강제공랭 M3·M4) W/K [가정]
         dmax=0.35)                                            # 공랭 연속 듀티 평탄 상한 [가정]
P['Qon'] = P['hA'] * (P['Tmax'] - P['Tamb']) / P['dmax']     # 듀티 1 발열 (W) — 평탄 상한과 자기일관 (977 W)
FMAX_M3 = 450.0        # M3 중형 모듈 상한(N)
L0 = 0.15              # 모듈 활성 길이(m) — 스트로크 0.40·0.15 = 60 mm (M3 상한)
R_PULLEY = 0.025       # 캠/풀리 반지름(T2.7 R25) — 90° 회전 = 39 mm = 26 % 변형률 (스트로크 안)
PRE = 0.40             # 중립 자세 예인장 변형률(= εL, 완전 신장)
A_WIRE = FMAX_M3 / 400e6   # 와이어 등가 단면(응력-문턱 이동용): F_max 에서 회복응력 400 MPa [가정] → ΔT = σ/C ≤ 50 K
K_MOD = FMAX_M3 / P['epsL']  # 모듈 수준 유효 강성 (N/단위변형률): ξ=0·ε=εL 에서 F_max
DT_MARGIN = 10.0       # u_max(T̂) 마진 (T3.3 [가정])


class SMA3:
    """T3.1 3상태 모델: 열 T · 상분율 ξ(히스테리시스, 반전점 스케일링) · 역학 σ = E(ξ)(ε − εL ξ)."""

    def __init__(self, p=P):
        self.p = p
        self.T = p['Tamb']; self.xi = 1.0; self.dirn = 0; self.xi_r = 1.0; self.T_r = p['Tamb']; self.T_prev = p['Tamb']

    def xi_heat(self, T, sig):
        p = self.p; As = p['As'] + sig / p['C']; Af = p['Af'] + sig / p['C']
        if T <= As: return 1.0
        if T >= Af: return 0.0
        return 0.5 * (1 + math.cos(math.pi * (T - As) / (Af - As)))

    def xi_cool(self, T, sig):
        p = self.p; Ms = p['Ms'] + sig / p['C']; Mf = p['Mf'] + sig / p['C']
        if T >= Ms: return 0.0
        if T <= Mf: return 1.0
        return 0.5 * (1 - math.cos(math.pi * (Ms - T) / (Ms - Mf)))   # T3.1: ξ = ½[1 − cos(π(Ms' − T)/(Ms − Mf))]

    def force(self, eps):
        p = self.p
        Kx = K_MOD * (self.xi * p['EM'] + (1 - self.xi) * p['EA']) / p['EA']   # 모듈 강성(상분율 의존)
        return max(0.0, Kx * (eps - p['epsL'] * self.xi))

    def step(self, u, eps, dt):
        p = self.p
        sig = self.force(eps) / A_WIRE
        self.T_prev = self.T
        self.T += (u * p['Qon'] - p['hA'] * (self.T - p['Tamb'])) / p['Ca'] * dt
        heating = self.T >= self.T_prev
        if heating:
            xi_major = self.xi_heat(self.T, sig)
            xi_new = min(self.xi, self.xi_r * xi_major / max(self.xi_heat(self.T_r, sig), 1e-9)) if self.dirn < 0 else min(self.xi, xi_major)
        else:
            xi_major = self.xi_cool(self.T, sig)
            xi_new = max(self.xi, 1 - (1 - self.xi_r) * (1 - xi_major) / max(1 - self.xi_cool(self.T_r, sig), 1e-9)) if self.dirn > 0 else max(self.xi, xi_major)
        d = 1 if heating else -1
        if d != self.dirn:
            self.T_r = self.T; self.xi_r = self.xi; self.dirn = d
        self.xi = float(np.clip(xi_new, 0, 1))
        return self.force(eps)

    def u_cap(self):
        return min(1.0, max(0.0, (self.p['Tmax'] - self.T) / DT_MARGIN))


class SMA1:
    """MuJoCo 쪽 모델(Wiener): 1차 열지연 a(가열 τ_h / 냉각 τ_c) → 온도 선형 항 F = F_max·sat((a − a0)/(a1 − a0)) + 스프링 k(ε − ε0)⁺.
    a 는 '정규화 온도' (a = 1 ↔ 듀티 1 정상온도)."""

    def __init__(self, tau_h, tau_c, a0, a1, Fmax=FMAX_M3, k=0.0, eps0=PRE):
        self.tau_h, self.tau_c, self.a0, self.a1, self.Fmax, self.k, self.eps0 = tau_h, tau_c, a0, a1, Fmax, k, eps0
        self.a = 0.0

    @property
    def T(self):
        return P['Tamb'] + self.a * (P['Tmax'] - P['Tamb']) / P['dmax']   # 듀티 1 정상온도로 환산한 추정 온도

    def step(self, u, eps, dt):
        tau = self.tau_h if u > self.a else self.tau_c
        self.a += (u - self.a) / tau * dt
        self.a = min(max(self.a, 0.0), 1.0)
        g = min(max((self.a - self.a0) / (self.a1 - self.a0), 0.0), 1.0)
        return self.Fmax * g + self.k * max(eps - self.eps0, 0.0)

    def u_cap(self):
        return min(1.0, max(0.0, (P['Tmax'] - self.T) / DT_MARGIN))


class SMA1H:
    """보강안: 1차 열지연(단일 τ) + 히스테리시스 정적 맵 — 가열 문턱(a0h,a1h)·냉각 문턱(a0c,a1c), 방향 플래그로 마이너 루프 근사."""

    def __init__(self, tau, a0h, a1h, a0c, a1c, Fmax=FMAX_M3):
        self.tau, self.a0h, self.a1h, self.a0c, self.a1c, self.Fmax = tau, a0h, a1h, a0c, a1c, Fmax
        self.a = 0.0; self.g = 0.0

    @property
    def T(self):
        return P['Tamb'] + self.a * (P['Tmax'] - P['Tamb']) / P['dmax']

    def step(self, u, eps, dt):
        a_prev = self.a
        self.a += (u - self.a) / self.tau * dt
        self.a = min(max(self.a, 0.0), 1.0)
        if self.a >= a_prev:
            gm = min(max((self.a - self.a0h) / max(self.a1h - self.a0h, 1e-6), 0.0), 1.0)
            self.g = max(self.g, gm)
        else:
            gm = min(max((self.a - self.a0c) / max(self.a1c - self.a0c, 1e-6), 0.0), 1.0)
            self.g = min(self.g, gm)
        return self.Fmax * self.g

    def u_cap(self):
        return min(1.0, max(0.0, (P['Tmax'] - self.T) / DT_MARGIN))


# ------------------------------------------------------------------ 리그 MJCF
XML = f"""
<mujoco model="elbow_rig">
  <compiler angle="radian"/>
  <option timestep="0.0005" gravity="0 -9.81 0" integrator="implicitfast"/>
  <worldbody>
    <body name="forearm" pos="0 0 0">
      <joint name="elbow" type="hinge" axis="0 0 1" range="-1.75 0.8" damping="0.3" frictionloss="0.2" armature="0.002" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1.0" rgba="0.6 0.6 0.65 1"/>
      <body name="load" pos="0.30 0 0"><geom type="sphere" size="0.04" mass="2.0" rgba="0.6 0.1 0.15 1"/></body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="flex"><joint joint="elbow" coef="{-R_PULLEY}"/></fixed>
    <fixed name="ext"><joint joint="elbow" coef="{R_PULLEY}"/></fixed>
  </tendon>
  <actuator>
    <general name="f_flex" tendon="flex" gainprm="-1" ctrlrange="0 5000"/>
    <general name="f_ext" tendon="ext" gainprm="-1" ctrlrange="0 5000"/>
  </actuator>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(XML)
Q_MID = -math.pi / 4          # 케이블 예인장 기준(중립) 자세


def rig_sim(act_flex, act_ext, u_fn, T, dt_c=0.001, q0=-math.pi / 2):
    """리그 순동역학. u_fn(t, q, qd) → (u_flex, u_ext) 듀티. 열 상한 u_max(T̂) 적용."""
    d = mujoco.MjData(m)
    d.qpos[0] = q0
    mujoco.mj_forward(m, d)
    nsub = int(round(dt_c / m.opt.timestep))
    log = []
    t = 0.0
    while t < T:
        q = d.qpos[0]; qd = d.qvel[0]
        uf, ue = u_fn(t, q, qd)
        uf = min(uf, act_flex.u_cap()); ue = min(ue, act_ext.u_cap())
        for _ in range(nsub):
            eps_f = PRE - R_PULLEY * (d.qpos[0] - Q_MID) / L0    # 굴곡(+q) 시 굴근 케이블 단축
            eps_e = PRE + R_PULLEY * (d.qpos[0] - Q_MID) / L0
            Ff = act_flex.step(uf, eps_f, m.opt.timestep)
            Fe = act_ext.step(ue, eps_e, m.opt.timestep)
            d.ctrl[0] = max(Ff, 0.0); d.ctrl[1] = max(Fe, 0.0)
            mujoco.mj_step(m, d)
        t += dt_c
        log.append([t, d.qpos[0], d.qvel[0], d.ctrl[0], d.ctrl[1], uf, ue, act_flex.T, getattr(act_flex, 'xi', float('nan'))])
    return np.array(log)


# ------------------------------------------------------------------ ① 식별 P4: 등척 스텝 ON/OFF × 듀티 3수준 (실물 = SMA3)
def isometric(make, u_levels, t_on=3.0, t_off=4.0, dt=0.001, eps=PRE):
    out = []
    for u in u_levels:
        a = make(); t = 0.0; rec = []
        while t < t_on + t_off:
            uu = min(u, a.u_cap()) if t < t_on else 0.0
            F = a.step(uu, eps, dt); t += dt
            rec.append([t, uu, F, a.T, getattr(a, 'xi', np.nan)])
        out.append(np.array(rec))
    return out


levels = [0.12, 0.18, 0.25, 0.35]
real = isometric(lambda: SMA3(), levels)


def model_resp(params, rec):
    th, tc, a0, a1 = params
    mdl = SMA1(th, tc, a0, a1)
    out = np.zeros(len(rec)); dt = rec[1, 0] - rec[0, 0]
    for i, row in enumerate(rec):
        out[i] = mdl.step(row[1], PRE, dt)
    return out


def resid(params):
    return np.concatenate([(model_resp(params, r) - r[:, 2]) / FMAX_M3 for r in real])


fit = least_squares(resid, x0=[0.5, 0.5, 0.1, 0.2], bounds=([0.02, 0.02, 0.0, 0.02], [5.0, 10.0, 0.5, 1.0]))
th, tc, a0, a1 = fit.x
def model_resp_h(params, rec):
    mdl = SMA1H(*params)
    out = np.zeros(len(rec)); dt = rec[1, 0] - rec[0, 0]
    for i, row in enumerate(rec):
        out[i] = mdl.step(row[1], PRE, dt)
    return out


def resid_h(params):
    return np.concatenate([(model_resp_h(params, r) - r[:, 2]) / FMAX_M3 for r in real])


fit_h = least_squares(resid_h, x0=[0.5, 0.1, 0.3, 0.05, 0.2], bounds=([0.05, 0.0, 0.02, 0.0, 0.01], [5.0, 0.6, 1.0, 0.6, 1.0]))
HP = fit_h.x
ident = {'tau_heat_s': float(th), 'tau_cool_s': float(tc), 'a0': float(a0), 'a1': float(a1), 'Fmax_N': FMAX_M3,
         'hyst_model': {'tau_s': float(HP[0]), 'a0h': float(HP[1]), 'a1h': float(HP[2]), 'a0c': float(HP[3]), 'a1c': float(HP[4]),
                        'fit_rms_pct_of_Fmax': float(np.sqrt(np.mean(fit_h.fun ** 2)) * 100)},
         'fit_rms_pct_of_Fmax': float(np.sqrt(np.mean(fit.fun ** 2)) * 100),
         'protocol': 'P4 등척 스텝 ON 3 s / OFF 4 s × 듀티 {0.12, 0.18, 0.25, 0.35} (공랭 hA 3.6 W/K)', 'Qon_W': P['Qon'],
         'thermal_tau_true_s': P['Ca'] / P['hA']}
for u, r in zip(levels, real):
    i = int(2.9 / 0.001)
    ident[f'steady_u{u}'] = {'F_N': float(r[i, 2]), 'T_C': float(r[i, 3]), 'xi': float(r[i, 4])}
print(json.dumps(ident, indent=1, ensure_ascii=False))

# ------------------------------------------------------------------ ② 개루프 검증: 식별에 안 쓴 파형 (실물 vs 모델 관절각)
def waveform(kind):
    rng = np.random.default_rng(1)
    if kind == 'sine0.3':
        return lambda t, q, qd: (0.16 + 0.08 * math.sin(2 * math.pi * 0.3 * t), 0.16 - 0.08 * math.sin(2 * math.pi * 0.3 * t))
    if kind == 'sine1.0':
        return lambda t, q, qd: (0.16 + 0.08 * math.sin(2 * math.pi * 1.0 * t), 0.16 - 0.08 * math.sin(2 * math.pi * 1.0 * t))
    if kind == 'prbs':
        seq = rng.integers(0, 2, 400)
        return lambda t, q, qd: (0.10 + 0.14 * seq[int(t / 0.4) % 400], 0.10 + 0.14 * (1 - seq[int(t / 0.4) % 400]))
    if kind == 'steps':
        return lambda t, q, qd: ((0.24, 0.08) if (t % 6) < 3 else (0.08, 0.24))


val = {}; logs = {}
for kind in ['sine0.3', 'sine1.0', 'prbs', 'steps']:
    T = 12.0
    lr = rig_sim(SMA3(), SMA3(), waveform(kind), T, q0=Q_MID)
    lm = rig_sim(SMA1(th, tc, a0, a1), SMA1(th, tc, a0, a1), waveform(kind), T, q0=Q_MID)
    lh = rig_sim(SMA1H(*HP), SMA1H(*HP), waveform(kind), T, q0=Q_MID)
    e = np.degrees(lr[:, 1] - lm[:, 1]); eh = np.degrees(lr[:, 1] - lh[:, 1])
    val[kind] = {'rmse_deg': float(np.sqrt(np.mean(e ** 2))), 'max_deg': float(np.abs(e).max()), 'range_deg': float(np.degrees(lr[:, 1].max() - lr[:, 1].min())),
                 'Tmax_real_C': float(np.nanmax(lr[:, 7])), 'Tmax_model_C': float(np.nanmax(lm[:, 7])),
                 'rmse_deg_hyst': float(np.sqrt(np.mean(eh ** 2))), 'max_deg_hyst': float(np.abs(eh).max())}
    logs[kind] = (lr, lm, lh)
    print(kind, {k: round(v, 2) for k, v in val[kind].items()})

# ------------------------------------------------------------------ ③ 폐루프 데모: PD + 길항 바이어스 (실물 플랜트로 재생) — 스텝 +45°, 정현파 대역, 열 마진
def pd_controller(q_ref_fn, kp=25.0, kd=1.5, bias=0.06, Fg=FMAX_M3):
    def u_fn(t, q, qd):
        qr, qdr = q_ref_fn(t)
        tau = kp * (qr - q) + kd * (qdr - qd)
        # 중력 보상 피드포워드 (전완 1 kg @0.15 + 하중 2 kg @0.3)
        tau += (1.0 * 0.15 + 2.0 * 0.30) * 9.81 * math.cos(q)
        f = tau / R_PULLEY
        uf = bias + max(f, 0) / Fg * (a1 - a0) + (a0 if f > 0 else 0)
        ue = bias + max(-f, 0) / Fg * (a1 - a0) + (a0 if f < 0 else 0)
        return min(uf, 1.0), min(ue, 1.0)
    return u_fn


def step_ref(t):
    return (-math.pi / 2 + (math.pi / 4 if t >= 1.0 else 0.0), 0.0)


lr = rig_sim(SMA3(), SMA3(), pd_controller(step_ref), 5.0, q0=-math.pi / 2)
lm = rig_sim(SMA1(th, tc, a0, a1), SMA1(th, tc, a0, a1), pd_controller(step_ref), 5.0, q0=-math.pi / 2)
lh = rig_sim(SMA1H(*HP), SMA1H(*HP), pd_controller(step_ref), 5.0, q0=-math.pi / 2)


def settle(log, target, tol=0.02):
    q = log[:, 1]; t = log[:, 0]
    idx = np.where((np.abs(q - target) > tol * (math.pi / 4)) & (t > 1.0))[0]
    return float(t[idx[-1]] - 1.0) if len(idx) else 0.0


demo = {'step_settle_2pct_real_s': settle(lr, -math.pi / 4), 'step_settle_2pct_model_s': settle(lm, -math.pi / 4),
        'step_ss_err_real_deg': float(np.degrees(abs(lr[-1, 1] + math.pi / 4))), 'step_ss_err_model_deg': float(np.degrees(abs(lm[-1, 1] + math.pi / 4))),
        'step_real_vs_model_rmse_deg': float(np.degrees(np.sqrt(np.mean((lr[:, 1] - lm[:, 1]) ** 2)))),
        'step_real_vs_hyst_rmse_deg': float(np.degrees(np.sqrt(np.mean((lr[:, 1] - lh[:, 1]) ** 2)))), 'step_settle_2pct_hyst_s': settle(lh, -math.pi / 4),
        'step_ss_err_hyst_deg': float(np.degrees(abs(lh[-1, 1] + math.pi / 4))),
        'step_Tmax_real_C': float(np.nanmax(lr[:, 7])), 'step_overshoot_real_deg': float(np.degrees(lr[:, 1].max() + math.pi / 4))}
band = {}
for f in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
    ref = lambda t, f=f: (-math.pi / 3 + 0.3 * math.sin(2 * math.pi * f * t), 0.3 * 2 * math.pi * f * math.cos(2 * math.pi * f * t))
    lg = rig_sim(SMA3(), SMA3(), pd_controller(ref), 6.0, q0=-math.pi / 3)
    seg = lg[lg[:, 0] > 2.0]
    # 기본파 진폭 (최소자승 정현파 적합) / 기준 진폭 0.3 rad
    tt = seg[:, 0]; y = seg[:, 1] - seg[:, 1].mean()
    X = np.column_stack([np.sin(2 * np.pi * f * tt), np.cos(2 * np.pi * f * tt)])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    amp = float(np.hypot(c[0], c[1]) / 0.3)
    phase = float(np.degrees(np.arctan2(c[1], c[0])))
    band[f] = {'amp_ratio': amp, 'phase_deg': phase, 'Tmax_C': float(np.nanmax(lg[:, 7])), 'duty_mean': float(seg[:, 5].mean())}
demo['band'] = band
fs = sorted(band)
ratios = [band[f]['amp_ratio'] for f in fs]
f3 = None
for i in range(1, len(fs)):
    if ratios[i] < 0.707 <= ratios[i - 1]:
        f3 = fs[i - 1] + (ratios[i - 1] - 0.707) / (ratios[i - 1] - ratios[i]) * (fs[i] - fs[i - 1]); break
demo['f_3dB_Hz'] = f3
print(json.dumps(demo, indent=1))
json.dump({'params': P, 'ident': ident, 'openloop_validation': val, 'closedloop_demo': demo,
           'rig': {'Fmax_M3_N': FMAX_M3, 'L0_m': L0, 'pulley_r_m': R_PULLEY, 'load_kg': 2.0, 'arm_m': 0.30, 'pre_strain': PRE}}, open(f'{outdir}/elbow_results.json', 'w'), indent=1, ensure_ascii=False)
np.savez(f'{outdir}/elbow_logs.npz', **{k + '_real': v[0] for k, v in logs.items()}, **{k + '_model': v[1] for k, v in logs.items()}, **{k + '_hyst': v[2] for k, v in logs.items()}, step_real=lr, step_model=lm, step_hyst=lh, levels=np.array(levels))

# ------------------------------------------------------------------ 그림
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f_ in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f_:
        font_manager.fontManager.addfont(f_)
plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
CR, BL, MU = '#9B1B30', '#2F5D8F', '#6E6870'
fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
ax = axes[0]
for u, r in zip(levels, real):
    ax.plot(r[:, 0], r[:, 2], color=CR, lw=1.5, label='가상 실물 (3상태)' if u == levels[0] else None)
    ax.plot(r[:, 0], model_resp(fit.x, r), color='k', lw=1, ls='--', label='1차 열지연+포화 (README 안)' if u == levels[0] else None)
    ax.plot(r[:, 0], model_resp_h(HP, r), color=BL, lw=1.2, ls='-.', label='1차 열지연+히스테리시스 (보강안)' if u == levels[0] else None)
    ax.text(3.05, r[int(2.9 / 0.001), 2] + 5, f'듀티 {u}', fontsize=8, color=MU)
ax.set_title(f'① 식별 P4 등척 스텝: README 안 잔차 {ident["fit_rms_pct_of_Fmax"]:.1f} % → 히스테리시스 보강안 {ident["hyst_model"]["fit_rms_pct_of_Fmax"]:.1f} % (τ {HP[0]:.2f} s)', fontsize=9)
ax.set_xlabel('time (s)'); ax.set_ylabel('힘 (N)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
lr_, lm_, lh_ = logs['prbs']
ax.plot(lr_[:, 0], np.degrees(lr_[:, 1]), color=CR, lw=1.5, label='가상 실물')
ax.plot(lm_[:, 0], np.degrees(lm_[:, 1]), color='k', lw=1.0, ls='--', label='README 안 (1차)')
ax.plot(lh_[:, 0], np.degrees(lh_[:, 1]), color=BL, lw=1.3, ls='-.', label='보강안 (1차+히스테리시스)')
ax.set_title(f'② 개루프 검증 (PRBS 듀티, 식별 미사용): 관절각 RMSE 1차 {val["prbs"]["rmse_deg"]:.1f}° → 보강안 {val["prbs"]["rmse_deg_hyst"]:.1f}° (범위 {val["prbs"]["range_deg"]:.0f}°)', fontsize=9)
ax.set_xlabel('time (s)'); ax.set_ylabel('팔꿈치 각도 (°)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[2]
ax.plot(lr[:, 0], np.degrees(lr[:, 1]), color=CR, lw=1.5, label='가상 실물 + PD')
ax.plot(lm[:, 0], np.degrees(lm[:, 1]), color='k', lw=1.0, ls='--', label='README 안 + PD')
ax.plot(lh[:, 0], np.degrees(lh[:, 1]), color=BL, lw=1.3, ls='-.', label='보강안 + PD')
ax.axhline(-45, color=MU, lw=0.8, ls=':')
f3s = 'n/a' if f3 is None else f'{f3:.2f}'
ax.set_title(f'③ 폐루프 스텝 +45° (하중 2 kg, 공랭·열 상한): 실물 대비 RMSE 1차 {demo["step_real_vs_model_rmse_deg"]:.0f}° → 보강안 {demo["step_real_vs_hyst_rmse_deg"]:.0f}° · −3 dB ≈ {f3s} Hz', fontsize=9)
ax.set_xlabel('time (s)'); ax.set_ylabel('팔꿈치 각도 (°)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle('팔꿈치 1자유도 리그 sim-to-real 준비 — 가상 실물(T3.1 3상태 SMA, 공랭 M3) vs MuJoCo 1차 열지연 모델: 식별 → 개루프 검증 → 폐루프 지표', fontsize=11)
fig.tight_layout()
fig.savefig(f'{outdir}/fig_elbow_sim2real.png', dpi=150)
print('saved')
