# -*- coding: utf-8 -*-
"""MuJoCo 역동역학 재생(mj_inverse) + 정적 케이블 검사.
사용: python3 -I id_replay.py <xml> <osim> <paths.json> <data_dir> <trial> <t0> <t1> <outdir> [grf_lowpass]
"""
import sys, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np, mujoco
from scipy.optimize import lsq_linear
from common import Leg16, lowpass, load_trial, COORDS

xml, osim, pj, data_dir, trial, t0, t1, outdir = sys.argv[1:9]
t0, t1 = float(t0), float(t1)
L = Leg16(xml, osim, pj)
m, d = L.m, L.d
hi, di, hg, dg = load_trial(data_dir, trial)
fs = 1.0 / np.mean(np.diff(di[:, 0]))
dif = di.copy()
dif[:, 1:] = lowpass(di[:, 1:], fs, 6.0)
if len(sys.argv) > 9 and float(sys.argv[9]) > 0:
    dg2 = dg.copy(); dg2[:, 1:] = lowpass(dg[:, 1:], 1.0 / np.mean(np.diff(dg[:, 0])), float(sys.argv[9]))
    dg = dg2
# GRF 를 IK 시간축에 보간
grf = np.column_stack([np.interp(dif[:, 0], dg[:, 0], dg[:, j]) for j in range(dg.shape[1])])

# qpos 궤적
Q = np.array([L.qpos_from_coords(dict(zip(hi, row))) for row in dif])
dt = 1.0 / fs
# qvel via mj_differentiatePos (쿼터니언 처리), 중앙차분
V = np.zeros((len(Q), m.nv))
for k in range(1, len(Q) - 1):
    mujoco.mj_differentiatePos(m, V[k], 2 * dt, Q[k - 1], Q[k + 1])
V[0] = V[1]; V[-1] = V[-2]
A = np.gradient(V, dt, axis=0)

# 역동역학 설정: 등식·접촉 끄고, 감쇠 0
m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_EQUALITY | mujoco.mjtDisableBit.mjDSBL_CONTACT)
m.dof_damping[:] = 0.0
m.dof_armature[:] = 0.0

mass = L.mass
weight = mass * 9.81
sel = (dif[:, 0] >= t0) & (dif[:, 0] <= t1)
idx = np.where(sel)[0]
tau = {s: [] for s in 'rl'}
resid = []
RA = {s: [] for s in 'rl'}
for k in idx:
    d.qpos[:] = Q[k]; d.qvel[:] = V[k]
    mujoco.mj_forward(m, d)          # 주의: mj_forward 가 qacc 를 덮어쓰므로 qacc 는 그 뒤에 넣는다
    qext = L.grf_qfrc(grf[k], hg)
    d.qacc[:] = A[k]
    mujoco.mj_inverse(m, d)
    f = d.qfrc_inverse.copy() - qext
    for s in 'rl':
        tau[s].append(L.joint_moment_eff(f, s))
        RA[s].append(L.moment_arms(s))
    resid.append(f[0:3].copy())
tau = {s: np.array(v) for s, v in tau.items()}
RA = {s: np.array(v) for s, v in RA.items()}
resid = np.array(resid)
vgrf = grf[:, hg.index('ground_force_calcn_r_vy')] + grf[:, hg.index('ground_force_calcn_l_vy')]
grf_peak_n = float(max(np.max(grf[:, hg.index('ground_force_calcn_r_vy')]), np.max(grf[:, hg.index('ground_force_calcn_l_vy')])))
res = {'trial': trial, 'mass_kg': mass, 'window': [t0, t1], 'grf_peak_bw': grf_peak_n / weight,
       'residual_pct': float(np.linalg.norm(resid, axis=1).max() / grf_peak_n * 100),
       'residual_mean_pct': float(np.linalg.norm(resid, axis=1).mean() / grf_peak_n * 100)}
for s in 'rl':
    for ci, c in enumerate(COORDS):
        res[f'peak_{c}_{s}_Nm_per_kg'] = float(np.max(np.abs(tau[s][:, ci])) / mass)
print(json.dumps(res, indent=1))

# ---------------- 정적 케이블 검사: 0 ≤ f ≤ Fmax, R f ≈ 1.1 τ (학생 2단계와 같은 잣대)
margin = 1.1
stat = {}
for s in 'rl':
    Fm = L.Fmax[L.act_idx[s]]
    err = np.zeros_like(tau[s]); u = np.zeros((len(idx), 8))
    for k in range(len(idx)):
        R = RA[s][k]; t = margin * tau[s][k]
        sol = lsq_linear(R, t, bounds=(np.zeros(8), Fm), lsmr_tol='auto')
        err[k] = R @ sol.x - t
        u[k] = sol.x / Fm
    peak = np.max(np.abs(margin * tau[s]), axis=0)
    worst = np.max(np.abs(err), axis=0) / peak * 100
    stat[s] = {'worst_pct_per_coord': worst.tolist(), 'worst_pct': float(worst.max()), 'mean_pct': float(np.mean(np.abs(err) / peak * 100)),
               'u_max_per_cable': u.max(axis=0).tolist(), 'u_mean': float(u.mean())}
    print(s, 'static check worst % per coord', np.round(worst, 2), 'u_max', np.round(u.max(axis=0), 2))
json.dump({'id': res, 'static': stat}, open(f'{outdir}/id_{trial}.json', 'w'), indent=1)
np.savez(f'{outdir}/id_{trial}.npz', t=dif[idx, 0], tau_r=tau['r'], tau_l=tau['l'], resid=resid, RA_r=RA['r'], RA_l=RA['l'], Q=Q[idx], V=V[idx], A=A[idx], grf=grf[idx], vgrf=vgrf[idx])

# ---------------- 그림
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f:
        font_manager.fontManager.addfont(f)
plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
BANDS = {'hip_flexion': (0.7, 1.2), 'knee_angle': (0.4, 0.8), 'ankle_angle': (1.2, 1.6)}
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
tt = dif[idx, 0]
for ax, (ci, c) in zip(axes[:3], [(0, 'hip_flexion'), (2, 'knee_angle'), (3, 'ankle_angle')]):
    lo, hi_ = BANDS[c]
    ax.axhspan(lo, hi_, color='0.9'); ax.axhspan(-hi_, -lo, color='0.9')
    ax.plot(tt, tau['r'][:, ci] / mass, color='#9B1B30', label='오른다리')
    ax.plot(tt, tau['l'][:, ci] / mass, color='#2F5D8F', label='왼다리')
    ax.axhline(0, color='0.6', lw=0.6)
    ax.set_title({'hip_flexion': '고관절 굴곡', 'knee_angle': '무릎', 'ankle_angle': '발목'}[c] + ' 모멘트 (Nm/kg)')
    ax.set_xlabel('time (s)'); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
ax = axes[3]
ax.plot(tt, np.linalg.norm(resid, axis=1) / grf_peak_n * 100, color='k')
ax.axhline(5, color='#9B1B30', ls='--', lw=0.8, label='밴드 5 %')
ax.set_title('골반 잔차력 (% peak GRF)'); ax.set_xlabel('time (s)'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle(f'MuJoCo 역동역학 재생 — subject3 {trial} (IK 6 Hz, GRF 실측, 구간 {t0:.2f}~{t1:.2f} s) · 회색 띠 = 검산 밴드', fontsize=11)
fig.tight_layout()
fig.savefig(f'{outdir}/fig_mj_id_{trial}.png', dpi=150)
print('saved')
