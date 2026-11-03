# -*- coding: utf-8 -*-
"""순동역학 보행 결과 그림·영상. 사용: python3 -I walk_report.py <outdir> [ppo_zip]"""
import sys, json, math
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np, mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from rl_env import ResidualWalkEnv, evaluate, make_sim
from walk_sim import SMA

for f_ in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f_:
        font_manager.fontManager.addfont(f_)
plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
CR, BL, GR, MU, GOLD = '#9B1B30', '#2F5D8F', '#2E7D4F', '#6E6870', '#B8860B'
outdir = sys.argv[1]
ppo = sys.argv[2] if len(sys.argv) > 2 else None

# ---------------- 세 조건: 이상 액추에이터 / SMA 열지연 / SMA + 발목 SES 강성(기본)  — 명목 제어기, 1 주기, yz 하네스
def run_cond(k_ankle, tau, label, cycles=1.0):
    env = ResidualWalkEnv(harness='yz', max_cycles=cycles, random_phase=False)
    L, m = env.L, env.L.m
    for s in 'rl':
        m.jnt_stiffness[L.jid[f'ankle_angle_{s}']] = k_ankle
    env.sim.sma = SMA(m.nu, *tau)
    if tau[0] < 0.01:
        env.sim.gamma = 0.0      # 지연 없는 액추에이터는 역보상 불필요 (u = a_des)
    res, log = evaluate(None, harness='yz', cycles=cycles) if False else (None, None)
    # evaluate() 는 새 env 를 만들므로 직접 루프
    obs, _ = env.reset()
    log = {'t': [], 'q': [], 'qref': [], 'a': [], 'Fh': [], 'pelvis': [], 'xpos': [], 'grf': []}
    rms = []
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(24))
        sim = env.sim; d = sim.d
        log['t'].append(sim.t); log['q'].append(d.qpos[sim.jq].copy()); log['qref'].append(env.ref.leg_ref(sim.t)[1]); log['a'].append(sim.sma.a.copy())
        log['Fh'].append(info['harness_BW']); log['pelvis'].append(d.xpos[sim.pelvis].copy()); log['xpos'].append(np.array([d.xpos[b].copy() for b in range(m.nbody)]))
        log['grf'].append([sim.contact_force('r'), sim.contact_force('l')]); rms.append(info['rmse_deg'])
        if term or trunc:
            break
    for k in log:
        log[k] = np.array(log[k])
    err = np.degrees(log['q'] - log['qref'])
    res = {'label': label, 'time_s': float(log['t'][-1] - log['t'][0]), 'fell': bool(info['fell']), 'rmse_deg_per_joint': np.sqrt(np.mean(err ** 2, axis=0)).tolist(),
           'rmse_deg': float(np.sqrt(np.mean(err ** 2))), 'harness_BW_rms': float(np.sqrt(np.mean(np.square(log['Fh'])))), 'dist_m': float(log['pelvis'][-1, 0] - log['pelvis'][0, 0]),
           'act_mean': float(log['a'].mean()), 'grf_peak_BW': (log['grf'].max(axis=0) / (env.W)).tolist()}
    return res, log, env


conds = [('이상 액추에이터(지연 없음)', 0.0, (1e-3, 1e-3)), ('SMA 1차 열지연 0.12/0.13 s', 0.0, (0.12, 0.13)), ('SMA 열지연 + 발목 병렬 강성 100 N·m/rad (SES)', 100.0, (0.12, 0.13))]
results = []; logs = []
for label, k, tau in conds:
    r, lg, env = run_cond(k, tau, label)
    results.append(r); logs.append(lg)
    print(json.dumps(r, ensure_ascii=False))
json.dump(results, open(f'{outdir}/walk_conditions.json', 'w'), indent=1, ensure_ascii=False)

# ---------------- 그림 1: 관절 추종 (오른다리 고관절 굴곡·무릎·발목) 세 조건 + 활성도 + 하네스
fig, axes = plt.subplots(2, 3, figsize=(15, 6.4))
names = ['고관절 굴곡', '고관절 내전', '무릎', '발목']
cols = [MU, GOLD, CR]
for ax, ji in zip(axes[0], [0, 2, 3]):
    lg = logs[2]
    ax.plot(lg['t'], np.degrees(lg['qref'][:, ji]), color='k', lw=1.2, ls='--', label='기준(IK)')
    for (r, lg2), c in zip(zip(results, logs), cols):
        ax.plot(lg2['t'], np.degrees(lg2['q'][:, ji]), color=c, lw=1.4, label=f"{r['label']} (RMSE {r['rmse_deg_per_joint'][ji]:.0f}°)")
    ax.set_title(f'오른다리 {names[ji]} (°)', fontsize=10); ax.set_xlabel('time (s)'); ax.grid(alpha=0.3)
axes[0][0].legend(fontsize=7, loc='best')
ax = axes[1][0]
lg = logs[2]
im = ax.imshow(lg['a'][:, ::2].T, aspect='auto', cmap='Reds', vmin=0, vmax=1, extent=[lg['t'][0], lg['t'][-1], 7.5, -0.5])
ax.set_yticks(range(8)); ax.set_yticklabels([f'M{i}' for i in range(8)]); ax.set_title('케이블 활성 a (오른다리, SMA+SES 조건)', fontsize=10); ax.set_xlabel('time (s)')
plt.colorbar(im, ax=ax, fraction=0.04)
ax = axes[1][1]
for (r, lg2), c in zip(zip(results, logs), cols):
    ax.plot(lg2['t'], lg2['Fh'], color=c, lw=1.3, label=f"{r['label'].split('(')[0].strip()} rms {r['harness_BW_rms']:.2f} BW")
ax.set_title('하네스(높이·좌우·자세만) 지지력 (BW)', fontsize=10); ax.set_xlabel('time (s)'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
ax = axes[1][2]
lg = logs[2]
W = results[2]['grf_peak_BW']
ax.plot(lg['t'], lg['grf'][:, 0] / (65.75 * 9.81), color=CR, label='오른발'); ax.plot(lg['t'], lg['grf'][:, 1] / (65.75 * 9.81), color=BL, label='왼발')
ax.set_title('접촉 수직력 (BW, SMA+SES 조건)', fontsize=10); ax.set_xlabel('time (s)'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle('MuJoCo 순동역학 보행 — 케이블 16개 + 1차 열지연 SMA, 명목 제어기(PD+ID 피드포워드 → 유계 최소자승 배분 → 열지연 보상), 1 보행주기', fontsize=11)
fig.tight_layout()
fig.savefig(f'{outdir}/fig_mj_walk_tracking.png', dpi=150)
print('fig saved')

# ---------------- 영상: 스틱피겨 (시뮬 vs 기준 뼈대) — SMA+SES 조건
lg = logs[2]; env_names = [mujoco.mj_id2name(env.L.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(env.L.m.nbody)]
bi = {n: i for i, n in enumerate(env_names)}
chains = [['pelvis', 'femur_r', 'tibia_r', 'calcn_r', 'toes_r'], ['pelvis', 'femur_l', 'tibia_l', 'calcn_l', 'toes_l'], ['pelvis', 'torso']]
# 기준 뼈대: Reference 좌표로 FK (MuJoCo 기구학)
L = env.L
ref_xpos = []
for t in lg['t']:
    q = env.ref.coords_at(t)
    L.set_pose(q)
    ref_xpos.append(np.array([L.d.xpos[b].copy() for b in range(L.m.nbody)]))
ref_xpos = np.array(ref_xpos)
import matplotlib.animation as animation
fig, ax = plt.subplots(figsize=(9, 5))
def draw(k):
    ax.clear()
    xp = lg['xpos'][k]; xr = ref_xpos[k]
    for ch in chains:
        pts = np.array([xr[bi[n]] for n in ch]); ax.plot(pts[:, 0], pts[:, 1], color='0.75', lw=6, solid_capstyle='round')
    for ch, c in zip(chains, [CR, BL, '0.3']):
        pts = np.array([xp[bi[n]] for n in ch]); ax.plot(pts[:, 0], pts[:, 1], color=c, lw=4, solid_capstyle='round', marker='o', ms=4)
    ax.axhline(0, color='k', lw=1)
    cx = xp[bi['pelvis']][0]
    ax.set_xlim(cx - 1.0, cx + 1.0); ax.set_ylim(-0.1, 1.5); ax.set_aspect('equal')
    ax.set_title(f'MuJoCo 순동역학 — 케이블 16개 + SMA 열지연 + 발목 SES (색) vs 기준 IK (회색)   t = {lg["t"][k]:.2f} s   a_mean {lg["a"][k].mean():.2f}', fontsize=10)
    ax.set_xlabel('[m]')
ani = animation.FuncAnimation(fig, draw, frames=range(0, len(lg['t']), 2), interval=66)
ani.save(f'{outdir}/video_mj_walk_nominal.mp4', writer='ffmpeg', fps=15, dpi=110)
draw(int(len(lg['t']) * 0.35)); fig.savefig(f'{outdir}/video_mj_walk_nominal_poster.png', dpi=110)
print('video saved')
