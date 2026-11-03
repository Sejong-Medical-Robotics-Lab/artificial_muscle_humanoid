# -*- coding: utf-8 -*-
"""최종 정책 다중 시작위상 평가 + 영상. 사용: python3 -I rl_eval.py <ppo_zip> <outdir> [harness=yz|none]"""
import sys, json, math
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np, mujoco
from stable_baselines3 import PPO
from rl_env import ResidualWalkEnv

zip_path, outdir = sys.argv[1:3]
harness = sys.argv[3] if len(sys.argv) > 3 else 'yz'
harness_arg = harness if harness != 'none' else False
model = PPO.load(zip_path, device='cpu')


def run(policy, t0, cycles=3.0):
    env = ResidualWalkEnv(harness=harness_arg, max_cycles=cycles, random_phase=False)
    env.sim.reset(t0); env.t_start = t0; env.steps = 0
    obs = env._obs()
    rms = []; hb = []; vx = []; log = {'t': [], 'xpos': [], 'a': [], 'q': [], 'qref': []}
    while True:
        a = np.zeros(24) if policy is None else policy.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, info = env.step(a)
        rms.append(info['rmse_deg']); hb.append(info['harness_BW']); vx.append(info['vx'])
        sim = env.sim; d = sim.d
        log['t'].append(sim.t); log['xpos'].append(np.array([d.xpos[b].copy() for b in range(sim.m.nbody)])); log['a'].append(sim.sma.a.copy())
        log['q'].append(d.qpos[sim.jq].copy()); log['qref'].append(env.ref.leg_ref(sim.t)[1])
        if term or trunc:
            break
    for k in log:
        log[k] = np.array(log[k])
    return {'t0': t0, 'time_s': float(env.sim.t - t0), 'fell': bool(info['fell']), 'rmse_deg': float(np.sqrt(np.mean(np.square(rms)))),
            'harness_BW_rms': float(np.sqrt(np.mean(np.square(hb)))), 'vx_mean': float(np.mean(vx))}, log, env


T = 1.0519696989312977
phases = [k * T / 8 for k in range(8)]
res = {'nominal': [], 'policy': []}
best = None
for t0 in phases:
    r0, _, _ = run(None, t0)
    r1, log, env = run(model, t0)
    res['nominal'].append(r0); res['policy'].append(r1)
    if best is None or (r1['time_s'], -r1['rmse_deg']) > (best[0]['time_s'], -best[0]['rmse_deg']):
        best = (r1, log, env)
    print(f"t0 {t0:.2f}: nominal {r0['time_s']:.2f}s fell {r0['fell']} rmse {r0['rmse_deg']:.1f} | policy {r1['time_s']:.2f}s fell {r1['fell']} rmse {r1['rmse_deg']:.1f} harness {r1['harness_BW_rms']:.2f} vx {r1['vx_mean']:.2f}")


def summ(lst):
    return {'time_mean_s': float(np.mean([x['time_s'] for x in lst])), 'time_min_s': float(np.min([x['time_s'] for x in lst])), 'time_max_s': float(np.max([x['time_s'] for x in lst])),
            'complete_frac': float(np.mean([not x['fell'] for x in lst])), 'rmse_mean_deg': float(np.mean([x['rmse_deg'] for x in lst])),
            'harness_mean_BW': float(np.mean([x['harness_BW_rms'] for x in lst])), 'vx_mean': float(np.mean([x['vx_mean'] for x in lst]))}


out = {'harness': harness, 'nominal': summ(res['nominal']), 'policy': summ(res['policy']), 'per_phase': res}
print(json.dumps({'nominal': out['nominal'], 'policy': out['policy']}, indent=1))
json.dump(out, open(f'{outdir}/rl_eval_{harness}.json', 'w'), indent=1)

# 영상: 최고 에피소드
r1, log, env = best
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import font_manager
for f_ in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f_:
        font_manager.fontManager.addfont(f_)
plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
CR, BL = '#9B1B30', '#2F5D8F'
m = env.L.m
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
bi = {n: i for i, n in enumerate(names)}
chains = [['pelvis', 'femur_r', 'tibia_r', 'calcn_r', 'toes_r'], ['pelvis', 'femur_l', 'tibia_l', 'calcn_l', 'toes_l'], ['pelvis', 'torso']]
L = env.L
ref_xpos = []
for t in log['t']:
    L.set_pose(env.ref.coords_at(t)); ref_xpos.append(np.array([L.d.xpos[b].copy() for b in range(m.nbody)]))
ref_xpos = np.array(ref_xpos)
fig, ax = plt.subplots(figsize=(9, 5))
def draw(k):
    ax.clear()
    xp = log['xpos'][k]; xr = ref_xpos[k]
    for ch in chains:
        pts = np.array([xr[bi[n]] for n in ch]); ax.plot(pts[:, 0], pts[:, 1], color='0.75', lw=6, solid_capstyle='round')
    for ch, c in zip(chains, [CR, BL, '0.3']):
        pts = np.array([xp[bi[n]] for n in ch]); ax.plot(pts[:, 0], pts[:, 1], color=c, lw=4, solid_capstyle='round', marker='o', ms=4)
    ax.axhline(0, color='k', lw=1)
    cx = xp[bi['pelvis']][0]
    ax.set_xlim(cx - 1.0, cx + 1.0); ax.set_ylim(-0.1, 1.5); ax.set_aspect('equal')
    ax.set_title(f'잔차 RL 정책(PPO) + SMA 열지연 케이블 16개 — 하네스 {harness}   t = {log["t"][k] - log["t"][0]:.2f} s   (회색 = 기준 IK)', fontsize=10)
    ax.set_xlabel('[m]')
ani = animation.FuncAnimation(fig, draw, frames=range(0, len(log['t']), 2), interval=66)
ani.save(f'{outdir}/video_mj_walk_rl_{harness}.mp4', writer='ffmpeg', fps=15, dpi=110)
draw(int(len(log['t']) * 0.5)); fig.savefig(f'{outdir}/video_mj_walk_rl_{harness}_poster.png', dpi=110)
print('video saved; best episode', r1)
