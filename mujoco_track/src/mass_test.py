# -*- coding: utf-8 -*-
"""질량 0.7배 가설 검증 — 동역학 모델의 바디 질량·관성을 바꾸고 명목 제어기·기존 RL 정책(질량 1.0에서 학습)을
8개 시작 위상(3주기 상한, yz 하네스)에서 재평가. 사용: python3 -I mass_test.py <ppo_zip> <outdir> [lam=0.7]

조건
  base : 질량 1.0 (기준 재실행)
  A    : 균일 λ, 제어기 그대로(ID 피드포워드·PD·발목 강성·하네스 모두 1.0 모델 값)
  B    : 균일 λ, ID 피드포워드만 λ배(가벼운 모델로 ID를 다시 계산한 것과 동일)
  C    : 균일 λ, 피드포워드·PD·발목 강성/감쇠·하네스 모두 λ배 (= 질량 1.0에서 케이블 힘 1/λ배와 수학적으로 등가)
  D    : 분포 변형(정강이·발 0.5, 대퇴 0.8, 골반·몸통은 총질량이 λ가 되도록) + 피드포워드 λ배(근사)
"""
import sys, json, time
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np, mujoco
from stable_baselines3 import PPO
from rl_env import ResidualWalkEnv

zip_path, outdir = sys.argv[1:3]
lam = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
model = PPO.load(zip_path, device='cpu')
T = 1.0519696989312977
phases = [k * T / 8 for k in range(8)]

DISTAL = ('tibia', 'talus', 'calcn', 'toes', 'patella')
FEMUR = ('femur',)


def scale_masses(env, cond):
    sim, L, m = env.sim, env.L, env.sim.m
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or '' for b in range(m.nbody)]
    M0 = float(m.body_mass.sum())
    fac = np.ones(m.nbody)
    if cond in ('A', 'B', 'C'):
        fac[:] = lam
    elif cond == 'D':
        md = sum(m.body_mass[b] for b, n in enumerate(names) if n.startswith(DISTAL))
        mf = sum(m.body_mass[b] for b, n in enumerate(names) if n.startswith(FEMUR))
        mt = M0 - md - mf
        ft = (lam * M0 - 0.5 * md - 0.8 * mf) / mt
        for b, n in enumerate(names):
            fac[b] = 0.5 if n.startswith(DISTAL) else 0.8 if n.startswith(FEMUR) else ft
    m.body_mass[:] = m.body_mass * fac
    m.body_inertia[:] = m.body_inertia * fac[:, None]
    L.mass = float(m.body_mass.sum())
    env.W = L.mass * 9.81
    if cond in ('B', 'C', 'D'):
        sim.ff_scale = lam
    if cond == 'C':
        sim.kp = sim.kp * lam
        sim.kd = sim.kd * lam
        sim.hg = tuple(g * lam for g in sim.hg)
        for s in 'rl':
            m.jnt_stiffness[L.jid[f'ankle_angle_{s}']] *= lam
            m.dof_damping[L.dof[f'ankle_angle_{s}']] *= lam
    info = {'mass_total': L.mass, 'mass_ratio': L.mass / M0}
    if cond == 'D':
        info['factors'] = {'distal': 0.5, 'femur': 0.8, 'trunk': float(ft)}
    return info


def run(env, policy, t0, cycles=3.0):
    env.sim.reset(t0); env.t_start = t0; env.steps = 0
    obs = env._obs()
    rms = []; hb = []; vx = []; amax = np.zeros(env.sim.m.nu); asum = np.zeros(env.sim.m.nu); n = 0
    while True:
        a = np.zeros(24) if policy is None else policy.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, info = env.step(a)
        rms.append(info['rmse_deg']); hb.append(info['harness_BW']); vx.append(info['vx'])
        act = env.sim.sma.a
        amax = np.maximum(amax, act); asum += act; n += 1
        if term or trunc:
            break
    return {'t0': t0, 'time_s': float(env.sim.t - t0), 'fell': bool(info['fell']), 'rmse_deg': float(np.sqrt(np.mean(np.square(rms)))),
            'harness_BW_rms': float(np.sqrt(np.mean(np.square(hb)))), 'vx_mean': float(np.mean(vx)),
            'act_max': amax.tolist(), 'act_mean': float(asum.sum() / n / len(asum))}


def summ(lst):
    return {'time_mean_s': float(np.mean([x['time_s'] for x in lst])), 'time_min_s': float(np.min([x['time_s'] for x in lst])),
            'complete_frac': float(np.mean([not x['fell'] for x in lst])), 'rmse_mean_deg': float(np.mean([x['rmse_deg'] for x in lst])),
            'harness_mean_BW': float(np.mean([x['harness_BW_rms'] for x in lst])), 'vx_mean': float(np.mean([x['vx_mean'] for x in lst])),
            'act_max_mean_per_cable': np.mean([x['act_max'] for x in lst], axis=0).round(2).tolist(),
            'n_cables_saturated': int(np.sum(np.max([x['act_max'] for x in lst], axis=0) > 0.98)),
            'act_mean': float(np.mean([x['act_mean'] for x in lst]))}


out = {'lam': lam, 'conditions': {}}
for cond in ['base', 'A', 'B', 'C', 'D']:
    t_w = time.time()
    env = ResidualWalkEnv(harness='yz', max_cycles=3.0, random_phase=False)
    minfo = scale_masses(env, cond) if cond != 'base' else {'mass_total': env.L.mass, 'mass_ratio': 1.0}
    res = {'nominal': [], 'policy': []}
    for t0 in phases:
        res['nominal'].append(run(env, None, t0))
        res['policy'].append(run(env, model, t0))
    out['conditions'][cond] = {'mass': minfo, 'nominal': summ(res['nominal']), 'policy': summ(res['policy']), 'per_phase': res}
    sn, sp = out['conditions'][cond]['nominal'], out['conditions'][cond]['policy']
    print(f"[{cond}] mass {minfo['mass_total']:.1f} kg ({minfo['mass_ratio']:.2f}) {minfo.get('factors','')}  ({time.time()-t_w:.0f}s)")
    print(f"   nominal: survive {sn['time_mean_s']:.2f} s (min {sn['time_min_s']:.2f}) complete {sn['complete_frac']*8:.0f}/8 rmse {sn['rmse_mean_deg']:.1f}° harness {sn['harness_mean_BW']:.2f} BW vx {sn['vx_mean']:.2f} sat {sn['n_cables_saturated']} act_mean {sn['act_mean']:.2f}")
    print(f"   policy : survive {sp['time_mean_s']:.2f} s (min {sp['time_min_s']:.2f}) complete {sp['complete_frac']*8:.0f}/8 rmse {sp['rmse_mean_deg']:.1f}° harness {sp['harness_mean_BW']:.2f} BW vx {sp['vx_mean']:.2f} sat {sp['n_cables_saturated']} act_mean {sp['act_mean']:.2f}")
    sys.stdout.flush()
json.dump(out, open(f'{outdir}/mass_test.json', 'w'), indent=1)
print('saved', f'{outdir}/mass_test.json')
