import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np, mujoco
from rl_env import ResidualWalkEnv
from walk_sim import SMA

def run(label, k_ankle, tau, cycles=3.0):
    env = ResidualWalkEnv(harness='yz', max_cycles=cycles, random_phase=False)
    L, m, sim = env.L, env.L.m, env.sim
    for s in 'rl':
        m.jnt_stiffness[L.jid[f'ankle_angle_{s}']] = k_ankle
    sim.sma = SMA(m.nu, *tau)
    if tau[0] < 0.01:
        sim.gamma = 0.0
    sim.reset(0.0); env.t_start = 0.0; env.steps = 0
    rows = []; amax = np.zeros(16)
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(24))
        d = sim.d
        amax = np.maximum(amax, sim.sma.a)
        rows.append((sim.t, d.xpos[sim.pelvis][1], info['harness_BW'], info['rmse_deg'], d.qvel[0], np.degrees(d.qpos[sim.jq][[2, 6]])))
        if term or trunc:
            break
    print(f'== {label}: end {sim.t:.2f} s fell {info["fell"]}  act_max>0.98: {int((amax>0.98).sum())} cables {np.where(amax>0.98)[0].tolist()}')
    for t, py, hb, rm, vx, kn in rows[::10]:
        print(f'   t {t:.2f}  pelvis_y {py:.3f}  harness {hb:.2f} BW  rmse {rm:5.1f}°  vx {vx:.2f}  knee R/L {kn[0]:5.1f}/{kn[1]:5.1f}')
    return env

env = run('이상 액추에이터 (k_ankle 0)', 0.0, (1e-3, 1e-3))
run('이상 + SES 100', 100.0, (1e-3, 1e-3))
run('SMA + SES 100', 100.0, (0.12, 0.13))
# 기준 궤적 이음새 연속성
ref = env.ref; T = ref.T
for dt in (0.02,):
    q1, p1, v1 = ref.leg_ref(T - dt); q2, p2, v2 = ref.leg_ref(T + dt)
    print('seam pos jump (deg):', np.round(np.degrees(p2 - p1), 1))
    print('seam vel before/after (deg/s):', np.round(np.degrees(v1), 0), '|', np.round(np.degrees(v2), 0))
    print('pelvis_ty seam', round(q1['pelvis_ty'], 3), round(q2['pelvis_ty'], 3), 'tilt', round(np.degrees(q1['pelvis_tilt']), 1), round(np.degrees(q2['pelvis_tilt']), 1))
