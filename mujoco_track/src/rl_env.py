# -*- coding: utf-8 -*-
"""잔차(residual) RL 환경 — 명목 제어기(PD+FF → 케이블 배분 → 열지연 보상) 위에 케이블 활성 잔차 Δu 를 학습.
관측: 다리 q(8)·qd(8)·골반 쿼터니언(4)·골반 속도(6)·활성 a(16)·위상(2) = 44
행동: Δu ∈ [-1,1]^16 × scale → u = clip(u_nom + Δu)
보상: 자세 추종 + 속도 추종 + 골반 높이/자세 + 전진 속도 + 하네스 힘 벌점 + 활성 에너지 벌점 + 생존
"""
import sys, json, math
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np, mujoco
import gymnasium as gym
from gymnasium import spaces
from common import Leg16, COORDS
from walk_sim import Reference, WalkSim, SMA

R = DATA
XML = model('leg16_dyn.xml')
OSIM = f'{R}/data/subject3/subject3_scaled.osim'
PJ = f'{R}/report/newmus_paths_k8.json'
DATA = f'{R}/data/subject3'
IDNPZ = result('id_walking1_full.npz')


def make_sim(harness='yz', dt_c=0.01, k_ankle=100.0, lead=0.08, fscale=1.0, tau=(0.12, 0.13)):
    L = Leg16(XML, OSIM, PJ)
    m = L.m
    L.Fmax[:] = L.Fmax * fscale
    m.actuator_gainprm[:, 0] = m.actuator_gainprm[:, 0] * fscale
    for s in 'rl':
        m.dof_damping[L.dof[f'ankle_angle_{s}']] = 1.0
        j = L.jid[f'ankle_angle_{s}']
        m.jnt_stiffness[j] = k_ankle
        m.qpos_spring[L.qadr[f'ankle_angle_{s}']] = 0.0
    ref = Reference(L, DATA, 'walking1', id_npz=IDNPZ)
    sim = WalkSim(L, ref, harness=harness, ff=True, kp=[150, 150, 120, 40] * 2, kd=[15, 15, 12, 1.5] * 2,
                  harness_gain=(1000, 150, 200, 20), dt_c=dt_c, gamma=2.0)
    sim.sma = SMA(m.nu, *tau)
    sim.ff_lead = lead
    return L, ref, sim


class ResidualWalkEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, harness='yz', max_cycles=3.0, act_scale=0.5, seed=0, w_harness=0.3, random_phase=True, q_scale=0.25):
        super().__init__()
        self.L, self.ref, self.sim = make_sim(harness=harness)
        self.max_t = max_cycles * self.ref.T
        self.act_scale = act_scale
        self.w_harness = w_harness
        self.random_phase = random_phase
        self.W = self.L.mass * 9.81
        self.rng = np.random.default_rng(seed)
        self.q_scale = q_scale
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(44,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(24,), dtype=np.float32)   # 16 케이블 잔차 + 8 관절 목표 잔차(rad)
        self.v_ref = self.ref.stride / self.ref.T

    def _obs(self):
        d = self.sim.d
        ph = (self.sim.t % self.ref.T) / self.ref.T
        o = np.concatenate([d.qpos[self.sim.jq], d.qvel[self.sim.jv] * 0.1, d.xquat[self.sim.pelvis], d.qvel[0:6] * 0.3, self.sim.sma.a,
                            [math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)]])
        return o.astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        t0 = float(self.rng.uniform(0, self.ref.T)) if self.random_phase else 0.0
        self.sim.reset(t0, perturb=0.0)
        self.t_start = t0
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        sim, ref = self.sim, self.ref
        action = np.asarray(action, float)
        sim.q_offset = self.q_scale * action[16:24] if len(action) >= 24 else np.zeros(8)
        u_nom, a_des, tau_des, qref = sim.nominal_control()
        u = np.clip(u_nom + self.act_scale * action[:16], 0, 1)
        Fh = sim.step(u, qref)
        d = sim.d
        q, pos, vel = ref.leg_ref(sim.t)
        eq = d.qpos[sim.jq] - pos
        ev = d.qvel[sim.jv] - vel
        p_ref_y = q['pelvis_ty']
        py = d.xpos[sim.pelvis][1]
        up = d.xmat[sim.pelvis].reshape(3, 3)[:, 1][1]
        vx = d.qvel[0]
        r_pose = math.exp(-2.0 * float(np.mean(eq ** 2)) / 0.1)      # ~ σ 0.22 rad
        r_vel = math.exp(-0.05 * float(np.mean(ev ** 2)))
        r_pelvis = math.exp(-20.0 * (py - p_ref_y) ** 2) * max(up, 0.0)
        r_fwd = math.exp(-2.0 * (vx - self.v_ref) ** 2)
        c_harness = self.w_harness * (np.linalg.norm(Fh) / self.W) ** 2
        c_act = 0.02 * float(np.mean(sim.sma.a ** 2))
        c_action = 0.005 * float(np.mean(np.square(action)))
        rew = 0.4 * r_pose + 0.1 * r_vel + 0.3 * r_pelvis + 0.2 * r_fwd + 0.3 - c_harness - c_act - c_action
        fell = sim.fallen()
        terminated = bool(fell)
        truncated = bool(sim.t - self.t_start >= self.max_t)
        if fell:
            rew -= 2.0
        self.steps += 1
        info = {'rmse_deg': float(np.degrees(np.sqrt(np.mean(eq ** 2)))), 'harness_BW': float(np.linalg.norm(Fh) / self.W), 'vx': float(vx), 'fell': fell}
        return self._obs(), float(rew), terminated, truncated, info


def evaluate(policy_fn, harness='yz', cycles=3.0, start=0.0, seed=0):
    """policy_fn(obs)->action 또는 None(명목 제어기)."""
    env = ResidualWalkEnv(harness=harness, max_cycles=cycles, seed=seed, random_phase=False)
    obs, _ = env.reset()
    rms = []; hb = []; vx = []; rew = 0; T = 0
    log = {'t': [], 'q': [], 'qref': [], 'a': [], 'Fh': [], 'pelvis': [], 'xpos': [], 'grf': []}
    while True:
        a = np.zeros(24) if policy_fn is None else policy_fn(obs)
        obs, r, term, trunc, info = env.step(a)
        rew += r; rms.append(info['rmse_deg']); hb.append(info['harness_BW']); vx.append(info['vx'])
        sim = env.sim; d = sim.d
        log['t'].append(sim.t); log['q'].append(d.qpos[sim.jq].copy()); log['qref'].append(env.ref.leg_ref(sim.t)[1]); log['a'].append(sim.sma.a.copy())
        log['Fh'].append(info['harness_BW']); log['pelvis'].append(d.xpos[sim.pelvis].copy()); log['xpos'].append(np.array([d.xpos[b].copy() for b in range(sim.m.nbody)]))
        log['grf'].append([sim.contact_force('r'), sim.contact_force('l')])
        if term or trunc:
            break
    for k in log:
        log[k] = np.array(log[k])
    return {'return': rew, 'time_s': float(env.sim.t - env.t_start), 'fell': bool(info['fell']), 'rmse_deg': float(np.sqrt(np.mean(np.square(rms)))),
            'harness_BW_rms': float(np.sqrt(np.mean(np.square(hb)))), 'vx_mean': float(np.mean(vx)), 'dist_m': float(log['pelvis'][-1, 0] - log['pelvis'][0, 0])}, log


if __name__ == '__main__':
    import time
    env = ResidualWalkEnv()
    obs, _ = env.reset()
    t0 = time.time(); n = 0; R_ = 0
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(24)); n += 1; R_ += r
        if term or trunc:
            break
    print('nominal: steps', n, 'return', round(R_, 2), 'time', round(env.sim.t - env.t_start, 2), 'fell', info['fell'], 'steps/s', round(n / (time.time() - t0)))
    res, _ = evaluate(None, cycles=3.0)
    print(res)
