# -*- coding: utf-8 -*-
"""순동역학 보행 시뮬 — SMA 1차 열지연 케이블 16개 + PD/피드포워드 추종 + 유계 최소자승 케이블 배분 + 골반 하네스(선택).
사용: python3 -I walk_sim.py <xml> <osim> <paths.json> <data_dir> <trial> <outdir> [harness=1|0] [cycles=1] [ff=1|0]
"""
import sys, json, math, time
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np, mujoco
from scipy.optimize import lsq_linear
from common import Leg16, lowpass, load_trial, COORDS
from osim_model import mat2quat, rot_x, rot_y, rot_z

TAU_HEAT, TAU_COOL = 0.12, 0.13   # T3.1 [가정]: t_h,min 0.12 s · 액랭 τ_c = C_a/hA = 1.8/14 = 0.129 s (M1 급 케이블 8개 전부 액랭)


class Reference:
    """IK(6 Hz 필터) 기반 기준 궤적 — 한 보행주기(오른발 뒤꿈치 접지→다음 접지)를 반복 재생."""

    def __init__(self, L, data_dir, trial, id_npz=None, hs=(1, 107), n_cycles=1):
        hi, di, hg, dg = load_trial(data_dir, trial)
        self.fs = 1.0 / np.mean(np.diff(di[:, 0]))
        dif = di.copy(); dif[:, 1:] = lowpass(di[:, 1:], self.fs, 6.0)
        self.hdr = hi
        i0, i1 = hs
        self.T = (i1 - i0) / self.fs                      # 주기
        self.stride = dif[i1, hi.index('pelvis_tx')] - dif[i0, hi.index('pelvis_tx')]
        seg = dif[i0:i1 + 1].copy()
        self.t = seg[:, 0] - seg[0, 0]
        self.seg = seg
        self.n_cycles = n_cycles
        self.L = L
        # 좌표별 기준 (보간)
        self.cols = {c: seg[:, hi.index(c)] for c in hi[1:]}
        # 주기 이음새 블렌딩: 끝 10 % 구간에서 시작값과 선형 혼합(골반 tx 제외)
        nb = max(3, int(0.1 * len(seg)))
        w = np.linspace(0, 1, nb)
        for c in hi[1:]:
            if c == 'pelvis_tx':
                continue
            a = self.cols[c]
            a[-nb:] = (1 - w) * a[-nb:] + w * (a[0] + (a[-1] - a[0]) * 0.0)
        # ID 피드포워드 (있으면)
        self.ff = None
        if id_npz is not None:
            z = np.load(id_npz)
            self.ff = {'t': z['t'] - seg[0, 0], 'r': z['tau_r'], 'l': z['tau_l']}

    def coords_at(self, t):
        """시각 t (0 ~ n_cycles·T) → OpenSim 좌표 dict (골반 tx 는 주기마다 stride 만큼 전진)."""
        k = int(t // self.T)
        tl = t - k * self.T
        q = {c: float(np.interp(tl, self.t, v)) for c, v in self.cols.items()}
        q['pelvis_tx'] += k * self.stride
        return q

    def leg_ref(self, t):
        q = self.coords_at(t)
        h = 1e-3
        q2 = self.coords_at(t + h)
        pos = np.array([q[f'{c}_{s}'] for s in 'rl' for c in COORDS])
        pos2 = np.array([q2[f'{c}_{s}'] for s in 'rl' for c in COORDS])
        return q, pos, (pos2 - pos) / h

    def ff_at(self, t, s, lead=0.0):
        if self.ff is None:
            return np.zeros(4)
        t = t + lead
        tl = t - int(t // self.T) * self.T
        tt = self.ff['t']
        if tl < tt[0] or tl > tt[-1]:
            return np.zeros(4)
        return np.array([np.interp(tl, tt, self.ff[s][:, i]) for i in range(4)])


class SMA:
    """1차 열지연(가열/냉각 비대칭) 활성 상태 a ∈ [0,1]."""

    def __init__(self, n, tau_h=TAU_HEAT, tau_c=TAU_COOL):
        self.a = np.zeros(n); self.tau_h = tau_h; self.tau_c = tau_c

    def step(self, u, dt):
        tau = np.where(u > self.a, self.tau_h, self.tau_c)
        self.a += (u - self.a) / tau * dt
        np.clip(self.a, 0, 1, out=self.a)
        return self.a

    def invert(self, a_des, dt_c, gamma=None):
        """열지연 역보상 명령 u. gamma=None 이면 제어주기 끝에 a_des 에 닿는 정확 역산(포화 심함),
        gamma 를 주면 u = a_des + gamma·(a_des − a) 의 부분 보상."""
        if gamma is None:
            tau = np.where(a_des > self.a, self.tau_h, self.tau_c)
            e = np.exp(-dt_c / tau)
            u = (a_des - self.a * e) / (1 - e)
        else:
            u = a_des + gamma * (a_des - self.a)
        return np.clip(u, 0, 1)


W_ROWS = np.array([1.0, 1.0, 1.0, 2.0])   # 발목(관성 작음)을 우선 실현


def allocate(R, tau, Fmax, reg=1e-4, w=W_ROWS):
    """0 ≤ f ≤ Fmax, min ||W(R f − τ)||² + reg·||f/Fmax||²."""
    A = np.vstack([w[:, None] * R, np.sqrt(reg) * np.diag(1.0 / Fmax)])
    b = np.concatenate([w * tau, np.zeros(len(Fmax))])
    sol = lsq_linear(A, b, bounds=(np.zeros(len(Fmax)), Fmax), method='bvls', tol=1e-8)
    return sol.x


def quat_err(q_ref, q):
    """회전 오차 벡터(월드) ≈ 2·vec(q_ref ⊗ q⁻¹)."""
    qi = np.array([q[0], -q[1], -q[2], -q[3]])
    w = q_ref[0] * qi[0] - q_ref[1] * qi[1] - q_ref[2] * qi[2] - q_ref[3] * qi[3]
    x = q_ref[0] * qi[1] + q_ref[1] * qi[0] + q_ref[2] * qi[3] - q_ref[3] * qi[2]
    y = q_ref[0] * qi[2] - q_ref[1] * qi[3] + q_ref[2] * qi[0] + q_ref[3] * qi[1]
    z = q_ref[0] * qi[3] + q_ref[1] * qi[2] - q_ref[2] * qi[1] + q_ref[3] * qi[0]
    v = np.array([x, y, z])
    return 2 * v * (1 if w >= 0 else -1)


class WalkSim:
    def __init__(self, L: Leg16, ref: Reference, harness=True, ff=True, kp=None, kd=None, dt_c=0.005, seed=0,
                 harness_gain=(1000.0, 150.0, 200.0, 20.0), gamma=2.0):
        self.L, self.ref = L, ref
        self.gamma = gamma
        self.ff_lead = 0.0
        self.ff_scale = 1.0
        self.vmc = None          # (kpy, kdy, kpx, kdx) 또는 None
        self.q_offset = np.zeros(8)
        self.m, self.d = L.m, L.d
        self.harness, self.ff = harness, ff
        self.dt_c = dt_c
        self.nsub = int(round(dt_c / self.m.opt.timestep))
        self.kp = np.array(kp if kp is not None else [150, 150, 120, 60] * 2, float)
        self.kd = np.array(kd if kd is not None else [15, 15, 12, 6] * 2, float)
        self.hg = harness_gain
        self.sma = SMA(self.m.nu)
        self.jq = [L.qadr[f'{c}_{s}'] for s in 'rl' for c in COORDS]
        self.jv = [L.dof[f'{c}_{s}'] for s in 'rl' for c in COORDS]
        self.pelvis = L.bid['pelvis']
        self.calcn = {s: L.bid[f'calcn_{s}'] for s in 'rl'}
        self.rng = np.random.default_rng(seed)

    def reset(self, t0=0.0, perturb=0.0):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        q, pos, vel = self.ref.leg_ref(t0)
        d.qpos[:] = self.L.qpos_from_coords(q)
        d.qpos[1] += 0.0
        # 기준 속도: 좌표 수치미분
        h = 1e-3
        q2 = self.ref.coords_at(t0 + h)
        qp2 = self.L.qpos_from_coords(q2)
        mujoco.mj_differentiatePos(m, d.qvel, h, d.qpos, qp2)
        if perturb > 0:
            d.qpos[7:] += self.rng.normal(0, perturb, m.nq - 7)
        self.sma.a[:] = 0.3
        self.t = t0
        mujoco.mj_forward(m, d)
        return self.obs()

    def obs(self):
        d = self.d
        q = d.qpos[self.jq]; v = d.qvel[self.jv]
        return np.concatenate([q, v, d.qpos[3:7], d.qvel[0:6], self.sma.a, [self.t % self.ref.T / self.ref.T]])

    def nominal_control(self):
        """PD + FF → 케이블 배분 → 열지연 역보상 명령 u (16). 반환 (u, a_des, tau_des)."""
        L, d = self.L, self.d
        qref, pos, vel = self.ref.leg_ref(self.t)
        pos = pos + self.q_offset
        q = d.qpos[self.jq]; v = d.qvel[self.jv]
        tau_pd = self.kp * (pos - q) + self.kd * (vel - v)
        # 가상 모델 제어(VMC): 접지 다리에 골반 높이·전진 오차 보정 지지력을 발 자코비안으로 변환해 추가 (가라앉음 방지)
        tau_vmc = np.zeros(8)
        if self.vmc:
            kpy, kdy, kpx, kdx = self.vmc
            y_err = qref['pelvis_ty'] - d.xpos[self.pelvis][1]
            vy = d.qvel[1]
            vx_ref = self.ref.stride / self.ref.T
            Fy = kpy * y_err - kdy * vy
            Fx = kpx * 0.0 + kdx * (vx_ref - d.qvel[0])
            contacts = [s for s in 'rl' if self.contact_force(s) > 30.0]
            for s in contacts:
                jacp = np.zeros((3, self.m.nv)); jacr = np.zeros((3, self.m.nv))
                mujoco.mj_jacBody(self.m, d, jacp, jacr, self.calcn[s])
                dofs = [L.dof[f'{c}_{s}'] for c in COORDS]
                J = jacp[:, dofs]                                   # 발 위치 / 다리 관절
                F_foot = -np.array([Fx, Fy, 0.0]) / len(contacts)    # 골반을 밀어올리려면 발을 아래로 미는 힘
                si = 0 if s == 'r' else 1
                tau_vmc[4 * si:4 * si + 4] = J.T @ F_foot
        u = np.zeros(self.m.nu); a_des = np.zeros(self.m.nu); tau_des = np.zeros(8)
        for si, s in enumerate('rl'):
            tau = tau_pd[4 * si:4 * si + 4] + tau_vmc[4 * si:4 * si + 4] + (self.ff_scale * self.ref.ff_at(self.t, s, self.ff_lead) if self.ff else 0)
            R = L.moment_arms(s)
            Fm = L.Fmax[L.act_idx[s]]
            f = allocate(R, tau, Fm)
            a_des[L.act_idx[s]] = f / Fm
            tau_des[4 * si:4 * si + 4] = tau
        u = self.sma.invert(a_des, self.dt_c, self.gamma)
        return u, a_des, tau_des, qref

    def apply_harness(self, qref):
        d = self.d
        d.xfrc_applied[:] = 0
        if not self.harness:
            return np.zeros(3)
        kp, kd, kr, kw = self.hg
        p_ref = np.array([qref['pelvis_tx'], qref['pelvis_ty'], qref['pelvis_tz']])
        R = rot_z(qref['pelvis_tilt']) @ rot_x(qref['pelvis_list']) @ rot_y(qref['pelvis_rotation'])
        q_ref = mat2quat(R)
        p = d.xpos[self.pelvis]; v = d.cvel[self.pelvis][3:6]  # cvel: [ang, lin] at body com? use qvel instead
        v = d.qvel[0:3]
        F = kp * (p_ref - p) - kd * v
        if self.harness == 'yz':            # 전진(x)은 자유 — 트레드밀 손잡이처럼 높이·좌우·자세만 보조
            F[0] = 0.0
        e = quat_err(q_ref, d.xquat[self.pelvis])
        w_world = d.xmat[self.pelvis].reshape(3, 3) @ d.qvel[3:6]
        T = kr * e - kw * w_world
        d.xfrc_applied[self.pelvis, :3] = F
        d.xfrc_applied[self.pelvis, 3:] = T
        return F

    def step(self, u_cmd, qref):
        """제어주기 1회: u_cmd 로 nsub 물리 스텝."""
        m, d = self.m, self.d
        Fh = self.apply_harness(qref)
        for _ in range(self.nsub):
            a = self.sma.step(u_cmd, m.opt.timestep)
            d.ctrl[:] = a
            mujoco.mj_step(m, d)
        self.t += self.dt_c
        return Fh

    def fallen(self):
        d = self.d
        up = d.xmat[self.pelvis].reshape(3, 3)[:, 1]  # pelvis local y in world
        return d.xpos[self.pelvis][1] < 0.6 or up[1] < 0.5

    def contact_force(self, s):
        """발 s 의 접촉 수직력 합."""
        d, m = self.d, self.m
        tot = np.zeros(6)
        f = np.zeros(6)
        body = self.calcn[s]
        toes = self.L.bid[f'toes_{s}']
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = m.geom_bodyid[c.geom1]; b2 = m.geom_bodyid[c.geom2]
            if b1 in (body, toes) or b2 in (body, toes):
                mujoco.mj_contactForce(m, d, i, f)
                tot[0] += abs(f[0])
        return tot[0]


def run(L, ref, outdir, tag, harness=True, ff=True, n_cycles=1.0, record=True, perturb=0.0):
    sim = WalkSim(L, ref, harness=harness, ff=ff)
    sim.reset(0.0, perturb)
    T_end = n_cycles * ref.T
    log = {'t': [], 'q': [], 'qref': [], 'a': [], 'u': [], 'Fh': [], 'pelvis': [], 'grf': [], 'tau_des': [], 'xpos': []}
    t0 = time.time()
    fell = False
    while sim.t < T_end:
        u, a_des, tau_des, qref = sim.nominal_control()
        Fh = sim.step(u, qref)
        d = sim.d
        log['t'].append(sim.t); log['q'].append(d.qpos[sim.jq].copy()); log['qref'].append(ref.leg_ref(sim.t)[1])
        log['a'].append(sim.sma.a.copy()); log['u'].append(u.copy()); log['Fh'].append(np.linalg.norm(Fh)); log['pelvis'].append(d.xpos[sim.pelvis].copy())
        log['grf'].append([sim.contact_force('r'), sim.contact_force('l')]); log['tau_des'].append(tau_des.copy())
        if record:
            log['xpos'].append(np.array([d.xpos[b].copy() for b in range(sim.m.nbody)]))
        if sim.fallen():
            fell = True
            break
    wall = time.time() - t0
    for k in log:
        log[k] = np.array(log[k])
    err = log['q'] - log['qref']
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    W = L.mass * 9.81
    res = {'tag': tag, 'harness': harness, 'ff': ff, 'sim_time_s': float(log['t'][-1]), 'fell': fell, 'wall_s': wall,
           'rmse_deg': (np.degrees(rmse)).tolist(), 'rmse_deg_mean': float(np.degrees(rmse).mean()),
           'harness_force_rms_BW': float(np.sqrt(np.mean(log['Fh'] ** 2)) / W), 'harness_force_peak_BW': float(log['Fh'].max() / W),
           'act_mean': float(log['a'].mean()), 'act_max_per_cable': log['a'].max(axis=0).tolist(),
           'sat_frac': float(np.mean((log['u'] > 0.999) | (log['u'] < 0.001))),
           'grf_peak_BW': (log['grf'].max(axis=0) / W).tolist(), 'dist_m': float(log['pelvis'][-1, 0] - log['pelvis'][0, 0])}
    print(json.dumps(res, indent=1))
    np.savez(f'{outdir}/walk_{tag}.npz', **log, names=np.array([mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(sim.m.nbody)]))
    json.dump(res, open(f'{outdir}/walk_{tag}.json', 'w'), indent=1)
    return res, log


if __name__ == '__main__':
    xml, osim, pj, data_dir, trial, outdir = sys.argv[1:7]
    harness = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    cycles = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    ff = int(sys.argv[9]) if len(sys.argv) > 9 else 1
    L = Leg16(xml, osim, pj)
    ref = Reference(L, data_dir, trial, id_npz=f'{outdir}/id_{trial}.npz')
    tag = f"{trial}_h{harness}_ff{ff}_c{cycles:g}"
    run(L, ref, outdir, tag, harness=bool(harness), ff=bool(ff), n_cycles=cycles)
