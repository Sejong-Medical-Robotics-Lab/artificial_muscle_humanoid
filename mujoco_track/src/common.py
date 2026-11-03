# -*- coding: utf-8 -*-
"""공통: MuJoCo 모델 로드, OpenSim 좌표 → qpos, 모멘트암(무릎 연쇄법칙 포함), GRF 적용, 필터."""
import sys, json, math
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np
import mujoco
from scipy.signal import butter, filtfilt
from osim_model import OsimModel, read_mot, mot_to_qdicts, mat2quat, rot_x, rot_y, rot_z

COORDS = ['hip_flexion', 'hip_adduction', 'knee_angle', 'ankle_angle']
KNEE_DEP = ('rotation2', 'rotation3', 'translation1', 'translation2')


class Leg16:
    def __init__(self, xml_path, osim_path, json_path):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.M = OsimModel(osim_path)
        self.P = json.load(open(json_path))
        self.meta = json.load(open(xml_path.replace('.xml', '_meta.json')))
        m = self.m
        self.jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)}
        self.bid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(m.nbody)}
        self.knee_axes = {s: {ax.label: ax for ax in self.M.joint_by_child[f'tibia_{s}'].axes} for s in ('r', 'l')}
        self.names = [mu['name'] for mu in self.P['muscles']]
        self.Fmax = np.array([mu['force_N'] for mu in self.P['muscles'] for s in ('r', 'l')])  # actuator order M0_r, M0_l, ...
        self.act_idx = {s: [2 * i + (0 if s == 'r' else 1) for i in range(8)] for s in ('r', 'l')}
        self.dof = {f'{c}_{s}': int(m.jnt_dofadr[self.jid[f'{c}_{s}']]) for s in ('r', 'l') for c in COORDS}
        self.qadr = {n: int(m.jnt_qposadr[j]) for n, j in self.jid.items()}
        self.mass = float(sum(m.body_mass))

    # ---------------- 자세
    def qpos_from_coords(self, q):
        """OpenSim 좌표 dict → qpos 벡터 (무릎 종속축은 스플라인 값)."""
        m = self.m
        qp = np.zeros(m.nq)
        R = rot_z(q.get('pelvis_tilt', 0)) @ rot_x(q.get('pelvis_list', 0)) @ rot_y(q.get('pelvis_rotation', 0))
        qp[0:3] = [q.get('pelvis_tx', 0), q.get('pelvis_ty', 0.95), q.get('pelvis_tz', 0)]
        qp[3:7] = mat2quat(R)
        for name, j in self.jid.items():
            if name == 'pelvis':
                continue
            adr = m.jnt_qposadr[j]
            if name in q:
                qp[adr] = q[name]
            elif name.startswith('walker_knee'):
                s = name.split('_')[2]
                lab = name.split('_')[-1]
                qp[adr] = self.knee_axes[s][lab].func(q.get(f'knee_angle_{s}', 0.0))
            elif name in self.meta['locked']:
                qp[adr] = q.get(name, self.meta['locked'][name])
        return qp

    def set_pose(self, q):
        self.d.qpos[:] = self.qpos_from_coords(q)
        mujoco.mj_forward(self.m, self.d)

    def knee_dq(self, s, ka):
        """종속축 dq_k/dθ (dict label→value)."""
        out = {}
        for lab in KNEE_DEP:
            f = self.knee_axes[s][lab].func
            h = 1e-5
            out[lab] = (f(ka + h) - f(ka - h)) / (2 * h)
        return out

    def ten_J_dense(self):
        m, d = self.m, self.d
        J = np.zeros((m.ntendon, m.nv))
        for t in range(m.ntendon):
            nnz = int(m.ten_J_rownnz[t]); adr = int(m.ten_J_rowadr[t])
            J[t, m.ten_J_colind[adr:adr + nnz]] = d.ten_J[adr:adr + nnz]
        return J

    def moment_arms(self, s):
        """(4, 8): 다리 s 의 케이블 8개 × 좌표 4개 유효 모멘트암 (현재 d 상태)."""
        J = self.ten_J_dense()
        R = np.zeros((4, 8))
        ka = self.d.qpos[self.qadr[f'knee_angle_{s}']]
        dq = self.knee_dq(s, ka)
        for ci, c in enumerate(COORDS):
            col = J[:, self.dof[f'{c}_{s}']].copy()
            if c == 'knee_angle':
                for lab in KNEE_DEP:
                    col += J[:, self.m.jnt_dofadr[self.jid[f'walker_knee_{s}_{lab}']]] * dq[lab]
            R[ci] = -col[self.act_idx[s]]
        return R

    def joint_moment_eff(self, qfrc, s):
        """일반화힘 벡터 → 좌표 4개의 유효 모멘트 (무릎은 종속축 기여 포함)."""
        ka = self.d.qpos[self.qadr[f'knee_angle_{s}']]
        dq = self.knee_dq(s, ka)
        out = np.zeros(4)
        for ci, c in enumerate(COORDS):
            v = qfrc[self.dof[f'{c}_{s}']]
            if c == 'knee_angle':
                for lab in KNEE_DEP:
                    v += qfrc[self.m.jnt_dofadr[self.jid[f'walker_knee_{s}_{lab}']]] * dq[lab]
            out[ci] = v
        return out

    # ---------------- GRF
    def grf_qfrc(self, grf_row, hdr):
        """OpenSim GRF .mot 한 행 → 일반화 외력 벡터 (mj_applyFT, calcn_r/l). 힘·COP·자유모멘트는 ground 좌표.
        mj_forward 가 끝난 상태에서 호출. 역동역학에서는 qfrc_inverse 에서 이 값을 빼야 관절 모멘트가 된다."""
        d = self.d
        qf = np.zeros(self.m.nv)
        for s in ('r', 'l'):
            F = np.array([grf_row[hdr.index(f'ground_force_calcn_{s}_v{a}')] for a in 'xyz'])
            Pt = np.array([grf_row[hdr.index(f'ground_force_calcn_{s}_p{a}')] for a in 'xyz'])
            Mf = np.array([grf_row[hdr.index(f'ground_force_calcn_{s}_m{a}')] for a in 'xyz'])
            b = self.bid[f'calcn_{s}']
            if np.linalg.norm(F) > 0 or np.linalg.norm(Mf) > 0:
                mujoco.mj_applyFT(self.m, d, F, Mf, Pt, b, qf)
        return qf

    def apply_grf(self, grf_row, hdr):
        """순동역학용: xfrc_applied 에 GRF 를 싣는다 (바디 원점 기준 토크로 환산)."""
        d = self.d
        d.xfrc_applied[:] = 0
        for s in ('r', 'l'):
            F = np.array([grf_row[hdr.index(f'ground_force_calcn_{s}_v{a}')] for a in 'xyz'])
            Pt = np.array([grf_row[hdr.index(f'ground_force_calcn_{s}_p{a}')] for a in 'xyz'])
            Mf = np.array([grf_row[hdr.index(f'ground_force_calcn_{s}_m{a}')] for a in 'xyz'])
            b = self.bid[f'calcn_{s}']
            xb = d.xpos[b]
            d.xfrc_applied[b, :3] = F
            d.xfrc_applied[b, 3:] = np.cross(Pt - xb, F) + Mf


def lowpass(data, fs, cutoff=6.0, order=4):
    b, a = butter(order, cutoff / (fs / 2.0))
    out = data.copy()
    for j in range(data.shape[1]):
        out[:, j] = filtfilt(b, a, data[:, j])
    return out


def load_trial(data_dir, trial):
    hi, di = read_mot(f'{data_dir}/{trial}_segment_0_ik.mot')
    hg, dg = read_mot(f'{data_dir}/{trial}_segment_0_grf.mot')
    return hi, di, hg, dg
