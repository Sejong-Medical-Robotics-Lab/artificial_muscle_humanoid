# -*- coding: utf-8 -*-
"""변환 검증: MuJoCo(엔진) vs numpy(OpenSim 기구학 독립 구현) — 사이트 위치·텐던 길이·모멘트암.
사용: python3 -I validate_conv.py <osim> <paths.json> <leg16.xml> <walking_ik.mot> <squat_ik.mot> <outdir>
"""
import sys, json, math
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np
import mujoco
from osim_model import OsimModel, read_mot, mot_to_qdicts, mat2quat, rot_x, rot_y, rot_z

osim_path, json_path, xml_path, walk_path, squat_path, outdir = sys.argv[1:7]
M = OsimModel(osim_path)
P = json.load(open(json_path))
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
meta = json.load(open(xml_path.replace('.xml', '_meta.json')))
COORDS = ['hip_flexion', 'hip_adduction', 'knee_angle', 'ankle_angle']

jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)}
knee_axes = {s: {ax.label: ax for ax in M.joint_by_child[f'tibia_{s}'].axes} for s in ('r', 'l')}


def set_pose(q):
    """OpenSim 좌표 dict → MuJoCo qpos (무릎 종속축은 스플라인으로 직접)."""
    d.qpos[:] = 0
    # pelvis free joint
    R = rot_z(q.get('pelvis_tilt', 0)) @ rot_x(q.get('pelvis_list', 0)) @ rot_y(q.get('pelvis_rotation', 0))
    d.qpos[0:3] = [q.get('pelvis_tx', 0), q.get('pelvis_ty', 0.95), q.get('pelvis_tz', 0)]
    d.qpos[3:7] = mat2quat(R)
    for name, j in jid.items():
        if name == 'pelvis':
            continue
        adr = m.jnt_qposadr[j]
        if name in q:
            d.qpos[adr] = q[name]
        elif name.startswith('walker_knee'):
            s = name.split('_')[2]
            lab = name.split('_')[-1]
            ka = q.get(f'knee_angle_{s}', 0.0)
            d.qpos[adr] = knee_axes[s][lab].func(ka)
        elif name in meta['locked']:
            d.qpos[adr] = q.get(name, meta['locked'][name])
    mujoco.mj_forward(m, d)


def mj_tendon_lengths():
    return d.ten_length.copy()


def mj_moment_arms():
    """(ntendon, 4 coords × 2 sides) : r = -dL/dq."""
    J = np.zeros((m.ntendon, m.nv))
    for t in range(m.ntendon):
        nnz = int(m.ten_J_rownnz[t]); adr = int(m.ten_J_rowadr[t])
        J[t, m.ten_J_colind[adr:adr + nnz]] = d.ten_J[adr:adr + nnz]
    out = {}
    for s in ('r', 'l'):
        for c in COORDS:
            dof = m.jnt_dofadr[jid[f'{c}_{s}']]
            col = J[:, dof].copy()
            if c == 'knee_angle':
                # 종속축(rotation2·3, translation1·2)의 기여를 연쇄법칙으로 더한다: dL/dθ = Σ_k dL/dq_k · dq_k/dθ
                ka = d.qpos[m.jnt_qposadr[jid[f'{c}_{s}']]]
                for lab in ('rotation2', 'rotation3', 'translation1', 'translation2'):
                    jn = f'walker_knee_{s}_{lab}'
                    f = knee_axes[s][lab].func
                    h = 1e-5
                    dq = (f(ka + h) - f(ka - h)) / (2 * h)
                    col += J[:, m.jnt_dofadr[jid[jn]]] * dq
            out[f'{c}_{s}'] = -col
    return out


def np_tendon_lengths(q):
    fk = M.fk(q)
    L = []
    for mu in P['muscles']:
        for s in ('r', 'l'):
            pts = []
            for bname, pt in zip(mu['chain'], mu['points']):
                bn = bname if bname == 'pelvis' else bname[:-2] + '_' + s
                pt = np.array(pt, float)
                if s == 'l':
                    pt = pt * np.array([1, 1, -1])
                pts.append(M.point_world(fk, bn, pt))
            L.append(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))
    return np.array(L)


def np_moment_arms(q, h=1e-5):
    out = {}
    for s in ('r', 'l'):
        for c in COORDS:
            k = f'{c}_{s}'
            qp = dict(q); qp[k] = q.get(k, 0) + h
            qm = dict(q); qm[k] = q.get(k, 0) - h
            out[k] = -(np_tendon_lengths(qp) - np_tendon_lengths(qm)) / (2 * h)
    return out


def site_positions_mj():
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i): d.site_xpos[i].copy() for i in range(m.nsite)}


def site_positions_np(q):
    fk = M.fk(q)
    out = {}
    for mu in P['muscles']:
        for s in ('r', 'l'):
            for k, (bname, pt) in enumerate(zip(mu['chain'], mu['points'])):
                bn = bname if bname == 'pelvis' else bname[:-2] + '_' + s
                pt = np.array(pt, float)
                if s == 'l':
                    pt = pt * np.array([1, 1, -1])
                out[f'{mu["name"]}_{s}_{k}'] = M.point_world(fk, bn, pt)
    return out


# ---------------- pose set: walking1 전체 + squats1 서브샘플 + ROM 꼭짓점 16
hw, dw = read_mot(walk_path)
hs, ds = read_mot(squat_path)
poses = mot_to_qdicts(hw, dw)
sq = mot_to_qdicts(hs, ds)
poses += sq[::max(1, len(sq) // 60)]
n_data = len(poses)
ROM = {'hip_flexion': (-0.35, 1.9), 'hip_adduction': (-0.5, 0.35), 'knee_angle': (0.0, 2.0), 'ankle_angle': (-0.5, 0.35)}
base = dict(poses[0])
corners = []
for b in range(16):
    q = dict(base)
    for i, c in enumerate(COORDS):
        v = ROM[c][(b >> i) & 1]
        q[f'{c}_r'] = v; q[f'{c}_l'] = v
    corners.append(q)
poses += corners
print('poses:', n_data, 'data +', len(corners), 'ROM corners =', len(poses))

site_err = []; len_err = []; len_rel = []; ma_err = []
L_all = []; MA_all = {k: [] for k in [f'{c}_{s}' for s in ('r', 'l') for c in COORDS]}
for q in poses:
    set_pose(q)
    sm = site_positions_mj(); sn = site_positions_np(q)
    site_err.append(max(np.linalg.norm(sm[k] - sn[k]) for k in sn))
    Lm = mj_tendon_lengths(); Ln = np_tendon_lengths(q)
    len_err.append(np.max(np.abs(Lm - Ln))); len_rel.append(np.max(np.abs(Lm - Ln) / Ln))
    Mm = mj_moment_arms(); Mn = np_moment_arms(q)
    ma_err.append(max(np.max(np.abs(Mm[k] - Mn[k])) for k in Mn))
    L_all.append(Lm)
    for k in MA_all:
        MA_all[k].append(Mm[k])
L_all = np.array(L_all)
res = {'n_poses': len(poses), 'site_pos_max_err_m': float(max(site_err)), 'tendon_len_max_err_m': float(max(len_err)),
       'tendon_len_max_rel': float(max(len_rel)), 'moment_arm_max_err_m': float(max(ma_err))}
print(json.dumps(res, indent=1))

# ---------------- 학생 json(OpenSim 산출)과 교차 비교: arm_min / arm_mean / r_mean / stroke  (오른다리, walking1 전체 + squats1 전체, MuJoCo 만)
full = mot_to_qdicts(hw, dw) + sq
Lf = []; MAf = {f'{c}_r': [] for c in COORDS}
for q in full:
    set_pose(q)
    Lf.append(mj_tendon_lengths())
    mm = mj_moment_arms()
    for c in COORDS:
        MAf[f'{c}_r'].append(mm[f'{c}_r'])
Lf = np.array(Lf)
names = [mu['name'] for mu in P['muscles']]
rows = []
for i, mu in enumerate(P['muscles']):
    ti = 2 * i  # right tendon index (order: M0_r, M0_l, ...)
    Ld = Lf[:, ti]
    stroke = (Ld.max() - Ld.min()) / Ld.max()
    r = np.array([MAf[f'{c}_r'] for c in COORDS])  # (4, n, ntendon)
    r = r[:, :, ti]
    rn = np.linalg.norm(r, axis=0)
    rows.append({'name': mu['name'], 'os_arm_min': mu['arm_min'], 'mj_arm_min': float(rn.min()), 'os_arm_mean': mu['arm_mean'], 'mj_arm_mean': float(rn.mean()),
                 'os_r_mean': mu['r_mean'], 'mj_r_mean': r.mean(axis=1).tolist(), 'os_stroke': mu['stroke'], 'mj_stroke': float(stroke),
                 'mj_L_min': float(Ld.min()), 'mj_L_max': float(Ld.max())})
for r in rows:
    print(f"{r['name']}: arm_min OS {r['os_arm_min']*1000:.1f} / MJ {r['mj_arm_min']*1000:.1f} mm · arm_mean OS {r['os_arm_mean']*1000:.1f} / MJ {r['mj_arm_mean']*1000:.1f} mm · "
          f"stroke OS {r['os_stroke']:.3f} / MJ {r['mj_stroke']:.3f} · r_mean OS {[round(x*1000,1) for x in r['os_r_mean']]} / MJ {[round(x*1000,1) for x in r['mj_r_mean']]}")
json.dump({'summary': res, 'cross': rows}, open(f'{outdir}/validate_conv.json', 'w'), indent=1)

# ---------------- figure: moment arms vs knee angle (knee sweep, other coords at mean walking) MuJoCo vs numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f or 'NanumGothic' in f:
        font_manager.fontManager.addfont(f)
plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'NanumGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
qm = {k: float(np.mean(dw[:, hw.index(k)])) for k in hw[1:]}
ks = np.linspace(0, 2.0, 41)
fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
for i, mu in enumerate(P['muscles']):
    ax = axes.flat[i]
    for c, col in zip(COORDS, ['#2F5D8F', '#B8860B', '#2E7D4F', '#9B1B30']):
        mj_v = []; np_v = []
        for ka in ks:
            q = dict(qm); q['knee_angle_r'] = ka
            set_pose(q)
            mj_v.append(mj_moment_arms()[f'{c}_r'][2 * i] * 100)
            np_v.append(np_moment_arms(q)[f'{c}_r'][2 * i] * 100)
        ax.plot(np.degrees(ks), mj_v, color=col, lw=2, label=f'{c} MuJoCo')
        ax.plot(np.degrees(ks), np_v, color='k', lw=0.8, ls='--')
    ax.set_title(f"{mu['name']} — 모멘트암 (cm) vs 무릎 각도", fontsize=10)
    ax.axhline(0, color='0.7', lw=0.6)
    ax.set_xlabel('무릎 각도 (°)'); ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc='best')
fig.suptitle('변환 검증 — MuJoCo(실선) vs OpenSim 기구학 numpy 구현(점선): 케이블 8개 모멘트암 (고관절 굴곡·내전·무릎·발목), 오른다리', fontsize=12)
fig.tight_layout()
fig.savefig(f'{outdir}/fig_mj_momentarm_check.png', dpi=150)
print('figure saved')
