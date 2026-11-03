# -*- coding: utf-8 -*-
"""영상 ① — 상체 확장 입력: Moore2015(AddBiomechanics) 팔 포함 트레드밀 보행을 MuJoCo로 옮겨 재생.
왼쪽: 팔 포함 모델(요추 3 + 팔 7×2 좌표) + 보정된 지면반력(초록) + 움직이는 벨트.  오른쪽: 요추·팔 좌표, 수직 지면반력, MuJoCo 역동역학(요추 포함).
사용: xvfb-run -a -s "-screen 0 1280x720x24" env MUJOCO_GL=glfw python3 -I video_upper.py <outdir>
"""
import sys, json, math
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np, mujoco
from osim_model import read_mot
from common import Leg16, lowpass, COORDS
from vid_common import MjView, FrameWriter, fig_frame, setup_fonts, CR, BL, GR, MU, INK
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

outdir = sys.argv[1]
R = DATA
setup_fonts()
SEGS = [dict(subj='subject9', trial='T026_segment_16', xml=model('moore9.xml'), t0=None, t1=None, reps=2, label='subject9 · 1.2 m/s (T026, 1 걸음주기 0.94 s)', stud_resid=2.05),
        dict(subj='subject3', trial='T048_segment_3', xml=model('moore3.xml'), t0=2.0, t1=5.0, reps=1, label='subject3 · 1.6 m/s (T048, 연속 20 s 조각 중 3 s)', stud_resid=5.63)]
UP = [('lumbar_extension', '요추 신전'), ('lumbar_bending', '요추 측굴'), ('lumbar_rotation', '요추 회전'), ('arm_flex_r', '오른팔 굴곡'), ('arm_flex_l', '왼팔 굴곡'), ('elbow_flex_r', '오른 팔꿈치')]
UPC = ['#9B1B30', '#E39A1C', '#2E7D4F', '#1f77b4', '#7fb3e0', '#9467bd']


def prepare(S):
    L = Leg16(S['xml'], f"{R}/data/moore2015/{S['subj']}/{S['subj']}_scaled.osim", model('nocables.json'))
    base = f"{R}/data/moore2015/{S['subj']}/{S['trial']}"
    hi, di = read_mot(base + '_ik.mot'); hg, dg = read_mot(base + '_grf.mot')
    fs = 1.0 / np.mean(np.diff(di[:, 0]))
    grf = np.column_stack([np.interp(di[:, 0], dg[:, 0], dg[:, j]) for j in range(dg.shape[1])])
    t0 = di[0, 0] if S['t0'] is None else di[0, 0] + S['t0']
    t1 = di[-1, 0] if S['t1'] is None else di[0, 0] + S['t1']
    # 역동역학 (15/15 Hz, 학생과 같은 컷오프) — 창보다 0.3 s 넓게 잡아 가장자리 영향 제거
    dif = di.copy(); dif[:, 1:] = lowpass(di[:, 1:], fs, 15.0) if len(di) > 40 else di[:, 1:]
    grf_f = grf.copy(); grf_f[:, 1:] = lowpass(grf[:, 1:], fs, 15.0) if len(di) > 40 else grf[:, 1:]
    m, d = L.m, L.d
    Q = np.array([L.qpos_from_coords(dict(zip(hi, row))) for row in dif])
    dt = 1.0 / fs
    V = np.zeros((len(Q), m.nv))
    for k in range(1, len(Q) - 1):
        mujoco.mj_differentiatePos(m, V[k], 2 * dt, Q[k - 1], Q[k + 1])
    V[0] = V[1]; V[-1] = V[-2]
    A = np.gradient(V, dt, axis=0)
    m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_EQUALITY | mujoco.mjtDisableBit.mjDSBL_CONTACT)
    damp, arm = m.dof_damping.copy(), m.dof_armature.copy()
    m.dof_damping[:] = 0; m.dof_armature[:] = 0
    lum = L.dof.get('lumbar_extension_x', None)
    lum = int(m.jnt_dofadr[L.jid['lumbar_extension']])
    tau = []; resid = []
    for k in range(len(Q)):
        d.qpos[:] = Q[k]; d.qvel[:] = V[k]; mujoco.mj_forward(m, d)
        qext = L.grf_qfrc(grf_f[k], hg)
        d.qacc[:] = A[k]; mujoco.mj_inverse(m, d)
        f = d.qfrc_inverse.copy() - qext
        tr = L.joint_moment_eff(f, 'r')
        tau.append([tr[0], tr[2], tr[3], f[lum]]); resid.append(np.linalg.norm(f[0:3]))
    tau = np.array(tau); resid = np.array(resid)
    m.dof_damping[:] = damp; m.dof_armature[:] = arm
    m.opt.disableflags = 0
    W = L.mass * 9.81
    vy = {s: grf[:, hg.index(f'ground_force_calcn_{s}_vy')] for s in 'rl'}
    peakg = max(vy['r'].max(), vy['l'].max())
    sel = (di[:, 0] >= t0) & (di[:, 0] <= t1)
    inner = sel & (di[:, 0] >= t0 + 0.12) & (di[:, 0] <= t1 - 0.12)      # 필터 가장자리 0.12 s 제외 (역동역학 통계·그림)
    # 벨트 속도 (지지 중 발 속도) · 진행 방향
    pos = []; toe = []
    for row in di:
        L.set_pose(dict(zip(hi, row))); pos.append(L.d.xpos[L.bid['calcn_r']].copy()); toe.append(L.d.xpos[L.bid['toes_r']].copy())
    pos = np.array(pos); toe = np.array(toe)
    vel = np.gradient(pos, dt, axis=0); st = vy['r'] > 300
    belt = float(np.median(np.linalg.norm(vel[st][:, [0, 2]], axis=1))) if st.sum() > 5 else 0.0
    f_dir = np.median(toe - pos, axis=0); f_dir[1] = 0; f_dir /= np.linalg.norm(f_dir)
    pel = np.median(np.array([L.qpos_from_coords(dict(zip(hi, row)))[0:3] for row in di[sel]]), axis=0)
    return dict(L=L, hi=hi, di=di, hg=hg, grf=grf, t=di[:, 0], sel=sel, t0=t0, t1=t1, tau=tau, resid=resid, W=W, peakg=peakg, vy=vy, belt=belt, fdir=f_dir, pel=pel,
                resid_pct=float(resid[inner].max() / peakg * 100), resid_mean_pct=float(resid[inner].mean() / peakg * 100), mass=L.mass, inner=inner, Qraw=np.array([L.qpos_from_coords(dict(zip(hi, row))) for row in di]))


segs = [prepare(S) for S in SEGS]
for S, P in zip(SEGS, segs):
    print(S['trial'], 'mass', round(P['mass'], 2), 'belt', round(P['belt'], 2), 'resid max/mean %', round(P['resid_pct'], 2), round(P['resid_mean_pct'], 2), 'peak', round(P['peakg'] / P['W'], 2),
          'tau peaks Nm/kg', np.round(np.abs(P['tau'][P['inner']]).max(axis=0) / P['mass'], 3))

W_, H_, FPS = 1540, 924, 30
SPEED = 1 / 3
fig = plt.figure(figsize=(W_ / 100, H_ / 100), dpi=100)
gs = GridSpec(3, 2, width_ratios=[1.12, 1], height_ratios=[1, 0.8, 1], left=0.015, right=0.985, top=0.9, bottom=0.065, wspace=0.14, hspace=0.6)
ax_img = fig.add_subplot(gs[:, 0]); ax_img.axis('off')
ax_up = fig.add_subplot(gs[0, 1]); ax_g = fig.add_subplot(gs[1, 1]); ax_tau = fig.add_subplot(gs[2, 1])
sup = fig.suptitle('', fontsize=12.5, color=INK, y=0.965)
wr = FrameWriter(f'{outdir}/video_mj_upper.mp4', W_, H_, FPS)
poster = False; n_total = 0
for si, (S, P) in enumerate(zip(SEGS, segs)):
    L = P['L']
    if si > 0:
        view.renderer.close(); del view        # GL 컨텍스트는 한 번에 하나만
    view = MjView(S['xml'], w=900, h=800)
    floor = mujoco.mj_name2id(view.m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    fdir = P['fdir']; right = np.cross(fdir, np.array([0, 1, 0.0]))
    dvec = 0.85 * right + 0.5 * fdir; dvec /= np.linalg.norm(dvec)
    az = math.degrees(math.atan2(dvec[0], dvec[2]))
    look = np.array([P['pel'][0], 0.82, P['pel'][2]])
    view.set_camera(look, az, 9, 2.35)
    ax_img.clear(); ax_img.axis('off')
    im = ax_img.imshow(np.zeros((view.h, view.w, 3), np.uint8))
    txt = ax_img.text(0.02, 0.975, '', transform=ax_img.transAxes, fontsize=11, va='top', color=INK)
    ax_img.text(0.02, 0.055, f"Rajagopal 팔 포함 모델 좌표 39개(요추 3 · 팔 7×2) · 질량 {P['mass']:.1f} kg · 벨트 {P['belt']:.2f} m/s · 케이블 없음(운동학·역동역학 입력 데이터)", transform=ax_img.transAxes, fontsize=9.5, color=MU)
    ax_img.text(0.02, 0.02, '초록 = 보정된 지면반력(압력중심에서; 30 N 미만 0 처리, 발 밖 COP 보정) · 바닥 = 트레드밀 벨트', transform=ax_img.transAxes, fontsize=9.5, color=MU)
    T = P['t'][P['sel']] - P['t0']; tt_all = P['t'] - P['t0']
    # 상체 좌표
    ax_up.clear()
    for (c, nm), col in zip(UP, UPC):
        ax_up.plot(tt_all[P['sel']], np.degrees(P['di'][P['sel'], P['hi'].index(c)]), color=col, lw=1.6, label=nm)
    cur1 = ax_up.axvline(0, color='k', lw=1)
    ax_up.set_xlim(T[0], T[-1]); ax_up.set_ylabel('각도 (°)'); ax_up.grid(alpha=0.3); ax_up.legend(fontsize=8, ncol=3, loc='upper right')
    ax_up.set_title('상체 좌표 — 요추 3자유도 · 팔 굴곡 · 팔꿈치 (Uhlrich2023 하지 데이터에는 없던 입력)', fontsize=10.5)
    # 지면반력
    ax_g.clear()
    ax_g.plot(tt_all[P['sel']], P['vy']['r'][P['sel']] / P['W'], color=CR, lw=1.8, label='오른발')
    ax_g.plot(tt_all[P['sel']], P['vy']['l'][P['sel']] / P['W'], color=BL, lw=1.8, label='왼발')
    ax_g.axhline(30 / P['W'], color='k', lw=0.8, ls='--'); ax_g.text(T[0] + 0.02, 30 / P['W'] + 0.04, '접촉 문턱 30 N', fontsize=8)
    cur2 = ax_g.axvline(0, color='k', lw=1)
    ax_g.set_xlim(T[0], T[-1]); ax_g.set_ylim(0, 1.6); ax_g.set_ylabel('BW'); ax_g.grid(alpha=0.3); ax_g.legend(fontsize=8.5, ncol=2, loc='upper right')
    ax_g.set_title(f"수직 지면반력 (벨트 2장 힘판, 보정 후) — 발 하나 피크 {P['vy']['r'][P['sel']].max() / P['W']:.2f} / {P['vy']['l'][P['sel']].max() / P['W']:.2f} BW", fontsize=10.5)
    # 역동역학
    ax_tau.clear()
    for j, (nm, col) in enumerate([('고관절 굴곡 R', '#1f77b4'), ('무릎 R', '#2ca02c'), ('발목 R', '#d62728'), ('요추 신전', CR)]):
        ax_tau.plot(tt_all[P['inner']], P['tau'][P['inner'], j] / P['mass'], color=col, lw=1.8 if j == 3 else 1.3, label=nm)
    cur3 = ax_tau.axvline(0, color='k', lw=1)
    ax_tau.set_xlim(T[0], T[-1]); ax_tau.set_ylabel('Nm/kg'); ax_tau.set_xlabel('time (s)'); ax_tau.grid(alpha=0.3); ax_tau.legend(fontsize=8, ncol=4, loc='lower left')
    ax_tau.set_title(f"MuJoCo 역동역학(mj_inverse, 15/15 Hz) — 골반 잔차 최대 {P['resid_pct']:.1f} % · 평균 {P['resid_mean_pct']:.1f} % (가장자리 0.12 s 제외; 학생 OpenSim ID {S['stud_resid']:.1f} %)", fontsize=9.5)
    sup.set_text(f"상체 확장 입력 — Moore2015 팔 포함 트레드밀 보행(AddBiomechanics → OpenSim → MuJoCo 변환) · {S['label']}   1/3 배속")
    n_frames = int(round((T[-1] - T[0]) / SPEED * FPS))
    for rep in range(S['reps']):
        for k in range(n_frames):
            t = k / FPS * SPEED
            j = int(np.argmin(np.abs(tt_all - t)))
            extra = []
            for s in 'rl':
                F = np.array([P['grf'][j, P['hg'].index(f'ground_force_calcn_{s}_v{a}')] for a in 'xyz'])
                Pt = np.array([P['grf'][j, P['hg'].index(f'ground_force_calcn_{s}_p{a}')] for a in 'xyz'])
                if np.linalg.norm(F) > 20:
                    extra.append({'type': 'arrow', 'from': Pt, 'to': Pt + 0.0005 * F, 'width': 0.014, 'rgba': (0.16, 0.6, 0.3, 1.0)})
            # 벨트: 바닥을 진행 방향 반대로 벨트 속도만큼 이동 (체커 주기 0.5 m)
            shift = (-P['belt'] * t) % 0.5
            view.m.geom_pos[floor] = np.array([0, 0, 0.0]) + fdir * shift
            img = view.render(P['Qraw'][j], extra=extra)
            im.set_data(img)
            txt.set_text(f"t = {t:.2f} s" + (f"   (반복 {rep + 1}/{S['reps']})" if S['reps'] > 1 else ''))
            cur1.set_xdata([t, t]); cur2.set_xdata([t, t]); cur3.set_xdata([t, t])
            fr = fig_frame(fig); wr.write(fr); n_total += 1
            if not poster and si == 0 and t >= 0.45:
                plt.imsave(f'{outdir}/video_mj_upper_poster.png', fr); poster = True
    print('segment', si, 'frames', n_frames * S['reps'])
wr.close()
json.dump({S['trial']: {'mass': P['mass'], 'belt': P['belt'], 'resid_max_pct': P['resid_pct'], 'resid_mean_pct': P['resid_mean_pct'], 'peak_BW': [float(P['vy'][s][P['sel']].max() / P['W']) for s in 'rl'],
                        'tau_peak_Nm_per_kg': {k: float(v) for k, v in zip(['hip_flexion_r', 'knee_r', 'ankle_r', 'lumbar_extension'], np.abs(P['tau'][P['inner']]).max(axis=0) / P['mass'])}} for S, P in zip(SEGS, segs)},
          open(f'{outdir}/upper_video_summary.json', 'w'), indent=1)
print('saved', n_total)
