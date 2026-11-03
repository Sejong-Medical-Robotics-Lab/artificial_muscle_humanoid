# -*- coding: utf-8 -*-
"""영상 ④ — MuJoCo 역동역학 재생(mj_inverse) + 정적 케이블 검사: subject3 walking1 (학생 ID 창 0.248~1.121 s).
왼쪽: 변환 모델 + 실측 GRF 화살표 + 케이블 색 = 힘 분율 u.  오른쪽: u 막대(양다리), 관절 모멘트, 케이블이 못 낸 몫.
사용: xvfb-run -a -s "-screen 0 1280x720x24" env MUJOCO_GL=glfw python3 -I video_id.py <outdir>
"""
import sys, json, math
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np, mujoco
from scipy.optimize import lsq_linear
from common import Leg16, load_trial, COORDS
from vid_common import MjView, FrameWriter, fig_frame, setup_fonts, act_rgba, CR, BL, GR, MU, INK
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

outdir = sys.argv[1]
R = DATA
OSIM, PJ, DATA = f'{R}/data/subject3/subject3_scaled.osim', f'{R}/report/newmus_paths_k8.json', f'{R}/data/subject3'
XML = model('leg16.xml')
setup_fonts()
L = Leg16(XML, OSIM, PJ)
z = np.load(result('id_walking1.npz'))
res = json.load(open(result('id_walking1.json')))
hi, di, hg, dg = load_trial(DATA, 'walking1')
T = z['t'] - z['t'][0]; Q = z['Q']; grf = z['grf']; vgrf = z['vgrf']
tau = {'r': z['tau_r'], 'l': z['tau_l']}; RA = {'r': z['RA_r'], 'l': z['RA_l']}
mass = L.mass; W = mass * 9.81
names = [mu['name'] for mu in L.P['muscles']]
view = MjView(XML, w=900, h=800, tendon_width=0.0085)

# 정적 케이블 검사 (학생 2단계와 같은 잣대: 0 ≤ f ≤ Fmax, R f = 1.1 τ)
margin = 1.1
U = {s: np.zeros((len(T), 8)) for s in 'rl'}; E = {s: np.zeros((len(T), 4)) for s in 'rl'}
for s in 'rl':
    Fm = L.Fmax[L.act_idx[s]]
    for k in range(len(T)):
        sol = lsq_linear(RA[s][k], margin * tau[s][k], bounds=(np.zeros(8), Fm), lsmr_tol='auto')
        U[s][k] = sol.x / Fm; E[s][k] = RA[s][k] @ sol.x - margin * tau[s][k]
peak = {s: np.max(np.abs(margin * tau[s]), axis=0) for s in 'rl'}
worst = {s: float((np.max(np.abs(E[s]), axis=0) / peak[s] * 100).max()) for s in 'rl'}
print('worst %', worst)

# ---------------- 그림
W_, H_, FPS = 1540, 924, 30
fig = plt.figure(figsize=(W_ / 100, H_ / 100), dpi=100)
gs = GridSpec(3, 2, width_ratios=[1.12, 1], height_ratios=[0.8, 1, 0.8], left=0.015, right=0.985, top=0.9, bottom=0.065, wspace=0.14, hspace=0.6)
ax_img = fig.add_subplot(gs[:, 0]); ax_img.axis('off')
ax_u = fig.add_subplot(gs[0, 1]); ax_tau = fig.add_subplot(gs[1, 1]); ax_res = fig.add_subplot(gs[2, 1])
fig.suptitle('MuJoCo 역동역학 재생 — subject3 walking1 (IK 6 Hz · 지면반력 실측 · mj_inverse) + 정적 케이블 검사 (0 ≤ f ≤ F_max, R f = 1.1 τ)   1/3 배속', fontsize=12.5, color=INK, y=0.965)
im = ax_img.imshow(np.zeros((view.h, view.w, 3), np.uint8))
txt = ax_img.text(0.02, 0.975, '', transform=ax_img.transAxes, fontsize=11, va='top', color=INK)
ax_img.text(0.02, 0.02, '초록 = 실측 지면반력(압력중심에서) · 케이블 색 = 힘 분율 u (연파랑 0 → 진홍 1)', transform=ax_img.transAxes, fontsize=9.5, color=MU)
foot2 = ax_img.text(0.02, 0.055, f"관절 모멘트 피크 (Nm/kg) MuJoCo vs OpenSim: 발목 {res['id']['peak_ankle_angle_r_Nm_per_kg']:.3f}/1.399 · 무릎 {res['id']['peak_knee_angle_r_Nm_per_kg']:.3f}/0.409 · 고관절 {res['id']['peak_hip_flexion_r_Nm_per_kg']:.3f}/0.945 · 골반 잔차력 최대 {res['id']['residual_pct']:.1f} %",
                    transform=ax_img.transAxes, fontsize=9.5, color=MU)

# u 막대
x = np.arange(8); bw = 0.38
bars_r = ax_u.bar(x - bw / 2, np.zeros(8), bw, color=CR, label='오른다리')
bars_l = ax_u.bar(x + bw / 2, np.zeros(8), bw, color=BL, label='왼다리')
ax_u.axhline(1.0, color='k', ls='--', lw=0.8)
ax_u.set_xticks(x); ax_u.set_xticklabels(names); ax_u.set_ylim(0, 1.15); ax_u.set_ylabel('u')
ax_u.set_title('이 순간 케이블 16개의 힘 분율 u = f/F_max (최소자승, 여유 10 %) — 상한 도달: 오른 M1·M2·M7, 왼 M1·M2', fontsize=9.5)
ax_u.legend(fontsize=8.5, loc='upper right', ncol=2)

# 관절 모멘트
JN = ['고관절 굴곡', '고관절 내전', '무릎', '발목']; JC = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for ci in range(4):
    ax_tau.plot(T, tau['r'][:, ci], color=JC[ci], lw=1.8, label=JN[ci])
    ax_tau.plot(T, tau['l'][:, ci], color=JC[ci], lw=1.0, ls=':')
cur1 = ax_tau.axvline(0, color='k', lw=1)
ax_tau.set_xlim(T[0], T[-1]); ax_tau.set_ylabel('Nm'); ax_tau.grid(alpha=0.3)
ax_tau.set_title(f'관절 모멘트 — MuJoCo mj_inverse (오른다리 실선 · 왼다리 점선), 구간 {z["t"][0]:.2f}~{z["t"][-1]:.2f} s', fontsize=10.5)
ax_tau.legend(fontsize=8.5, ncol=4, loc='lower left')

# 못 낸 몫
for ci in range(4):
    ax_res.plot(T, E['r'][:, ci], color=JC[ci], lw=1.6)
    ax_res.plot(T, E['l'][:, ci], color=JC[ci], lw=0.9, ls=':')
cur2 = ax_res.axvline(0, color='k', lw=1)
ax_res.set_xlim(T[0], T[-1]); ax_res.set_ylim(-10, 10); ax_res.set_ylabel('Nm'); ax_res.set_xlabel('time (s)'); ax_res.grid(alpha=0.3)
ax_res.set_title(f'케이블이 못 낸 몫 R f − 1.1 τ — 최악 오른 {worst["r"]:.1f} % · 왼 {worst["l"]:.1f} % (ID 창 · 전 구간 0~1.3 s 기준 3.9 · 0.8 %)', fontsize=9.5)
now_txt = ax_res.text(0.99, 0.95, '', transform=ax_res.transAxes, fontsize=9, ha='right', va='top', color=MU)

wr = FrameWriter(f'{outdir}/video_mj_id.mp4', W_, H_, FPS)
speed = 1 / 3
n_frames = int(round(T[-1] / speed * FPS))
reps = 3
poster = False; n_total = 0
pel = [np.array(Q[k][0:3]) for k in range(len(T))]
for rep in range(reps):
    for k in range(n_frames):
        t = k / FPS * speed
        j = int(np.argmin(np.abs(T - t)))
        qpos = Q[j]
        # GRF 화살표
        extra = []
        for s in 'rl':
            F = np.array([grf[j, hg.index(f'ground_force_calcn_{s}_v{a}')] for a in 'xyz'])
            Pt = np.array([grf[j, hg.index(f'ground_force_calcn_{s}_p{a}')] for a in 'xyz'])
            if np.linalg.norm(F) > 20:
                extra.append({'type': 'arrow', 'from': Pt, 'to': Pt + 0.0005 * F, 'width': 0.014, 'rgba': (0.16, 0.6, 0.3, 1.0)})
        u16 = np.zeros(16)
        for s in 'rl':
            u16[L.act_idx[s]] = U[s][j]
        rgba = np.array([act_rgba(u) for u in u16])
        view.m.tendon_width[:] = 0.006 + 0.007 * u16
        view.set_camera([pel[j][0] + 0.05, 0.82, 0.0], 25, 9, 2.25)
        img = view.render(qpos, extra=extra, tendon_rgba=rgba)
        im.set_data(img)
        txt.set_text(f"t = {z['t'][j]:.2f} s   (반복 {rep + 1}/{reps})   vGRF = {vgrf[j] / W:.2f} BW")
        for b, v in zip(bars_r, U['r'][j]):
            b.set_height(v)
        for b, v in zip(bars_l, U['l'][j]):
            b.set_height(v)
        cur1.set_xdata([t, t]); cur2.set_xdata([t, t])
        now_txt.set_text('지금(오른): ' + '  '.join(f'{n} {e:+.1f}' for n, e in zip(JN, E['r'][j])) + ' Nm')
        fr = fig_frame(fig); wr.write(fr); n_total += 1
        if not poster and t >= 0.33:
            plt.imsave(f'{outdir}/video_mj_id_poster.png', fr); poster = True
wr.close()
print('saved', n_total, 'frames')
