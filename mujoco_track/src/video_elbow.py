# -*- coding: utf-8 -*-
"""영상 ⑥ — 팔꿈치 1자유도 리그 sim-to-real 리허설: 가상 실물(T3.1 3상태 SMA) vs README 1차 모델 vs 히스테리시스 보강안, 3개 리그 나란히.
구간 A: 폐루프 스텝 +45° (PD + 중력 보상 + 열 상한, 하중 2 kg)  구간 B: 개루프 PRBS 듀티 (식별에 안 쓴 파형)
사용: xvfb-run -a -s "-screen 0 1280x720x24" env MUJOCO_GL=glfw python3 -I video_elbow.py <outdir>
"""
import sys, json, math
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np, mujoco
from vid_common import MjView, FrameWriter, fig_frame, setup_fonts, act_rgba, CR, BL, GR, MU, INK
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

outdir = sys.argv[1]
setup_fonts()
z = np.load(result('elbow_logs.npz'))
res = json.load(open(result('elbow_results.json')))
R_P, FMAX = 0.025, 450.0
XS = [-0.95, 0.0, 0.95]
LABELS = ['가상 실물 (T3.1 3상태 SMA: 열·상분율·히스테리시스)', 'README 안: 1차 열지연 + 온도 포화', '보강안: 1차 열지연 + 히스테리시스 정적 맵']
COLS = [CR, 'k', BL]


def rig_xml():
    body = ''
    for i, x in enumerate(XS):
        body += f'''
    <geom type="capsule" fromto="{x} 0.46 0 {x} 0 0" size="0.022" rgba="0.72 0.7 0.7 1"/>
    <geom type="cylinder" pos="{x} 0 0" size="0.03 0.014" rgba="0.45 0.45 0.5 1"/>
    <geom type="box" pos="{x - 0.09} 0.33 0" size="0.03 0.07 0.03" rgba="0.55 0.55 0.6 1"/>
    <geom type="box" pos="{x + 0.09} 0.33 0" size="0.03 0.07 0.03" rgba="0.55 0.55 0.6 1"/>
    <body name="forearm{i}" pos="{x} 0 0">
      <joint name="elbow{i}" type="hinge" axis="0 0 1" range="-1.75 0.8" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" rgba="0.6 0.6 0.65 1"/>
      <body name="load{i}" pos="0.30 0 0"><geom type="sphere" size="0.04" rgba="0.6 0.1 0.15 1"/></body>
    </body>'''
    return f'''<mujoco model="elbow_rigs_vis">
  <compiler angle="radian"/>
  <option gravity="0 -9.81 0"/>
  <worldbody>
    <light pos="0 3 2" dir="0 -1 -0.5" directional="true"/>
    <geom name="floor" type="plane" pos="0 -0.55 0" size="6 6 0.1" quat="0.70710678 -0.70710678 0 0" rgba="0.9 0.9 0.9 1"/>{body}
  </worldbody>
</mujoco>'''


view = MjView(xml_string=rig_xml(), w=1500, h=470)
view.set_camera([0.0, 0.03, 0.0], 0, 6, 1.45)


def cable_geoms(i, Ff, Fe):
    x = XS[i]
    out = []
    for sgn, F in [(-1, Ff), (1, Fe)]:
        u = min(F / FMAX, 1.0)
        c = act_rgba(u)
        out.append({'type': 'capsule', 'from': [x + sgn * 0.09, 0.26, 0.0], 'to': [x + sgn * R_P, 0.0, 0.0], 'width': 0.006 + 0.008 * u, 'rgba': c})
    return out


def render(qs, forces):
    qpos = np.array(qs)
    extra = []
    for i in range(3):
        extra += cable_geoms(i, *forces[i])
    return view.render(qpos, extra=extra)


W_, H_, FPS = 1540, 924, 30
fig = plt.figure(figsize=(W_ / 100, H_ / 100), dpi=100)
gs = GridSpec(2, 3, height_ratios=[1.0, 0.78], left=0.045, right=0.985, top=0.9, bottom=0.075, wspace=0.26, hspace=0.3)
ax_img = fig.add_subplot(gs[0, :]); ax_img.axis('off')
ax_q = fig.add_subplot(gs[1, 0]); ax_T = fig.add_subplot(gs[1, 1]); ax_u = fig.add_subplot(gs[1, 2])
sup = fig.suptitle('', fontsize=12.5, color=INK, y=0.965)
im = ax_img.imshow(np.zeros((view.h, view.w, 3), np.uint8))
lab_txt = [ax_img.text(0.17 + 0.333 * i, 0.985, LABELS[i], transform=ax_img.transAxes, fontsize=10.5, ha='center', va='top', color=COLS[i], fontweight='bold') for i in range(3)]
t_txt = ax_img.text(0.01, 0.06, '', transform=ax_img.transAxes, fontsize=11, color=INK)
ax_img.text(0.99, 0.06, '케이블 색/굵기 = 힘 / F_max (M3 450 N) · 왼쪽 = 굴근, 오른쪽 = 신근 · 풀리 R25 · 말단 하중 2 kg @ 0.30 m', transform=ax_img.transAxes, fontsize=9.5, ha='right', color=MU)

SEGS = [dict(key='step', t0=0.5, t1=3.6, speed=0.5, title='폐루프 스텝 +45° (PD 25/1.5 + 중력 보상 + 열 상한 u_max(T̂)) — 같은 제어기를 세 플랜트에', ref=lambda t: -90 + (45 if t >= 1.0 else 0),
                 note=f"실물: 오버슈트 {res['closedloop_demo']['step_overshoot_real_deg']:.1f}° · 정상 오차 {res['closedloop_demo']['step_ss_err_real_deg']:.1f}° · T_max {res['closedloop_demo']['step_Tmax_real_C']:.0f} °C   |   README 모델 정상 오차 {res['closedloop_demo']['step_ss_err_model_deg']:.0f}° (발산) · 보강안 실물 대비 RMSE {res['closedloop_demo']['step_real_vs_hyst_rmse_deg']:.1f}°"),
        dict(key='prbs', t0=0.0, t1=6.0, speed=1.0, title='개루프 PRBS 듀티 (0.4 s 간격, 식별에 안 쓴 파형) — 같은 듀티를 세 플랜트에', ref=None,
                 note=f"실물 대비 관절각 RMSE: README 1차 모델 {res['openloop_validation']['prbs']['rmse_deg']:.1f}° → 보강안 {res['openloop_validation']['prbs']['rmse_deg_hyst']:.1f}° (동작 범위 {res['openloop_validation']['prbs']['range_deg']:.0f}°)")]
wr = FrameWriter(f'{outdir}/video_mj_elbow.mp4', W_, H_, FPS)
poster = False; n_total = 0
HOLD = 1.2
for si, S in enumerate(SEGS):
    lr, lm, lh = z[f"{S['key']}_real"], z[f"{S['key']}_model"], z[f"{S['key']}_hyst"]
    logs3 = [lr, lm, lh]
    t0, t1 = S['t0'], S['t1']
    sel = (lr[:, 0] >= t0) & (lr[:, 0] <= t1)
    tt = lr[sel, 0]
    for ax in (ax_q, ax_T, ax_u):
        ax.clear()
    for lg, c, nm, ls in zip(logs3, COLS, ['가상 실물', 'README 1차', '보강안'], ['-', '--', '-.']):
        ax_q.plot(lg[sel, 0], np.degrees(lg[sel, 1]), color=c, lw=1.9 if c == CR else 1.4, ls=ls, label=nm)
    if S['ref'] is not None:
        ax_q.plot(tt, [S['ref'](t) for t in tt], color=MU, lw=1.0, ls=':', label='기준')
    ax_q.set_xlim(t0, t1); ax_q.set_xlabel('time (s)'); ax_q.set_ylabel('팔꿈치 각도 (°)'); ax_q.grid(alpha=0.3); ax_q.legend(fontsize=8.5, loc='lower right' if si == 0 else 'upper right')
    ax_q.set_title('관절 각도 — 세 플랜트', fontsize=10.5)
    ax_T.plot(lr[sel, 0], lr[sel, 7], color=CR, lw=1.9, label='가상 실물 굴근 온도')
    ax_T.axhline(120, color='k', lw=0.9, ls='--'); ax_T.text(t0 + 0.05 * (t1 - t0), 121.5, 'T_max 120 °C (열 상한 u_max = (120 − T̂)/10)', fontsize=8, color=INK)
    ax_T.axhspan(55, 72, color=CR, alpha=0.08); ax_T.text(t1 - 0.02 * (t1 - t0), 56, 'As~Af 55~72', fontsize=7.5, ha='right', color=CR)
    ax_T.axhspan(38, 52, color=BL, alpha=0.08); ax_T.text(t1 - 0.02 * (t1 - t0), 39, 'Mf~Ms 38~52', fontsize=7.5, ha='right', color=BL)
    ax_T.set_xlim(t0, t1); ax_T.set_ylim(20, 135); ax_T.set_xlabel('time (s)'); ax_T.set_ylabel('°C'); ax_T.grid(alpha=0.3)
    ax_T.set_title('가상 실물 SMA 온도 (3상태 모델) — 변태 문턱과 열 상한', fontsize=10.5)
    ax_u.plot(lr[sel, 0], lr[sel, 5], color=CR, lw=1.6, label='굴근 듀티 (실물에 입력)'); ax_u.plot(lr[sel, 0], lr[sel, 6], color=BL, lw=1.6, label='신근 듀티')
    ax_u.axhline(0.35, color='k', lw=0.8, ls='--'); ax_u.text(t0 + 0.05 * (t1 - t0), 0.36, '공랭 연속 듀티 상한 0.35', fontsize=8)
    ax_u.set_xlim(t0, t1); ax_u.set_ylim(0, 1.05); ax_u.set_xlabel('time (s)'); ax_u.set_ylabel('duty'); ax_u.grid(alpha=0.3); ax_u.legend(fontsize=8.5, loc='upper right')
    ax_u.set_title('듀티 명령 (가열 PWM)', fontsize=10.5)
    curs = [ax.axvline(t0, color='k', lw=1) for ax in (ax_q, ax_T, ax_u)]
    note = ax_q.text(0.0, 1.14, '', transform=ax_q.transAxes, fontsize=9.5, color=INK, va='bottom')
    sup.set_text(f"팔꿈치 리그 sim-to-real 리허설 {si + 1}/2 — {S['title']}   {'1/2 배속' if S['speed'] < 1 else '실시간'}")
    n_frames = int(round((t1 - t0) / S['speed'] * FPS)) + 1
    for k in range(n_frames + int(HOLD * FPS)):
        t = min(t0 + k / FPS * S['speed'], t1)
        j = int(np.argmin(np.abs(lr[:, 0] - t)))
        if k <= n_frames:
            qs = [lg[j, 1] for lg in logs3]
            forces = [(lg[j, 3], lg[j, 4]) for lg in logs3]
            im.set_data(render(qs, forces))
        t_txt.set_text(f"t = {t:.2f} s" + (f"   각도: 실물 {math.degrees(lr[j, 1]):.1f}° · README {math.degrees(lm[j, 1]):.1f}° · 보강안 {math.degrees(lh[j, 1]):.1f}°"))
        for c_ in curs:
            c_.set_xdata([t, t])
        note.set_text(S['note'] if k >= n_frames - 1 else '')
        fr = fig_frame(fig); wr.write(fr); n_total += 1
        if not poster and si == 0 and t >= 1.6:
            plt.imsave(f'{outdir}/video_mj_elbow_poster.png', fr); poster = True
    print('segment', si, n_frames)
wr.close(); print('saved', n_total)
