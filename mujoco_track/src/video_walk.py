# -*- coding: utf-8 -*-
"""영상 ⑤ — 순동역학 보행: 명목 제어기 3조건(이상 / SMA 지연 / SMA+SES) + 잔차 RL 정책, yz 하네스.
영상 A video_mj_walk.mp4: 4개 구간 순차 (왼쪽 3D 렌더 + 고스트 기준 IK + 활성 색 + 하네스·접촉 화살표, 오른쪽 활성 막대·관절 추종·골반/하네스)
영상 B video_mj_walk_rl.mp4: 명목 제어기 vs 잔차 RL 정책 나란히 (같은 시작 위상 0)
사용: xvfb-run -a -s "-screen 0 1280x720x24" env MUJOCO_GL=glfw python3 -I video_walk.py <outdir> <ppo_zip>
"""
import sys, json, math, os
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, MODELS, RESULTS, model, result
import numpy as np, mujoco
from rl_env import ResidualWalkEnv
from walk_sim import SMA
from vid_common import MjView, FrameWriter, fig_frame, setup_fonts, act_rgba, ghost_capsules, foot_contact_arrows, CR, BL, GR, MU, INK, GOLD
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

outdir, ppo_zip = sys.argv[1:3]
setup_fonts()
XML = model('leg16_dyn.xml')
COND = [('이상 액추에이터 (열지연 없음)', 0.0, (1e-3, 1e-3), None),
        ('SMA 1차 열지연 0.12 / 0.13 s', 0.0, (0.12, 0.13), None),
        ('SMA 열지연 + 발목 병렬 강성 100 N·m/rad (SES)', 100.0, (0.12, 0.13), None),
        ('잔차 RL 정책 (PPO 1.2 M 스텝) + SMA 열지연 + SES', 100.0, (0.12, 0.13), 'policy')]
CACHE = f'{outdir}/walk_video_logs.npz'


def simulate(label, k_ankle, tau, mode, cycles, t0=0.0, policy=None):
    env = ResidualWalkEnv(harness='yz', max_cycles=cycles, random_phase=False)
    L, m, sim = env.L, env.L.m, env.sim
    for s in 'rl':
        m.jnt_stiffness[L.jid[f'ankle_angle_{s}']] = k_ankle
    sim.sma = SMA(m.nu, *tau)
    if tau[0] < 0.01:
        sim.gamma = 0.0
    sim.reset(t0); env.t_start = t0; env.steps = 0
    obs = env._obs()
    feet = [{L.bid['calcn_r'], L.bid['toes_r']}, {L.bid['calcn_l'], L.bid['toes_l']}]
    log = {'t': [], 'qpos': [], 'qref_pos': [], 'a': [], 'Fh': [], 'q': [], 'qref': [], 'pelvis_y': [], 'pref_y': [], 'arrows': [], 'rmse': []}
    while True:
        act = np.zeros(24) if policy is None else policy.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, info = env.step(act)
        d = sim.d
        qd, pos, vel = env.ref.leg_ref(sim.t)
        log['t'].append(sim.t); log['qpos'].append(d.qpos.copy()); log['qref_pos'].append(L.qpos_from_coords(qd))
        log['a'].append(sim.sma.a.copy()); log['Fh'].append(d.xfrc_applied[sim.pelvis, :3].copy())
        log['q'].append(d.qpos[sim.jq].copy()); log['qref'].append(pos.copy()); log['pelvis_y'].append(d.xpos[sim.pelvis][1]); log['pref_y'].append(qd['pelvis_ty'])
        arr = foot_contact_arrows(m, d, feet, scale=0.0005)
        log['arrows'].append([(a['from'], a['to']) for a in arr]); log['rmse'].append(info['rmse_deg'])
        if term or trunc:
            break
    out = {k: (np.array(v) if k != 'arrows' else v) for k, v in log.items()}
    W = env.W
    out['W'] = W
    out['fell'] = bool(info['fell']); out['label'] = label
    out['rmse'] = float(np.sqrt(np.mean(np.square(out['rmse']))))
    out['harness_BW'] = float(np.sqrt(np.mean(np.sum(out['Fh'] ** 2, axis=1))) / W)
    out['time_s'] = float(out['t'][-1] - out['t'][0])
    out['T'] = env.ref.T
    print(label, 'time', round(out['time_s'], 2), 'fell', out['fell'], 'rmse', round(out['rmse'], 1), 'harness', round(out['harness_BW'], 2))
    return out


from stable_baselines3 import PPO
policy = PPO.load(ppo_zip, device='cpu')
logs = []
for label, k, tau, mode in COND:
    logs.append(simulate(label, k, tau, mode, cycles=3.0 if mode == 'policy' else 1.0, policy=policy if mode == 'policy' else None))
nominal_vs = simulate('명목 제어기 (SMA + SES), 3주기 시도', 100.0, (0.12, 0.13), None, cycles=3.0)

view = MjView(XML, w=900, h=800, tendon_width=0.0085)
W_, H_, FPS = 1540, 924, 30
SPEED = 1 / 3
HOLD = 1.0
COORD_NAMES = ['고관절 굴곡', '고관절 내전', '무릎', '발목']
names = [f'M{i}' for i in range(8)]


def render_frame(view, lg, j, extra_hold=False):
    qpos = lg['qpos'][j]
    extra = ghost_capsules(view, lg['qref_pos'][j])
    for fr_, to_ in lg['arrows'][j]:
        extra.append({'type': 'arrow', 'from': fr_, 'to': to_, 'width': 0.014, 'rgba': (0.16, 0.6, 0.3, 1.0)})
    Fh = lg['Fh'][j]; p = qpos[0:3]
    if np.linalg.norm(Fh) > 5:
        extra.append({'type': 'arrow', 'from': p, 'to': p + 0.0008 * Fh, 'width': 0.016, 'rgba': (0.93, 0.6, 0.1, 1.0)})
    a = lg['a'][j]
    view.m.tendon_width[:] = 0.006 + 0.008 * a
    rgba = np.array([act_rgba(x) for x in a])
    view.set_camera([p[0] + 0.05, 0.82, 0.0], 25, 9, 2.3)
    return view.render(qpos, extra=extra, tendon_rgba=rgba)


# ======================= 영상 A: 4조건 순차
fig = plt.figure(figsize=(W_ / 100, H_ / 100), dpi=100)
gs = GridSpec(3, 2, width_ratios=[1.12, 1], height_ratios=[0.75, 1.05, 0.85], left=0.015, right=0.985, top=0.9, bottom=0.065, wspace=0.14, hspace=0.62)
ax_img = fig.add_subplot(gs[:, 0]); ax_img.axis('off')
ax_u = fig.add_subplot(gs[0, 1]); ax_q = fig.add_subplot(gs[1, 1]); ax_p = fig.add_subplot(gs[2, 1])
sup = fig.suptitle('', fontsize=12.5, color=INK, y=0.965)
im = ax_img.imshow(np.zeros((view.h, view.w, 3), np.uint8))
txt = ax_img.text(0.02, 0.975, '', transform=ax_img.transAxes, fontsize=11.5, va='top', color=INK)
res_txt = ax_img.text(0.02, 0.905, '', transform=ax_img.transAxes, fontsize=11.5, va='top', color=CR, fontweight='bold')
ax_img.text(0.02, 0.02, '회색 = 기준 IK(고스트) · 케이블 색/굵기 = 활성 a · 초록 = 접촉력 · 주황 = 하네스 힘(높이·좌우·자세만, 전진은 자유)', transform=ax_img.transAxes, fontsize=9.5, color=MU)
x = np.arange(8); bw = 0.38
bars_r = ax_u.bar(x - bw / 2, np.zeros(8), bw, color=CR, label='오른다리'); bars_l = ax_u.bar(x + bw / 2, np.zeros(8), bw, color=BL, label='왼다리')
ax_u.axhline(1.0, color='k', ls='--', lw=0.8); ax_u.set_xticks(x); ax_u.set_xticklabels(names); ax_u.set_ylim(0, 1.15); ax_u.set_ylabel('a')
ax_u.set_title('이 순간 케이블 16개의 활성 a (열지연 뒤 실제 힘 / F_max)', fontsize=10.5); ax_u.legend(fontsize=8.5, loc='upper right', ncol=2)
wr = FrameWriter(f'{outdir}/video_mj_walk.mp4', W_, H_, FPS)
poster = False; n_total = 0
for ci, lg in enumerate(logs):
    T = lg['t'] - lg['t'][0]
    # 관절 추종 패널
    ax_q.clear()
    for jj, (ci_, c) in enumerate([(0, '고관절 굴곡'), (2, '무릎'), (3, '발목')]):
        col = ['#1f77b4', '#2ca02c', '#d62728'][jj]
        ax_q.plot(T, np.degrees(lg['qref'][:, ci_]), color=col, lw=1.2, ls=':')
        ax_q.plot(T, np.degrees(lg['q'][:, ci_]), color=col, lw=1.8, label=c)
    cur1 = ax_q.axvline(0, color='k', lw=1)
    ax_q.set_xlim(0, max(T[-1], 1.0)); ax_q.set_ylabel('각도 (°)'); ax_q.grid(alpha=0.3); ax_q.legend(fontsize=8.5, ncol=3, loc='upper right')
    ax_q.set_title(f'오른다리 관절 각도 — 로봇(실선) vs 기준 IK(점선) · RMSE {lg["rmse"]:.1f}°', fontsize=10.5)
    ax_p.clear()
    ax_p.plot(T, lg['pelvis_y'], color=INK, lw=1.8, label='골반 높이 (m)'); ax_p.plot(T, lg['pref_y'], color=INK, lw=1.0, ls=':', label='기준')
    hb = np.linalg.norm(lg['Fh'], axis=1) / lg['W']
    ax_p.plot(T, hb, color='#E39A1C', lw=1.8, label='하네스 힘 (BW)')
    ax_p.axhline(0.6, color=CR, lw=0.8, ls='--'); ax_p.text(0.01, 0.61, '넘어짐 판정 0.6 m', fontsize=8, color=CR)
    cur2 = ax_p.axvline(0, color='k', lw=1)
    ax_p.set_xlim(0, max(T[-1], 1.0)); ax_p.set_ylim(0, 1.2); ax_p.set_xlabel('time (s)'); ax_p.grid(alpha=0.3); ax_p.legend(fontsize=8.5, ncol=3, loc='upper right')
    ax_p.set_title(f'골반 높이 · 하네스 지지력 (rms {lg["harness_BW"]:.2f} BW)', fontsize=10.5)
    sup.set_text(f"순동역학 보행 {ci + 1}/4 — {lg['label']} · {'명목 제어기(PD + ID 피드포워드 → 케이블 배분 → 열지연 보상)' if ci < 3 else '명목 제어기 + 학습된 잔차(활성 16 · 관절 목표 8)'} · yz 하네스   1/3 배속")
    n_frames = int(round(T[-1] / SPEED * FPS)) + 1
    verdict = f"결과: {'넘어짐 ' + format(lg['time_s'], '.2f') + ' s' if lg['fell'] else ('완주 ' + format(lg['time_s'] / lg['T'], '.0f') + '주기')} · RMSE {lg['rmse']:.1f}° · 하네스 {lg['harness_BW']:.2f} BW"
    for k in range(n_frames + int(HOLD * FPS)):
        t = min(k / FPS * SPEED, T[-1])
        j = int(np.argmin(np.abs(T - t)))
        if k < n_frames or k == n_frames:
            img = render_frame(view, lg, j)
            im.set_data(img)
        txt.set_text(f"t = {t:.2f} s   주기 {int(t // lg['T']) + 1}")
        res_txt.set_text(verdict if k >= n_frames - 1 else '')
        for b, v in zip(bars_r, lg['a'][j][0::2]):
            b.set_height(v)
        for b, v in zip(bars_l, lg['a'][j][1::2]):
            b.set_height(v)
        cur1.set_xdata([t, t]); cur2.set_xdata([t, t])
        fr = fig_frame(fig); wr.write(fr); n_total += 1
        if not poster and ci == 2 and t >= 0.5:
            plt.imsave(f'{outdir}/video_mj_walk_poster.png', fr); poster = True
    print('segment', ci, 'frames', n_frames)
wr.close(); print('video A saved', n_total)

# ======================= 영상 B: 명목 vs 정책 나란히 (시작 위상 0, 3주기 상한)
plt.close(fig)
viewB = MjView(XML, w=760, h=600, tendon_width=0.0085)
fig = plt.figure(figsize=(W_ / 100, H_ / 100), dpi=100)
gs = GridSpec(2, 2, height_ratios=[1.25, 0.75], left=0.03, right=0.985, top=0.9, bottom=0.07, wspace=0.08, hspace=0.3)
axL = fig.add_subplot(gs[0, 0]); axR = fig.add_subplot(gs[0, 1]); axL.axis('off'); axR.axis('off')
axp = fig.add_subplot(gs[1, 0]); axh = fig.add_subplot(gs[1, 1])
fig.suptitle('잔차 RL — 명목 제어기(왼쪽) vs 학습된 잔차 정책(오른쪽), 같은 시작 위상 0 · SMA 열지연 + SES · yz 하네스 · 3주기 상한   1/3 배속', fontsize=12.5, color=INK, y=0.965)
imL = axL.imshow(np.zeros((viewB.h, viewB.w, 3), np.uint8)); imR = axR.imshow(np.zeros((viewB.h, viewB.w, 3), np.uint8))
tL = axL.text(0.02, 0.97, '', transform=axL.transAxes, fontsize=11.5, va='top', color=INK); tR = axR.text(0.02, 0.97, '', transform=axR.transAxes, fontsize=11.5, va='top', color=INK)
A, B = nominal_vs, logs[3]
TA = A['t'] - A['t'][0]; TB = B['t'] - B['t'][0]
axp.plot(TA, A['pelvis_y'], color=MU, lw=1.8, label='명목'); axp.plot(TB, B['pelvis_y'], color=CR, lw=1.8, label='잔차 정책'); axp.plot(TB, B['pref_y'], color='k', lw=0.9, ls=':', label='기준')
axp.axhline(0.6, color=CR, lw=0.8, ls='--'); axp.set_ylim(0.4, 1.1); axp.set_xlim(0, TB[-1]); axp.set_ylabel('골반 높이 (m)'); axp.set_xlabel('time (s)'); axp.grid(alpha=0.3); axp.legend(fontsize=8.5, ncol=3, loc='lower left')
axp.set_title(f"골반 높이 — 명목: {A['time_s']:.2f} s 넘어짐 · 정책: {'완주 3주기' if not B['fell'] else format(B['time_s'], '.2f') + ' s'}", fontsize=10.5)
axh.plot(TA, np.linalg.norm(A['Fh'], axis=1) / A['W'], color=MU, lw=1.8, label='명목'); axh.plot(TB, np.linalg.norm(B['Fh'], axis=1) / B['W'], color=CR, lw=1.8, label='잔차 정책')
axh.set_ylim(0, 1.0); axh.set_xlim(0, TB[-1]); axh.set_ylabel('하네스 힘 (BW)'); axh.set_xlabel('time (s)'); axh.grid(alpha=0.3); axh.legend(fontsize=8.5, loc='upper right')
axh.set_title(f"하네스 지지력 rms — 명목 {A['harness_BW']:.2f} BW · 정책 {B['harness_BW']:.2f} BW (관절 RMSE {A['rmse']:.1f}° → {B['rmse']:.1f}°)", fontsize=10.5)
c1 = axp.axvline(0, color='k', lw=1); c2 = axh.axvline(0, color='k', lw=1)
wr = FrameWriter(f'{outdir}/video_mj_walk_rl.mp4', W_, H_, FPS)
n_frames = int(round(TB[-1] / SPEED * FPS)) + 1
posterB = False; n_total = 0
for k in range(n_frames + int(HOLD * FPS)):
    t = min(k / FPS * SPEED, TB[-1])
    jB = int(np.argmin(np.abs(TB - t))); jA = int(np.argmin(np.abs(TA - min(t, TA[-1]))))
    if k <= n_frames:
        imL.set_data(render_frame(viewB, A, jA)); imR.set_data(render_frame(viewB, B, jB))
    tL.set_text(f"명목 제어기   t = {min(t, TA[-1]):.2f} s" + ('   넘어짐' if t >= TA[-1] - 1e-6 else ''))
    tR.set_text(f"잔차 RL 정책   t = {t:.2f} s   주기 {int(t // B['T']) + 1}" + ('   완주' if (k >= n_frames - 1 and not B['fell']) else ''))
    c1.set_xdata([t, t]); c2.set_xdata([t, t])
    fr = fig_frame(fig); wr.write(fr); n_total += 1
    if not posterB and t >= 1.3:
        plt.imsave(f'{outdir}/video_mj_walk_rl_poster.png', fr); posterB = True
wr.close(); print('video B saved', n_total)
json.dump({lg['label']: {k: lg[k] for k in ('time_s', 'fell', 'rmse', 'harness_BW')} for lg in logs + [nominal_vs]}, open(f'{outdir}/walk_video_summary.json', 'w'), indent=1, ensure_ascii=False)
