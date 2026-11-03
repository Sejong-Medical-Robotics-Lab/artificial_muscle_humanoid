# README 1차 모델의 폐루프 진동 원인 확인: 냉각 시정수만 바꿔 본다
import sys, math
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import DATA, MODELS, RESULTS, model, result
sys.argv = ['x', RESULTS]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'elbow_rig.py'), encoding='utf-8').read().split('\n')
# rig_sim 정의까지만 실행 (식별·검증 본문은 건너뜀)
end = next(i for i, l in enumerate(src) if l.startswith('# ------------------------------------------------------------------ ① 식별'))
exec('\n'.join(src[:end]))
import numpy as np
a0, a1 = 0.1269664833956663, 0.3526688361153039
HP = (0.49365476334437014, 0.12618850066323337, 0.35253736926551454, 0.3370390196909862, 0.33763650066299244)

def pd_controller(q_ref_fn, kp=25.0, kd=1.5, bias=0.06, Fg=FMAX_M3):
    def u_fn(t, q, qd):
        qr, qdr = q_ref_fn(t)
        tau = kp * (qr - q) + kd * (qdr - qd) + (1.0 * 0.15 + 2.0 * 0.30) * 9.81 * math.cos(q)
        f = tau / R_PULLEY
        uf = bias + max(f, 0) / Fg * (a1 - a0) + (a0 if f > 0 else 0)
        ue = bias + max(-f, 0) / Fg * (a1 - a0) + (a0 if f < 0 else 0)
        return min(uf, 1.0), min(ue, 1.0)
    return u_fn

step_ref = lambda t: (-math.pi / 2 + (math.pi / 4 if t >= 1.0 else 0.0), 0.0)
def report(name, mk):
    lg = rig_sim(mk(), mk(), pd_controller(step_ref), 5.0, q0=-math.pi / 2)
    seg = lg[lg[:, 0] > 2.5]
    q = np.degrees(seg[:, 1])
    print(f'{name:38s} 2.5~5 s: 평균 {q.mean():6.1f}°  진폭(p-p) {q.max() - q.min():5.1f}°  정상오차 {abs(q[-1] + 45):4.1f}°  굴근 듀티 평균 {seg[:, 5].mean():.2f}')

report('가상 실물 (3상태)', lambda: SMA3())
report('README 1차  τc = 0.02 s (식별값)', lambda: SMA1(0.49, 0.02, a0, a1))
report('README 1차  τc = 0.10 s', lambda: SMA1(0.49, 0.10, a0, a1))
report('README 1차  τc = 0.25 s', lambda: SMA1(0.49, 0.25, a0, a1))
report('README 1차  τc = 0.49 s (가열과 같게)', lambda: SMA1(0.49, 0.49, a0, a1))
report('보강안 (1차 + 히스테리시스 맵)', lambda: SMA1H(*HP))
