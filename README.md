# artificial_muscle_humanoid — 세종대 의료로봇연구실

미래판기술프로젝트(RS-2026-25541200) 인공근육 휴머노이드 플랫폼 — 세종대 담당분(설계 해석·시뮬레이션·제어)의 공유 저장소.

| 폴더 | 내용 | 담당 |
| --- | --- | --- |
| [`mujoco_track/`](mujoco_track/README.md) | OpenSim 설계안 → MuJoCo 변환 · 역동역학 재생 · 순동역학 보행 · 잔차 RL · 팔꿈치 sim-to-real 리허설 · 발표 영상 (2026-10-31 ~ 11-03) | 우현수 |

설계 파이프라인(데이터 → 검산 → 시너지 → 근육 수 → 방향·용량 → 경로 → 재현 검증)과 Moore2015 변환은 나유빈 학생 저장소
[nauqin/Artificial-muscle-humanoid](https://github.com/nauqin/Artificial-muscle-humanoid)에 있다. `mujoco_track/`의 스크립트는 그 저장소의 데이터를
`AMH_DATA` 환경변수로 가리켜 읽는다(복사하지 않음). 두 저장소를 나란히 clone 하면 된다:

```bash
git clone https://github.com/nauqin/Artificial-muscle-humanoid
git clone https://github.com/Sejong-Medical-Robotics-Lab/artificial_muscle_humanoid
export AMH_DATA=$PWD/Artificial-muscle-humanoid
cd artificial_muscle_humanoid/mujoco_track && python3 -I src/paths.py
```
