# -*- coding: utf-8 -*-
"""경로 한 곳에서 관리.
- SRC / MODELS / RESULTS : 이 저장소(mujoco_track) 안의 폴더
- DATA : 나유빈 학생 저장소(nauqin/Artificial-muscle-humanoid)의 루트 — data/subject3/*.osim, *.mot, report/newmus_paths_k8.json, data/moore2015/… 를 여기서 읽는다.
  환경변수 AMH_DATA 로 지정하거나, 없으면 ../Artificial-muscle-humanoid, ../, ../../ 순으로 찾는다.
"""
import os

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
MODELS = os.path.join(ROOT, 'models')
RESULTS = os.path.join(ROOT, 'results')


def _find_data():
    cands = []
    if os.environ.get('AMH_DATA'):
        cands.append(os.environ['AMH_DATA'])
    cands += [os.path.join(ROOT, '..', 'Artificial-muscle-humanoid'), os.path.join(ROOT, '..'), os.path.join(ROOT, '..', '..')]
    for c in cands:
        if os.path.isfile(os.path.join(c, 'data', 'subject3', 'subject3_scaled.osim')):
            return os.path.abspath(c)
    return os.path.abspath(cands[0])


DATA = _find_data()


def data(*p):
    return os.path.join(DATA, *p)


def model(*p):
    return os.path.join(MODELS, *p)


def result(*p):
    return os.path.join(RESULTS, *p)


if __name__ == '__main__':
    print('SRC    ', SRC)
    print('MODELS ', MODELS)
    print('RESULTS', RESULTS)
    print('DATA   ', DATA, '(exists)' if os.path.isfile(data('data', 'subject3', 'subject3_scaled.osim')) else '(data/subject3/subject3_scaled.osim 없음 — AMH_DATA 를 지정하세요)')
