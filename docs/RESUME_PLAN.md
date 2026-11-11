# Master-Sim 재개 계획

> 기준일: 2026-11-11
> 작업선: `main` only

## 1. 이 프로젝트가 현재 하는 일

Master-Sim의 장기 목표는 산업용 정밀 조립을 위한 Sim-to-Real 플랫폼이다.
현재 구현은 그 전체가 아니라, MuJoCo 안에서 Panda 7축 팔의 end-effector가
고정된 목표점에 도달하는 **reach-task용 Behavior Cloning(BC) MVP**다.

현재 파이프라인은 다음과 같다.

```text
PandaEnv + 고정 target_box
  → IK expert trajectory 생성
  → HDF5 state/action 데이터
  → NumPy MLP BC 학습
  → joint delta 추론
  → EE-to-target 거리로 평가
```

`delta` action label의 기존 실험은 50개 에피소드에서 100% 성공률을
기록했다. 이는 고정 reach-task의 결과이며, Peg-in-Hole, 일반화, 실제
로봇 성능을 뜻하지 않는다. 자세한 결과는
[delta_vs_qpos_report.md](experiments/delta_vs_qpos_report.md)를 본다.

## 2. 현재 자산과 한계

### 이미 있는 것

- MuJoCo Panda 팔, 7개 관절 position control, viewer 및 IK controller
- 키보드 기반 관절 teleoperation과 trajectory 저장 도구
- HDF5 전처리·정규화·BC MLP·학습·평가·데모 스크립트
- `qpos` 대신 `delta` action을 쓰는 학습/평가 경로
- 데이터 증강, diverse expert-data 생성, 마찰·질량 randomization 초안
- 기본 단위 테스트와 과거 모델·평가 결과

### 아직 없는 것 또는 신뢰할 수 없는 것

- 실제 Peg, Hole, 그리퍼, 삽입 성공 판정. 현재 기본 장면은 `target_box`로
  향하는 reach-task다.
- 고정 목표 이외 조건에서의 재현 가능한 benchmark 및 일반화 성능
- 완전한 수동 시연 수집(마우스 IK/그리퍼)과 실제 시연 데이터셋
- 웹 대시보드, RL, ONNX/quantization, Isaac Sim, ROS 2, 실제 로봇, API,
  클라우드 배포
- 실행 가능한 고정 개발 환경과 통과가 확인된 전체 테스트 스위트

## 3. 첫 번째 릴리즈 목표

첫 릴리즈 후보는 플랫폼이나 SaaS가 아니라 **`v0.1.0`: 재현 가능한
MuJoCo Peg-in-Hole BC benchmark**로 한정한다.

이 릴리즈는 다음을 만족해야 한다.

- Panda, 그리퍼, Peg, Hole이 있는 하나의 명시적인 MuJoCo 과제
- reset, observation, action, reward, termination, success의 계약 문서화
- seed를 받아 동일한 결과를 재현할 수 있는 데이터 생성·학습·평가 CLI
- holdout seed 100개에서 성공률, 삽입 시간, 충돌/접촉 지표를 JSON으로 저장
- 설치부터 평가까지의 명령이 README에서 동작
- headless 단위/통합 테스트와 최소 CI 품질 게이트

## 4. 우선순위 백로그

### M0 — 재현 가능한 기준선 복구

목표: 지금의 reach-task 기준선을 누구나 다시 실행·검증할 수 있게 한다.

1. Python 3.11 환경을 고정하고 `requirements.txt`의 누락/버전을 정리한다.
   HDF5 사용 패키지와 MuJoCo headless 실행 요구사항을 명시한다.
2. 현재 테스트를 실행해 실패 목록을 만든다. `load_normalizers()` 반환값과
   테스트의 불일치처럼 API와 테스트가 어긋난 항목을 수정한다.
3. `ruff`, `black`, `pytest`를 실행하는 CI를 추가한다.
4. 데이터 생성 → 학습 → 평가의 smoke test를 작은 seed/작은 데이터로 만든다.
5. 모델·체크포인트·대형 실험 파일의 추적 정책을 정하고 `.gitignore`와
   README를 맞춘다.

완료 기준: 새 Python 3.11 환경에서 한 명령으로 테스트와 작은 end-to-end
smoke test를 재현할 수 있다.

### M1 — 과제를 실제 Peg-in-Hole로 정의

목표: 성공률이 실제 조립 의미를 가지도록 환경을 교체한다.

1. 기본 XML에 table, peg, hole plate, gripper와 필요한 actuator를 추가한다.
2. 관측값·액션 형식·접촉/충돌·삽입 깊이·성공 조건을 명세한다.
3. reset에서 object pose와 목표 조건을 seed 기반으로 초기화한다.
4. 성공, timeout, self/collision, peg drop을 포함한 환경 테스트를 작성한다.
5. 기존 `ContactPhysicsEnv`가 기본 장면과 호환되는지 통합하거나 제거한다.

완료 기준: 임의 정책과 scripted/IK expert가 같은 환경 계약 아래에서
실행되며, 삽입 성공 여부를 자동 판정한다.

### M2 — 재현 가능한 BC benchmark

목표: “학습됐다”가 아닌 측정 가능한 기준 모델을 만든다.

1. scripted expert와 수동 시연의 역할을 정하고, 데이터 포맷을 HDF5 하나로
   통일한다.
2. train/validation/test seed를 분리하고, test 환경의 목표 pose를 학습에
   노출하지 않는다.
3. `delta` BC를 baseline으로 재학습한다.
4. 100 holdout episodes에 대해 성공률, 삽입 시간, final error, 최대 접촉력,
   failure reason을 저장한다.
5. 명령, seed, commit SHA, 결과 JSON을 포함한 benchmark 보고서를 작성한다.

완료 기준: baseline 결과가 버전 관리된 설정과 결과 JSON으로 재현된다.

### M3 — 강건성은 측정 후에 추가

목표: 이미 `main`에 들어온 randomization 초안을 검증 가능한 기능으로 만든다.

1. `generate_diverse_data.py`의 사용되지 않는 `--target-randomization`을
   구현하거나 제거한다.
2. randomization 범위를 config 파일로 관리하고, reset마다 원본 파라미터에서
   새 샘플을 뽑는지 테스트한다.
3. 마찰, 질량, object pose, action noise를 한 항목씩 도입한다.
4. clean/shifted 조건의 benchmark를 분리하여 성능 저하를 보고한다.

완료 기준: 범위·seed·평가 조건이 명확하고, 강건성 결과를 baseline과 비교할
수 있다.

## 5. 의도적으로 뒤로 미룰 일

M0~M2의 benchmark가 안정되기 전에는 다음을 시작하지 않는다.

- React/Streamlit dashboard
- PPO 등 RL 도입
- ONNX export 또는 INT8 quantization
- Isaac Sim, ROS 2, 실제 Panda 하드웨어
- FastAPI, DB, 인증, AWS 배포, 결제·마케팅 기능

이들은 기존 장기 계획에는 남기되, 현재의 v0.1.0 완료 기준에는 포함하지
않는다.

## 6. 다음 작업 순서

1. M0-1: Python 3.11 및 의존성 lock/설치 절차 복구
2. M0-2: 전체 테스트 실패를 기록하고 테스트/API 불일치 수정
3. M0-4: 작은 E2E smoke test와 CI 추가
4. M1: Peg-in-Hole 환경 계약과 장면 구현

`main`에서 이 순서대로 작은 검증 단위로 커밋한다. 릴리즈 후보가 생기기
전까지 새 작업 브랜치는 만들지 않는다.
