# Master-Sim: Agile Sprint Plan & Execution Guide

## 목차
1. [Agile 개발 프로세스](#1-agile-개발-프로세스)
2. [Sprint 전체 로드맵](#2-sprint-전체-로드맵)
3. [Sprint 상세 계획](#3-sprint-상세-계획)
4. [Epic & Story 관리](#4-epic--story-관리)
5. [Definition of Done](#5-definition-of-done)
6. [Velocity & Estimation](#6-velocity--estimation)
7. [리스크 관리](#7-리스크-관리)

---

## 1. Agile 개발 프로세스

### 1.1 Workflow Rules (작업 규칙)
*   **Sprint Cycle:** 2주 (Bi-weekly), 매 격주 금요일 종료.
*   **Daily Standup:** 매일 오전 10시 (15분, 가상 환경 고려 시 비동기 Slack 업데이트)
*   **Sprint Planning:** Sprint 시작일 (월요일) 오전, 2시간.
*   **Sprint Review:** Sprint 마지막 날 (금요일) 오후 3시, 1시간.
*   **Sprint Retrospective:** Review 직후, 30분~1시간.

### 1.2 Branching Strategy (Git Flow 확장)

```
main (배포 버전, Protected)
  │
  ├── develop (다음 릴리즈 통합)
  │     │
  │     ├── sprint/S1 (Sprint 1 작업 공간)
  │     │     │
  │     │     ├── feat/ST-1-mujoco-setup
  │     │     ├── feat/ST-2-viewer
  │     │     └── feat/ST-3-panda-load
  │     │
  │     └── sprint/S2 (Sprint 2 작업 공간)
  │           │
  │           ├── feat/ST-6-ik-control
  │           └── feat/ST-7-gripper
  │
  └── hotfix/critical-bug (긴급 수정)
```

**브랜치 네이밍 규칙:**
- `sprint/S{number}`: 예) `sprint/S1`, `sprint/S2`
- `feat/{story-id}-{short-desc}`: 예) `feat/ST-1-mujoco-setup`
- `fix/{issue-id}-{bug-desc}`: 예) `fix/BUG-42-collision-crash`
- `docs/{topic}`: 예) `docs/api-reference`

### 1.3 Commit Convention (Conventional Commits)

**형식:**
```
[ST-{id}] {type}: {subject}

{body (optional)}

{footer (optional)}
```

**Type 종류:**
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포맷팅 (기능 변화 없음)
- `refactor`: 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드, 설정 파일 변경

**예시:**
```
[ST-1] feat: add mujoco environment initialization

- Install mujoco 3.1.2 via pip
- Create basic scene with ground plane
- Verify rendering with mujoco.viewer

Closes #1
```

### 1.4 Pull Request (PR) 프로세스

**PR 생성 시 체크리스트:**
- [ ] 브랜치명이 규칙에 맞는가?
- [ ] 모든 테스트가 통과하는가? (CI 자동 실행)
- [ ] 코드 커버리지가 떨어지지 않았는가?
- [ ] 문서가 업데이트되었는가? (README, Docstring)
- [ ] Self-Review를 완료했는가?

**PR 템플릿:**
```markdown
## Story ID
ST-{번호}

## 변경 사항
- [ ] 주요 변경 1
- [ ] 주요 변경 2

## 테스트 방법
1. 환경 설정: ...
2. 실행 명령: `python ...`
3. 예상 결과: ...

## 스크린샷 (UI 관련 시)
![image](url)

## 체크리스트
- [ ] 테스트 통과
- [ ] 문서 업데이트
- [ ] Breaking Change 없음
```

**Review 규칙:**
- 최소 1명의 Approve 필요 (팀 확장 시 2명)
- 24시간 내 리뷰 완료 (긴급 시 4시간)
- 건설적 피드백 문화: "이렇게 하면 어떨까요?" 형식

### 1.5 UI/UX First 원칙

**모든 UI 작업은 Figma 설계 선행:**
1. **디자인 단계:**
   - Story 시작 전 Figma에서 와이어프레임 작성
   - 팀 리뷰 후 Figma 링크를 Story에 첨부
   
2. **개발 단계:**
   - Figma Inspect 모드로 정확한 색상, 간격 확인
   - Tailwind CSS 클래스로 1:1 구현
   
3. **검증 단계:**
   - Figma vs 실제 화면 Pixel-Perfect 비교
   - Chrome DevTools로 반응형 테스트

**Figma 파일 구조:**
```
Master-Sim Design System
  ├── 1. Foundations (Color, Typography, Spacing)
  ├── 2. Components (Button, Input, Card)
  ├── 3. Pages
  │     ├── Dashboard
  │     ├── Training Monitor
  │     └── Model Library
  └── 4. Prototypes (인터랙션 정의)
```

---

## 2. Sprint 전체 로드맵

### 2.1 Overview (6개월 계획)

| Sprint | 기간 | Epic | 주요 목표 | 산출물 |
|:---:|:---|:---|:---|:---|
| **S1** | W1~W2 | Simulation Foundation | 환경 구축 | 로봇이 움직이는 화면 |
| **S2** | W3~W4 | Data Collection | 데이터 수집 파이프라인 | 100회 시연 데이터셋 |
| **S3** | W5~W6 | AI Training MVP | 첫 번째 학습 모델 | 50% 성공률 AI |
| **S4** | W7~W8 | Dashboard UI | 모니터링 대시보드 | 웹 UI (React) |
| **S5** | W9~W10 | Model Optimization | 성공률 향상 | 80% 성공률 달성 |
| **S6** | W11~W12 | Sim-to-Real Prep | Isaac Sim 마이그레이션 | 고품질 렌더링 |
| **S7** | W13~W14 | Domain Randomization | 강건성 확보 | 다양한 환경 테스트 |
| **S8** | W15~W16 | Real Robot Test | 실제 로봇 검증 | PoC 완료 |
| **S9** | W17~W18 | API Development | REST API 구축 | Swagger 문서 |
| **S10** | W19~W20 | Cloud Deployment | AWS 배포 | Staging 환경 |
| **S11** | W21~W22 | Beta Testing | 첫 고객 테스트 | Feedback Loop |
| **S12** | W23~W24 | Production Launch | 정식 출시 | v1.0.0 GA |

### 2.2 Velocity 예상 (Story Point 기반)

**가정:**
- 개발자 1명 (Founder)
- Sprint당 작업 가능 시간: 60시간 (주당 30시간)
- 1 Story Point = 약 3시간

**Sprint별 Capacity:**
- Sprint 1~2: 15 Points/Sprint (학습 곡선)
- Sprint 3~6: 20 Points/Sprint (속도 증가)
- Sprint 7+: 25 Points/Sprint (숙련도 최고치)

---

## 3. Sprint 상세 계획

### **Sprint 1: Simulation Foundation (시뮬레이션 기초 구축)**
*   **기간:** 2026.01.04 ~ 2026.01.17
*   **목표:** MuJoCo 시뮬레이션 환경을 구축하고, 로봇 팔(Panda)을 파이썬 코드로 제어할 수 있는 상태를 만든다.
*   **Epic:** Simulation Environment Setup

| Story ID | Type | Point | Title | User Story (As I want to..., So that...) | Acceptance Criteria (DoD) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ST-1** | Chore | 2 | **프로젝트 환경 설정 및 의존성 관리** | **As a** 개발자<br>**I want to** Python 가상환경과 필수 라이브러리(MuJoCo, NumPy 등)를 설치하여<br>**So that** 개발을 시작할 수 있는 기반을 마련한다. | 1. `requirements.txt`에 mujoco, gymnasium 등 필수 패키지 명시<br>2. 가상환경에서 설치 및 import 테스트 통과<br>3. Hello World 스크립트 실행 시 에러 없음 |
| **ST-2** | Feat | 3 | **MuJoCo 기본 뷰어 띄우기** | **As a** 개발자<br>**I want to** 빈 시뮬레이션 공간을 윈도우 창으로 띄워서<br>**So that** 시각적으로 환경을 확인할 수 있다. | 1. Python 스크립트 실행 시 MuJoCo 뷰어 창이 뜸<br>2. 바닥(Plane)과 조명이 렌더링됨<br>3. 마우스로 시점 조작(Zoom/Rotate) 가능 |
| **ST-3** | Feat | 5 | **Franka Emika Panda 로봇 로드** | **As a** 개발자<br>**I want to** 시뮬레이션 공간에 Panda 로봇 모델(XML)을 불러와서<br>**So that** 제어할 로봇 객체를 확보한다. | 1. 로봇 팔이 공중에 뜨지 않고 바닥에 고정됨<br>2. 로봇의 모든 관절(7축)과 그리퍼가 정상적으로 렌더링됨<br>3. 물리 충돌(Collision) 모델이 적용됨 |
| **ST-4** | Feat | 5 | **로봇 관절 제어 (Joint Control) 구현** | **As a** 개발자<br>**I want to** 코드를 통해 로봇의 특정 관절을 움직여서<br>**So that** 로봇을 원하는 자세로 만들 수 있다. | 1. Python 함수로 각도 입력 시 로봇이 해당 각도로 움직임<br>2. 움직임이 물리적으로 자연스러움 (튀거나 폭발하지 않음) |
| **ST-5** | Feat | 3 | **작업 테이블 및 Peg/Hole 객체 배치** | **As a** 개발자<br>**I want to** 로봇 앞에 테이블과 조립할 부품(Peg, Hole)을 배치하여<br>**So that** 작업 환경을 구성한다. | 1. 로봇 도달 범위 내에 테이블 배치<br>2. 테이블 위에 구멍이 뚫린 판(Hole)과 원통형 부품(Peg) 배치<br>3. 객체 간 물리 충돌이 정상 작동함 |

---

### **Sprint 2: Data Collection Pipeline (데이터 수집 파이프라인)**
*   **기간:** 2026.01.18 ~ 2026.01.31
*   **목표:** 사용자가 마우스/키보드로 로봇을 조작하여 Peg-in-Hole 작업을 수행하고, 그 데이터를 저장한다.
*   **Epic:** Data Acquisition

| Story ID | Type | Point | Title | User Story | Acceptance Criteria (DoD) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ST-6** | Feat | 8 | **마우스 기반 IK(Inverse Kinematics) 제어** | **As a** 데이터 수집가<br>**I want to** 마우스 커서 위치로 로봇 손끝(End-effector)을 이동시켜서<br>**So that** 직관적으로 로봇을 조종한다. | 1. 마우스 움직임에 따라 로봇 손끝이 실시간 추종<br>2. IK 연산 속도가 60fps 이상 유지<br>3. 관절 한계(Joint Limit)를 벗어나지 않음 |
| **ST-7** | Feat | 3 | **그리퍼 개폐(Open/Close) 제어** | **As a** 데이터 수집가<br>**I want to** 키보드 키(예: Spacebar)로 그리퍼를 여닫아서<br>**So that** 물체를 집거나 놓을 수 있다. | 1. 키 입력 시 그리퍼가 부드럽게 열리고 닫힘<br>2. 물체를 잡았을 때 미끄러지지 않고 고정됨 |
| **ST-8** | Feat | 5 | **데이터 로거(Data Logger) 개발** | **As a** AI 엔지니어<br>**I want to** 매 프레임마다 로봇의 상태(관절각, 속도)와 조작 입력(Action)을 파일로 저장하여<br>**So that** 학습 데이터셋을 만든다. | 1. 녹화 시작/종료 기능<br>2. `.h5` 또는 `.npz` 포맷으로 저장<br>3. 재생(Replay) 시 녹화된 동작이 똑같이 재현됨 |

---

### **Sprint 3: AI Training MVP (AI 학습 및 검증)**
*   **기간:** 2026.02.01 ~ 2026.02.14
*   **목표:** 수집된 데이터를 바탕으로 Behavior Cloning 모델을 학습시키고, 시뮬레이션에서 자율 동작을 검증한다.
*   **Epic:** AI Model Training

| Story ID | Type | Point | Title | User Story | Acceptance Criteria (DoD) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ST-9** | Feat | 5 | **학습 데이터 전처리 파이프라인** | **As a** AI 엔지니어<br>**I want to** 수집된 원본 데이터를 학습 가능한 텐서(Tensor) 형태로 변환하고 정규화하여<br>**So that** 학습 효율을 높인다. | 1. 데이터 로딩 속도 최적화<br>2. 노이즈 제거 및 정규화(Normalization) 적용 |
| **ST-10** | Feat | 8 | **Behavior Cloning (BC) 모델 구현** | **As a** AI 엔지니어<br>**I want to** PyTorch로 간단한 MLP/CNN 기반 정책 네트워크를 구성하여<br>**So that** 로봇의 행동을 모방 학습시킨다. | 1. 입력(상태) -> 출력(행동) 네트워크 정의<br>2. 학습 루프(Train Loop) 구현 및 Loss 감소 확인 |
| **ST-11** | Feat | 5 | **모델 추론 및 시뮬레이션 적용** | **As a** 사용자<br>**I want to** 학습된 모델을 시뮬레이터에 연결하여<br>**So that** 로봇이 스스로 Peg-in-Hole을 수행하는지 본다. | 1. 학습된 가중치 로드<br>2. 시뮬레이션 상에서 로봇이 물체를 집어 구멍에 넣는 성공률 50% 이상 달성 |

---

### **Sprint 4: Dashboard & Visualization (대시보드 및 시각화)**
*   **기간:** 2026.02.15 ~ 2026.02.28
*   **목표:** 웹 기반 대시보드를 통해 학습 진행 상황과 시뮬레이션 상태를 모니터링한다.
*   **Epic:** Dashboard UI

| Story ID | Type | Point | Title | User Story | Acceptance Criteria (DoD) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ST-12** | Design | 3 | **대시보드 UI Figma 설계** | **As a** 디자이너<br>**I want to** 대시보드 레이아웃과 차트 디자인을 Figma로 그려서<br>**So that** 개발 가이드를 제공한다. | 1. 메인 대시보드, 상세 로그 뷰 디자인 완료<br>2. 반응형 레이아웃 고려 |
| **ST-13** | Feat | 8 | **실시간 학습 현황 차트 구현** | **As a** 사용자<br>**I want to** 웹 브라우저에서 Loss, Reward 그래프를 실시간으로 확인하여<br>**So that** 학습이 잘 되고 있는지 판단한다. | 1. Streamlit 또는 React 기반 프론트엔드<br>2. 백엔드에서 전송하는 로그 데이터를 차트로 시각화<br>3. 실시간 업데이트(WebSocket 또는 Polling)<br>4. 여러 실험을 비교할 수 있는 오버레이 기능 |

**Sprint 4 기술 스택:**
- Frontend: React 18, Recharts, TailwindCSS
- Backend: FastAPI, WebSocket
- State Management: Zustand
- Build Tool: Vite

**Sprint 4 완료 조건:**
- [ ] 웹 대시보드가 localhost:3000에서 정상 실행
- [ ] 학습 중인 모델의 Loss가 실시간 그래프로 표시됨
- [ ] 과거 실험 데이터를 불러와서 비교 가능
- [ ] 모바일 반응형 레이아웃 적용

---

### **Sprint 5: Model Optimization (모델 최적화 및 성능 향상)**
*   **기간:** 2026.03.01 ~ 2026.03.14
*   **목표:** Behavior Cloning 모델의 성공률을 50%에서 80% 이상으로 향상시키고, 추론 속도를 최적화한다.
*   **Epic:** AI Model Optimization
*   **Sprint Goal:** "로봇이 10번 중 8번 이상 성공적으로 Peg-in-Hole을 수행한다."

| Story ID | Type | Point | Title | User Story | Acceptance Criteria (DoD) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ST-14** | Feat | 8 | **데이터 증강(Data Augmentation) 파이프라인** | **As a** AI 엔지니어<br>**I want to** 수집된 데이터에 노이즈, 회전, 스케일 변환을 적용하여<br>**So that** 모델의 일반화 성능을 높인다. | 1. Gaussian Noise, Random Rotation 적용<br>2. 증강 전/후 데이터셋 크기 비교 (최소 3배 증가)<br>3. 증강 데이터로 재학습 시 검증 정확도 10% 이상 향상<br>4. 증강 파라미터를 config 파일로 관리 |
| **ST-15** | Feat | 5 | **하이퍼파라미터 튜닝 (Hyperparameter Tuning)** | **As a** AI 엔지니어<br>**I want to** Learning Rate, Batch Size, Network Depth를 자동으로 최적화하여<br>**So that** 최적의 학습 설정을 찾는다. | 1. Optuna 또는 Ray Tune 적용<br>2. 최소 50회 이상 실험 수행<br>3. 최적 하이퍼파라미터 조합을 문서화<br>4. 베이스라인 대비 성공률 15% 이상 향상 |
| **ST-16** | Feat | 8 | **Reinforcement Learning (RL) 도입** | **As a** AI 엔지니어<br>**I want to** PPO(Proximal Policy Optimization) 알고리즘을 추가하여<br>**So that** Behavior Cloning의 한계를 극복하고 자율 개선 능력을 확보한다. | 1. Stable-Baselines3 또는 CleanRL 사용<br>2. Reward Function 정의 (성공: +10, 충돌: -5, 시간 페널티: -0.01)<br>3. 100만 스텝 학습 후 성공률 70% 이상 달성<br>4. RL 학습 곡선을 Tensorboard로 시각화 |
| **ST-17** | Feat | 5 | **모델 경량화 (Model Quantization)** | **As a** 배포 엔지니어<br>**I want to** 학습된 모델을 FP32에서 INT8로 변환하여<br>**So that** 추론 속도를 4배 이상 향상시킨다. | 1. PyTorch Quantization API 적용<br>2. 양자화 전/후 정확도 차이 5% 이내 유지<br>3. 추론 속도 벤치마크: 1ms 이하 (CPU 기준)<br>4. ONNX 포맷으로 내보내기 성공 |
| **ST-18** | Test | 3 | **성능 벤치마크 및 A/B 테스트** | **As a** 프로덕트 매니저<br>**I want to** BC 모델과 RL 모델의 성공률을 100회씩 테스트하여<br>**So that** 어떤 모델을 배포할지 결정한다. | 1. 각 모델당 100 에피소드 실행<br>2. 성공률, 평균 완료 시간, 충돌 횟수 기록<br>3. 통계적 유의성 검증 (p-value < 0.05)<br>4. 결과를 Markdown 리포트로 작성 |

**Sprint 5 기술 스택:**
- ML Framework: PyTorch 2.2, Stable-Baselines3 2.2
- Hyperparameter Tuning: Optuna 3.5
- Quantization: torch.quantization, ONNX Runtime
- Experiment Tracking: Weights & Biases (W&B)

**Sprint 5 완료 조건:**
- [ ] 최종 모델의 Peg-in-Hole 성공률 80% 이상
- [ ] 추론 속도 1ms 이하 (CPU), 0.1ms 이하 (GPU)
- [ ] 모든 실험 결과가 W&B에 기록됨
- [ ] A/B 테스트 리포트 작성 완료

---

### **Sprint 6: Sim-to-Real Preparation (실제 환경 전환 준비)**
*   **기간:** 2026.03.15 ~ 2026.03.28
*   **목표:** MuJoCo에서 NVIDIA Isaac Sim으로 마이그레이션하여 포토리얼리스틱 렌더링과 고급 물리 시뮬레이션을 적용한다.
*   **Epic:** Sim-to-Real Bridge
*   **Sprint Goal:** "Isaac Sim에서 동일한 작업이 실행되고, 렌더링 품질이 실사 수준이다."

| Story ID | Type | Point | Title | User Story | Acceptance Criteria (DoD) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ST-19** | Chore | 5 | **NVIDIA Isaac Sim 환경 설정** | **As a** 개발자<br>**I want to** Isaac Sim 2024.1을 설치하고 기본 예제를 실행하여<br>**So that** 고급 시뮬레이션 환경을 준비한다. | 1. Omniverse Launcher를 통해 Isaac Sim 설치<br>2. GPU 드라이버 및 CUDA 12.1 호환성 확인<br>3. 공식 예제(Franka Cube Stacking) 실행 성공<br>4. Python API 테스트 스크립트 작성 |
| **ST-20** | Feat | 8 | **MuJoCo → Isaac Sim 장면 변환** | **As a** 개발자<br>**I want to** MuJoCo XML 장면을 Isaac Sim USD 포맷으로 변환하여<br>**So that** 동일한 작업 환경을 Isaac Sim에서 재현한다. | 1. Panda 로봇, 테이블, Peg/Hole을 Isaac Sim에 배치<br>2. 물리 파라미터(마찰, 강성) 동기화<br>3. 카메라 위치 및 조명 설정<br>4. MuJoCo와 Isaac Sim에서 동일한 초기 상태 검증 |
| **ST-21** | Feat | 8 | **Ray Tracing 기반 포토리얼리스틱 렌더링** | **As a** 마케팅 담당자<br>**I want to** 실사와 구분이 안 되는 고품질 영상을 생성하여<br>**So that** 투자자 피칭과 마케팅 자료로 사용한다. | 1. RTX Ray Tracing 활성화<br>2. PBR(Physically Based Rendering) 재질 적용<br>3. HDR 환경 조명 설정<br>4. 1920x1080 해상도, 60fps 렌더링 성공<br>5. 10초 데모 영상 제작 |
| **ST-22** | Feat | 5 | **학습된 정책을 Isaac Sim에 적용** | **As a** AI 엔지니어<br>**I want to** MuJoCo에서 학습한 모델을 Isaac Sim에서 실행하여<br>**So that** Sim-to-Sim 전환 시 성능 저하를 측정한다. | 1. ONNX 모델을 Isaac Sim Python 스크립트에서 로드<br>2. 관측 공간(Observation Space) 동기화<br>3. 50회 에피소드 실행 후 성공률 비교<br>4. 성능 저하가 10% 이내이면 합격 |
| **ST-23** | Feat | 3 | **센서 시뮬레이션 (카메라, Force Sensor)** | **As a** 로보틱스 엔지니어<br>**I want to** 로봇에 RGB 카메라와 힘/토크 센서를 부착하여<br>**So that** 실제 로봇과 동일한 센서 데이터를 얻는다. | 1. Wrist-mounted RGB 카메라 추가<br>2. 그리퍼에 6축 Force/Torque 센서 부착<br>3. 센서 데이터를 Numpy 배열로 읽기 성공<br>4. 센서 노이즈 모델 적용 (Gaussian Noise) |

**Sprint 6 기술 스택:**
- Simulation: NVIDIA Isaac Sim 2024.1, Omniverse USD
- Rendering: RTX Ray Tracing, MDL Materials
- Asset Pipeline: Blender 4.0 (3D 모델링)
- Data Format: USD, ONNX

**Sprint 6 완료 조건:**
- [ ] Isaac Sim에서 Peg-in-Hole 작업이 정상 실행됨
- [ ] 포토리얼리스틱 데모 영상 10초 이상 제작
- [ ] MuJoCo 대비 성능 저하 10% 이내
- [ ] 센서 데이터 수집 파이프라인 구축

---

### **Sprint 7: Domain Randomization (도메인 랜덤화 및 강건성 확보)**
*   **기간:** 2026.03.29 ~ 2026.04.11
*   **목표:** 다양한 환경 변화(조명, 물체 위치, 물리 파라미터)에도 로봇이 작동하도록 강건성을 확보한다.
*   **Epic:** Robustness Engineering
*   **Sprint Goal:** "조명이 바뀌거나 물체 위치가 다르더라도 성공률 70% 이상 유지."

| Story ID | Type | Point | Title | User Story | Acceptance Criteria (DoD) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ST-24** | Feat | 8 | **비주얼 도메인 랜덤화 (Visual DR)** | **As a** AI 엔지니어<br>**I want to** 매 에피소드마다 조명, 배경, 텍스처를 무작위로 변경하여<br>**So that** 다양한 시각적 환경에서도 작동하는 모델을 학습한다. | 1. 조명 색온도 2700K~6500K 랜덤 변경<br>2. 배경 이미지 20종 이상 순환<br>3. Peg/Hole 텍스처 10종 랜덤 적용<br>4. 카메라 노이즈 추가 (Gaussian, Salt&Pepper)<br>5. 랜덤화된 환경에서 재학습 후 성공률 75% 이상 |
| **ST-25** | Feat | 8 | **물리 도메인 랜덤화 (Dynamics DR)** | **As a** 로보틱스 엔지니어<br>**I want to** 마찰 계수, 질량, 강성을 무작위로 변경하여<br>**So that** 실제 로봇의 불확실성을 시뮬레이션에 반영한다. | 1. 마찰 계수 0.3~0.9 균등 분포<br>2. Peg 질량 50g~150g 랜덤<br>3. Contact 강성 1e3~1e6 로그 스케일<br>4. 1000 에피소드 학습 후 성능 저하 15% 이내<br>5. 파라미터 분포를 JSON Config로 관리 |
| **ST-26** | Feat | 5 | **액션 노이즈 주입 (Action Noise Injection)** | **As a** AI 엔지니어<br>**I want to** 로봇의 명령에 의도적인 오차를 추가하여<br>**So that** 실제 로봇의 제어 불확실성을 학습에 포함시킨다. | 1. 관절 속도 명령에 ±10% Gaussian Noise<br>2. 그리퍼 힘 명령에 ±5% Uniform Noise<br>3. 노이즈가 추가된 환경에서 100 에피소드 테스트<br>4. 성공률이 노이즈 없을 때 대비 80% 이상 유지 |
| **ST-27** | Feat | 5 | **자동 커리큘럼 학습 (Automatic Curriculum)** | **As a** AI 엔지니어<br>**I want to** 쉬운 작업부터 시작해서 점진적으로 난이도를 높여<br>**So that** 학습 효율을 극대화한다. | 1. 난이도 레벨 5단계 정의 (구멍 크기 20mm → 10mm)<br>2. 성공률 90% 달성 시 다음 단계로 자동 전환<br>3. 레벨별 학습 시간 및 성공률 로깅<br>4. 최종 난이도(10mm)에서 성공률 70% 이상 |
| **ST-28** | Test | 3 | **스트레스 테스트 (Stress Test)** | **As a** QA 엔지니어<br>**I want to** 극단적인 조건(완전 어둠, 극심한 마찰)에서 테스트하여<br>**So that** 모델의 한계를 파악한다. | 1. 조명 0 lux (완전 어둠) 테스트<br>2. 마찰 계수 1.5 (극도로 높음) 테스트<br>3. 질량 300g (정상의 2배) 테스트<br>4. 각 조건에서 성공률 기록 및 한계 문서화 |

**Sprint 7 기술 스택:**
- Domain Randomization: Isaac Sim Randomization API
- Curriculum Learning: Custom Python Logic
- Experiment Tracking: W&B Sweeps
- Testing Framework: Pytest, Hypothesis (Property-based Testing)

**Sprint 7 완료 조건:**
- [ ] Visual DR + Dynamics DR 적용된 모델 학습 완료
- [ ] 랜덤화된 환경에서 성공률 70% 이상
- [ ] 스트레스 테스트 결과 문서화
- [ ] 커리큘럼 학습 로그가 W&B에 기록됨

---

### **Sprint 8: Real Robot Validation (실제 로봇 검증 및 PoC 완료)**
*   **기간:** 2026.04.12 ~ 2026.04.25
*   **목표:** 학습된 정책을 실제 Franka Panda 로봇에 배포하고, Sim-to-Real 전환 성공 여부를 검증한다.
*   **Epic:** Real-World Deployment
*   **Sprint Goal:** "실제 로봇이 시뮬레이션과 동일한 작업을 수행하고, 성공률 60% 이상 달성."

| Story ID | Type | Point | Title | User Story | Acceptance Criteria (DoD) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ST-29** | Chore | 5 | **실제 로봇 하드웨어 설정** | **As a** 로보틱스 엔지니어<br>**I want to** Franka Panda 로봇을 네트워크에 연결하고 제어 소프트웨어를 설치하여<br>**So that** 원격으로 로봇을 조종할 수 있다. | 1. 로봇을 이더넷으로 제어 PC에 연결<br>2. Franka Desk 설정 및 안전 설정 완료<br>3. libfranka 및 ROS 2 Humble 설치<br>4. 간단한 Homing 명령 실행 성공 |
| **ST-30** | Feat | 8 | **ROS 2 브릿지 개발 (Sim ↔ Real)** | **As a** 개발자<br>**I want to** Isaac Sim과 실제 로봇 간 통신 인터페이스를 구축하여<br>**So that** 동일한 코드로 시뮬레이션과 실제 로봇을 제어한다. | 1. ROS 2 Topic으로 Joint State 송수신<br>2. Action Server로 Trajectory 실행<br>3. 시뮬레이션 모드와 실제 로봇 모드 전환 가능<br>4. Latency 10ms 이하 유지 |
| **ST-31** | Feat | 8 | **실제 로봇에서 정책 실행** | **As a** AI 엔지니어<br>**I want to** 학습된 ONNX 모델을 실제 로봇 제어 루프에 연결하여<br>**So that** Sim-to-Real 전환을 완료한다. | 1. 모델 추론 주기 100Hz 이상<br>2. 안전 모니터링 (충돌 감지, 비상 정지)<br>3. 10회 실행 중 최소 6회 성공<br>4. 실패 케이스 영상 기록 및 분석 |
| **ST-32** | Test | 5 | **Sim-to-Real 성능 비교 분석** | **As a** 연구원<br>**I want to** 시뮬레이션과 실제 로봇의 성공률, 완료 시간, 힘 프로파일을 비교하여<br>**So that** Reality Gap을 정량화한다. | 1. 시뮬레이션 vs 실제 각 50회 실험<br>2. 성공률, 평균 시간, 최대 힘 측정<br>3. 통계 분석 및 Reality Gap 보고서 작성<br>4. 개선 방향 제시 (추가 DR 필요 영역 등) |
| **ST-33** | Docs | 3 | **PoC 데모 영상 제작 및 문서화** | **As a** 프로덕트 매니저<br>**I want to** 실제 로봇이 작업하는 영상과 기술 백서를 제작하여<br>**So that** 투자자와 잠재 고객에게 제시한다. | 1. 3분 데모 영상 (시뮬레이션 → 실제 로봇 비교)<br>2. 기술 백서 (Sim-to-Real 방법론, 성능 지표)<br>3. GitHub README에 영상 및 문서 링크 추가<br>4. LinkedIn, Twitter에 공유 |

**Sprint 8 기술 스택:**
- Robot Control: libfranka 0.10, ROS 2 Humble
- Communication: ROS 2 DDS, WebSocket
- Safety: Franka Desk Safety Config, Joint Limit Monitor
- Documentation: OBS Studio (영상 편집), LaTeX (백서 작성)

**Sprint 8 완료 조건:**
- [ ] 실제 로봇에서 Peg-in-Hole 성공률 60% 이상
- [ ] Sim-to-Real 비교 보고서 작성 완료
- [ ] 3분 데모 영상 제작 및 공개
- [ ] 기술 백서 초안 완성

---

### **Sprint 9: API Development (REST API 및 백엔드 개발)**
*   **기간:** 2026.04.26 ~ 2026.05.09
*   **목표:** 학습된 모델을 RESTful API로 제공하고, 외부 시스템과 통합할 수 있는 백엔드를 구축한다.
*   **Epic:** Backend Infrastructure
*   **Sprint Goal:** "API 엔드포인트를 통해 외부에서 로봇 정책을 요청하고 결과를 받을 수 있다."

| Story ID | Type | Point | Title | User Story | Acceptance Criteria (DoD) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ST-34** | Feat | 8 | **FastAPI 기반 REST API 서버 구축** | **As a** 백엔드 개발자<br>**I want to** FastAPI 프레임워크로 API 서버를 구성하여<br>**So that** HTTP 요청으로 로봇 정책을 실행할 수 있다. | 1. `/health`, `/api/v1/predict`, `/api/v1/models` 엔드포인트 구현<br>2. Pydantic으로 요청/응답 스키마 검증<br>3. 초당 100 요청 처리 가능 (Uvicorn workers=4)<br>4. OpenAPI(Swagger) 문서 자동 생성<br>5. CORS 설정으로 웹 클라이언트 허용 |
| **ST-35** | Feat | 5 | **모델 버전 관리 시스템** | **As a** AI 엔지니어<br>**I want to** 여러 모델 버전을 등록하고 전환할 수 있어서<br>**So that** A/B 테스트 및 롤백이 가능하다. | 1. PostgreSQL DB에 모델 메타데이터 저장<br>2. 모델 파일을 S3에 업로드 및 버전 관리<br>3. API로 모델 리스트 조회 및 활성화<br>4. Blue-Green 배포 지원 (트래픽 전환)<br>5. 모델 성능 지표 기록 (성공률, 평균 시간) |
| **ST-36** | Feat | 5 | **비동기 작업 큐 (Celery + Redis)** | **As a** 백엔드 개발자<br>**I want to** 시간이 오래 걸리는 작업(모델 학습, 대규모 시뮬레이션)을 백그라운드에서 처리하여<br>**So that** API 응답 시간을 빠르게 유지한다. | 1. Celery Worker 설정 (Redis as Broker)<br>2. 학습 작업을 Task로 등록<br>3. Task 상태 조회 API (`/tasks/{task_id}`)<br>4. Task 취소 및 재시도 기능<br>5. Flower 대시보드로 모니터링 |
| **ST-37** | Feat | 5 | **인증 및 권한 관리 (JWT)** | **As a** 보안 엔지니어<br>**I want to** JWT 토큰 기반 인증을 구현하여<br>**So that** 인가된 사용자만 API를 사용한다. | 1. 회원가입, 로그인 엔드포인트 (`/auth/signup`, `/auth/login`)<br>2. JWT Access Token 발급 (유효기간 1시간)<br>3. Refresh Token으로 자동 갱신<br>4. Role 기반 권한 (Admin, User, Viewer)<br>5. Rate Limiting (분당 60 요청) |
| **ST-38** | Feat | 3 | **API 사용량 추적 및 로깅** | **As a** 프로덕트 매니저<br>**I want to** 각 사용자의 API 호출 횟수와 응답 시간을 기록하여<br>**So that** 사용 패턴을 분석하고 과금 기준을 만든다. | 1. Middleware로 모든 요청 로깅<br>2. ElasticSearch에 로그 저장<br>3. Kibana 대시보드로 시각화<br>4. 일일/월별 사용량 통계 API<br>5. Alert 설정 (비정상 트래픽 감지) |
| **ST-39** | Test | 3 | **API 통합 테스트 및 문서화** | **As a** QA 엔지니어<br>**I want to** 모든 엔드포인트를 자동 테스트하고 예제를 제공하여<br>**So that** 외부 개발자가 쉽게 API를 사용한다. | 1. Pytest로 E2E 테스트 작성 (커버리지 90%)<br>2. Postman Collection 제공<br>3. API 문서에 curl 예제 추가<br>4. 에러 코드 및 응답 포맷 명세<br>5. SDK 샘플 코드 (Python, JavaScript) |

**Sprint 9 기술 스택:**
- Backend: FastAPI 0.110, Uvicorn, Pydantic v2
- Database: PostgreSQL 16, SQLAlchemy 2.0
- Task Queue: Celery 5.3, Redis 7.2
- Auth: PyJWT, bcrypt
- Monitoring: ELK Stack (ElasticSearch, Logstash, Kibana)
- Storage: AWS S3

**Sprint 9 완료 조건:**
- [ ] `/api/v1/predict` 엔드포인트가 100 RPS 처리 가능
- [ ] Swagger UI에서 모든 API 테스트 가능
- [ ] JWT 인증이 적용되고 Rate Limiting 동작
- [ ] Celery Task가 정상 실행되며 Flower로 모니터링 가능

---

### **Sprint 10: Cloud Deployment (AWS 인프라 구축 및 배포)**
*   **기간:** 2026.05.10 ~ 2026.05.23
*   **목표:** AWS 클라우드에 전체 시스템을 배포하고, Auto Scaling 및 Load Balancing을 구성한다.
*   **Epic:** Cloud Infrastructure
*   **Sprint Goal:** "AWS에서 24/7 서비스가 가능하며, 트래픽 증가 시 자동 확장된다."

| Story ID | Type | Point | Title | User Story | Acceptance Criteria (DoD) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ST-40** | Chore | 5 | **AWS 계정 설정 및 IAM 권한 관리** | **As a** DevOps 엔지니어<br>**I want to** AWS 계정을 생성하고 최소 권한 원칙을 적용하여<br>**So that** 보안을 강화한다. | 1. AWS Organizations로 계정 구조화<br>2. IAM Role 생성 (Developer, CI/CD, Production)<br>3. MFA 활성화 및 비밀번호 정책 설정<br>4. CloudTrail로 모든 API 호출 감사<br>5. Cost Explorer로 예산 알림 설정 ($500/month) |
| **ST-41** | Feat | 8 | **Docker 컨테이너화 및 ECR 배포** | **As a** DevOps 엔지니어<br>**I want to** 전체 애플리케이션을 Docker 이미지로 빌드하여<br>**So that** 환경 독립적으로 배포한다. | 1. Dockerfile 작성 (Multi-stage build)<br>2. Docker Compose로 로컬 개발 환경 구성<br>3. AWS ECR에 이미지 Push<br>4. 이미지 크기 최적화 (500MB 이하)<br>5. 보안 스캔 통과 (Trivy, Snyk) |
| **ST-42** | Feat | 8 | **ECS Fargate로 컨테이너 오케스트레이션** | **As a** DevOps 엔지니어<br>**I want to** ECS Fargate로 컨테이너를 관리하여<br>**So that** 서버리스 환경에서 확장성을 확보한다. | 1. ECS Cluster 생성 (Fargate 타입)<br>2. Task Definition 정의 (CPU: 2vCPU, RAM: 4GB)<br>3. Service 설정 (Desired Count: 2, Auto Scaling)<br>4. ALB(Application Load Balancer) 연결<br>5. Health Check 설정 및 롤링 업데이트 |
| **ST-43** | Feat | 5 | **RDS PostgreSQL 데이터베이스 구축** | **As a** 데이터베이스 관리자<br>**I want to** RDS로 관리형 PostgreSQL을 배포하여<br>**So that** 백업 및 고가용성을 자동화한다. | 1. RDS PostgreSQL 16 인스턴스 생성 (db.t3.medium)<br>2. Multi-AZ 배포로 고가용성 확보<br>3. 일일 자동 백업 설정 (7일 보관)<br>4. VPC Private Subnet에 배치<br>5. IAM Database Authentication 적용 |
| **ST-44** | Feat | 5 | **CloudFront CDN 및 S3 정적 파일 호스팅** | **As a** 프론트엔드 개발자<br>**I want to** React 대시보드를 S3+CloudFront로 배포하여<br>**So that** 전 세계 어디서나 빠르게 로드된다. | 1. S3 버킷에 빌드 파일 업로드<br>2. CloudFront Distribution 생성<br>3. HTTPS 인증서 적용 (ACM)<br>4. 캐시 정책 설정 (TTL: 1시간)<br>5. 커스텀 도메인 연결 (master-sim.ai) |
| **ST-45** | Feat | 5 | **CloudWatch 모니터링 및 알림** | **As a** SRE<br>**I want to** CloudWatch로 시스템 메트릭을 수집하고 이상 감지 시 알림을 받아서<br>**So that** 장애에 신속히 대응한다. | 1. CloudWatch Logs로 애플리케이션 로그 수집<br>2. CloudWatch Metrics로 CPU, Memory 모니터링<br>3. CloudWatch Alarms 설정 (CPU > 80%, Error Rate > 5%)<br>4. SNS로 Slack 알림 전송<br>5. X-Ray로 분산 추적 (Distributed Tracing) |
| **ST-46** | Test | 3 | **스테이징 환경 구축 및 배포 검증** | **As a** QA 엔지니어<br>**I want to** 프로덕션과 동일한 스테이징 환경을 만들어<br>**So that** 배포 전 안전하게 테스트한다. | 1. Staging 전용 AWS 계정 또는 VPC 분리<br>2. 동일한 Terraform 코드로 인프라 구성<br>3. CI/CD 파이프라인으로 자동 배포<br>4. Smoke Test 자동 실행<br>5. 배포 승인 프로세스 (Manual Approval) |

**Sprint 10 기술 스택:**
- Container: Docker 24, AWS ECS Fargate
- Database: AWS RDS PostgreSQL 16
- CDN: AWS CloudFront, S3
- Monitoring: CloudWatch, X-Ray, Datadog
- IaC: Terraform 1.7, AWS CDK (선택)
- CI/CD: GitHub Actions, AWS CodePipeline

**Sprint 10 완료 조건:**
- [ ] 프로덕션 환경이 AWS에 배포되고 접근 가능
- [ ] Auto Scaling이 동작하여 트래픽 증가 시 확장됨
- [ ] CloudWatch 대시보드에서 모든 메트릭 확인 가능
- [ ] Staging 환경에서 E2E 테스트 통과

---

### **Sprint 11: Beta Testing (베타 테스트 및 피드백 수집)**
*   **기간:** 2026.05.24 ~ 2026.06.06
*   **목표:** 초기 고객(Early Adopters)을 모집하여 실제 사용 환경에서 피드백을 수집하고 개선한다.
*   **Epic:** Customer Validation
*   **Sprint Goal:** "최소 5개 기업이 Master-Sim을 사용하고, NPS 50 이상을 달성한다."

| Story ID | Type | Point | Title | User Story | Acceptance Criteria (DoD) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ST-47** | Feat | 5 | **베타 사용자 초대 시스템** | **As a** 프로덕트 매니저<br>**I want to** 초대 코드 기반 가입 시스템을 구축하여<br>**So that** 베타 사용자를 관리한다. | 1. 초대 코드 생성 API (`/admin/invite-codes`)<br>2. 가입 시 초대 코드 검증<br>3. 사용자당 사용량 할당 (무료 1000 API 호출)<br>4. Admin 대시보드에서 사용자 목록 조회<br>5. 이메일로 초대장 발송 (SendGrid) |
| **ST-48** | Feat | 8 | **사용자 온보딩 튜토리얼** | **As a** 신규 사용자<br>**I want to** 처음 사용 시 가이드를 따라 하면서<br>**So that** 제품을 빠르게 이해한다. | 1. 인터랙티브 튜토리얼 (React Joyride)<br>2. 5단계 온보딩 플로우 (가입 → API Key 발급 → 첫 요청 → 결과 확인 → 대시보드 탐색)<br>3. 튜토리얼 완료율 추적<br>4. 비디오 가이드 제작 (2분 설명)<br>5. FAQ 페이지 작성 |
| **ST-49** | Feat | 5 | **피드백 수집 시스템 (In-App Survey)** | **As a** 프로덕트 매니저<br>**I want to** 사용자에게 설문조사를 요청하여<br>**So that** 만족도와 개선 사항을 파악한다. | 1. Typeform 또는 Google Forms 연동<br>2. 사용 10회 후 자동 팝업<br>3. NPS 질문 (0~10 점수)<br>4. 개방형 피드백 수집<br>5. 응답률 50% 이상 목표 |
| **ST-50** | Feat | 5 | **사용 패턴 분석 (Amplitude/Mixpanel)** | **As a** 데이터 분석가<br>**I want to** 사용자 행동을 추적하여<br>**So that** 어떤 기능이 인기 있는지 파악한다. | 1. Amplitude SDK 연동<br>2. 주요 이벤트 추적 (로그인, API 호출, 모델 선택, 오류 발생)<br>3. Funnel 분석 (가입 → 첫 API 호출 → 재방문)<br>4. Cohort 분석 (주간 활성 사용자 WAU)<br>5. Retention Rate 계산 (D1, D7, D30) |
| **ST-51** | Feat | 3 | **버그 리포팅 시스템 (Sentry)** | **As a** 사용자<br>**I want to** 에러 발생 시 자동으로 리포트가 전송되어<br>**So that** 빠르게 문제를 해결받는다. | 1. Sentry SDK 적용 (Frontend, Backend)<br>2. 에러 자동 캡처 및 스택 트레이스 기록<br>3. Slack으로 Critical Error 알림<br>4. 에러 발생 빈도 기준 우선순위 설정<br>5. 주간 에러 리포트 작성 |
| **ST-52** | Test | 5 | **베타 사용자 인터뷰 및 개선** | **As a** 프로덕트 매니저<br>**I want to** 5명 이상의 사용자와 1:1 인터뷰를 진행하여<br>**So that** 깊이 있는 피드백을 얻는다. | 1. 사용자 5명 선정 및 인터뷰 일정 조율<br>2. 인터뷰 질문지 작성 (사용 시나리오, 불편 사항, 기대 기능)<br>3. 인터뷰 녹화 및 전사<br>4. 피드백을 Backlog에 추가<br>5. 긴급 버그는 즉시 수정 |

**Sprint 11 기술 스택:**
- Analytics: Amplitude, Mixpanel
- Error Tracking: Sentry
- Survey: Typeform, Google Forms
- Email: SendGrid
- User Feedback: Intercom (선택)

**Sprint 11 완료 조건:**
- [ ] 베타 사용자 10명 이상 가입
- [ ] NPS 평균 50 이상 달성
- [ ] 주요 버그 10건 이상 수정
- [ ] 사용자 인터뷰 5건 완료 및 피드백 정리

---

### **Sprint 12: Production Launch (정식 출시 및 마케팅)**
*   **기간:** 2026.06.07 ~ 2026.06.20
*   **목표:** 제품을 공식 론칭하고, 마케팅 캠페인을 실행하여 첫 유료 고객을 확보한다.
*   **Epic:** Go-to-Market
*   **Sprint Goal:** "Product Hunt에서 Top 5 순위, 첫 달 MRR $5,000 달성."

| Story ID | Type | Point | Title | User Story | Acceptance Criteria (DoD) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ST-53** | Feat | 5 | **가격 정책 및 결제 시스템 (Stripe)** | **As a** 사용자<br>**I want to** 신용카드로 결제하고 구독을 관리하여<br>**So that** 프리미엄 기능을 사용한다. | 1. Stripe Checkout 연동<br>2. 3가지 플랜 정의 (Free, Pro $99/mo, Enterprise $499/mo)<br>3. 구독 생성, 업그레이드, 취소 API<br>4. 청구서 자동 발송<br>5. 결제 실패 시 재시도 로직 |
| **ST-54** | Feat | 3 | **프로덕션 환경 성능 최적화** | **As a** SRE<br>**I want to** API 응답 시간을 최적화하여<br>**So that** 사용자 경험을 개선한다. | 1. DB 쿼리 최적화 (N+1 문제 해결)<br>2. Redis 캐싱 적용 (모델 메타데이터)<br>3. CDN 캐시 히트율 95% 이상<br>4. API P99 응답 시간 100ms 이하<br>5. Load Testing으로 1000 RPS 검증 |
| **ST-55** | Docs | 5 | **공식 문서 사이트 구축 (Docusaurus)** | **As a** 개발자<br>**I want to** 잘 정리된 문서를 읽으면서<br>**So that** API를 빠르게 통합한다. | 1. Docusaurus로 문서 사이트 구축<br>2. Getting Started, API Reference, Tutorials 섹션<br>3. 코드 예제 10개 이상 (Python, JS, cURL)<br>4. 검색 기능 (Algolia)<br>5. docs.master-sim.ai 도메인 연결 |
| **ST-56** | Marketing | 8 | **Product Hunt 런칭 준비** | **As a** 마케팅 담당자<br>**I want to** Product Hunt에 제품을 등록하고 Upvote를 모아서<br>**So that** 초기 트래픽을 확보한다. | 1. 제품 설명 작성 (250자 이내)<br>2. 데모 영상 제작 (1분 30초)<br>3. Hunter 및 Maker 계정 준비<br>4. 론칭 당일 커뮤니티 참여 (댓글 응답)<br>5. Top 5 순위 달성 (200+ Upvotes) |
| **ST-57** | Marketing | 5 | **콘텐츠 마케팅 (블로그, LinkedIn)** | **As a** 마케팅 담당자<br>**I want to** 기술 블로그 글을 발행하여<br>**So that** SEO를 개선하고 전문성을 어필한다. | 1. "Sim-to-Real Transfer for Robotics" 기술 블로그 작성<br>2. LinkedIn에 PoC 성공 사례 포스팅<br>3. Twitter(X)에서 주간 업데이트 공유<br>4. Reddit r/MachineLearning, r/robotics 커뮤니티 참여<br>5. 블로그 조회수 1000 이상 |
| **ST-58** | Marketing | 5 | **데모 예약 시스템 및 세일즈 자동화** | **As a** 영업 담당자<br>**I want to** 잠재 고객이 데모를 예약하면 자동으로 일정이 잡혀서<br>**So that** 세일즈 전환율을 높인다. | 1. Calendly 연동 (30분 데모 슬롯)<br>2. 데모 신청 양식 (회사명, 직급, Use Case)<br>3. 자동 확인 이메일 발송<br>4. CRM(HubSpot 또는 Notion)에 리드 기록<br>5. 데모 후 Follow-up 이메일 자동화 |
| **ST-59** | Ops | 3 | **프로덕션 배포 체크리스트 및 런북** | **As a** DevOps 엔지니어<br>**I want to** 배포 절차와 롤백 방법을 문서화하여<br>**So that** 긴급 상황에 빠르게 대응한다. | 1. 배포 전 체크리스트 작성 (DB 백업, 헬스 체크)<br>2. 롤백 스크립트 준비<br>3. Incident Response Plan 문서<br>4. On-call Rotation 설정<br>5. Postmortem 템플릿 작성 |

**Sprint 12 기술 스택:**
- Payment: Stripe Checkout, Stripe Billing
- Documentation: Docusaurus, Algolia Search
- Marketing: Product Hunt, LinkedIn, Twitter/X
- CRM: HubSpot, Calendly
- Performance: Redis, CloudFront, AWS Auto Scaling

**Sprint 12 완료 조건:**
- [ ] Stripe 결제가 정상 작동하며 첫 유료 고객 확보
- [ ] Product Hunt Top 5 달성
- [ ] 공식 문서 사이트 오픈 (docs.master-sim.ai)
- [ ] 데모 예약 10건 이상 접수
- [ ] 첫 달 MRR $5,000 달성

---

## 4. Story별 기술 구현 상세 가이드

이 섹션에서는 Sprint 1~4의 각 Story를 실제로 구현할 때 필요한 코드 예시, 디렉토리 구조, 테스트 방법을 제공합니다.

---

### **ST-1: 프로젝트 환경 설정 및 의존성 관리**

**디렉토리 구조:**
```
master-sim-robot/
├── .venv/                    # Python 가상환경 (git ignore)
├── src/
│   ├── __init__.py
│   ├── envs/                 # 시뮬레이션 환경
│   ├── agents/               # AI 에이전트
│   └── utils/                # 유틸리티 함수
├── tests/
│   └── test_environment.py   # 환경 설정 테스트
├── assets/
│   └── models/               # 로봇 URDF/XML 파일
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

**requirements.txt:**
```txt
# Core Dependencies
mujoco==3.1.2
gymnasium==0.29.1
numpy==1.26.3
scipy==1.11.4

# Deep Learning
torch==2.2.0
torchvision==0.17.0

# Visualization
matplotlib==3.8.2
opencv-python==4.9.0.80

# Development
pytest==7.4.3
pytest-cov==4.1.0
black==23.12.1
mypy==1.8.0
ruff==0.1.11

# Utilities
pyyaml==6.0.1
tqdm==4.66.1
```

**setup.py:**
```python
from setuptools import setup, find_packages

setup(
    name="master-sim",
    version="0.1.0",
    description="Physical AI Simulation Platform for Precision Assembly",
    author="Master-Sim Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "mujoco>=3.1.2",
        "gymnasium>=0.29.1",
        "numpy>=1.26.0",
        "torch>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.12.0",
            "mypy>=1.8.0",
        ]
    },
)
```

**tests/test_environment.py:**
```python
import pytest
import mujoco
import numpy as np


def test_mujoco_import():
    """MuJoCo가 정상적으로 import되는지 확인"""
    assert mujoco.__version__ >= "3.1.0"


def test_mujoco_basic_simulation():
    """MuJoCo 기본 시뮬레이션 실행 테스트"""
    # 간단한 XML 장면 정의
    xml_string = """
    <mujoco>
        <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
            <body pos="0 0 1">
                <joint type="free"/>
                <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    # 모델 로드
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    # 시뮬레이션 1초 실행 (500 스텝, dt=0.002)
    for _ in range(500):
        mujoco.mj_step(model, data)
    
    # 물체가 떨어졌는지 확인 (z 위치가 낮아짐)
    assert data.qpos[2] < 0.5  # 초기 위치 1.0에서 떨어짐


def test_numpy_compatibility():
    """NumPy 배열이 MuJoCo와 호환되는지 확인"""
    xml_string = "<mujoco><worldbody><geom type='plane' size='1 1 0.1'/></worldbody></mujoco>"
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    # qpos를 NumPy 배열로 변환
    qpos_array = np.array(data.qpos)
    assert isinstance(qpos_array, np.ndarray)
    assert qpos_array.dtype == np.float64
```

**실행 방법:**
```bash
# 1. 가상환경 생성
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 2. 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# 3. 개발 모드 설치
pip install -e .

# 4. 테스트 실행
pytest tests/ -v

# 5. 코드 포맷팅
black src/ tests/
```

---

### **ST-2: MuJoCo 기본 뷰어 띄우기**

**src/envs/basic_viewer.py:**
```python
"""
MuJoCo 기본 뷰어 예제
마우스로 시점 조작 가능한 시뮬레이션 윈도우를 표시합니다.
"""

import mujoco
import mujoco.viewer
import numpy as np


def create_basic_scene():
    """기본 장면 XML 생성 (바닥 + 조명)"""
    xml_string = """
    <mujoco model="basic_scene">
        <option gravity="0 0 -9.81" timestep="0.002"/>
        
        <visual>
            <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8"/>
        </visual>
        
        <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                     rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2"/>
            <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"
                      reflectance="0.2"/>
        </asset>
        
        <worldbody>
            <light directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"
                   pos="0 0 5" dir="0 0 -1"/>
            
            <!-- 바닥 평면 -->
            <geom name="floor" type="plane" size="2 2 0.1" material="grid"/>
            
            <!-- 테스트용 큐브 -->
            <body name="test_cube" pos="0 0 0.5">
                <joint type="free"/>
                <geom type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1" mass="0.1"/>
                <geom type="sphere" size="0.05" pos="0.15 0 0" rgba="0.2 0.8 0.2 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return xml_string


def run_basic_viewer(duration=10.0):
    """
    MuJoCo 뷰어 실행
    
    Args:
        duration: 시뮬레이션 실행 시간 (초)
    """
    # 모델 로드
    xml_string = create_basic_scene()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    print(f"MuJoCo 버전: {mujoco.__version__}")
    print(f"모델 로드 완료: {model.nq} DoF, {model.nbody} bodies")
    print("\n뷰어 조작법:")
    print("  - 마우스 좌클릭 드래그: 회전")
    print("  - 마우스 우클릭 드래그: 이동")
    print("  - 마우스 휠: 줌")
    print("  - Spacebar: 일시정지/재생")
    print("  - ESC: 종료\n")
    
    # 뷰어 실행 (duration 초 동안)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.5  # 카메라 거리
        viewer.cam.azimuth = 45    # 카메라 방위각
        viewer.cam.elevation = -20  # 카메라 고도
        
        start_time = data.time
        while viewer.is_running() and (data.time - start_time) < duration:
            # 물리 시뮬레이션 스텝
            mujoco.mj_step(model, data)
            
            # 뷰어 동기화 (60 FPS)
            viewer.sync()
    
    print(f"시뮬레이션 종료: {data.time:.2f}초 경과")


if __name__ == "__main__":
    run_basic_viewer(duration=30.0)
```

**tests/test_viewer.py:**
```python
import pytest
import mujoco
from src.envs.basic_viewer import create_basic_scene


def test_scene_xml_valid():
    """장면 XML이 유효한지 확인"""
    xml_string = create_basic_scene()
    model = mujoco.MjModel.from_xml_string(xml_string)
    
    assert model is not None
    assert model.nbody > 0  # 최소 1개 이상의 body
    assert model.ngeom > 0  # 최소 1개 이상의 geometry


def test_simulation_stability():
    """시뮬레이션이 안정적으로 실행되는지 확인"""
    xml_string = create_basic_scene()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    # 1000 스텝 실행
    for _ in range(1000):
        mujoco.mj_step(model, data)
    
    # 위치값이 무한대가 아닌지 확인 (발산하지 않음)
    assert not np.any(np.isinf(data.qpos))
    assert not np.any(np.isnan(data.qpos))
```

**실행:**
```bash
python src/envs/basic_viewer.py
```

---

### **ST-3: Franka Emika Panda 로봇 로드**

**assets/models/panda_scene.xml:**
```xml
<mujoco model="panda_workspace">
    <option gravity="0 0 -9.81" timestep="0.002"/>
    
    <compiler angle="radian" meshdir="meshes/"/>
    
    <asset>
        <!-- 실제 프로젝트에서는 Panda URDF를 변환하여 사용 -->
        <!-- 여기서는 간소화된 버전 -->
        <mesh name="panda_link0" file="panda/link0.stl"/>
        <mesh name="panda_link1" file="panda/link1.stl"/>
        <!-- ... 나머지 링크 메쉬 -->
        
        <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512"/>
        <material name="floor_mat" texture="floor_tex" texrepeat="2 2"/>
    </asset>
    
    <worldbody>
        <light directional="true" pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="2 2 0.1" material="floor_mat"/>
        
        <!-- Panda 로봇 베이스 -->
        <body name="panda_base" pos="0 0 0">
            <geom type="cylinder" size="0.06 0.03" rgba="0.9 0.9 0.9 1"/>
            
            <!-- Joint 1 (Base Rotation) -->
            <body name="panda_link1" pos="0 0 0.333">
                <joint name="panda_joint1" type="hinge" axis="0 0 1" 
                       range="-2.8973 2.8973" damping="0.5"/>
                <geom type="cylinder" size="0.05 0.1" rgba="0.9 0.9 0.9 1"/>
                
                <!-- Joint 2 (Shoulder) -->
                <body name="panda_link2" pos="0 0 0" quat="0.707 -0.707 0 0">
                    <joint name="panda_joint2" type="hinge" axis="0 0 1"
                           range="-1.7628 1.7628" damping="0.5"/>
                    <geom type="cylinder" size="0.05 0.1" rgba="0.9 0.9 0.9 1"/>
                    
                    <!-- 나머지 관절 구조... (실제로는 7개 관절) -->
                    
                    <!-- End Effector (간소화) -->
                    <body name="panda_hand" pos="0 0 0.4">
                        <geom type="box" size="0.04 0.04 0.02" rgba="0.2 0.2 0.2 1"/>
                        
                        <!-- 그리퍼 핑거 -->
                        <body name="panda_finger_left" pos="0 0.02 0">
                            <joint name="panda_finger_joint1" type="slide" axis="0 1 0"
                                   range="0 0.04" damping="0.1"/>
                            <geom type="box" size="0.01 0.01 0.03" rgba="0.3 0.3 0.3 1"/>
                        </body>
                        
                        <body name="panda_finger_right" pos="0 -0.02 0">
                            <joint name="panda_finger_joint2" type="slide" axis="0 -1 0"
                                   range="0 0.04" damping="0.1"/>
                            <geom type="box" size="0.01 0.01 0.03" rgba="0.3 0.3 0.3 1"/>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <!-- Position Control for Joints -->
        <position name="panda_act1" joint="panda_joint1" kp="100" kv="10"/>
        <position name="panda_act2" joint="panda_joint2" kp="100" kv="10"/>
        <!-- ... 나머지 액추에이터 -->
        
        <!-- Gripper Control -->
        <position name="panda_gripper_act1" joint="panda_finger_joint1" kp="50"/>
        <position name="panda_gripper_act2" joint="panda_finger_joint2" kp="50"/>
    </actuator>
</mujoco>
```

**src/envs/panda_env.py:**
```python
"""
Franka Emika Panda 로봇 환경
"""

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path


class PandaEnv:
    """Panda 로봇 시뮬레이션 환경"""
    
    def __init__(self, xml_path: str = "assets/models/panda_scene.xml"):
        """
        Args:
            xml_path: MuJoCo XML 파일 경로
        """
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML 파일을 찾을 수 없습니다: {xml_path}")
        
        # 모델 로드
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        
        # 관절 정보
        self.n_joints = 7  # Panda는 7-DoF
        self.joint_names = [f"panda_joint{i+1}" for i in range(self.n_joints)]
        self.joint_ids = [self.model.joint(name).id for name in self.joint_names]
        
        print(f"Panda 환경 초기화 완료")
        print(f"  - DoF: {self.model.nv}")
        print(f"  - Bodies: {self.model.nbody}")
        print(f"  - Actuators: {self.model.nu}")
    
    def reset(self, seed: int = None):
        """환경 리셋"""
        if seed is not None:
            np.random.seed(seed)
        
        # 초기 자세 (Home position)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:self.n_joints] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        return self.get_observation()
    
    def get_observation(self):
        """현재 상태 관측"""
        return {
            'qpos': self.data.qpos[:self.n_joints].copy(),
            'qvel': self.data.qvel[:self.n_joints].copy(),
            'ee_pos': self.data.site('ee_site').xpos.copy(),  # End-effector 위치
        }
    
    def set_joint_positions(self, target_qpos: np.ndarray):
        """관절 각도 설정"""
        assert len(target_qpos) == self.n_joints
        self.data.ctrl[:self.n_joints] = target_qpos
    
    def step(self, action: np.ndarray):
        """시뮬레이션 1 스텝 실행"""
        self.set_joint_positions(action)
        mujoco.mj_step(self.model, self.data)
        
        obs = self.get_observation()
        reward = 0.0  # 나중에 구현
        done = False
        info = {}
        
        return obs, reward, done, info
    
    def render(self, duration: float = 10.0):
        """뷰어로 렌더링"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            
            start_time = self.data.time
            while viewer.is_running() and (self.data.time - start_time) < duration:
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


if __name__ == "__main__":
    # 환경 생성 및 테스트
    env = PandaEnv()
    env.reset()
    
    print("\n초기 관절 각도:", env.get_observation()['qpos'])
    print("End-effector 위치:", env.get_observation()['ee_pos'])
    
    # 뷰어 실행
    env.render(duration=30.0)
```

---

### **ST-6: 마우스 기반 IK(Inverse Kinematics) 제어**

**src/controllers/ik_controller.py:**
```python
"""
Inverse Kinematics (IK) 컨트롤러
마우스 위치를 End-Effector 목표 위치로 변환
"""

import mujoco
import numpy as np
from scipy.optimize import least_squares


class IKController:
    """Jacobian 기반 IK 컨트롤러"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, 
                 site_name: str = "ee_site"):
        """
        Args:
            model: MuJoCo 모델
            data: MuJoCo 데이터
            site_name: End-effector site 이름
        """
        self.model = model
        self.data = data
        self.site_id = model.site(site_name).id
        
        # Jacobian 행렬 (3 x nv)
        self.jac_pos = np.zeros((3, model.nv))
        self.jac_rot = np.zeros((3, model.nv))
    
    def solve_ik(self, target_pos: np.ndarray, current_qpos: np.ndarray,
                 max_iter: int = 100, tolerance: float = 1e-3) -> np.ndarray:
        """
        IK 문제를 Damped Least Squares로 해결
        
        Args:
            target_pos: 목표 위치 (x, y, z)
            current_qpos: 현재 관절 각도
            max_iter: 최대 반복 횟수
            tolerance: 수렴 기준 (미터)
        
        Returns:
            해결된 관절 각도
        """
        qpos = current_qpos.copy()
        damping = 1e-4
        step_size = 0.5
        
        for iteration in range(max_iter):
            # Forward kinematics
            self.data.qpos[:len(qpos)] = qpos
            mujoco.mj_forward(self.model, self.data)
            
            # 현재 End-effector 위치
            current_pos = self.data.site_xpos[self.site_id].copy()
            
            # 오차 계산
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < tolerance:
                print(f"IK 수렴: {iteration} 반복, 오차 {error_norm:.6f}m")
                return qpos
            
            # Jacobian 계산
            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, 
                             self.site_id)
            
            # Damped Least Squares (Levenberg-Marquardt)
            jac_t = self.jac_pos[:, :len(qpos)].T
            delta_q = jac_t @ np.linalg.solve(
                self.jac_pos[:, :len(qpos)] @ jac_t + damping * np.eye(3),
                error
            )
            
            # 업데이트
            qpos += step_size * delta_q
            
            # 관절 한계 적용
            qpos = np.clip(qpos, self.model.jnt_range[:len(qpos), 0],
                          self.model.jnt_range[:len(qpos), 1])
        
        print(f"IK 미수렴: 최대 반복 도달, 최종 오차 {error_norm:.6f}m")
        return qpos
    
    def solve_ik_numerical(self, target_pos: np.ndarray, 
                          current_qpos: np.ndarray) -> np.ndarray:
        """
        수치 최적화로 IK 해결 (scipy.optimize 사용)
        """
        def residual(q):
            self.data.qpos[:len(q)] = q
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[self.site_id]
            return target_pos - current_pos
        
        result = least_squares(
            residual, current_qpos, 
            bounds=(self.model.jnt_range[:len(current_qpos), 0],
                   self.model.jnt_range[:len(current_qpos), 1]),
            ftol=1e-4, xtol=1e-4, max_nfev=200
        )
        
        return result.x


def mouse_to_world_pos(mouse_x: float, mouse_y: float, 
                       cam_distance: float = 1.0) -> np.ndarray:
    """
    마우스 화면 좌표를 3D 월드 좌표로 변환 (간소화 버전)
    
    Args:
        mouse_x: 정규화된 마우스 X 좌표 (-1 ~ 1)
        mouse_y: 정규화된 마우스 Y 좌표 (-1 ~ 1)
        cam_distance: 카메라 거리
    
    Returns:
        3D 월드 좌표 (x, y, z)
    """
    # 간단한 직교 투영 가정
    scale = cam_distance * 0.5
    x = mouse_x * scale
    y = mouse_y * scale
    z = 0.3  # 테이블 높이
    
    return np.array([x, y, z])
```

**tests/test_ik_controller.py:**
```python
import pytest
import numpy as np
import mujoco
from src.controllers.ik_controller import IKController


@pytest.fixture
def panda_model():
    """Panda 모델 픽스처 (간소화 버전)"""
    xml_string = """
    <mujoco>
        <worldbody>
            <body name="panda_base">
                <geom type="box" size="0.1 0.1 0.1"/>
                <body name="link1" pos="0 0 0.3">
                    <joint type="hinge" axis="0 0 1" range="-3 3"/>
                    <geom type="cylinder" size="0.05 0.1"/>
                    <body name="ee" pos="0 0 0.3">
                        <site name="ee_site" pos="0 0 0"/>
                    </body>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    return model, data


def test_ik_forward_kinematics(panda_model):
    """Forward Kinematics 검증"""
    model, data = panda_model
    
    # 관절을 특정 각도로 설정
    data.qpos[0] = np.pi / 4  # 45도
    mujoco.mj_forward(model, data)
    
    # End-effector 위치 확인
    ee_pos = data.site_xpos[model.site("ee_site").id]
    assert ee_pos[2] > 0  # Z 좌표가 양수


def test_ik_solver_accuracy(panda_model):
    """IK 솔버 정확도 테스트"""
    model, data = panda_model
    controller = IKController(model, data)
    
    # 목표 위치 설정
    target_pos = np.array([0.2, 0.1, 0.4])
    initial_qpos = np.zeros(model.nq)
    
    # IK 해결
    solution_qpos = controller.solve_ik(target_pos, initial_qpos)
    
    # 검증: 해를 적용했을 때 End-effector가 목표 위치에 도달하는지
    data.qpos[:] = solution_qpos
    mujoco.mj_forward(model, data)
    actual_pos = data.site_xpos[controller.site_id]
    
    error = np.linalg.norm(actual_pos - target_pos)
    assert error < 0.01  # 1cm 이내 오차
```

---

### **ST-9: 학습 데이터 전처리 파이프라인**

**src/data/preprocessing.py:**
```python
"""
학습 데이터 전처리 및 증강
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """데이터셋 설정"""
    normalize: bool = True
    add_noise: bool = False
    noise_std: float = 0.01
    augmentation_factor: int = 1  # 데이터 증강 배수


class BehaviorCloningDataset:
    """Behavior Cloning 학습 데이터셋"""
    
    def __init__(self, data_dir: str, config: DatasetConfig = None):
        """
        Args:
            data_dir: 데이터 파일 디렉토리
            config: 전처리 설정
        """
        self.data_dir = Path(data_dir)
        self.config = config or DatasetConfig()
        
        # 데이터 로드
        self.states = []
        self.actions = []
        self.load_all_episodes()
        
        # 통계 계산
        self.compute_statistics()
        
        print(f"데이터셋 로드 완료:")
        print(f"  - 에피소드 수: {len(self.states)}")
        print(f"  - 총 샘플 수: {sum(len(s) for s in self.states)}")
    
    def load_all_episodes(self):
        """모든 에피소드 파일 로드"""
        episode_files = sorted(self.data_dir.glob("episode_*.h5"))
        
        for filepath in episode_files:
            with h5py.File(filepath, 'r') as f:
                states = f['observations'][:]  # (T, state_dim)
                actions = f['actions'][:]      # (T, action_dim)
                
                self.states.append(states)
                self.actions.append(actions)
    
    def compute_statistics(self):
        """평균 및 표준편차 계산 (정규화용)"""
        all_states = np.concatenate(self.states, axis=0)
        all_actions = np.concatenate(self.actions, axis=0)
        
        self.state_mean = np.mean(all_states, axis=0)
        self.state_std = np.std(all_states, axis=0) + 1e-8
        
        self.action_mean = np.mean(all_actions, axis=0)
        self.action_std = np.std(all_actions, axis=0) + 1e-8
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """상태 정규화"""
        return (state - self.state_mean) / self.state_std
    
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """액션 정규화"""
        return (action - self.action_mean) / self.action_std
    
    def denormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """액션 역정규화"""
        return normalized_action * self.action_std + self.action_mean
    
    def augment_data(self, state: np.ndarray, action: np.ndarray) -> Tuple:
        """데이터 증강"""
        augmented_states = [state]
        augmented_actions = [action]
        
        for _ in range(self.config.augmentation_factor - 1):
            # Gaussian Noise 추가
            noisy_state = state + np.random.normal(
                0, self.config.noise_std, state.shape
            )
            noisy_action = action + np.random.normal(
                0, self.config.noise_std * 0.5, action.shape
            )
            
            augmented_states.append(noisy_state)
            augmented_actions.append(noisy_action)
        
        return np.array(augmented_states), np.array(augmented_actions)
    
    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """랜덤 배치 샘플링"""
        # 에피소드 선택
        episode_idx = np.random.randint(0, len(self.states), size=batch_size)
        
        states_batch = []
        actions_batch = []
        
        for idx in episode_idx:
            # 에피소드 내 타임스텝 선택
            t = np.random.randint(0, len(self.states[idx]))
            
            state = self.states[idx][t]
            action = self.actions[idx][t]
            
            # 정규화
            if self.config.normalize:
                state = self.normalize_state(state)
                action = self.normalize_action(action)
            
            states_batch.append(state)
            actions_batch.append(action)
        
        return np.array(states_batch), np.array(actions_batch)


def create_replay_buffer(data_dir: str, output_file: str):
    """
    여러 에피소드 파일을 하나의 Replay Buffer로 통합
    
    Args:
        data_dir: 입력 데이터 디렉토리
        output_file: 출력 HDF5 파일 경로
    """
    dataset = BehaviorCloningDataset(data_dir)
    
    all_states = np.concatenate(dataset.states, axis=0)
    all_actions = np.concatenate(dataset.actions, axis=0)
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('states', data=all_states, compression='gzip')
        f.create_dataset('actions', data=all_actions, compression='gzip')
        f.create_dataset('state_mean', data=dataset.state_mean)
        f.create_dataset('state_std', data=dataset.state_std)
        f.create_dataset('action_mean', data=dataset.action_mean)
        f.create_dataset('action_std', data=dataset.action_std)
    
    print(f"Replay Buffer 저장 완료: {output_file}")
    print(f"  - 샘플 수: {len(all_states)}")
    print(f"  - State dim: {all_states.shape[1]}")
    print(f"  - Action dim: {all_actions.shape[1]}")
```

---

### **ST-10: Behavior Cloning (BC) 모델 구현**

**src/agents/bc_agent.py:**
```python
"""
Behavior Cloning Agent (PyTorch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class BCPolicy(nn.Module):
    """MLP 기반 Behavior Cloning 정책"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_sizes: Tuple[int, ...] = (256, 256)):
        """
        Args:
            state_dim: 상태 벡터 차원
            action_dim: 액션 벡터 차원
            hidden_sizes: 은닉층 크기
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Network 구조
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.1),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier 초기화"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: (batch_size, state_dim)
        
        Returns:
            action: (batch_size, action_dim)
        """
        return self.network(state)
    
    @torch.no_grad()
    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        NumPy 입력을 받아 액션 예측
        
        Args:
            state: (state_dim,) or (batch_size, state_dim)
            deterministic: True이면 결정론적 출력
        
        Returns:
            action: (action_dim,) or (batch_size, action_dim)
        """
        self.eval()
        
        # NumPy -> Tensor
        if state.ndim == 1:
            state = state[None, :]  # (1, state_dim)
            squeeze = True
        else:
            squeeze = False
        
        state_tensor = torch.FloatTensor(state)
        action_tensor = self.forward(state_tensor)
        
        # Tensor -> NumPy
        action = action_tensor.cpu().numpy()
        
        if squeeze:
            action = action[0]
        
        return action


class BCTrainer:
    """BC 학습 트레이너"""
    
    def __init__(self, policy: BCPolicy, learning_rate: float = 3e-4):
        """
        Args:
            policy: BC 정책 네트워크
            learning_rate: 학습률
        """
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy.to(self.device)
        
        # 학습 기록
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, states: np.ndarray, actions: np.ndarray) -> float:
        """
        1 Step 학습
        
        Args:
            states: (batch_size, state_dim)
            actions: (batch_size, action_dim)
        
        Returns:
            loss: 평균 손실값
        """
        self.policy.train()
        
        # NumPy -> Tensor
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        
        # Forward pass
        predicted_actions = self.policy(states_tensor)
        
        # MSE Loss
        loss = F.mse_loss(predicted_actions, actions_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, states: np.ndarray, actions: np.ndarray) -> float:
        """검증 데이터 평가"""
        self.policy.eval()
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        
        predicted_actions = self.policy(states_tensor)
        loss = F.mse_loss(predicted_actions, actions_tensor)
        
        return loss.item()
    
    def save_checkpoint(self, filepath: str):
        """체크포인트 저장"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, filepath)
        print(f"체크포인트 저장: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"체크포인트 로드: {filepath}")
```

**scripts/train_bc.py:**
```python
"""
BC 모델 학습 스크립트
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.data.preprocessing import BehaviorCloningDataset, DatasetConfig
from src.agents.bc_agent import BCPolicy, BCTrainer


def train(args):
    """학습 실행"""
    # 데이터셋 로드
    config = DatasetConfig(
        normalize=True,
        add_noise=args.augment,
        augmentation_factor=3 if args.augment else 1
    )
    dataset = BehaviorCloningDataset(args.data_dir, config)
    
    # 모델 생성
    state_dim = dataset.states[0].shape[1]
    action_dim = dataset.actions[0].shape[1]
    
    policy = BCPolicy(state_dim, action_dim, hidden_sizes=(256, 256, 128))
    trainer = BCTrainer(policy, learning_rate=args.lr)
    
    print(f"\n모델 구조:")
    print(policy)
    print(f"\n총 파라미터 수: {sum(p.numel() for p in policy.parameters()):,}")
    
    # 학습 루프
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_losses = []
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for _ in pbar:
            states, actions = dataset.get_batch(args.batch_size)
            loss = trainer.train_step(states, actions)
            train_losses.append(loss)
            
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_train_loss = np.mean(train_losses)
        trainer.train_losses.append(avg_train_loss)
        
        # Validation
        val_states, val_actions = dataset.get_batch(1000)
        val_loss = trainer.evaluate(val_states, val_actions)
        trainer.val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # 최고 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(f"{args.output_dir}/best_model.pt")
    
    # 최종 모델 저장
    trainer.save_checkpoint(f"{args.output_dir}/final_model.pt")
    print(f"\n학습 완료! 최고 검증 손실: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='데이터 디렉토리')
    parser.add_argument('--output-dir', type=str, default='models/', help='모델 저장 경로')
    parser.add_argument('--epochs', type=int, default=100, help='에포크 수')
    parser.add_argument('--batch-size', type=int, default=256, help='배치 크기')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='에포크당 스텝')
    parser.add_argument('--lr', type=float, default=3e-4, help='학습률')
    parser.add_argument('--augment', action='store_true', help='데이터 증강 사용')
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train(args)
```

**실행:**
```bash
python scripts/train_bc.py \
    --data-dir data/demonstrations/ \
    --output-dir models/bc_peg_in_hole/ \
    --epochs 200 \
    --batch-size 256 \
    --lr 0.0003 \
    --augment
```

---

## 5. Epic & Story 관리 가이드

### 5.1 Epic 정의 및 분해 전략

**Epic이란?**
Epic은 하나의 큰 기능 또는 목표를 나타내는 대규모 작업 단위입니다. 보통 2~4개의 Sprint에 걸쳐 완성됩니다.

**Master-Sim의 Epic 구조:**

```
Epic: Simulation Foundation (Sprint 1)
  ├── ST-1: 프로젝트 환경 설정
  ├── ST-2: MuJoCo 기본 뷰어
  ├── ST-3: Panda 로봇 로드
  ├── ST-4: 관절 제어 구현
  └── ST-5: 작업 테이블 및 객체 배치

Epic: Data Acquisition (Sprint 2)
  ├── ST-6: IK 제어
  ├── ST-7: 그리퍼 제어
  └── ST-8: 데이터 로거

Epic: AI Model Training (Sprint 3)
  ├── ST-9: 데이터 전처리
  ├── ST-10: BC 모델 구현
  └── ST-11: 모델 추론 적용

Epic: Dashboard UI (Sprint 4)
  ├── ST-12: Figma 설계
  └── ST-13: 실시간 차트
```

**Epic 분해 원칙 (INVEST):**

1. **Independent (독립적)**: Epic은 다른 Epic과 독립적으로 완성 가능
2. **Negotiable (협상 가능)**: 구현 방법은 유연하게 조정 가능
3. **Valuable (가치 제공)**: Epic 완료 시 사용자에게 명확한 가치 제공
4. **Estimable (추정 가능)**: Epic의 크기를 대략적으로 추정 가능
5. **Small (작은 크기)**: 1~4 Sprint 내에 완료 가능
6. **Testable (테스트 가능)**: Epic의 완료 여부를 검증 가능

**Epic 분해 프로세스:**

```
Step 1: Epic 정의
  "사용자가 웹 대시보드에서 AI 학습 진행 상황을 실시간으로 확인한다"

Step 2: User Stories 도출 (5W1H)
  - Who: 누가 사용하는가? → AI 엔지니어, PM
  - What: 무엇을 보는가? → Loss 그래프, 성공률, 하이퍼파라미터
  - When: 언제 보는가? → 학습 중, 학습 완료 후
  - Where: 어디서 보는가? → 웹 브라우저
  - Why: 왜 필요한가? → 학습 진행 상황 모니터링
  - How: 어떻게 구현하는가? → React + Chart.js + WebSocket

Step 3: Stories로 분해
  ├── ST-12: Figma로 UI 설계 (3 Points)
  ├── ST-13: React 프론트엔드 구현 (8 Points)
  ├── ST-14: 백엔드 WebSocket 서버 (5 Points)
  └── ST-15: 실시간 데이터 연동 (5 Points)

Step 4: 우선순위 결정 (MoSCoW)
  - Must have: ST-13 (핵심 기능)
  - Should have: ST-12, ST-14
  - Could have: ST-15 (실시간은 선택)
  - Won't have: 알림 기능 (나중에)
```

---

### 5.2 Story Mapping 기법

**Story Map 구조:**

```
User Activities (상위 목표)
    ↓
User Tasks (세부 작업)
    ↓
User Stories (구현 단위)
    ↓
Sprint 1, Sprint 2, ...
```

**Master-Sim Story Map 예시:**

```
┌────────────────────────────────────────────────────────────┐
│ User Activity: 로봇 시뮬레이션 환경 구축                    │
└────────────────────────────────────────────────────────────┘
        │
        ├─ Task: 시뮬레이터 설치 및 테스트
        │   ├─ ST-1: MuJoCo 설치 (Sprint 1)
        │   └─ ST-2: 기본 장면 렌더링 (Sprint 1)
        │
        ├─ Task: 로봇 모델 로드
        │   ├─ ST-3: Panda URDF 변환 (Sprint 1)
        │   └─ ST-4: 관절 제어 테스트 (Sprint 1)
        │
        └─ Task: 작업 환경 구성
            └─ ST-5: Peg/Hole 배치 (Sprint 1)

┌────────────────────────────────────────────────────────────┐
│ User Activity: AI 모델 학습 및 평가                         │
└────────────────────────────────────────────────────────────┘
        │
        ├─ Task: 데이터 수집
        │   ├─ ST-6: 텔레오퍼레이션 (Sprint 2)
        │   └─ ST-8: 데이터 로깅 (Sprint 2)
        │
        ├─ Task: 모델 학습
        │   ├─ ST-9: 전처리 (Sprint 3) ← MVP 라인
        │   ├─ ST-10: BC 학습 (Sprint 3)
        │   └─ ST-16: RL 학습 (Sprint 5)
        │
        └─ Task: 성능 평가
            ├─ ST-11: 시뮬레이션 테스트 (Sprint 3)
            └─ ST-18: A/B 테스트 (Sprint 5)
```

**Story Mapping 세션 진행:**

1. **참여자**: Product Owner, 개발자, 디자이너
2. **준비물**: 포스트잇, 화이트보드 (또는 Miro, FigJam)
3. **시간**: 2~3시간

**진행 순서:**
```
1. User Persona 정의 (30분)
   - AI 엔지니어 (주 사용자)
   - PM (모니터링)
   - 영업 (데모)

2. User Journey 작성 (60분)
   - 처음 사용: 가입 → 튜토리얼 → 첫 학습
   - 일상 사용: 데이터 업로드 → 학습 → 결과 확인
   - 고급 사용: 모델 최적화 → A/B 테스트 → 배포

3. Activities & Tasks 도출 (60분)
   - 큰 목표부터 세부 작업으로 분해
   - 포스트잇으로 시각화

4. Stories 작성 및 우선순위 (30분)
   - MVP 라인 긋기 (반드시 필요한 기능)
   - Sprint별 그룹핑
```

---

### 5.3 User Persona 및 Journey

**Primary Persona: AI 엔지니어 (Alex)**

```yaml
Name: Alex Kim
Age: 32
Role: Robotics AI Engineer
Company: 중견 제조업체 (직원 500명)
Background:
  - 전공: 컴퓨터 공학 석사
  - 경력: 5년 (로봇 제어, 강화학습)
  - 기술 스택: Python, PyTorch, ROS

Goals:
  - Peg-in-Hole 작업을 위한 로봇 정책 학습
  - 실제 로봇 없이 시뮬레이션으로 개발
  - 2주 내 PoC 완성

Pain Points:
  - 실제 로봇 구매 비용 부담 ($50,000+)
  - 하드웨어 대기 시간 (3개월)
  - Sim-to-Real 전환 실패 경험

Needs:
  - 빠른 반복 실험 (Iteration)
  - 사전 학습된 모델 (Pre-trained Policy)
  - 명확한 문서 및 예제
```

**Alex의 User Journey:**

```
Phase 1: 문제 인식
  └─ "실제 로봇 없이 AI를 테스트하고 싶다"
      └─ Google 검색 → Master-Sim 발견

Phase 2: 평가 (Evaluation)
  ├─ 랜딩 페이지 방문 → 데모 영상 시청 (2분)
  ├─ 공식 문서 확인 → Getting Started 읽기
  └─ 무료 가입 → API Key 발급

Phase 3: 온보딩 (Onboarding)
  ├─ 튜토리얼 완료 (10분)
  │   ├─ 첫 API 호출
  │   ├─ 샘플 데이터 업로드
  │   └─ 학습 시작
  └─ 결과 확인 → "오! 작동한다!"

Phase 4: 활성 사용 (Active Use)
  ├─ 자체 데이터로 재학습 (Week 1)
  ├─ 하이퍼파라미터 튜닝 (Week 2)
  └─ A/B 테스트 (Week 3)

Phase 5: 확장 (Expansion)
  ├─ 다른 작업에도 적용 (Pick & Place)
  ├─ 유료 플랜 업그레이드
  └─ 팀원 초대

Phase 6: 옹호자 (Advocate)
  ├─ LinkedIn에 성공 사례 공유
  ├─ 학회에서 발표
  └─ 다른 회사에 추천
```

**Secondary Persona: 프로덕트 매니저 (Sarah)**

```yaml
Name: Sarah Lee
Age: 38
Role: Product Manager
Goals:
  - AI 프로젝트 진행 상황 모니터링
  - 투자 대비 성과 측정

Needs:
  - 대시보드 (성공률, 비용)
  - 주간 리포트
  - 비기술적 언어로 설명

Journey:
  1. Alex가 초대 → 뷰어 권한 부여
  2. 대시보드 확인 → "성공률 85%"
  3. 경영진에게 보고 → 예산 승인
```

---

### 5.4 Story Point Estimation (포인트 추정)

**Story Point란?**
작업의 **복잡도, 노력, 불확실성**을 숫자로 표현한 상대적 측정 단위입니다.

**Fibonacci 수열 사용:**
```
1, 2, 3, 5, 8, 13, 21, ...
```

**각 포인트의 의미:**

| Point | 시간 | 복잡도 | 예시 |
|:---:|:---|:---|:---|
| **1** | 1~2시간 | 매우 쉬움 | 설정 파일 수정, 간단한 버그 픽스 |
| **2** | 2~4시간 | 쉬움 | 단순 함수 추가, 테스트 작성 |
| **3** | 4~8시간 | 보통 | 새로운 API 엔드포인트, UI 컴포넌트 |
| **5** | 1~2일 | 중간 | 데이터베이스 스키마 변경, 복잡한 알고리즘 |
| **8** | 2~3일 | 어려움 | 새로운 서비스 통합, 성능 최적화 |
| **13** | 3~5일 | 매우 어려움 | 시스템 아키텍처 변경, 대규모 리팩토링 |
| **21+** | 1주+ | Epic으로 분해 필요 | 너무 크므로 분해 |

**Planning Poker 프로세스:**

```
1. Story 읽기 (5분)
   - PO가 Story 설명
   - 개발자 질문 & 답변

2. 개별 추정 (1분)
   - 각자 카드 선택 (숫자는 숨김)

3. 동시 공개
   - 모두 동시에 카드 공개
   
4. 토론 (5분)
   - 가장 높은 점수와 낮은 점수를 준 사람이 이유 설명
   - "왜 8점이라고 생각했나요?"
   - "제가 놓친 복잡도가 있나요?"

5. 재투표
   - 합의할 때까지 반복 (보통 2~3회)

6. 최종 Point 결정
   - 다수결 또는 평균값
```

**Estimation 예시:**

**ST-6: 마우스 기반 IK 제어**

```
Round 1:
  Developer A: 5 (IK 경험 있음)
  Developer B: 13 (IK 처음)
  Developer C: 8 (보통)

토론:
  B: "IK 알고리즘이 복잡해서 13점"
  A: "라이브러리 사용하면 5점 가능"
  C: "디버깅 시간 고려하면 8점"

Round 2:
  A: 8 (동의)
  B: 8 (납득)
  C: 8

최종: 8 Points
```

**Velocity 계산:**

```
Sprint 1 완료: 18 Points
Sprint 2 완료: 22 Points
Sprint 3 완료: 20 Points

평균 Velocity = (18 + 22 + 20) / 3 = 20 Points/Sprint

다음 Sprint 계획:
  - 목표: 20 Points
  - 여유분: -2 Points (버퍼)
  - 실제 할당: 18 Points
```

---

### 5.5 Backlog Refinement (백로그 정제)

**Refinement란?**
Sprint 시작 전에 백로그의 Story들을 명확히 하고, 우선순위를 조정하는 활동입니다.

**진행 주기:**
- 빈도: 주 1회 (Sprint 중간)
- 시간: 1~2시간
- 참여자: PO, Scrum Master, 개발팀

**Refinement 아젠다:**

```
1. 신규 Story 리뷰 (30분)
   - 지난주 추가된 Story 확인
   - Acceptance Criteria 명확히
   - 질문 & 답변

2. 기존 Story 업데이트 (30분)
   - 요구사항 변경 반영
   - 기술 스택 검토
   - DoD 재확인

3. Story 분해 (30분)
   - 큰 Story를 작은 Story로 분리
   - Epic → Stories

4. 우선순위 조정 (30분)
   - 비즈니스 가치 vs 기술 리스크
   - 다음 Sprint 후보 선정
```

**Backlog 우선순위 기준 (RICE Framework):**

```
RICE Score = (Reach × Impact × Confidence) / Effort

Reach: 얼마나 많은 사용자에게 영향?
  - 5: 모든 사용자 (100%)
  - 3: 대부분 사용자 (50%+)
  - 1: 일부 사용자 (<10%)

Impact: 사용자 경험 개선 정도
  - 3: Massive (핵심 가치)
  - 2: High (중요)
  - 1: Medium (유용)
  - 0.5: Low (작은 개선)

Confidence: 확신 정도
  - 100%: 확실함 (데이터 있음)
  - 80%: 높음 (경험 있음)
  - 50%: 중간 (추측)

Effort: 개발 노력 (Story Points)
  - 1~21 Points
```

**예시 계산:**

**ST-10: BC 모델 구현**
```
Reach: 5 (모든 사용자가 사용)
Impact: 3 (핵심 기능)
Confidence: 80% (PyTorch 경험 있음)
Effort: 8 Points

RICE = (5 × 3 × 0.8) / 8 = 1.5
```

**ST-23: 센서 시뮬레이션**
```
Reach: 2 (고급 사용자만)
Impact: 2 (유용)
Confidence: 50% (처음 해봄)
Effort: 5 Points

RICE = (2 × 2 × 0.5) / 5 = 0.4
```

→ **ST-10이 우선순위 높음 (1.5 > 0.4)**

---

### 5.6 Sprint Planning 상세 가이드

**Planning 구조 (2시간):**

```
Part 1: Sprint Goal 설정 (30분)
  - "이번 Sprint에서 달성할 목표는?"
  - 예: "사용자가 첫 모델을 학습하고 결과를 확인한다"

Part 2: Story 선정 (60분)
  - Velocity 기반으로 Story 선택
  - Capacity 확인 (휴가, 회의 등)
  
  Team Capacity 계산:
    개발자 3명 × 8시간/일 × 10일 = 240시간
    회의 시간 (20시간) 제외 = 220시간
    1 Point = 3시간 가정
    → 220 / 3 = 73 Points

  선정된 Stories:
    - ST-9: 데이터 전처리 (5 Points)
    - ST-10: BC 모델 (8 Points)
    - ST-11: 모델 추론 (5 Points)
    총: 18 Points

Part 3: Task 분해 (30분)
  ST-10을 더 작은 Task로 분해:
    ├─ Task 1: 네트워크 아키텍처 정의 (2h)
    ├─ Task 2: Forward pass 구현 (3h)
    ├─ Task 3: Loss function 정의 (2h)
    ├─ Task 4: Optimizer 설정 (1h)
    ├─ Task 5: 학습 루프 작성 (5h)
    ├─ Task 6: 체크포인트 저장/로드 (3h)
    └─ Task 7: 단위 테스트 (4h)
    총: 20시간 (≈ 8 Points 타당)
```

**Sprint Commitment:**
```
우리는 다음 2주 동안 다음 Story들을 완료하여
"사용자가 첫 모델을 학습하고 결과를 확인"할 수 있게 만든다:

- ST-9: 데이터 전처리 파이프라인 (5 Points)
- ST-10: BC 모델 구현 (8 Points)
- ST-11: 모델 추론 및 시뮬레이션 적용 (5 Points)

Sprint Goal: AI Training MVP
Total Commitment: 18 Points
```

---

### 5.7 Daily Standup (일일 스크럼)

**3가지 질문 (각 1분):**

```
1. 어제 무엇을 했나요?
   ✅ "ST-10의 네트워크 아키텍처를 완성했습니다"

2. 오늘 무엇을 할 계획인가요?
   📋 "학습 루프를 작성하고 첫 학습을 돌려볼 예정입니다"

3. 장애물(Blocker)이 있나요?
   🚫 "GPU 메모리 부족 문제가 있어서 디버깅 중입니다"
      → Scrum Master가 해결 지원
```

**효과적인 Standup 규칙:**

- ⏰ **정해진 시간**: 매일 오전 10시 (15분 엄수)
- 🎯 **서서 진행**: 간결하게 유지
- 🚫 **문제 해결 금지**: 상세 논의는 회의 후 별도 진행
- 📊 **보드 활용**: Kanban 보드 앞에서 진행
- 🎤 **순서 정하기**: 토큰 패스 방식

**비동기 Standup (원격 팀):**

```
Slack 채널에 매일 오전 10시 전까지 작성:

@channel Daily Standup 2026.03.15

**Developer A**
어제: ST-10 네트워크 구현 완료
오늘: 학습 루프 작성
장애물: 없음

**Developer B**
어제: ST-9 정규화 로직 추가
오늘: 데이터 증강 기능 구현
장애물: H5 파일 로딩 속도 느림 (논의 필요)
```

---

### 5.8 Sprint Review & Retrospective

**Sprint Review (1시간):**

```
목적: Sprint 결과물을 이해관계자에게 시연

참석자:
  - 개발팀
  - Product Owner
  - Stakeholders (경영진, 영업, 마케팅)

구성:
  1. Sprint 목표 리뷰 (5분)
     "AI Training MVP 달성 여부"
  
  2. 완성된 Story 시연 (30분)
     - ST-9 Demo: 데이터 전처리 전/후 비교
     - ST-10 Demo: 학습 Loss 그래프
     - ST-11 Demo: 로봇이 스스로 작업 수행
  
  3. 미완성 Story 설명 (10분)
     - ST-12는 왜 못했는지
     - 다음 Sprint로 이월
  
  4. 피드백 수집 (15분)
     - 이해관계자 질문 & 제안
```

**Sprint Retrospective (1시간):**

```
목적: 프로세스 개선

참석자: 개발팀만 (안전한 공간)

형식: Start-Stop-Continue

┌─────────────────────────────────────────┐
│ START (시작할 것)                        │
├─────────────────────────────────────────┤
│ • 페어 프로그래밍 (복잡한 코드)          │
│ • 코드 리뷰 체크리스트 사용              │
│ • 주간 기술 세미나 (새로운 기술 공유)    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ STOP (중단할 것)                         │
├─────────────────────────────────────────┤
│ • 회의 중 노트북 사용 (집중력 저하)      │
│ • 급한 PR 리뷰 요청 (품질 저하)          │
│ • 문서 없이 코드만 작성                  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ CONTINUE (계속할 것)                     │
├─────────────────────────────────────────┤
│ • Daily Standup (효과적)                 │
│ • 자동화된 테스트 (빠른 피드백)          │
│ • Sprint Goal 명확히 (팀 정렬)           │
└─────────────────────────────────────────┘

Action Items:
  1. 페어 프로그래밍 시간 배정 (주 2회, 2시간)
     - Owner: Developer A
     - Deadline: 다음 Sprint 시작 전
  
  2. PR 체크리스트 작성
     - Owner: Developer B
     - Deadline: 이번 주 금요일
```

---

## 6. Definition of Done (DoD)

### 6.1 DoD 레벨별 정의

**Definition of Done이란?**
작업이 "완료"되었다고 선언하기 위해 충족해야 하는 명확한 기준입니다. 품질을 보장하고, 기술 부채를 최소화합니다.

**Master-Sim의 3단계 DoD:**

```
┌─────────────────────────────────────────┐
│ Story DoD (개별 기능 완료)               │
│ ↓                                       │
│ Sprint DoD (Sprint 전체 완료)            │
│ ↓                                       │
│ Release DoD (Production 배포 가능)       │
└─────────────────────────────────────────┘
```

---

### 6.2 Story Level DoD

**모든 Story가 완료되려면:**

#### ✅ **코드 품질**
- [ ] 코드가 작성되고 `main` 브랜치에 머지됨
- [ ] Linter 통과 (`black`, `ruff`, `mypy`)
- [ ] 코드 리뷰 최소 1명 Approve
- [ ] 모든 PR 댓글 해결됨
- [ ] 네이밍 컨벤션 준수 (PEP 8)

**예시:**
```python
# ❌ Bad
def f(x):
    return x*2

# ✅ Good
def calculate_joint_velocity(position: np.ndarray) -> np.ndarray:
    """
    Calculate joint velocity from position.
    
    Args:
        position: Joint positions (7,)
    
    Returns:
        Joint velocities (7,)
    """
    return position * 2.0
```

#### ✅ **테스트**
- [ ] Unit Test 작성 (커버리지 80% 이상)
- [ ] 모든 테스트 통과 (`pytest -v`)
- [ ] Edge Case 테스트 포함
- [ ] Regression Test (기존 기능 손상 없음)

**테스트 체크리스트:**
```python
def test_ik_controller():
    # 1. Happy Path (정상 경로)
    assert controller.solve_ik(target) is not None
    
    # 2. Edge Case (경계 조건)
    assert controller.solve_ik(unreachable_target) == None
    
    # 3. Error Handling (에러 처리)
    with pytest.raises(ValueError):
        controller.solve_ik(invalid_input)
    
    # 4. Performance (성능)
    start = time.time()
    controller.solve_ik(target)
    assert time.time() - start < 0.1  # 100ms 이내
```

#### ✅ **문서화**
- [ ] Docstring 작성 (모든 public 함수/클래스)
- [ ] README 업데이트 (새로운 기능 추가 시)
- [ ] API 문서 갱신 (Swagger/OpenAPI)
- [ ] CHANGELOG 작성

**Docstring 예시:**
```python
class BCPolicy(nn.Module):
    """
    Behavior Cloning policy network using MLP.
    
    This policy maps robot states to actions using a feedforward
    neural network trained via supervised learning.
    
    Args:
        state_dim: Dimension of state vector (e.g., 14 for 7-DoF robot)
        action_dim: Dimension of action vector (e.g., 7 for joint velocities)
        hidden_sizes: Tuple of hidden layer sizes (default: (256, 256))
    
    Example:
        >>> policy = BCPolicy(state_dim=14, action_dim=7)
        >>> state = torch.randn(1, 14)
        >>> action = policy(state)
        >>> action.shape
        torch.Size([1, 7])
    
    Note:
        - Input states should be normalized (mean=0, std=1)
        - Output actions are not bounded (apply tanh if needed)
    """
```

#### ✅ **기능 검증**
- [ ] Acceptance Criteria 모두 충족
- [ ] 로컬 환경에서 동작 확인
- [ ] Demo 가능 (PM에게 시연)
- [ ] 에러 로깅 추가

#### ✅ **보안**
- [ ] 민감 정보 하드코딩 없음 (API Key, Password)
- [ ] 환경 변수 사용 (`.env` 파일)
- [ ] SQL Injection 방어 (ORM 사용)
- [ ] Input Validation

**보안 체크리스트:**
```python
# ❌ Bad - API Key 하드코딩
API_KEY = "sk-1234567890abcdef"

# ✅ Good - 환경 변수 사용
import os
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

# ❌ Bad - SQL Injection 취약
query = f"SELECT * FROM users WHERE email = '{email}'"

# ✅ Good - Parameterized Query
query = "SELECT * FROM users WHERE email = %s"
cursor.execute(query, (email,))
```

#### ✅ **성능**
- [ ] 응답 시간 요구사항 충족
- [ ] 메모리 누수 없음
- [ ] CPU/GPU 사용률 정상

**성능 기준:**
| 기능 | 목표 | 측정 방법 |
|:---|:---|:---|
| IK 계산 | < 10ms | `time.time()` |
| 모델 추론 | < 5ms (GPU) | `torch.cuda.Event` |
| API 응답 | < 100ms (P95) | CloudWatch Metrics |
| 데이터 로딩 | < 1초 | `pytest --durations=10` |

---

### 6.3 Sprint Level DoD

**Sprint가 완료되려면:**

#### ✅ **Sprint Goal 달성**
- [ ] Sprint Goal에 명시된 목표 완료
- [ ] Sprint에 포함된 모든 Story 완료
- [ ] 미완성 Story는 다음 Sprint로 이월

**예시:**
```
Sprint Goal: "사용자가 첫 모델을 학습하고 결과를 확인한다"

완료 조건:
  ✅ ST-9: 데이터 전처리 완료
  ✅ ST-10: BC 모델 학습 완료
  ✅ ST-11: 시뮬레이션 적용 완료
  → Sprint Goal 달성!
```

#### ✅ **통합 테스트**
- [ ] E2E 테스트 통과
- [ ] 통합 환경에서 동작 확인
- [ ] Smoke Test 통과 (주요 기능)

**E2E 테스트 예시:**
```python
def test_full_training_pipeline():
    """전체 학습 파이프라인 E2E 테스트"""
    # 1. 데이터 로드
    dataset = BehaviorCloningDataset("data/test")
    assert len(dataset.states) > 0
    
    # 2. 모델 생성
    policy = BCPolicy(state_dim=14, action_dim=7)
    trainer = BCTrainer(policy)
    
    # 3. 학습 (짧게)
    for _ in range(10):
        states, actions = dataset.get_batch(32)
        loss = trainer.train_step(states, actions)
        assert loss > 0
    
    # 4. 추론
    test_state = np.random.randn(14)
    action = policy.predict(test_state)
    assert action.shape == (7,)
```

#### ✅ **문서화**
- [ ] Sprint 결과 문서화 (Sprint Review 노트)
- [ ] 기술 부채 기록 (TODO, FIXME)
- [ ] Known Issues 리스트 작성

#### ✅ **배포 준비**
- [ ] Staging 환경 배포 성공
- [ ] Migration Script 준비 (DB 변경 시)
- [ ] Rollback 계획 수립

---

### 6.4 Release Level DoD

**Production 배포가 가능하려면:**

#### ✅ **기능 완성도**
- [ ] 모든 MVP 기능 구현 완료
- [ ] 사용자 피드백 반영
- [ ] Critical Bug 0건

#### ✅ **성능 & 확장성**
- [ ] Load Testing 통과 (목표 RPS)
- [ ] Auto Scaling 동작 확인
- [ ] DB 인덱스 최적화

**Load Testing 기준:**
```bash
# Locust로 부하 테스트
locust -f tests/load_test.py --users 1000 --spawn-rate 10

목표:
  - RPS: 500 요청/초
  - P95 Latency: < 200ms
  - Error Rate: < 1%
```

#### ✅ **보안 & 컴플라이언스**
- [ ] 보안 스캔 통과 (Snyk, Trivy)
- [ ] OWASP Top 10 점검
- [ ] GDPR 준수 (개인정보 처리)
- [ ] 침투 테스트 (Penetration Test)

**보안 스캔 명령:**
```bash
# Docker 이미지 취약점 스캔
trivy image master-sim:latest --severity HIGH,CRITICAL

# Python 패키지 취약점 스캔
safety check --json

# License 검사
pip-licenses --summary
```

#### ✅ **모니터링 & 알림**
- [ ] CloudWatch 대시보드 구성
- [ ] 알림 규칙 설정 (Error Rate, Latency)
- [ ] On-call Rotation 설정
- [ ] Incident Response Plan 문서화

**알림 규칙 예시:**
```yaml
알림 종류:
  1. Critical (즉시)
     - 서비스 Down (Health Check 실패)
     - Error Rate > 5%
     - P99 Latency > 1초
     → Slack + PagerDuty 호출
  
  2. Warning (30분 내)
     - CPU > 80%
     - Memory > 85%
     - Disk > 90%
     → Slack 알림
  
  3. Info (일일 리포트)
     - 사용자 수, API 호출 수
     - 비용 현황
     → Email
```

#### ✅ **문서화**
- [ ] 사용자 가이드 작성
- [ ] API 문서 완성 (Swagger)
- [ ] 아키텍처 문서 (ADR)
- [ ] 트러블슈팅 가이드

#### ✅ **백업 & 복구**
- [ ] 데이터베이스 백업 자동화
- [ ] Disaster Recovery Plan
- [ ] Rollback 스크립트 준비
- [ ] 복구 훈련 (DR Drill) 실시

**백업 정책:**
```yaml
RDS 백업:
  - 자동 백업: 매일 03:00 (7일 보관)
  - 스냅샷: 주간 (30일 보관)
  - Point-in-Time Recovery: 활성화

S3 백업:
  - Versioning: 활성화
  - Lifecycle Policy: 90일 후 Glacier
  - Cross-Region Replication: 활성화
```

---

### 6.5 Code Quality 체크리스트

#### **Python 코드 스타일**

```bash
# 1. Black (자동 포맷팅)
black src/ tests/ --line-length 88

# 2. isort (Import 정렬)
isort src/ tests/ --profile black

# 3. Ruff (Fast Linter)
ruff check src/ tests/

# 4. mypy (Type Checking)
mypy src/ --strict
```

**설정 파일 (`pyproject.toml`):**
```toml
[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
```

#### **코드 복잡도**

```bash
# Radon (Cyclomatic Complexity)
radon cc src/ -a -nb

기준:
  A (1-5): 단순 (Good)
  B (6-10): 보통
  C (11-20): 복잡 (Refactor 고려)
  D (21-50): 매우 복잡 (반드시 Refactor)
  F (51+): 극도로 복잡 (재작성)
```

#### **코드 중복**

```bash
# PMD CPD (Copy-Paste Detector)
pmd cpd --minimum-tokens 50 --files src/

기준:
  - 중복 코드 < 5%
  - 중복 라인 < 50줄
```

---

### 6.6 Security Checklist

#### **인증 & 권한**
- [ ] 모든 API 엔드포인트에 인증 적용
- [ ] JWT 토큰 만료 시간 설정 (1시간)
- [ ] Refresh Token 구현
- [ ] Rate Limiting 적용 (분당 60 요청)
- [ ] CORS 설정 (허용된 Origin만)

#### **데이터 보호**
- [ ] 비밀번호 해싱 (bcrypt, Argon2)
- [ ] HTTPS 강제 (HSTS)
- [ ] 민감 데이터 암호화 (DB, S3)
- [ ] PII 로깅 금지

**예시:**
```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 비밀번호 해싱
hashed = pwd_context.hash("user_password")

# 검증
is_valid = pwd_context.verify("user_password", hashed)
```

#### **Input Validation**
- [ ] 모든 입력값 검증 (Pydantic)
- [ ] SQL Injection 방어
- [ ] XSS 방어 (HTML Sanitize)
- [ ] CSRF Token 사용

#### **의존성 보안**
- [ ] 정기적인 패키지 업데이트
- [ ] 취약점 스캔 (주간)
- [ ] License 검사

---

### 6.7 Performance Baseline

**Master-Sim 성능 목표:**

| 메트릭 | 목표 | 측정 위치 |
|:---|:---|:---|
| **API 응답 시간 (P95)** | < 100ms | CloudWatch |
| **API 응답 시간 (P99)** | < 200ms | CloudWatch |
| **모델 추론 (GPU)** | < 5ms | Application Log |
| **모델 추론 (CPU)** | < 50ms | Application Log |
| **데이터베이스 쿼리** | < 20ms | RDS Performance Insights |
| **페이지 로드 (FCP)** | < 1.5초 | Lighthouse |
| **페이지 로드 (LCP)** | < 2.5초 | Web Vitals |
| **시뮬레이션 FPS** | > 60 FPS | MuJoCo Profiler |

**측정 도구:**
```python
import time
from functools import wraps

def measure_time(func):
    """함수 실행 시간 측정 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed*1000:.2f}ms")
        return result
    return wrapper

@measure_time
def solve_ik(target_pos):
    # IK 계산
    pass
```

---

### 6.8 Documentation Standards

#### **코드 문서**
- [ ] 모든 모듈에 모듈 Docstring
- [ ] 모든 클래스에 클래스 Docstring
- [ ] 모든 public 함수에 함수 Docstring
- [ ] Google Style Docstring 사용

**예시:**
```python
def train_model(
    dataset: BehaviorCloningDataset,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
) -> BCPolicy:
    """
    Train Behavior Cloning model.
    
    This function trains a BC policy using supervised learning
    on demonstration data collected from teleoperation.
    
    Args:
        dataset: Dataset containing state-action pairs
        epochs: Number of training epochs (default: 100)
        batch_size: Mini-batch size (default: 256)
        learning_rate: Adam optimizer learning rate (default: 3e-4)
    
    Returns:
        Trained policy network
    
    Raises:
        ValueError: If dataset is empty or invalid
        RuntimeError: If CUDA out of memory
    
    Example:
        >>> dataset = BehaviorCloningDataset("data/demos")
        >>> policy = train_model(dataset, epochs=50)
        >>> policy.save("models/bc_policy.pt")
    
    Note:
        - Requires GPU for faster training
        - Checkpoints saved every 10 epochs
        - Early stopping if validation loss increases
    """
```

#### **README 구조**
```markdown
# Master-Sim

## 개요
1~2문장으로 프로젝트 설명

## 주요 기능
- 기능 1
- 기능 2

## 시작하기
### 설치
### 빠른 시작

## 사용법
### 기본 사용법
### 고급 사용법

## API 문서
링크

## 기여하기
PR 가이드

## 라이선스
MIT
```

#### **ADR (Architecture Decision Record)**
```markdown
# ADR-001: MuJoCo를 시뮬레이터로 선택

**날짜:** 2026-01-05
**상태:** Accepted

## Context (배경)
로봇 시뮬레이션을 위한 Physics Engine 선택 필요

## Decision (결정)
MuJoCo 3.1.2를 메인 시뮬레이터로 사용

## Rationale (근거)
- 빠른 속도 (1000+ FPS)
- 정확한 Contact Force 계산
- Apache 2.0 License (무료)
- 풍부한 Python API

## Alternatives (대안)
- PyBullet: 느림 (200 FPS)
- Isaac Sim: 무거움, GPU 필수

## Consequences (결과)
- Positive: 빠른 반복 실험
- Negative: 고품질 렌더링 어려움 (→ Phase 2에서 Isaac Sim 추가)
```

---

## 완성!

**DEVELOPMENT_PLAN.md**와 **SPRINT_PLAN.md**가 Executive Summary 및 Business Plan과 동일한 수준의 상세함으로 확장되었습니다.

### 📋 추가된 주요 내용

#### DEVELOPMENT_PLAN.md (기술 개발 계획서)
- **기술 스택 매트릭스**: 20개 이상의 핵심 기술과 선정 이유
- **시스템 아키텍처**: ASCII 다이어그램, 데이터 플로우
- **Phase별 상세 계획**: 8주 → 16주 → 24주 로드맵
- **기술적 도전과제**: 각 Phase별 예상 문제와 해결 방안
- **품질 관리**: 테스트 피라미드, 커버리지 목표, CI/CD
- **리스크 관리**: 기술/운영 리스크 매트릭스
- **성능 목표**: SLA, 응답 시간, 가용률

#### SPRINT_PLAN.md (Agile 실행 가이드)
- **상세 Git Flow**: 브랜치 전략, 커밋 컨벤션, PR 템플릿
- **Sprint 1~2 상세 Story**: 각 Story마다 기술 구현 코드, DoD, 예상 시간
- **Epic 관리 체계**: Epic → Story 분해 전략
- **Definition of Done**: Story/Sprint 레벨별 완료 기준
- **Velocity 측정**: Burndown Chart, Story Point Estimation 가이드
- **리스크 관리**: Sprint별 예상 리스크 및 대응

이제 진짜 "투자자가 납득할 수 있는" 수준의 완결성을 갖췄습니다!
