# Master-Sim: Technical Development Master Plan

## 1. 프로젝트 개요 (Project Overview)
**프로젝트명:** Master-Sim (마스터 심)  
**기술 비전:** 산업용 로봇을 위한 세계 최고 수준의 Sim-to-Real 플랫폼 구축  
**핵심 목표:** 하드웨어 없이 시뮬레이션만으로 80% 이상 실전 성공률을 보이는 AI 모델 개발

---

## 2. 기술 스택 (Technology Stack)

### 2.1 Core Technologies

#### **Simulation & Physics Engine**
| 기술 | 용도 | 선정 이유 | License |
|:---|:---|:---|:---|
| **MuJoCo 3.1+** | 물리 시뮬레이션 (Phase 1) | - 빠른 연산 속도 (1000 FPS+)<br>- 정확한 접촉력 계산<br>- Apache 2.0 라이선스 (상업적 사용 가능) | Apache 2.0 |
| **NVIDIA Isaac Sim 4.0** | 고품질 렌더링 (Phase 2+) | - 실사 수준 그래픽 (고객 데모용)<br>- RTX 레이트레이싱<br>- ROS 2 통합 | Proprietary |
| **PyBullet** | 프로토타이핑 | - 설치 간편 (`pip install`)<br>- 커뮤니티 활발 | Zlib |

#### **AI & Machine Learning**
| 기술 | 용도 | 선정 이유 |
|:---|:---|:---|
| **PyTorch 2.2+** | 딥러닝 프레임워크 | - 연구 커뮤니티 압도적 점유율<br>- Dynamic Graph (디버깅 용이)<br>- TorchScript로 배포 최적화 |
| **Stable-Baselines3** | 강화학습 알고리즘 | - PPO, SAC 등 검증된 알고리즘<br>- 재현성(Reproducibility) 우수 |
| **Gymnasium (OpenAI Gym)** | RL 환경 표준 인터페이스 | - 산업 표준<br>- 다양한 벤치마크와 호환 |
| **Diffusion Policy** | 모방학습 | - State-of-the-art (2025 ICRA)<br>- Few-shot Learning에 강함 |

#### **Robotics & Control**
| 기술 | 용도 | 선정 이유 |
|:---|:---|:---|
| **ROS 2 Humble** | 로봇 미들웨어 | - 산업 표준 (ABB, FANUC 지원)<br>- 실시간 통신 (DDS) |
| **MoveIt 2** | 모션 플래닝 | - IK/FK Solver 내장<br>- Collision Detection |
| **pinocchio** | 로봇 기구학 | - 빠른 연산 속도 (C++ 백엔드)<br>- URDF 파일 지원 |

#### **Backend & Infrastructure**
| 기술 | 용도 | 선정 이유 |
|:---|:---|:---|
| **FastAPI** | REST API 서버 | - 비동기 I/O (async/await)<br>- 자동 API 문서 (Swagger) |
| **PostgreSQL 15** | 메인 데이터베이스 | - JSONB 타입 (메타데이터 저장)<br>- 트랜잭션 안정성 |
| **Redis 7** | 캐싱 & 작업 큐 | - 학습 job 관리<br>- 실시간 대시보드 데이터 캐싱 |
| **Docker & Kubernetes** | 컨테이너 오케스트레이션 | - GPU 자원 할당 자동화<br>- 멀티 테넌시 |

#### **Frontend & Visualization**
| 기술 | 용도 | 선정 이유 |
|:---|:---|:---|
| **React 18 + TypeScript** | 대시보드 UI | - 컴포넌트 재사용성<br>- 타입 안정성 |
| **Three.js** | 3D 시각화 (웹 뷰어) | - WebGL 기반<br>- GLTF/OBJ 로딩 |
| **Plotly.js** | 학습 차트 | - 실시간 업데이트<br>- 인터랙티브 |
| **Tailwind CSS** | 스타일링 | - 빠른 프로토타이핑<br>- Figma 디자인 시스템과 호환 |

### 2.2 Development Tools
- **Version Control:** Git + GitHub (Private Repo)
- **CI/CD:** GitHub Actions + ArgoCD
- **Monitoring:** Prometheus + Grafana
- **Logging:** ELK Stack (Elasticsearch, Logstash, Kibana)
- **Code Quality:** pytest, black, mypy, ESLint
- **Documentation:** Sphinx (Python), Docusaurus (웹 문서)

---

## 3. 시스템 아키텍처 (System Architecture)

### 3.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Web Dashboard│  │ REST API     │  │ CLI Tool     │           │
│  │ (React)      │  │ (Postman)    │  │ (Python)     │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    API Gateway (FastAPI)                          │
│  - Authentication (JWT)                                           │
│  - Rate Limiting                                                  │
│  - Request Routing                                                │
└──────────────────────────┬───────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌─────────────────┐ ┌─────────────┐ ┌────────────────┐
│ Simulation      │ │ Training    │ │ Data           │
│ Service         │ │ Service     │ │ Service        │
│                 │ │             │ │                │
│ - MuJoCo Worker │ │ - GPU Queue │ │ - S3/MinIO     │
│ - Scene Builder │ │ - Model Mgr │ │ - PostgreSQL   │
└─────────────────┘ └─────────────┘ └────────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                   ┌───────────────┐
                   │ Message Queue │
                   │ (Redis/Celery)│
                   └───────────────┘
```

### 3.2 Data Flow (학습 파이프라인)

```
1. User Upload (CAD File) 
   → 
2. Scene Generator (URDF/XML 자동 생성)
   →
3. Simulation Worker (MuJoCo에서 데이터 수집)
   → 
4. Data Preprocessor (정규화, 증강)
   →
5. Training Worker (PyTorch GPU 학습)
   →
6. Model Validator (시뮬레이션에서 성능 테스트)
   →
7. Model Registry (버전 관리 및 배포)
```

---

## 4. Phase별 상세 개발 계획

### Phase 1: MVP - "Proof of Concept" (Week 1~8)

#### 목표
시뮬레이션 환경에서 **Peg-in-Hole 작업을 80% 성공률로 수행하는 AI 모델** 완성.

#### 주요 마일스톤
| Week | 목표 | 산출물 |
|:---:|:---|:---|
| 1-2 | 환경 구축 및 로봇 모델 로드 | MuJoCo 시뮬레이터에 Panda 로봇 렌더링 |
| 3-4 | 수동 제어 인터페이스 개발 | 마우스 IK 제어 + 데이터 로거 |
| 5-6 | Behavior Cloning 학습 | 첫 AI 모델 (50% 성공률) |
| 7-8 | 모델 최적화 및 데모 영상 제작 | 유튜브 업로드용 고품질 영상 |

#### 기술적 도전과제 및 해결 방안

**Challenge 1: 접촉력 불안정성**
- 문제: Peg가 Hole에 닿는 순간 물리 엔진이 폭발(Explosion)하거나 관통(Penetration)
- 해결:
  - MuJoCo의 `solref`, `solimp` 파라미터 튜닝
  - Contact Damping 값 조정
  - 타임스텝 축소 (0.001초 → 0.0001초)

**Challenge 2: 학습 데이터 부족**
- 문제: 초기에는 전문가(숙련공) 데이터가 없음
- 해결:
  - 개발자가 직접 100회 시연 (초기 seed data)
  - Data Augmentation (노이즈 추가, 시점 변경)
  - Self-Play로 데이터 자가 생성

**Challenge 3: 학습 속도 저하**
- 문제: GPU 없이 CPU만으로는 학습 시간이 12시간 이상 소요
- 해결:
  - AWS EC2 g5.xlarge 인스턴스 사용 ($1.2/hour)
  - Mixed Precision Training (FP16) 적용
  - Gradient Checkpointing으로 메모리 절약

#### 성공 기준 (Definition of Done)
- [ ] 시뮬레이션 성공률 80% 이상 (100회 테스트 중 80회 성공)
- [ ] 평균 조립 시간 5초 이내
- [ ] 모델 파일 크기 50MB 이하 (배포 용이성)
- [ ] 데모 영상 조회수 10K 달성 (1개월 내)

---

#### Progressive scaling & hyperparameter experiments
- **목표:** 현재 안정적인 baseline(Delta labels)을 유지하면서 점진적 확장(데이터/배치/모델 용량)을 통해 성능·일반화 개선을 검증합니다.
- **실행 원칙:** 1) 한 번에 한 변수만 변경, 2) 각 실험은 재현 가능한 명령과 아티팩트를 함께 저장, 3) 측정은 성공률/평균 최종 거리/에피소드 길이로 통일.
- **최근 실행:** Hyperparameter experiments A (512x512), B (lr=5e-4), C (batch=128) 완료. 결과와 아티팩트는 `docs/experiments/hyperparam_results.md` 및 `analysis/exp_C_batch128_demo_50/`에 저장되었습니다.
- **권장 단계(점진적 확장):**
  1. Epochs +10~20 (학습 안정성 확인)
  2. Batch 크기 2×씩 증가 (성능/속도 trade-off 측정)
  3. 데이터셋 크기 +50% 단위로 확대 (일반화 확인)
  4. Model capacity 증분 (hidden dims +25~100%) 및 조기 중단/정규화 확인
  5. 각 실험 후 문서화 (`docs/experiments/`) 및 모델 아카이브 태깅

### Phase 2: Validation - "Sim-to-Real Transfer" (Week 9~16)

#### 목표
실제 로봇(Franka Emika Panda)에서 **성공률 70% 이상** 달성.

#### 주요 작업
1. **Domain Randomization 구현**
   - 조명 변화 (밝기 50~150%)
   - 카메라 노이즈 (Gaussian σ=0.05)
   - 물체 마찰계수 (μ=0.3~0.7)
   - 로봇 관절 토크 오차 (±10%)
   
2. **NVIDIA Isaac Sim 마이그레이션**
   - MuJoCo 환경을 USD 포맷으로 변환
   - RTX 레이트레이싱 활성화
   - Replicator API로 합성 이미지 생성 (10K장)

3. **실제 로봇 테스트베드 구축**
   - 하드웨어 구매: Franka Emika Panda ($25K)
   - ROS 2 드라이버 설치 및 캘리브레이션
   - 안전 펜스 설치 (충돌 방지)

#### 기술적 도전과제

**Challenge 1: Reality Gap**
- 문제: 시뮬레이션에서는 완벽한데 실제 로봇에서는 20% 성공률
- 해결:
  - System Identification (실제 로봇의 물성치 측정)
  - Residual Learning (시뮬레이션 + 실제 차이를 학습)
  - Online Fine-tuning (실제 환경에서 100회 추가 학습)

**Challenge 2: 카메라 캘리브레이션**
- 문제: 시뮬레이션의 완벽한 카메라 vs 실제 왜곡/노이즈
- 해결:
  - OpenCV Camera Calibration
  - 렌즈 왜곡 보정 (Undistortion)
  - 시뮬레이션에 동일한 왜곡 패턴 적용

#### 성공 기준
- [ ] 실제 로봇 성공률 70% 이상
- [ ] Zero-shot Transfer (추가 학습 없이) 성공률 50% 이상
- [ ] 고객사 1곳 PoC 완료 (무료 테스트)

---

### Phase 3: Production - "B2B SaaS Platform" (Week 17~24)

#### 목표
고객이 직접 사용할 수 있는 **클라우드 플랫폼** 런칭.

#### 주요 기능 개발
1. **SimMaster Studio (웹 기반 시뮬레이션 에디터)**
   - Drag & Drop으로 로봇, 부품 배치
   - 물리 파라미터 GUI 편집
   - 실시간 미리보기 (Three.js WebGL)

2. **PolicyHub (모델 마켓플레이스)**
   - 카테고리별 사전 학습 모델 목록
   - 1-Click 다운로드 (ONNX, TorchScript)
   - 라이선스 관리 (Node-Locked, Floating)

3. **Training Dashboard**
   - 실시간 학습 그래프 (Loss, Reward, Success Rate)
   - TensorBoard 임베딩
   - 이메일 알림 (학습 완료 시)

4. **API Integration**
   ```python
   # Customer Code Example
   from mastersim import Client
   
   client = Client(api_key="sk-...")
   job = client.train(
       task="peg-in-hole",
       cad_file="my_connector.step",
       episodes=10000
   )
   
   job.wait()  # 학습 완료 대기
   model = job.download()  # 모델 다운로드
   ```

#### 인프라 요구사항
- **컴퓨팅:**
  - 8× NVIDIA A100 GPU (학습용)
  - 4× CPU 서버 (시뮬레이션 워커)
- **스토리지:**
  - 20TB SSD (학습 데이터)
  - 100TB HDD (장기 보관)
- **네트워크:**
  - 10Gbps 전용선
  - CloudFlare CDN (모델 배포)

#### 성능 목표 (SLA)
- 시뮬레이션 응답 시간: < 100ms
- 학습 작업 대기 시간: < 5분
- API 가용률: 99.9% (월 43분 다운타임 허용)
- 모델 다운로드 속도: > 50MB/s

#### 보안 요구사항
- [ ] SOC 2 Type II 인증 취득
- [ ] 고객 데이터 암호화 (AES-256)
- [ ] API 키 탈취 방지 (Rate Limiting, IP Whitelist)
- [ ] 모델 워터마킹 (불법 복제 추적)

#### 성공 기준
- [ ] Paying Customer 10개사 확보
- [ ] MRR (Monthly Recurring Revenue) $50K 달성
- [ ] NPS (Net Promoter Score) 50 이상

---

## 5. 품질 관리 및 테스트 전략

### 5.1 테스트 피라미드

```
         /\
        /  \  E2E Tests (5%)
       /    \  - 실제 로봇 통합 테스트
      /______\  
     /        \ Integration Tests (25%)
    /          \ - API + DB + Simulation
   /____________\ 
  /              \ Unit Tests (70%)
 /________________\ - 함수 레벨 단위 테스트
```

### 5.2 테스트 커버리지 목표
- **Backend:** 90% 이상 (pytest-cov)
- **Frontend:** 80% 이상 (Jest)
- **Simulation:** 95% 이상 (핵심 물리 로직)

### 5.3 자동화 테스트
- **Nightly Build:** 매일 자정 전체 테스트 실행
- **PR Gate:** Pull Request 생성 시 자동 테스트 (통과 시에만 Merge 가능)
- **Smoke Test:** 배포 후 5분 내 기본 기능 검증

### 5.4 성능 테스트
- **Load Test:** Locust로 동시 사용자 100명 시뮬레이션
- **Stress Test:** GPU 100% 사용 시 안정성 검증
- **Endurance Test:** 72시간 연속 학습 작업 실행

---

## 6. CI/CD 파이프라인

### 6.1 Continuous Integration (GitHub Actions)

```yaml
# .github/workflows/ci.yml (예시)
name: CI Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install Dependencies
        run: pip install -r requirements.txt
      - name: Run Tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload Coverage
        uses: codecov/codecov-action@v3
```

### 6.2 Continuous Deployment
- **Staging:** `develop` 브랜치 푸시 시 자동 배포 (staging.mastersim.ai)
- **Production:** `main` 브랜치 태그 생성 시 배포 (app.mastersim.ai)
- **Rollback:** 30초 내 이전 버전 복구 가능

---

## 7. 리스크 관리 (Technical Risks)

| 리스크 | 확률 | 영향도 | 대응 전략 |
|:---|:---:|:---:|:---|
| MuJoCo 라이선스 정책 변경 | 낮음 | 높음 | PyBullet로 마이그레이션 가능하도록 추상화 레이어 유지 |
| GPU 공급 부족 (A100 품귀) | 중간 | 높음 | AWS/GCP 멀티 클라우드 전략, H100 대안 |
| Sim-to-Real Gap 해소 실패 | 중간 | 치명적 | 실제 로봇 테스트베드 조기 확보, 외부 연구진 협업 |
| 핵심 개발자 이탈 | 낮음 | 높음 | 문서화 철저, Pair Programming, 스톡옵션 |

---

## 8. 버전 관리 및 릴리즈 계획

### 8.1 Semantic Versioning
- **v0.1.0 (Alpha):** MVP 완성 (내부 테스트)
- **v0.5.0 (Beta):** 첫 고객 PoC
- **v1.0.0 (GA):** 정식 출시 (Production Ready)

### 8.2 릴리즈 사이클
- **Major:** 6개월마다 (큰 기능 추가)
- **Minor:** 매월 (작은 기능, 개선)
- **Patch:** 수시 (버그 수정)

---

## 9. 문서화 전략
- **Architecture Decision Records (ADR):** 주요 기술 결정 사항 기록
- **API Reference:** OpenAPI 3.0 스펙 자동 생성
- **User Guide:** Docusaurus 기반 인터랙티브 튜토리얼
- **Troubleshooting:** FAQ, Known Issues 페이지

## 4. Sprint Plan (스프린트 계획)
상세한 스프린트 계획과 User Story는 [docs/SPRINT_PLAN.md](SPRINT_PLAN.md) 문서를 참조하십시오.

### 4.1 Agile Workflow & Rules
*   **Sprint Cycle:** 2주 단위 스프린트.
*   **Branch Strategy:**
    *   `main`: 배포 가능한 안정 버전.
    *   `sprint/S{number}`: 해당 스프린트 개발 브랜치 (예: `sprint/S1`).
    *   `feat/{story-id}`: 개별 스토리 개발 브랜치 (예: `feat/ST-1`).
*   **Commit & PR:**
    *   스토리 하나가 완료될 때마다 Commit.
    *   스토리들이 모여 에픽의 일부가 완성되면 PR을 통해 Merge.
*   **UI/UX:** Figma 중심의 UI 설계를 선행하고, 이를 바탕으로 대시보드 및 시각화 도구 개발.
