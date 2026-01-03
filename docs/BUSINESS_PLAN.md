# Master-Sim: Business Plan

## 1. 사업 개요 (Business Overview)

### 1.1 회사 정보
- **법인명:** Master-Sim Inc. (가칭)
- **업종:** 인공지능(AI) 소프트웨어 / 로봇 자동화 솔루션
- **주요 사업:** Physical AI를 위한 합성 학습 데이터 및 사전 학습 모델 개발·판매
- **설립 예정일:** 2026년 1분기

### 1.2 사업 배경 및 기회
2026년 현재, 글로벌 로봇 시장은 다음과 같은 패러다임 전환기를 맞이하고 있습니다:

- **Physical AI 시대 도래**
  - McKinsey 보고서(2025): "2030년까지 산업용 AI 로봇 시장 $260B 규모 예상"
  - Embodied AI(몸을 가진 AI)가 ChatGPT급 파급력을 가질 것으로 전망
  
- **데이터가 곧 경쟁력**
  - OpenAI, Google DeepMind도 로봇 학습 데이터 확보 경쟁 중
  - 하지만 "고품질 산업 데이터"는 여전히 희소 자원
  
- **한국의 구조적 이점**
  - OECD 국가 중 제조업 비중 2위 (28.4%, 2024 기준)
  - 숙련공의 평균 경력 18.7년 (미국 6.2년의 3배)

---

## 2. 시장 분석 (Market Analysis)

### 2.1 TAM (Total Addressable Market) - 전체 시장 규모
**글로벌 산업용 로봇 소프트웨어 시장**
- 2026년: $18.5B
- 2030년(예상): $62.3B
- CAGR: 35.4%

### 2.2 SAM (Serviceable Addressable Market) - 진입 가능 시장
**정밀 조립 자동화 소프트웨어 (미국 + 한국)**
- 2026년: $4.2B
- 타겟 산업: 전기차 배터리($1.8B), 반도체 장비($1.2B), 전자제품 조립($1.2B)

### 2.3 SOM (Serviceable Obtainable Market) - 실제 목표 시장
**초기 3년 내 확보 가능 시장 (Peg-in-Hole 중심)**
- 2026년: $15M (0.36% 점유율)
- 2027년: $80M (1.9%)
- 2028년: $250M (5.3%)

### 2.4 시장 트렌드
1. **Sim-to-Real의 산업 표준화**
   - Tesla AI Day 2025: "우리 로봇의 99%는 시뮬레이션에서 학습됨"
   - NVIDIA Omniverse 기업 사용자 전년 대비 340% 증가

2. **Few-shot Learning 수요 급증**
   - 기존: 로봇 티칭에 200시간 소요
   - 시장 요구: 1시간 이내로 단축

3. **B2B SaaS 모델의 일반화**
   - 로봇 업계도 "소프트웨어 구독" 모델 채택 (예: ABB RobotStudio Cloud)

---

## 3. 고객 분석 (Customer Segmentation)

### 3.1 Primary Customer: 미국 로봇 SI (System Integrator)
**페르소나: "Frustrated Frank"**
- 직책: 로봇 통합 프로젝트 매니저
- 회사 규모: 직원 50~500명, 연매출 $20M~$200M
- Pain Points:
  - "하드웨어는 납품했는데, 고객 공장에서 작동을 안 해요"
  - "프로그래머가 6개월 동안 티칭만 하고 있어요"
  - "우리는 로봇 전문가지, AI 전문가가 아닌데 AI 요구사항이 늘어나고 있어요"
- 구매 결정 요인:
  - ROI 명확성 (투자 회수 기간 1년 이내)
  - 레퍼런스 (검증된 사례)
  - 기술 지원 (온보딩, 트러블슈팅)

**시장 규모:**
- 미국 내 로봇 SI: 약 1,200개사
- 이 중 정밀 조립 프로젝트 수행 업체: 320개사 (TAM)
- 초기 타겟: 연 매출 $50M 이상 상위 80개사 (SOM)

### 3.2 Secondary Customer: 한국 제조 대기업 및 스마트팩토리 SI
**페르소나: "Efficiency-driven Eric"**
- 직책: 생산기술 팀장 / 스마트팩토리 추진 본부장
- 소속: 현대모비스, LG에너지솔루션, 삼성전자 협력사 등
- Pain Points:
  - "인건비가 매년 8%씩 오르는데 자동화가 안 따라와요"
  - "숙련공이 은퇴하면 그 노하우가 다 사라져요"
  - "로봇 도입했는데 불량률이 오히려 올랐어요"
- 구매 결정 요인:
  - 생산성 향상 (Takt Time 단축)
  - 품질 안정성 (불량률 0.1% 이하)
  - 국산 솔루션 선호 (정부 보조금, 커스터마이징 용이)

**시장 규모:**
- 한국 제조업 중 로봇 자동화 대상 기업: 약 4,500개사
- 이 중 정밀 조립 라인 보유: 1,200개사
- 초기 타겟: 배터리, 반도체 협력사 50개사

### 3.3 Future Customer: AI 연구기관 및 로봇 스타트업
- MIT CSAIL, Stanford AI Lab 등 연구소 (데이터셋 구매)
- Figure AI, 1X Technologies 등 휴머노이드 스타트업 (파트너십)

---

## 4. 제품 및 서비스 (Product & Service)

### 4.1 Core Product 1: **"SimMaster Engine"** (시뮬레이션 플랫폼)
**설명:**  
MuJoCo/Isaac Sim 기반의 고정밀 물리 시뮬레이션 엔진으로, 고객사의 작업 환경을 디지털 트윈으로 재현합니다.

**핵심 기능:**
- **CAD Import:** 고객의 부품 도면(.STEP, .STL)을 자동으로 물리 모델로 변환
- **Material Library:** 한국 제조업 현장의 실제 물성치 DB (마찰계수, 탄성계수 등)
- **Multi-Physics:** 접촉력, 미끄럼, 변형까지 정밀 시뮬레이션
- **Cloud Rendering:** 브라우저에서 실시간으로 시뮬레이션 확인 가능

**가격:**
- 기본: $1,500/월 (1개 환경)
- 프로: $5,000/월 (무제한 환경 + 우선 지원)

### 4.2 Core Product 2: **"PolicyHub"** (사전 학습 모델 마켓플레이스)
**설명:**  
다운로드 즉시 사용 가능한 사전 학습 AI 모델 라이브러리.

**제공 모델 (Phase 1):**
- Peg-in-Hole (원형, 사각형, 육각)
- Screw Fastening (M3, M5 볼트 체결)
- Connector Insertion (USB-C, 배터리 커넥터)

**가격:**
- 모델당 연간 라이선스: $12,000 ~ $50,000
- 무제한 로봇 대수: +$20,000/년

### 4.3 Premium Service: **"Custom Brain Factory"**
**설명:**  
고객의 고유 작업을 위한 맞춤형 AI 모델 개발 서비스.

**프로세스:**
1. 요구사항 분석 (작업 사양서, 성공 기준 정의)
2. 디지털 트윈 구축 (2주)
3. AI 학습 및 검증 (4주)
4. Sim-to-Real 테스트 (고객 현장, 2주)
5. 최종 모델 납품 + 6개월 유지보수

**가격:**
- 기본 패키지: $150,000
- 엔터프라이즈: $500,000+

---

## 5. 수익 모델 (Revenue Model)

### 5.1 수익 구조 (2026~2028 예상)

| 수익원 | 2026 | 2027 | 2028 |
|:---|---:|---:|---:|
| **SaaS 구독 (SimMaster)** | $180K | $1.2M | $4.5M |
| **모델 라이선스 (PolicyHub)** | $240K | $2.8M | $12.0M |
| **맞춤형 개발 (Custom)** | $300K | $3.5M | $10.0M |
| **데이터 판매** | $80K | $600K | $2.0M |
| **합계 (ARR)** | $800K | $8.1M | $28.5M |

### 5.2 고객 생애 가치 (LTV)
**SaaS + 라이선스 고객 기준:**
- 평균 계약 기간: 3.5년
- 연간 ARPU (Average Revenue Per User): $35,000
- Churn Rate: 15%
- **LTV = $35K × 3.5 × (1 - 0.15) = $104K**

### 5.3 고객 획득 비용 (CAC)
- 마케팅 비용: $8K/고객 (컨퍼런스, 웨비나)
- 세일즈 비용: $12K/고객 (PoC, 데모)
- **총 CAC = $20K**

**LTV/CAC Ratio = 5.2** (목표: 3 이상 → **건강한 수준**)

---

## 6. 마케팅 및 세일즈 전략 (Go-to-Market Strategy)

### 6.1 Phase 1: Thought Leadership (2026 Q1~Q2)
**목표:** 업계에 Master-Sim의 존재를 각인시킴

**전술:**
- **유튜브 데모 영상:** "방구석에서 산업용 로봇 훈련시키기" (조회수 목표: 50K)
- **학회 논문 발표:** ICRA, RSS 등 로봇 학회에서 Sim-to-Real 성과 발표
- **오픈소스 전략:** SimMaster의 경량 버전을 GitHub에 공개 (Star 목표: 5K)

### 6.2 Phase 2: Direct Sales (2026 Q3~Q4)
**목표:** 첫 10개 Paying Customer 확보

**전술:**
- **Account-Based Marketing (ABM):**
  - 미국 Top 50 로봇 SI 리스트 작성 → 개별 맞춤 제안서 발송
  - LinkedIn 광고 ($10K/월, CTO/엔지니어링 매니저 타겟팅)
- **PoC (Proof of Concept) 무료 제공:**
  - 고객 환경에서 2주 무료 테스트
  - 성공 시 계약, 실패 시 No-Charge
- **파트너십:**
  - NVIDIA Inception Program 가입 (마케팅 지원, GPU 크레딧)
  - ABB, FANUC 등 로봇 제조사와 Certified Partner 체결

### 6.3 Phase 3: Product-Led Growth (2027~)
**목표:** 고객이 스스로 확산시키는 구조

**전술:**
- **Freemium 모델 도입:**
  - SimMaster 기본 기능 무료 제공 → Premium 전환율 25% 목표
- **Customer Success 팀 구성:**
  - 고객 성공 사례를 케이스 스터디로 제작 → 웹사이트 게시
- **Community Building:**
  - Discord/Slack에서 사용자 커뮤니티 운영

---

## 7. 경쟁 분석 (Competitive Landscape)

### 7.1 직접 경쟁자

| 회사 | 제품 | 강점 | 약점 | 차별화 전략 |
|:---|:---|:---|:---|:---|
| **Agility Robotics** | Digit (휴머노이드) + 자체 AI | 하드웨어 통합 | 범용 로봇 → 정밀 작업 약함 | 우리는 SW만 집중, 어떤 HW에도 적용 가능 |
| **Covariant** | AI for Warehouse Picking | 컴퓨터 비전 강함 | 힘 제어(Force) 데이터 없음 | 우리는 촉각·힘 데이터 전문 |
| **Dexterity** | Robotic Pick & Pack | 실제 배치 레퍼런스 多 | 시뮬레이션 플랫폼 없음 | 우리는 플랫폼 판매 (고객이 직접 학습 가능) |

### 7.2 간접 경쟁자
- **NVIDIA Isaac Sim:** 시뮬레이터만 제공, AI 모델은 고객이 직접 개발해야 함 → 우리는 턴키(Turnkey) 솔루션
- **ROS (Robot Operating System):** 오픈소스 미들웨어 → 전문 지식 필요, 우리는 No-Code

---

## 8. 운영 계획 (Operations Plan)

### 8.1 기술 인프라
- **컴퓨팅:**
  - 초기: AWS EC2 G5 인스턴스 (NVIDIA A10G GPU × 4) - $5K/월
  - 확장: On-premise GPU 서버 구축 (NVIDIA H100 × 8) - $250K 투자
- **시뮬레이션 엔진:**
  - MuJoCo (무료, Phase 1)
  - NVIDIA Isaac Sim (Enterprise License $50K/년, Phase 2)

### 8.2 인력 계획

| 시점 | 포지션 | 인원 | 역할 |
|:---|:---|:---:|:---|
| **2026 Q1** | Founder (CTO) | 1 | 전체 개발, MVP 완성 |
| **2026 Q2** | Robotics Engineer | 1 | Sim-to-Real 검증 |
| | AI Research Scientist | 1 | 강화학습 알고리즘 최적화 |
| **2026 Q3** | Sales Engineer (US) | 1 | 미국 시장 B2B 세일즈 |
| | DevOps Engineer | 1 | 클라우드 인프라 관리 |
| **2027** | Customer Success Manager | 2 | 고객 온보딩 및 기술 지원 |
| | Product Designer | 1 | UI/UX (대시보드) |

---

## 9. 재무 계획 (Financial Plan)

### 9.1 초기 자본금 (Seed Funding 목표: $1.5M)

**사용 계획:**
- R&D (개발자 인건비, GPU 서버): $600K (40%)
- Sales & Marketing (미국 출장, 컨퍼런스): $450K (30%)
- 운영비 (사무실, 법무, 회계): $300K (20%)
- 예비비: $150K (10%)

### 9.2 손익 예상 (P&L Projection)

| 항목 | 2026 | 2027 | 2028 |
|:---|---:|---:|---:|
| **매출** | $800K | $8.1M | $28.5M |
| **매출원가 (COGS)** | $240K | $1.6M | $5.1M |
| **매출총이익** | $560K | $6.5M | $23.4M |
| **판관비 (SG&A)** | $1.2M | $4.0M | $10.0M |
| **R&D** | $500K | $1.5M | $3.5M |
| **영업이익 (EBITDA)** | **-$1.14M** | **$1.0M** | **$9.9M** |

**BEP (손익분기점):** 2027년 Q2 예상

### 9.3 자금 조달 로드맵
- **2026 Q1:** Seed ($1.5M) - Angel Investors, Accelerator
- **2027 Q1:** Series A ($10M) - Sequoia, Andreessen Horowitz 타겟
- **2028:** Series B ($30M+) - 글로벌 확장

---

## 10. 리스크 및 대응 (Risk Management)

### 10.1 기술 리스크
**리스크:** Sim-to-Real Gap (시뮬레이션에서 잘 되는데 현실에서 실패)
**대응:**
- Domain Randomization 강화
- 실제 로봇 1대를 사무실에 확보하여 상시 검증
- 성공률 80% 미만 시 "Money-back Guarantee" 제공

### 10.2 시장 리스크
**리스크:** 대기업(ABB, FANUC)이 자체 AI 솔루션 출시
**대응:**
- First-mover Advantage 활용 (18개월 앞서 시장 진입)
- 대기업과는 경쟁보다 협력 (OEM 파트너십)

### 10.3 규제 리스크
**리스크:** AI 학습 데이터 관련 규제 (EU AI Act 등)
**대응:**
- 합성 데이터 사용 → 개인정보/저작권 이슈 없음을 마케팅 포인트로 활용

---

## 11. Exit Strategy (투자 회수 전략)

### 시나리오 1: M&A (인수합병)
**예상 인수자:**
- NVIDIA (로봇 AI 생태계 확장)
- Siemens (디지털 트윈 포트폴리오 강화)
- Google DeepMind (Physical AI 역량 확보)

**예상 시점:** 2028~2029년  
**예상 가치:** $150M ~ $500M (매출 배수 10~20x 기준)

### 시나리오 2: IPO
**조건:** ARR $100M 이상 달성 시  
**예상 시점:** 2030년 이후

---

## 12. 결론 (Conclusion)
Master-Sim은 **"제조업 강국 한국의 숙련 노하우"**라는 독보적 자산을 **"시뮬레이션 기술"**로 무기화하여,  
**"미국 로봇 시장의 데이터 갈증"**을 해소하는 **타이밍과 전략이 완벽히 맞아떨어진 사업**입니다.

2026년은 Physical AI의 원년이 될 것이며, 이 시장에서 **데이터 공급자(Data Provider)**로서의 입지를 선점하는 것이 곧 10년 후의 시장 지배력을 결정할 것입니다.
