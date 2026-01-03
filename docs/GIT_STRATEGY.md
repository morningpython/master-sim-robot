# Master-Sim: GitHub & Git Strategy Guide

## 목차
1. [Git Workflow 개요](#1-git-workflow-개요)
2. [Branch Strategy](#2-branch-strategy)
3. [Commit Convention](#3-commit-convention)
4. [Pull Request Process](#4-pull-request-process)
5. [Code Review Guidelines](#5-code-review-guidelines)
6. [Release Management](#6-release-management)
7. [GitHub Project 활용](#7-github-project-활용)
8. [보안 및 접근 권한](#8-보안-및-접근-권한)

---

## 1. Git Workflow 개요

### 1.1 핵심 원칙
- **"Never break main"**: `main` 브랜치는 항상 배포 가능한 상태 유지
- **"One Story, One Commit"**: 하나의 Story는 하나의 명확한 커밋으로 완성
- **"Sprint = Branch"**: Sprint가 곧 브랜치, Story는 커밋으로 관리
- **"Preserve History"**: 완료된 Sprint 브랜치는 삭제하지 않고 영구 보존
- **"Test Before Merge"**: CI 통과 없이는 Merge 불가

### 1.2 전체 흐름 (Single Developer → Team 확장 대비)

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Sprint Planning                                    │
│  - Story 정의 (ST-1, ST-2, ...)                            │
│  - Story Point 할당                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Branch 생성                                        │
│  main → develop → sprint/S1 → feat/ST-1-description        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 개발 & Commit                                     │
│  - Story 작업 진행                                          │
│  - [ST-1] feat: ... 형식으로 커밋                          │
│  - 로컬 테스트 수행                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Push & Pull Request                                │
│  - feat/ST-1 → sprint/S1 으로 PR 생성                       │
│  - CI 자동 실행 (테스트, 린트)                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Code Review (Self or Peer)                        │
│  - 체크리스트 확인                                          │
│  - 피드백 반영                                              │
│  - Approve 획득                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 6: Merge                                              │
│  - Squash & Merge (Sprint 브랜치로)                        │
│  - Story를 "Done"으로 이동 (GitHub Project)                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 7: Sprint 종료 시                                     │
│  - sprint/S1 → develop 으로 PR                             │
│  - Release Note 작성                                        │
│  - develop → main (릴리즈 시)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Branch Strategy

### 2.1 브랜치 구조 (Sprint-Based Flow)

```
main (Protected)
  ├── Production 배포 버전 (v1.0.0, v1.1.0, ...)
  │
develop (Protected)
  ├── 다음 릴리즈 통합 브랜치
  │
  ├── sprint/S1 (Sprint 1 작업 공간, 영구 보존)
  │     │
  │     ├── [ST-1] feat: setup mujoco environment (commit)
  │     ├── [ST-2] feat: add basic viewer (commit)
  │     ├── [ST-3] feat: load panda robot (commit)
  │     ├── [ST-4] feat: implement joint control (commit)
  │     └── [ST-5] feat: add peg-hole scene (commit)
  │     → Sprint 완료 시 PR: sprint/S1 → develop
  │
  ├── sprint/S2 (Sprint 2 작업 공간, 영구 보존)
  │     │
  │     ├── [ST-6] feat: implement ik control (commit)
  │     ├── [ST-7] feat: add gripper control (commit)
  │     └── [ST-8] feat: implement data logger (commit)
  │     → Sprint 완료 시 PR: sprint/S2 → develop
  │
  ├── hotfix/critical-bug-name (긴급 수정, main에서 분기)
  │
  └── release/v1.0.0 (릴리즈 준비)
```

### 2.2 브랜치 네이밍 규칙

| 브랜치 타입 | 패턴 | 예시 | 용도 |
|:---|:---|:---|:---|
| **main** | `main` | `main` | Production 배포 |
| **develop** | `develop` | `develop` | 개발 통합 |
| **sprint** | `sprint/S{number}` | `sprint/S1` | 2주 스프린트 작업 (Story들을 커밋으로 관리) |
| **hotfix** | `hotfix/{severity}-{desc}` | `hotfix/critical-memory-leak` | 긴급 수정 |
| **release** | `release/v{major}.{minor}.{patch}` | `release/v1.0.0` | 릴리즈 준비 |

**Note:** Feature 브랜치는 사용하지 않습니다. 모든 Story는 Sprint 브랜치에서 직접 커밋됩니다.

### 2.3 브랜치 보호 규칙 (GitHub Settings)

#### **main 브랜치**
```yaml
Protection Rules:
  - Require pull request reviews: ✅ (최소 1명)
  - Require status checks to pass: ✅
    - CI/CD Pipeline
    - Code Coverage (>= 80%)
  - Require branches to be up to date: ✅
  - Include administrators: ✅
  - Restrict who can push: ✅ (Release Manager만)
  - Allow force pushes: ❌
  - Allow deletions: ❌
```

#### **develop 브랜치**
```yaml
Protection Rules:
  - Require pull request reviews: ✅
  - Require status checks to pass: ✅
  - Allow force pushes: ❌
  - Allow deletions: ❌
```

### 2.4 브랜치 라이프사이클

```bash
# Sprint 시작 시
git checkout develop
git pull origin develop
git checkout -b sprint/S1
git push -u origin sprint/S1

# Story 작업 (ST-1 시작)
# Sprint 브랜치에서 직접 작업
# 코드 작성...

# Story 완료 시 커밋 (하나의 Story = 하나의 커밋)
git add .
git commit -m "[ST-1] feat: setup mujoco environment

- Add mujoco 3.1.2 to requirements.txt
- Create Python 3.11 virtual environment
- Verify import success in tests

Closes #1"

# 커밋 후 즉시 Push
git push origin sprint/S1

# 다음 Story 작업 (ST-2)
# 같은 브랜치에서 계속 작업...
git add .
git commit -m "[ST-2] feat: add basic mujoco viewer"
git push origin sprint/S1

# Sprint의 모든 Story 완료 후
# GitHub Web UI에서 PR 생성
# sprint/S1 → develop

# PR Merge 완료 후
# ⚠️ 브랜치는 삭제하지 않고 보존 (히스토리 추적용)
git checkout develop
git pull origin develop

# 다음 Sprint 시작
git checkout -b sprint/S2
git push -u origin sprint/S2
```

---

## 3. Commit Convention

### 3.1 Commit Message 형식 (Conventional Commits)

```
[ST-{id}] {type}({scope}): {subject}

{body}

{footer}
```

**예시:**
```
[ST-1] feat(env): install mujoco dependencies

- Add mujoco 3.1.2 to requirements.txt
- Create .venv with Python 3.11
- Verify import success in test

Closes #1
```

### 3.2 Type 분류

| Type | 설명 | 예시 |
|:---|:---|:---|
| **feat** | 새로운 기능 추가 | `[ST-1] feat: add mujoco viewer` |
| **fix** | 버그 수정 | `[ST-3] fix: resolve robot floating issue` |
| **docs** | 문서 변경만 | `[ST-5] docs: update README installation guide` |
| **style** | 코드 포맷팅 (기능 변화 없음) | `[ST-2] style: format with black` |
| **refactor** | 리팩토링 (기능 변화 없음) | `[ST-4] refactor: extract controller class` |
| **test** | 테스트 추가/수정 | `[ST-1] test: add mujoco import test` |
| **chore** | 빌드, 설정 파일 변경 | `[ST-1] chore: update .gitignore` |
| **perf** | 성능 개선 | `[ST-6] perf: optimize IK calculation` |

### 3.3 Scope (선택사항)

- `env`: 환경 설정
- `sim`: 시뮬레이션
- `ctrl`: 컨트롤러
- `ai`: AI 모델
- `ui`: 사용자 인터페이스
- `api`: REST API
- `db`: 데이터베이스
- `ci`: CI/CD

### 3.4 Subject 작성 규칙
- 50자 이내 (제목줄)
- 명령형 동사로 시작 (add, fix, update, remove)
- 첫 글자 소문자
- 마침표 없음

### 3.5 Body 작성 규칙 (선택사항)
- 72자마다 줄바꿈
- "왜" 변경했는지 설명
- Bullet Points (-, *) 사용 가능

### 3.6 Footer
- `Closes #123`: Issue 종료
- `Refs #456`: Issue 참조
- `BREAKING CHANGE:`: Breaking Change 명시

### 3.7 Atomic Commit 원칙
- **하나의 커밋 = 하나의 논리적 변경**
- 너무 큰 커밋은 분할
- 커밋 메시지만 보고도 변경 내용 파악 가능해야 함

**Bad Example:**
```bash
git commit -m "update code"  # ❌ 너무 모호함
```

**Good Example:**
```bash
git commit -m "[ST-1] feat(env): add mujoco 3.1.2 to requirements.txt"  # ✅ 명확함
```

---

## 4. Pull Request Process

### 4.1 PR 생성 전 체크리스트
- [ ] 로컬에서 모든 테스트 통과 (`pytest`)
- [ ] 코드 포맷팅 완료 (`black .`, `isort .`)
- [ ] 타입 체크 통과 (`mypy src/`)
- [ ] 불필요한 파일 제거 (`.pyc`, `__pycache__`)
- [ ] `.gitignore` 확인
- [ ] Commit 메시지 규칙 준수

### 4.2 PR 제목 형식
```
[ST-{id}] {Type}: {Short Description}
```

**예시:**
- `[ST-1] Feat: Setup MuJoCo Environment`
- `[ST-6] Feat: Implement IK Controller`
- `[BUG-42] Fix: Resolve Collision Detection Issue`

### 4.3 PR 템플릿 (`.github/pull_request_template.md`)

```markdown
## Story / Issue
- Story ID: ST-{번호}
- Sprint: S{번호}
- Epic: {Epic 이름}

## 변경 사항 요약
<!-- 무엇을 변경했는지 3줄 이내로 요약 -->

## 주요 변경 파일
- [ ] `src/envs/basic_viewer.py` - MuJoCo 뷰어 추가
- [ ] `requirements.txt` - mujoco 3.1.2 추가
- [ ] `tests/test_environment.py` - import 테스트

## 테스트 방법
<!-- 리뷰어가 직접 테스트할 수 있는 명령어 -->
```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 테스트 실행
pytest tests/test_environment.py

# 3. 데모 실행
python src/envs/basic_viewer.py
```

## 예상 결과
<!-- 정상 동작 시 어떤 결과가 나와야 하는지 -->
- MuJoCo 뷰어 창이 뜨고, 회색 바닥이 렌더링됨
- 마우스로 시점 조작 가능

## 스크린샷 / 데모 영상 (UI 관련 시)
<!-- 이미지 또는 GIF 첨부 -->
![demo](https://...)

## 체크리스트
- [ ] 로컬 테스트 통과 (`pytest`)
- [ ] 코드 포맷팅 완료 (`black`, `isort`)
- [ ] 타입 체크 통과 (`mypy`)
- [ ] 문서 업데이트 (Docstring, README)
- [ ] Breaking Change 없음 (있다면 명시)
- [ ] Acceptance Criteria 모두 충족

## 관련 링크
- Figma 디자인: https://...
- 참고 문서: https://...
```

### 4.4 PR 생성 후 흐름 (Sprint 완료 시)

```
1. Sprint의 모든 Story 완료 후 PR 생성
   sprint/S1 → develop
   ↓
2. CI 자동 실행 (GitHub Actions)
   - pytest (Unit Tests)
   - black --check (Formatting)
   - mypy (Type Check)
   - codecov (Coverage Report)
   ↓
3. CI 통과 확인
   ✅ All checks passed
   ❌ CI Failed → Sprint 브랜치에서 수정 후 재푸시
   ↓
4. Sprint Review
   - Sprint Goal 달성 확인
   - 모든 Story의 DoD 충족 확인
   - Demo 준비 (선택)
   ↓
5. Approve 획득
   ✅ Approved (Self-review 또는 Peer)
   ↓
6. Merge
   - "Create a Merge Commit" 선택 (히스토리 보존)
   - Merge 커밋 메시지: "Merge sprint/S1: Simulation Foundation"
   ↓
7. ⚠️ 브랜치 보존
   - GitHub에서 브랜치 삭제 옵션 비활성화
   - sprint/S1 브랜치는 영구 보존 (히스토리 추적용)
   - 로컬에서도 삭제하지 않음
   ↓
8. 다음 Sprint 준비
   - develop에서 sprint/S2 브랜치 생성
```

### 4.5 Merge 전략

**Master-Sim에서 사용하는 방식: Create a Merge Commit**

```bash
# Sprint 브랜치의 각 Story는 개별 커밋으로 관리
sprint/S1:
  - [ST-1] feat: setup mujoco environment
  - [ST-2] feat: add basic mujoco viewer
  - [ST-3] feat: load panda robot
  - [ST-4] feat: implement joint control
  - [ST-5] feat: add peg-hole scene

# develop으로 Merge 시 모든 커밋 히스토리 보존
develop:
  - Merge sprint/S1: Simulation Foundation
    - [ST-1] feat: setup mujoco environment
    - [ST-2] feat: add basic mujoco viewer
    - [ST-3] feat: load panda robot
    - [ST-4] feat: implement joint control
    - [ST-5] feat: add peg-hole scene
```

**장점:**
- 각 Story별 작업 내용이 명확히 기록됨
- Sprint 단위 히스토리와 Story 단위 히스토리 모두 추적 가능
- 특정 Story만 Revert 가능 (git revert <commit-hash>)
- 브랜치 구조가 단순해짐 (feat 브랜치 불필요)

**사용 시기:**
- `sprint/S1` → `develop`: **Create a Merge Commit** ✅
- `develop` → `main`: **Create a Merge Commit** (릴리즈 태그)

---

## 5. Code Review Guidelines

### 5.1 Self-Review 체크리스트 (1인 개발 시)

**코드 품질:**
- [ ] 함수/클래스명이 명확한가?
- [ ] 주석이 필요한 복잡한 로직이 있는가?
- [ ] 중복 코드가 없는가?
- [ ] 에러 핸들링이 적절한가?

**테스트:**
- [ ] 모든 함수에 단위 테스트가 있는가?
- [ ] Edge Case를 테스트했는가?
- [ ] Coverage가 떨어지지 않았는가?

**성능:**
- [ ] 불필요한 루프가 없는가?
- [ ] 메모리 누수 가능성은 없는가?

**보안:**
- [ ] API 키, 비밀번호가 커밋되지 않았는가?
- [ ] 사용자 입력 검증이 있는가?

### 5.2 Peer Review 체크리스트 (팀 확장 시)

**리뷰어의 책임:**
- 24시간 내 리뷰 완료 (긴급 시 4시간)
- 건설적 피드백 ("이렇게 하면 어떨까요?" 형식)
- 코드뿐만 아니라 설계 관점에서도 검토

**리뷰 우선순위:**
1. **P0 (Blocker)**: 버그, 보안 이슈 → 반드시 수정
2. **P1 (Major)**: 성능, 가독성 → 수정 권장
3. **P2 (Minor)**: 네이밍, 스타일 → 선택 사항

**Comment 예시:**
```markdown
# ❌ Bad Comment
"이 코드 이상해요"

# ✅ Good Comment
"이 루프에서 O(n²) 복잡도가 발생할 수 있습니다. 
dict를 사용하여 O(n)으로 최적화하면 어떨까요?

\```python
# 제안 코드
lookup = {item.id: item for item in items}
result = lookup.get(target_id)
\```
"
```

### 5.3 리뷰 승인 기준
- [ ] CI 전체 통과
- [ ] 코드가 Story의 Acceptance Criteria 충족
- [ ] 테스트 커버리지 유지/향상
- [ ] Breaking Change가 없거나, 있다면 문서화됨
- [ ] 리뷰어가 "LGTM (Looks Good To Me)" 코멘트

---

## 6. Release Management

### 6.1 Semantic Versioning

```
v{MAJOR}.{MINOR}.{PATCH}
```

- **MAJOR**: Breaking Change (v1.0.0 → v2.0.0)
- **MINOR**: 새로운 기능 추가, 하위 호환 (v1.0.0 → v1.1.0)
- **PATCH**: 버그 수정 (v1.0.0 → v1.0.1)

### 6.2 릴리즈 프로세스

```bash
# Sprint 종료 → develop에 Merge 완료
# v1.0.0 릴리즈 준비

# 1. Release 브랜치 생성
git checkout develop
git pull origin develop
git checkout -b release/v1.0.0

# 2. 버전 정보 업데이트
# - pyproject.toml, __version__.py 등
echo "1.0.0" > VERSION

# 3. Release Note 작성
# CHANGELOG.md 업데이트

# 4. 최종 테스트
pytest
black --check .
mypy src/

# 5. Release 브랜치 → main PR
# GitHub에서 PR 생성 및 Merge

# 6. Tag 생성 (GitHub Releases)
git checkout main
git pull origin main
git tag -a v1.0.0 -m "Release v1.0.0: MVP Launch"
git push origin v1.0.0

# 7. main → develop 백머지
git checkout develop
git merge main
git push origin develop
```

### 6.3 CHANGELOG.md 형식

```markdown
# Changelog

## [1.0.0] - 2026-03-01

### Added
- [ST-1] MuJoCo simulation environment setup
- [ST-6] IK-based mouse control
- [ST-10] Behavior Cloning model training

### Fixed
- [BUG-42] Fixed collision detection crash

### Changed
- [ST-4] Improved joint control PD gains

### Removed
- Deprecated PyBullet support

## [0.5.0] - 2026-02-15 (Beta)
...
```

---

## 7. GitHub Project 활용

### 7.1 Project Board 구조 (Kanban)

```
┌─────────────┬─────────────┬─────────────┬──────────┬──────────┐
│  Backlog    │  Todo       │ In Progress │  Review  │   Done   │
├─────────────┼─────────────┼─────────────┼──────────┼──────────┤
│ ST-15       │ ST-1        │ ST-6        │ ST-2     │ ST-3     │
│ ST-16       │ ST-4        │             │          │ ST-5     │
│ ST-17       │ ST-7        │             │          │          │
└─────────────┴─────────────┴─────────────┴──────────┴──────────┘
```

**컬럼 정의:**
- **Backlog**: Sprint 백로그 (우선순위 정렬)
- **Todo**: 이번 Sprint에서 할 일
- **In Progress**: 현재 작업 중 (1인당 최대 2개)
- **Review**: PR 생성됨, 리뷰 대기
- **Done**: Merge 완료, Sprint 내 완료

### 7.2 Issue Template

**`.github/ISSUE_TEMPLATE/user_story.md`:**
```markdown
---
name: User Story
about: Create a new user story
title: '[ST-XX] '
labels: story
assignees: ''
---

## User Story
**As a** {role}  
**I want to** {goal}  
**So that** {benefit}

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Story Points
{1, 2, 3, 5, 8, 13}

## Epic
Epic #{number}

## Technical Notes
<!-- 구현 힌트, 참고 자료 -->
```

### 7.3 Labels 체계

| Label | Color | 용도 |
|:---|:---:|:---|
| `story` | 🟦 Blue | User Story |
| `bug` | 🟥 Red | 버그 |
| `epic` | 🟪 Purple | Epic |
| `P0` | 🟥 Red | 최우선 |
| `P1` | 🟧 Orange | 높음 |
| `P2` | 🟨 Yellow | 보통 |
| `enhancement` | 🟩 Green | 개선 |
| `documentation` | 📘 Blue | 문서 |
| `blocked` | 🟥 Red | 차단됨 |

---

## 8. 보안 및 접근 권한

### 8.1 Repository 설정
- **Visibility**: Private (초기), Public (오픈소스 전환 시)
- **Collaborators**: Founder (Owner), 팀원 (Write)

### 8.2 Secrets 관리
- **GitHub Secrets 사용**: Settings → Secrets and variables
- **민감 정보 목록:**
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `OPENAI_API_KEY`
  - `CODECOV_TOKEN`

**절대 커밋 금지:**
- `.env` 파일
- `credentials.json`
- API 키, 비밀번호

**`.gitignore` 필수 항목:**
```
.env
*.key
*.pem
credentials.json
secrets/
```

### 8.3 Code Scanning
```yaml
# .github/workflows/codeql.yml
name: "CodeQL"
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: github/codeql-action/init@v2
        with:
          languages: python
      - uses: github/codeql-action/analyze@v2
```

---

## 9. 실전 시나리오

### 시나리오 1: 새로운 Sprint 시작

```bash
# 월요일 아침, Sprint Planning 완료
# Sprint 2 시작 (ST-6, ST-7, ST-8)

# 1. Sprint 브랜치 생성
git checkout develop
git pull origin develop
git checkout -b sprint/S2
git push -u origin sprint/S2

# 2. GitHub Project에서 Story들을 "Todo"로 이동
# ST-6, ST-7, ST-8

# 3. 첫 번째 Story (ST-6) 작업
# sprint/S2 브랜치에서 직접 코드 작성
# src/controllers/ik_controller.py 개발...

# 4. ST-6 완료 시 커밋
git add src/controllers/ik_controller.py tests/test_ik.py
git commit -m "[ST-6] feat: implement IK controller

- Add inverse kinematics solver using Jacobian
- Implement mouse position to end-effector mapping
- Achieve 60fps IK calculation speed

Closes #6"
git push origin sprint/S2

# 5. ST-7 작업 시작 (같은 브랜치에서 계속)
# src/controllers/gripper_controller.py 개발...

# 6. ST-7 완료 시 커밋
git add src/controllers/gripper_controller.py
git commit -m "[ST-7] feat: add gripper control"
git push origin sprint/S2

# 7. 모든 Story 완료 후 PR 생성
# GitHub UI: sprint/S2 → develop
```

### 시나리오 2: Hotfix 처리

```bash
# Production에서 Critical Bug 발견

# 1. main에서 hotfix 브랜치 생성
git checkout main
git pull origin main
git checkout -b hotfix/critical-memory-leak

# 2. 버그 수정
# ...

# 3. 테스트
pytest tests/

# 4. main에 직접 PR (긴급)
# hotfix/critical-memory-leak → main

# 5. Merge 후 Tag
git tag -a v1.0.1 -m "Hotfix: Memory leak in IK solver"

# 6. develop에도 백머지
git checkout develop
git merge main
git push origin develop
```

---

## 10. FAQ

**Q1. 1인 개발인데 브랜치 전략이 너무 복잡하지 않나요?**
> A: 초기에는 복잡해 보일 수 있지만, 팀 확장 시 문화가 이미 정착되어 있으면 온보딩이 쉽습니다. 또한 나중에 과거 작업을 추적하기 훨씬 수월합니다.

**Q2. Sprint 브랜치를 삭제하지 않으면 너무 많아지지 않나요?**
> A: Sprint는 2주 단위이므로 연간 약 26개 브랜치만 생성됩니다. 이는 전혀 많지 않으며, 과거 Sprint의 작업 내용을 언제든 확인할 수 있어 오히려 장점입니다. GitHub에서 필터링(Active branches)으로 현재 작업 중인 브랜치만 볼 수 있습니다.

**Q3. CI가 실패하면 어떻게 하나요?**
> A: 절대 "Skip CI"하지 마세요. 실패 원인을 파악하고 수정한 뒤 재푸시합니다. CI는 품질의 최후 방어선입니다.

**Q4. Sprint 중간에 긴급한 작업이 들어오면?**
> A: Hotfix 브랜치를 사용하거나, Sprint Backlog를 재조정합니다. Product Owner(본인)와 협의 후 결정.

---

## 부록: 유용한 Git 명령어

```bash
# 브랜치 상태 한눈에 보기
git branch -a

# 최근 10개 커밋 로그
git log --oneline -10

# 특정 Story 관련 커밋만 보기
git log --grep="ST-1"

# Unstaged 변경사항 임시 저장
git stash
git stash pop

# 실수로 잘못된 브랜치에 커밋한 경우
git checkout correct-branch
git cherry-pick <commit-hash>

# PR Merge 전 최신 develop 반영
git checkout feat/ST-1
git fetch origin
git rebase origin/develop

# Conflict 해결 후
git rebase --continue
```

---

**작성일:** 2026-01-03  
**버전:** 1.0.0  
**관리자:** Founder
