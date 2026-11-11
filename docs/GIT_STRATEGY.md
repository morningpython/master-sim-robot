# Master-Sim Git 작업 방식

## 현재 정책: main-first

Master-Sim은 현재 1인 개발 프로젝트다. 배포 가능한 첫 번째 결과물이
나오기 전까지는 **`main` 하나에서 직접 작업**한다. `develop`, `sprint/*`,
`feature/*` 브랜치는 만들지 않는다.

이 정책의 목적은 작업 맥락을 한 곳에 모으고, 아직 작은 연구용 MVP에
브랜치 운영 비용을 추가하지 않는 것이다.

```text
main
 ├─ [M0] 재현 가능한 개발 환경
 ├─ [M1] 실제 Peg-in-Hole 과제
 └─ [M2] 재현 가능한 BC 기준선
```

## main에서 작업하는 규칙

1. 작업 시작 전 `git status`가 깨끗한지 확인한다.
2. 하나의 논리적 작업 단위가 끝날 때마다 테스트와 정적 검사를 실행한다.
3. 검증 가능한 작은 단위로 커밋한다. 커밋 메시지는 Conventional Commits를
   사용하되, 과거 Story ID를 재사용하지 않는다.
4. 커밋 후 원격 `main`에 push한다.
5. 실험 산출물은 재현 명령, seed, 입력 데이터 버전, 결과 요약을 문서에
   남긴다. 대형 데이터셋·체크포인트는 Git에 새로 추가하지 않는다.

예시:

```bash
git add src/envs/panda_env.py tests/test_panda_env.py
git commit -m "feat(env): add peg insertion success condition"
git push origin main
```

## 품질 게이트

`main`에 커밋하기 전에, 가능한 범위에서 다음을 확인한다.

```bash
python -m pytest
ruff check src scripts tests
black --check src scripts tests
```

MuJoCo처럼 GUI 또는 시스템 의존성이 있는 검증은 headless 테스트와 별도로
기록한다. 실행하지 못한 검증은 커밋 본문 또는 작업 기록에 이유를 남긴다.

## 브랜치를 다시 도입하는 시점

아래 둘 중 하나가 충족될 때만 브랜치 정책을 다시 도입한다.

- 외부 사용자가 설치·실행할 수 있는 첫 릴리즈 후보가 생겼을 때
  (`v0.1.0` 등)
- 동시에 유지해야 할 안정 버전과 다음 개발 버전이 생겼을 때

그때의 최소 구조는 다음과 같다.

```text
main                 # 안정/배포 버전
develop              # 다음 릴리즈 통합선
release/vX.Y.Z       # 릴리즈 검증 및 문서화가 필요할 때만 생성
```

긴급 수정은 안정 버전이 실제로 존재한 뒤에만 `hotfix/<description>`으로
분리한다. 기능 브랜치와 스프린트 브랜치는 협업 또는 병렬 실험이 필요해질
때 다시 검토한다.

## 기존 브랜치 정리

`sprint/S2`는 이미 `main`에 포함되어 있고, `sprint/S3`도 2026-11-11에
`main`으로 fast-forward 되었다. `feat/docs-hyperparam`에는 `main`에 없는
변경이 없다. 원격 브랜치 삭제는 공유 참조를 제거하는 작업이므로, 원격에
push한 뒤 별도 확인 후 수행한다.
