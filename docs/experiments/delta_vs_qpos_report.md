# Delta vs Qpos Action Label Experiment

요약
- qpos (absolute joint targets)로 학습한 모델(`expert_trained_20.npz`)은 평가에서 실패: 성공률 0% (50 episodes).
- delta (q_target - q_current)로 학습한 모델(`expert_trained_delta.npz`)은 성공률 100% (50 episodes), 평균 에피소드 길이 95 steps, 평균 최종 거리 0.0494.

실험 환경
- 데이터: IK 기반 전문가 시뮬레이션 (200 trajectories × 200 steps), obs flattened (20 dim)
- 모델: NumPy MLP (hidden_dims=[256,256])
- 학습: 100 epochs, batch_size=64, lr=1e-3

결과 (주요 지표)
- `expert_trained_20.npz` (qpos): success_rate=0.0%, mean_final_distance ≈ 3.12 (50 eps)
- `expert_trained_delta.npz` (delta): success_rate=100.0%, mean_final_distance ≈ 0.0494 (50 eps)

분석 요약
- 원인: 에이전트가 절대 관절 목표(qpos)를 직접 예측하면 현재 관절 상태와의 상호작용을 고려하지 못해 과도한 동작이나 불안정한 응답을 보임.
- delta 레이블은 "현재 상태에서의 변화량"을 학습하게 하여 폐회로 제어(online correction)에 더 적합함.

권장 적용
- 데이터 및 학습 파이프라인에서 **delta를 기본(action_type=delta)** 으로 사용하도록 권장.
- 기존 저장된 모델과 비교 시 delta가 더 안정적이고 성공률이 높음.

아카이브된 파일
- Qpos 모델: `models/expert_trained_20.npz`
- Delta 모델: `models/expert_trained_delta.npz`
- 평가: `demo_run_expert20_v5/`, `demo_run_expert_delta/`
- 실패 비디오/플롯: `analysis/failure_videos/`

다음 단계
- `delta`를 기본으로 코드 및 README 업데이트 (진행 중)
- 하이퍼파라미터 실험 및 문서화 (진행 예정)
