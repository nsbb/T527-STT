# 실험 관리 정책 — 모든 실험은 기록하고 보존한다

**날짜:** 2026-03-31
**배경:** QAT 실험 과정에서 데이터 덮어쓰기 사고 발생. 재발 방지를 위한 정책 수립.

---

## 1. 사고 경위

### 1.1 무슨 일이 있었나

AIHub 100k(95,000개, 84시간) 데이터로 QAT 학습 후 CER 5.3% 달성.
이후 AIHub 전체(4,092,104개, 4,356시간) 데이터로 QAT를 진행하면서
**같은 파일명(`aihub_train_manifest.json`)에 전체 데이터 manifest를 덮어씀.**

```
사고 전:
  aihub_train_manifest.json → 95,000개 (100k 실험용)
  aihub_val_manifest.json   → 5,000개

사고 후:
  aihub_train_manifest.json → 4,092,104개 (전체 데이터로 덮어씀)
  aihub_val_manifest.json   → 215,373개 (전체 데이터로 덮어씀)
```

### 1.2 왜 문제인가

- **100k QAT(CER 5.3%)와 동일한 데이터로 후속 실험을 비교할 수 없게 됨**
- seed(42)와 필터 조건이 같으면 이론적으로 동일 데이터가 나오지만, 코드 변경(필터 조건 추가 등)으로 100% 보장 불가
- 실험 재현성(reproducibility) 상실 — 과학적 실험의 기본 원칙 위반

### 1.3 원인

1. `prepare_manifest.py`에서 `MAX_SAMPLES`를 100,000 → 99,999,999로 변경
2. 출력 파일명을 바꾸지 않고 같은 이름으로 저장
3. 이전 파일을 백업하지 않음
4. **"기존 데이터를 보존해야 한다"는 의식 부재**

---

## 2. 실험 관리 정책

### 2.1 핵심 원칙

> **한 번 사용한 실험 데이터와 결과물은 절대 삭제하거나 덮어쓰지 않는다.**

### 2.2 파일 명명 규칙

모든 실험 관련 파일에는 **실험 ID**를 포함한다.

```
명명 패턴: {데이터소스}_{데이터크기}_{기법}_{날짜}_{버전}

예시:
  manifest:
    aihub_100k_train_260326.json        (100k 실험, 3/26 생성)
    aihub_full_train_260327.json         (전체 실험, 3/27 생성)
    custom_294_train_260327.json         (자체 데이터, 3/27 생성)

  출력 모델:
    conformer_qat_aihub100k_ep10_260326.nemo
    conformer_qat_aihubfull_ep09_260329.nemo
    conformer_qat_kd_aihub100k_ep10_260331.nemo

  출력 폴더:
    qat_aihub_100k_margin03_260326/
    qat_aihub_full_margin03_260327/
    qat_kd_aihub_100k_margin05_260331/
```

### 2.3 실험 기록 필수 항목

모든 실험을 시작할 때 아래 항목을 기록한다.

```yaml
# 실험 기록 템플릿
experiment:
  id: "qat_aihub_100k_margin03_260326"
  date: "2026-03-26"
  
  data:
    source: "train_base_4356hr.csv"
    filter:
      cer: 0.0
      duration_min: 0.5
      duration_max: 15.0
      exclude_paths: ["/nas03/data_audio/005.명령어 음성"]
    sampling:
      method: "random"
      seed: 42
      max_samples: 100000
    split:
      train: 95000
      val: 5000
      test: null
    manifest_files:
      train: "aihub_100k_train_260326.json"
      val: "aihub_100k_val_260326.json"
    total_hours: 84.63

  model:
    base: "stt_kr_conformer_ctc_medium.1.nemo"
    params: "122.5M (114M trainable, 7.6M frozen)"
    
  hyperparameters:
    lr: 1e-5
    batch_size: 16
    epochs: 10
    optimizer: "AdamW (weight_decay=0.01)"
    scheduler: "CosineAnnealingLR"
    gradient_clip: 1.0
    precision: "FP32"
    margin_target: 0.3
    margin_lambda: 0.1
    kd_lambda: null  # KD 미사용
    kd_temperature: null

  environment:
    server: "gpu-108"
    gpu: "RTX 6000 Ada × 1"
    slurm_job_id: 18337
    training_time: "~2 hours"

  results:
    val_loss_best: 0.1416 (epoch 4)
    val_loss_final: 0.151 (epoch 10)
    t527_cer:
      ep04_best_val: "100% (실패)"
      ep10_final: "5.3%"
    
  output_files:
    - "qat_aihub_output/conformer_qat_100k_84hr_final.nemo"
    - "qat_aihub_output/conformer_qat_100k_84hr_final.onnx"
    - "qat_aihub_output/conformer_qat_100k_best_ep04.nemo"
    - "qat_aihub_output/conformer_qat_epoch=03_val_loss=0.1446.ckpt"
    - "qat_aihub_output/conformer_qat_epoch=04_val_loss=0.1416.ckpt"
    - "qat_aihub_output/conformer_qat_epoch=05_val_loss=0.1438.ckpt"

  lessons:
    - "val_loss best ≠ T527 CER best"
    - "ep04 val_loss 0.1416(best)이 T527 CER 100%로 실패"
    - "ep10 val_loss 0.151(worse)이 T527 CER 5.3%로 성공"
```

### 2.4 데이터 보존 규칙

| 규칙 | 설명 |
|------|------|
| **절대 덮어쓰지 않기** | 새 실험은 새 파일명으로 생성. 이전 파일 유지 |
| **manifest 보존** | 실험에 사용된 manifest는 해당 실험 출력 폴더에 복사본 저장 |
| **스크립트 보존** | 실험에 사용된 학습 스크립트도 출력 폴더에 복사 |
| **로그 보존** | SLURM 로그, 학습 로그 전부 보존 |
| **삭제 금지** | checkpoint가 디스크를 많이 차지해도 실험 완료 전 삭제 금지 |

### 2.5 실험 시작 전 체크리스트

```
[ ] 새 출력 폴더명에 실험 ID 포함했는가?
[ ] manifest 파일명이 이전 실험과 겹치지 않는가?
[ ] 하이퍼파라미터를 전부 기록했는가?
[ ] 비교 대상 실험의 데이터가 동일한지 확인했는가?
[ ] 이전 실험 파일이 보존되어 있는가?
```

---

## 3. 실험 비교 시 주의사항

### 3.1 공정한 비교의 조건

두 실험을 비교하려면 **변경한 변수 외에 모든 조건이 동일**해야 함.

```
공정한 비교:
  실험 A: AIHub 100k, lr=1e-5, margin=0.3, epoch 10
  실험 B: AIHub 100k, lr=1e-5, margin=0.5, epoch 10
  → margin_target만 다르고 나머지 동일 → 비교 가능

불공정한 비교:
  실험 A: AIHub 100k (seed=42로 생성)
  실험 B: AIHub 100k (다른 시점에 다른 코드로 생성)
  → 데이터가 다를 수 있음 → 비교 불가
```

### 3.2 동일 데이터 보장 방법

1. **manifest 파일 자체를 보존** (가장 확실)
2. seed + 코드 버전 기록 (차선)
3. manifest의 md5 hash 기록 (검증용)

```bash
# manifest hash 기록
md5sum aihub_100k_train_260326.json >> experiment_log.txt
```

### 3.3 비교 결과 기록

```
비교: margin=0.3 vs margin=0.5
조건: AIHub 100k, lr=1e-5, batch=16, epoch 10, 동일 manifest

| 변수 | margin=0.3 | margin=0.5 |
|------|-----------|-----------|
| val_loss (best) | 0.1416 | ? |
| val_loss (final) | 0.151 | ? |
| T527 CER (final) | 5.3% | ? |
| margin_mean | ? | ? |
```

---

## 4. 과거 실험 복구 상태

### 4.1 복구 가능한 실험

| 실험 | manifest 보존 | 모델 보존 | 결과 보존 |
|------|-------------|---------|---------|
| AIHub 100k QAT | **X (덮어씀)** | O (.nemo, .onnx) | O (CER 5.3%) |
| 자체 데이터 QAT | O | O | O (CER ~6%) |
| AIHub 전체 QAT | O (현재 파일) | O | 테스트 필요 |

### 4.2 100k manifest 복구 시도

같은 조건(seed=42, CER=0.0, duration 0.5~15초, 005 경로 제외)으로 재생성.
`aihub_100k_train.json`, `aihub_100k_val.json`으로 저장됨.

**주의: 이전과 100% 동일하다고 보장할 수 없음.**
- prepare_manifest.py의 필터 조건이 중간에 변경되었을 수 있음 (005 경로 제외 등)
- CSV 읽는 순서가 OS/파일시스템에 따라 다를 수 있음

### 4.3 교훈

```
"데이터를 덮어쓰는 순간, 그 실험은 재현 불가능해진다."
```

---

## 5. 향후 적용

### 5.1 즉시 적용

- [x] 실험 관리 정책 문서 작성 (이 문서)
- [ ] 현재 보존된 manifest에 실험 ID 부여하여 정리
- [ ] 앞으로 모든 새 실험에 정책 적용

### 5.2 자동화 검토

- 실험 시작 시 자동으로 manifest 복사 + hash 기록하는 스크립트
- 출력 폴더에 실험 조건 자동 기록 (config.yaml)
- manifest 파일명 충돌 시 경고
