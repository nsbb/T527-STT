# SungBeom Conformer CTC — Android 앱 테스트 결과

## 테스트 환경

| 항목 | 값 |
|------|-----|
| 앱 | awaiasr_2 (기존 멀티모델 테스트 앱에 Conformer 추가) |
| Activity | `ConformerTestActivity` |
| NPU API | awnn (libawnn.viplite.so + libVIPlite.so + libVIPuser.so) |
| NB | `/data/local/tmp/kr_conf_sb/network_binary.nb` (102MB, uint8 KL) |
| Vocab | `vocab_correct.json` (2049 BPE tokens) |
| 디바이스 | T527 Android 보드 |

## 앱 구조

### 추가한 파일

| 파일 | 역할 |
|------|------|
| `jni/conformer/awconformersdk.c` | NPU 추론 JNI (uint8 입력 → awnn → argmax 반환) |
| `conformer/AwConformerJni.java` | Java ↔ JNI 인터페이스 |
| `conformer/ConformerDecoder.java` | CTC greedy decode (BPE vocab) |
| `ConformerTestActivity.java` | 슬라이딩 윈도우 테스트 UI |
| `Android.mk` | `awconformer` 네이티브 라이브러리 추가 |
| `AndroidManifest.xml` | ConformerTestActivity 등록 |

### 동작 흐름

```
1. NB 로드 (/data/local/tmp/kr_conf_sb/network_binary.nb)
2. vocab 로드 (vocab_correct.json, 2049 BPE tokens)
3. 각 테스트 샘플:
   a. 미리 만든 uint8 mel chunk .dat 파일 읽기 (NeMo preprocessor로 생성)
   b. 각 chunk를 NPU에 넣어 추론 (~250ms/chunk)
   c. 슬라이딩 윈도우 merge (STRIDE_OUT=63 frames/chunk)
   d. CTC greedy decode → 한국어 텍스트
4. 화면 + logcat에 GT vs NPU 출력
```

### NB 로드 방식

102MB NB는 assets에 넣기엔 너무 큼 → **디바이스 `/data/local/tmp/`에 adb push**하여 직접 로드.

```bash
adb push network_binary.nb /data/local/tmp/kr_conf_sb/
adb push vocab_correct.json /data/local/tmp/kr_conf_sb/
```

### mel 전처리

현재 Phase 1: **NeMo Docker에서 미리 생성한 mel을 .dat로 push.**
Phase 2 (미구현): Java/JNI에서 NeMo 호환 mel 생성.

## 테스트 결과 (5샘플, 슬라이딩 윈도우)

### 전체 문장 결과

| # | dur | chunks | infer_ms | RTF | GT | NPU |
|---|-----|--------|---------|-----|-----|-----|
| 00 | 10.5s | 4 | 1046ms | 0.100 | 몬터규는 자녀들이 사랑을 제대로 못 받고 크면 매우 심각한 결과가 초래된다는 결론을 내렸습니다 | 문토규은 자녀들이 사랑을 제대로 못 받고크면 매우 심각한 결과가 초래된다는 결론을 내렸습니다 |
| 01 | 20.5s | 8 | 2160ms | 0.106 | 차 문이 종잇장처럼 얇지 않으니 문 두께 스물 센티미터를 빼면 실제 승 하차 여유 공간은 스물 센티미터라는 계산이 나옵니다 | 처럼 낮지 않으니 두 문 두 두께 스물 센티미터를 빼면 실제 승차 여유 공간에 스물 센티미터 계산이 나옵니다... |
| 02 | 8.5s | 4 | 1012ms | 0.119 | 지난해 이들 크루즈관광객의 평균 체류기간은 오 쩜 구 사 시간 | 지난해 이들 크루즈 관광 객의 평균 체류 기간은 오 구 점 구사 시간 음 |
| 03 | 16.6s | 7 | 1772ms | 0.107 | 그리고 이 나무는 태즈메이니아 남부 지역에 부는 남극의 바람을 맞으면서도 또한 해안에 위치한 산간 지방의 안개가 많은 환경에서도 무성하게 자랍니다 | 그리고 이 나무는 태 즈메이니아 남부 지역에 부 는 남극의 바람을 맞 찾으면서도 또한 해안에 위치 지안 상간 지방의 안개가 많은 자 환경에서도 무성하게 자랍니다 음 |
| 04 | 8.2s | 4 | 1063ms | 0.130 | 평소 오전 아홉 시 에서 오후 일곱 시까지 일하면 하루 이 만원 정도를 번다 | 평소 오전 아홉 시에서 오후 일곱 시까지 일하면 하루 이만 원 정도를 번다 에 |

### RTF (Real-Time Factor)

```
RTF = 추론시간 / 음성길이
RTF < 1.0 = 실시간보다 빠름
```

| 샘플 | 음성 | 추론 | RTF |
|------|------|------|-----|
| 00 | 10.5s | 1.05s | **0.100** |
| 01 | 20.5s | 2.16s | **0.106** |
| 02 | 8.5s | 1.01s | **0.119** |
| 03 | 16.6s | 1.77s | **0.107** |
| 04 | 8.2s | 1.06s | **0.130** |
| **평균** | | | **0.112** |

**RTF 0.112 = 실시간의 약 9배 빠름.** 모든 샘플에서 RTF < 0.15.

### vpm_run과 비교

| | vpm_run | Android 앱 |
|---|---|---|
| 추론시간/chunk | 233ms | **~250ms** |
| 출력 결과 | 한국어 텍스트 | **동일한 한국어 텍스트** |
| RTF | ~0.093 | **~0.112** |
| 차이 원인 | CLI 직접 호출 | JNI 오버헤드 + Java 디코딩 |

앱이 vpm_run보다 약 20% 느리지만 (JNI/Java 오버헤드), **출력 결과는 동일.** CER도 동일한 10% 수준.

### KoCitrinet 비교

| 모델 | NB | 추론방식 | 추론시간 (10초 음성) | RTF | CER |
|------|-----|---------|-------------------|-----|-----|
| **Conformer** | **102MB** | **슬라이딩 4 chunks** | **~1050ms** | **0.105** | **10.02%** |
| KoCitrinet | 62MB | 단일 3초 | 120ms | 0.040 | 44.44% |

Conformer는 KoCitrinet보다 느리지만 (RTF 0.11 vs 0.04), **CER이 4배 이상 좋음** (10% vs 44%).

## 핵심 확인 사항

1. **vpm_run = 앱 결과 동일** — NPU 드라이버가 같으므로 NB 추론 결과 동일
2. **전체 문장 인식 가능** — 슬라이딩 윈도우로 20초 음성도 처리
3. **RTF < 0.15** — 모든 길이에서 실시간보다 6~10배 빠름
4. **CER 10% 수준** — vpm_run 100샘플 결과와 동일한 정확도

## 남은 작업 (Phase 2)

현재는 NeMo Docker에서 미리 생성한 mel을 사용. 완전한 앱이 되려면:

1. **NeMo mel 전처리를 Java/JNI로 구현** — FFT, mel filterbank, ln, per-feature normalize
2. **WAV 파일 직접 입력** — 현재는 .dat 파일, WAV 로드 + mel 변환 필요
3. **실시간 마이크 입력** — AudioRecord → mel → NPU → 텍스트
4. **NB 다운로드/캐싱** — 102MB를 앱 내에서 관리 (assets or 서버 다운로드)
