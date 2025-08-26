# QAI Hub를 사용한 Zonemaker AI 모델 변환

이 디렉토리는 QAI Hub를 사용하여 Zonemaker AI의 PyTorch 모델을 ONNX로 변환하는 스크립트를 포함합니다.



## 사용법


### 1. 기본 변환

```bash
# 체크포인트를 ONNX로 변환
python pth2onnxcompiler.py ../data/models/final.pth
# 출력 파일명 지정
python pth2onnxcompiler.py ../data/models/final.pth -o ../data/models/final.onnx
```

### 2. 상세 옵션

```bash
# 상세 로그와 함께 변환
python pth2onnxcompiler.py ../data/models/final.pth -v
# 도움말 보기
python pth2onnxcompiler.py --help
```


## 변환 과정


1. **모델 로드**: PyTorch 체크포인트 파일 로드
2. **호환 모델 생성**: ONNX 변환을 위한 텐서 입력 모델 생성
3. **JIT 트레이싱**: TorchScript 모델로 변환
4. **QAI Hub 컴파일**: Snapdragon X Elite CRD에서 ONNX 컴파일
5. **프로파일링**: 모델 성능 측정
6. **추론 테스트**: 변환된 모델 테스트
7. **모델 다운로드**: ONNX 파일 다운로드


## 입력/출력 형태


### 입력
- **형태**: `[batch_size, seq_len, feature_dim]`
- **기본값**: `[1, 30, 192]`
  - `batch_size`: 1 (배치 크기)
  - `seq_len`: 30 (30초 시퀀스)
  - `feature_dim`: 192 (윈도우 특징 128 + 활동 특징 64)

### 출력
- **predicted_positions**: `[batch_size, max_windows, 4]`
  - 윈도우별 x, y, width, height 예측
- **window_existence**: `[batch_size, max_windows]`
  - 윈도우별 존재 확률


## 디버깅


```bash
# 상세 로그와 함께 실행
python pth2onnxcompiler.py model.pth -v
# 모델 테스트만 실행
python -c "
from backend.ml.model import create_model
model = create_model()
print('모델 생성 성공')
"
```