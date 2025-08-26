# Zonemaker AI - AI 기반 윈도우 배열 최적화 시스템

### Team 정보 : Jae2
- 이재용 / eugssmixx@gmail.com(wodydy0507@korea.ac.kr)
- 이재모 / samjaemo@korea.ac.kr

## 개요

Zonemaker AI는 Microsoft Copilot PC의 Snapdragon NPU를 활용하여 사용자의 윈도우 사용 패턴을 학습하고, 실시간으로 최적의 윈도우 배열을 제안하는 AI 시스템입니다. 
Vision Transformer 기반의 경량화된 모델을 사용하여 로컬에서 빠른 추론을 수행합니다.

## 주요 기능

- **실시간 윈도우 모니터링**: Windows API를 통한 지속적인 윈도우 상태 추적
- **AI 기반 배열 예측**: 30초 관찰 후 다음 순간의 윈도우 위치 예측
- **연속 최적화**: 1초마다 새로운 예측으로 지속적인 최적화
- **NPU 최적화**: Snapdragon NPU 전용 모델 변환 및 최적화
- **멀티 모델 지원**: PyTorch (.pth) 및 ONNX (.onnx) 모델 지원

## 핵심 명령어 정리(CLI 명령어)

### **학습용 데이터 수집**
```bash
python run.py --mode data-collect --duration T(초; 정수입력)
```
산출물 : data/window_acticity_data_*.json

### **Train**
```bash
python run.py --mode train --data-file data/window_acticity_data_*.json --epochs 50
```
산출물 : data/models/best.pth, data/models/final.pth

### **NPU Converter**
```bash
python qai-hub/pth2onnxcompiler.py data/models/final.pth -o data/models/npu_converted/finial.onnx
```
산출물 : data/models/npu_conveted/final.onnx

### **Inference**
```bash
python run.py --mode inference --model-path data/models/final.onnx --duration T(초)
```

### **CPU 환경에서 즉시 Inference**
```bash
python run.py --mode inference --model-path data/models/best.pth --duration T(초)
```

## Dependency 및 가상환경 세팅

### 1. 가상환경 생성
```bash
conda create -n zonemakeraienv python=3.11 -y
conda activate zonemakeraienv
```
### 2. 디펜던시 직접 설치(권장)
```bash
conda install pytorch torchvision cpuonly -c pytorch -y
conda install numpy scikit-learn matplotlib tqdm psutil requests -y
conda install -c conda-forge fastapi uvicorn pydantic python-dotenv onnx onnxruntime -y
pip install pywin32 PySide6[all]>=6.4.0 PySide6-Addons>=6.4.0
pip install qai-hub
```
### 2-1. 또는 requirements.txt로 설치
```bash
pip install -r requirements.txt
```
### 3. Qai-hub 세팅
```bash
qai-hub configure --api-token YOUR_API_TOKEN
```
### 4. CLI 명령어 실행
### 5. 가상환경 종료
```bash
conda deactivate
```


# 기타 상세 내용


## 시스템 요구사항

### 하드웨어 요구사항
- **메모리**: 최소 8GB RAM (16GB 권장)
- **저장공간**: 최소 2GB 여유 공간
- **NPU**: Snapdragon NPU (선택사항, 성능 향상)

### 소프트웨어 요구사항
- **OS**: Windows 10/11 (64비트)
- **Python**: 3.8 이상
- **가상환경**: conda 권장

## 설치 및 설정

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/zonemakerai.git
cd zonemakerai
```

### 2. 가상환경 생성 및 활성화
```bash
# conda 사용
conda create -n zonemakeraiconda python=3.9
conda activate zonemakeraiconda

# 또는 venv 사용
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

## ML 파이프라인 상세 가이드

### 데이터 수집 파이프라인

#### 구조 및 원리
```
사용자 활동 → 윈도우 상태 모니터링 → 시퀀스 생성 → 실시간 버퍼링
```

#### 핵심 컴포넌트

**1. DataCollector 클래스**
```python
from backend.core.data_collector import DataCollector

# 데이터 수집기 생성
collector = DataCollector()

# 30초간 연속 데이터 수집
activities = collector.collect_data_sample(duration_seconds=30)

# 비동기 연속 수집 (콜백 기반)
def on_data_collected(samples):
    print(f"수집된 샘플: {len(samples)}개")

collector.start_continuous_collection(30, callback=on_data_collected)
```

**2. 수집되는 데이터 유형**
- **윈도우 정보**: 제목, 클래스명, 프로세스명, 위치, 크기, 상태
- **사용자 활동**: 마우스 위치, 키보드 활동, 윈도우 변화
- **시계열 데이터**: 0.1초마다 샘플링하여 30초 시퀀스 구성

#### 데이터 수집 실행
```bash
# 30초간 데이터 수집
python run.py --mode data-collect --duration 30

# 결과: data/window_activity_data_[timestamp].json
```

### ML 모델 아키텍처

#### 모델 구조
```
입력: 30초 윈도우 시퀀스 + 활동 시퀀스
  ↓
WindowFeatureExtractor: 윈도우 정보 → 특징 벡터
  ↓
ActivityFeatureExtractor: 사용자 활동 → 특징 벡터
  ↓
Transformer Encoder: 시퀀스 처리
  ↓
출력: 윈도우별 위치/크기 + 존재 여부
```

#### 모델 생성 및 사용
```python
from backend.ml.model import create_model, WindowArrangementPredictor

# 모델 생성
model = create_model({
    'window_feature_dim': 128,
    'activity_feature_dim': 64,
    'hidden_dim': 256,
    'num_heads': 8,
    'num_layers': 6,
    'max_windows': 20
})

# 예측기 생성
predictor = WindowArrangementPredictor("path/to/model.pth")

# 다음 순간 윈도우 배열 예측
predicted_positions = predictor.predict_next_arrangement(
    window_sequence, activity_sequence
)
```

#### 모델 훈련
```bash
# 기본 훈련 (50 에포크)
python run.py --mode train --epochs 50

# 특정 데이터 파일로 훈련
python run.py --mode train --data-file data/my_data.json --epochs 100

# 결과: data/models/best.pth, data/models/final.pth
```

### 실시간 추론 파이프라인

#### 추론 엔진 동작 원리
```
1. 30초 데이터 버퍼 수집
2. 버퍼가 가득 차면 예측 수행
3. 예측 결과를 실제 윈도우에 적용
4. 1초 후 새로운 데이터로 버퍼 업데이트
5. 반복하여 연속 최적화
```

#### 실시간 추론 실행
```bash
# 60초간 실시간 추론
python run.py --mode inference --duration 60

# 특정 모델로 추론
python run.py --mode inference --model-path data/models/best.pth --duration 120
```

### 1. 모델 경량화
- **양자화**: int8 정밀도로 모델 크기 감소
- **프루닝**: 불필요한 가중치 제거
- **지식 증류**: 작은 모델로 성능 유지

### 2. 추론 최적화
- **배치 처리**: 여러 윈도우 동시 예측
- **비동기 처리**: UI 블로킹 방지
- **메모리 관리**: 효율적인 버퍼 관리

### 3. NPU 최적화
- **모델 융합**: 연산 레이어 결합
- **메모리 최적화**: NPU 메모리 효율적 사용
- **병렬 처리**: NPU 병렬 연산 활용

## 📚 API 참조

### 핵심 클래스

#### DataCollector
- `collect_data_sample(duration)`: 지정된 시간 동안 데이터 수집
- `start_continuous_collection(duration, callback)`: 연속 데이터 수집
- `save_data(activities, filename)`: 데이터 저장

#### RealTimeWindowPredictor
- `predict_next_arrangement(window_seq, activity_seq)`: 다음 순간 예측
- `apply_prediction(window_handles, positions)`: 예측 결과 적용

#### RealTimeInferenceEngine
- `start_inference()`: 추론 시작
- `stop_inference()`: 추론 중지
- `get_inference_status()`: 상태 정보 반환

#### ModelTrainer
- `prepare_training_data(data_file, ...)`: 훈련 데이터 준비
- `train(train_loader, val_loader, ...)`: 모델 훈련
- `export_to_onnx(save_path)`: ONNX 모델 내보내기

**Zonemaker AI** - Arrange Less, Achieve More.
