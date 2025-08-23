# Zonemaker AI

> Edge AI Developers Hackathon - Windows AI-powered Window Arrangement System

## 프로젝트 개요

Zonemaker AI는 MS Copilot PC의 Snapdragon NPU를 활용하여 실시간으로 사용자의 작업 환경에 최적화된 윈도우 배열을 수행하는 AI 시스템입니다.

## ✨ 주요 기능

- **실시간 데이터 수집**: Windows API를 통한 사용자 행동 모니터링
- **AI 모델 학습**: Vision Transformer 기반 윈도우 배열 예측 모델
- **NPU 최적화**: Snapdragon NPU 최적화된 실시간 추론
- **직관적 UI**: PyQt 기반 사용자 친화적 인터페이스
- **Workstation 관리**: 사용자 정의 작업 환경 생성 및 관리

## 🏗️ 아키텍처
zonemaker-ai/
Zonemaker AI/
├── backend/
│   ├── api/
│   │   └── main.py
│   ├── core/
│   │   ├── data_collector.py
│   │   ├── workstation.py
│   │   └── window_manager.py
│   ├── ml/
│   │   ├── model.py
│   │   ├── trainer.py
│   │   ├── inference.py
│   │   └── npu_converter.py
│   └── config/
│       └── settings.py
├── frontend/
│   ├── pages/
│   │   ├── home_page.py
│   │   ├── create_workstation_page.py
│   │   ├── train_run_page.py
│   │   ├── train_page.py
│   │   └── run_page.py
│   ├── main.py
│   └── api_client.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│       └── npu_converted/
├── deploy/
│   ├── dist/
│   └── installer.nsi
├── benchmark_results/
├── run.py
├── test_integration.py
├── benchmark_performance.py
├── deploy.py
├── final_deploy.py
├── requirements.txt
└── README.md



## 🚀 실행 방법

### 1. 개발 환경에서 실행

```bash
# 1. 의존성 설치 : ARM 기반 의존성 문제 발생가능하므로, 직접 하나씩 설치하는 것을 권장. psutil의 경우 별도의 빌드 툴 설치 필요.
pip install -r requirements.txt

# 2. 백엔드 서버 시작
cd backend
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 3. 새 터미널에서 프론트엔드 시작
cd frontend
python main.py
```

### 2. 통합 실행

```bash
# 전체 시스템을 한 번에 실행
python run.py
```

### 3. 테스트 및 벤치마킹

```bash
# 통합 테스트 실행
python test_integration.py

# 성능 벤치마킹
python benchmark_performance.py
```

### 4. 배포

```bash
# 배포 패키지 생성
python deploy.py

# 최종 배포 실행 (테스트 + 배포)
python final_deploy.py
```

## �� 주요 컴포넌트

### 백엔드
- **DataCollector**: Windows API를 통한 실시간 데이터 수집
- **WorkstationManager**: 작업 환경 생명주기 관리
- **WindowManager**: 윈도우 배열 및 제어
- **ZonemakerViT**: NPU 최적화된 Vision Transformer
- **NPUConverter**: ONNX 변환 및 NPU 최적화

### 프론트엔드
- **HomePage**: Workstation 목록 및 관리
- **CreateWorkstationPage**: 새 작업 환경 생성
- **TrainRunPage**: 학습/실행 선택
- **TrainPage**: 모델 학습 진행률
- **RunPage**: 실시간 추론 실행

### ML 파이프라인
- **데이터 수집**: 1초 간격 실시간 모니터링
- **모델 학습**: 10분 데이터 기반 시계열 학습
- **NPU 최적화**: Snapdragon NPU 최적화
- **실시간 추론**: 500ms 이내 응답

## 📊 성능 지표

- **추론 시간**: < 500ms
- **모델 크기**: < 10MB
- **메모리 사용량**: < 100MB
- **데이터 수집 주기**: 1초 간격
- **학습 시간**: 10분 (데이터 수집 포함)

## �� 사용 시나리오

1. **Workstation 생성**: 필요한 프로그램들을 선택하여 작업 환경 생성
2. **모델 학습**: 10분간 사용자 행동 데이터 수집 및 AI 모델 학습
3. **실시간 실행**: 학습된 모델을 사용하여 윈도우 배열 자동화
4. **성능 모니터링**: 실시간 성능 지표 및 시스템 상태 확인

## 🔍 문제 해결

### 백엔드 서버 연결 오류
```bash
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 의존성 오류
```bash
pip install -r requirements.txt
```

### 프론트엔드 실행 오류
```bash
cd frontend
python main.py
```

## 📚 추가 문서

- `docs/README.md`: 사용자 가이드
- `docs/API.md`: API 문서
- `benchmark_results/`: 성능 벤치마크 결과
- `deploy/dist/`: 배포 파일들

## 🔧 시스템 요구사항

- **OS**: Windows 10/11
- **Python**: 3.8+
- **하드웨어**: Snapdragon NPU 지원 (권장)
- **메모리**: 8GB+ RAM
- **저장공간**: 2GB+ 여유 공간

**Made with ❤️ for Edge AI Innovation**
