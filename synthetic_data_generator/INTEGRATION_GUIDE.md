# 🔗 **메인 프로젝트 훈련기 연결 가이드**

## 📋 **연결 과정 개요**

이 가이드는 **합성 데이터 생성기**에서 생성된 데이터를 **메인 프로젝트의 훈련기**와 연결하는 과정을 단계별로 설명합니다.

### 🎯 **연결 목표**
- 합성 데이터를 메인 프로젝트 형식으로 변환
- 기존 훈련기에서 합성 데이터 사용 가능하도록 설정
- 모델 성능 향상을 위한 대용량 훈련 데이터 확보

## 🔄 **전체 데이터 흐름**

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   합성 데이터 생성기  │───▶│    데이터 어댑터     │───▶│   메인 프로젝트     │
│                     │    │                     │    │     훈련기         │
│ • 시나리오별 생성    │    │ • 형식 변환         │    │ • 모델 훈련         │
│ • 다양한 패턴        │    │ • 데이터 검증       │    │ • 성능 평가         │
│ • 대용량 데이터      │    │ • 연결 테스트       │    │ • 모델 저장         │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## 📁 **프로젝트 구조 및 파일 관계**

```
zonemakerai/                          # 메인 프로젝트 루트
├── backend/
│   ├── core/
│   │   └── data_collector.py         # WindowInfo, UserActivity 클래스
│   └── ml/
│       ├── model.py                  # RealTimeWindowPredictor 모델
│       ├── trainer.py                # ModelTrainer 클래스
│       └── inference.py              # 추론 엔진
├── data/                             # 훈련 데이터 저장소
│   ├── window_activity_data_*.json   # 기존 수집 데이터
│   └── converted_synthetic_*.json    # 변환된 합성 데이터
└── synthetic_data_generator/         # 합성 데이터 생성기
    ├── config/
    │   └── data_config.py            # 데이터 생성 설정
    ├── scripts/
    │   ├── synthetic_data_generator.py
    │   ├── generate_data.py
    │   └── data_adapter.py           # 🔑 핵심 연결 도구
    ├── output/                       # 생성된 합성 데이터
    └── logs/                         # 로그 파일
```

## 🚀 **단계별 연결 과정**

### **1단계: 합성 데이터 생성**

#### **1.1 기본 데이터 생성 (테스트)**
```bash
cd synthetic_data_generator

# 테스트 모드로 5개 시퀀스 생성
python scripts/generate_data.py --test-mode

# 결과 확인
ls output/
# → synthetic_window_activity_YYYYMMDD_HHMMSS.json
```

#### **1.2 대용량 데이터 생성**
```bash
# 1000개 시퀀스 생성 (기본값)
python scripts/generate_data.py

# 5000개 시퀀스 생성 (고품질 훈련용)
python scripts/generate_data.py --num-sequences 5000

# 특정 시나리오만 생성
python scripts/generate_data.py \
    --num-sequences 2000 \
    --scenarios browsing development
```

#### **1.3 생성된 데이터 확인**
```bash
# 파일 크기 확인
ls -lh output/

# 데이터 구조 확인 (첫 몇 줄)
head -20 output/synthetic_window_activity_*.json
```

### **2단계: 데이터 형식 변환**

#### **2.1 기본 변환**
```bash
# 합성 데이터를 메인 프로젝트 형식으로 변환
python scripts/data_adapter.py output/synthetic_window_activity_*.json

# 결과 확인
ls ../data/
# → converted_synthetic_window_activity_*.json
```

#### **2.2 사용자 정의 출력 파일명**
```bash
# 사용자 정의 파일명으로 저장
python scripts/data_adapter.py output/synthetic_data.json \
    --output-file my_training_data.json
```

#### **2.3 변환 과정 모니터링**
```bash
# 상세한 로그와 함께 변환
python scripts/data_adapter.py output/synthetic_data.json --test-training
```

**예상 출력:**
```
INFO - 합성 데이터 어댑터 초기화: output/synthetic_data.json
INFO - 합성 데이터 로드 완료: output/synthetic_data.json
INFO - 데이터 정보: 1000개 시퀀스
INFO - 데이터 형식 변환 시작...
INFO - 데이터 형식 변환 완료: 30000개 활동
INFO - 변환된 데이터 저장 완료: ../data/converted_synthetic_data.json
INFO - 훈련 데이터 준비 테스트 시작...
INFO - 훈련 데이터 준비 테스트 성공!
INFO - 훈련 배치 수: 125
INFO - 검증 배치 수: 31
```

### **3단계: 메인 프로젝트에서 사용**

#### **3.1 변환된 데이터 확인**
```bash
cd ..  # 메인 프로젝트 루트로 이동

# 변환된 데이터 파일 확인
ls data/converted_*.json

# 데이터 구조 확인
python -c "
import json
with open('data/converted_synthetic_data.json', 'r') as f:
    data = json.load(f)
print(f'총 활동 수: {len(data)}')
print(f'첫 번째 활동 키: {list(data[0].keys())}')
print(f'윈도우 수: {len(data[0].get(\"all_windows\", []))}')
"
```

#### **3.2 기존 훈련기와 통합**
```bash
# 기존 훈련 명령어 사용 (변환된 데이터 파일 지정)
python run.py --mode train \
    --data-file data/converted_synthetic_data.json \
    --epochs 100

# 또는 기존 데이터와 합성 데이터를 함께 사용
python run.py --mode train \
    --data-file data/window_activity_data_*.json \
    --epochs 100
```

## 🔧 **고급 연결 옵션**

### **1. 데이터 어댑터 직접 사용**

#### **Python 스크립트에서 직접 사용**
```python
# train_with_synthetic.py
import sys
import os

# 합성 데이터 생성기 경로 추가
sys.path.append('synthetic_data_generator/scripts')

from data_adapter import SyntheticDataAdapter
from backend.ml.trainer import ModelTrainer
from backend.ml.model import create_model

def main():
    # 1. 합성 데이터 어댑터 생성
    adapter = SyntheticDataAdapter('synthetic_data_generator/output/synthetic_data.json')
    
    # 2. 데이터 로드 및 변환
    adapter.load_synthetic_data()
    converted_data = adapter.convert_to_main_format()
    
    # 3. 훈련 데이터 준비
    train_loader, val_loader = adapter.prepare_training_data(
        train_split=0.8,
        batch_size=16,
        sequence_length=30
    )
    
    # 4. 모델 생성 및 훈련
    model = create_model()
    trainer = ModelTrainer(model)
    
    # 5. 훈련 실행
    trainer.train(train_loader, val_loader, epochs=100)
    
    # 6. 모델 저장
    trainer.save_model('data/models/synthetic_trained_model.pth')

if __name__ == "__main__":
    main()
```

#### **실행**
```bash
python train_with_synthetic.py
```

### **2. 배치 처리 및 스케줄링**

#### **여러 합성 데이터 파일 처리**
```bash
# 여러 파일을 순차적으로 처리
for file in synthetic_data_generator/output/synthetic_*.json; do
    echo "처리 중: $file"
    python synthetic_data_generator/scripts/data_adapter.py "$file"
done
```

#### **자동화 스크립트 생성**
```bash
# auto_integrate.sh
#!/bin/bash

echo "=== 합성 데이터 자동 통합 시작 ==="

# 1. 합성 데이터 생성
cd synthetic_data_generator
python scripts/generate_data.py --num-sequences 2000
cd ..

# 2. 모든 합성 데이터 파일 변환
for file in synthetic_data_generator/output/synthetic_*.json; do
    echo "변환 중: $file"
    python synthetic_data_generator/scripts/data_adapter.py "$file"
done

# 3. 통합된 데이터로 훈련
echo "통합 훈련 시작..."
python run.py --mode train --epochs 100

echo "=== 자동 통합 완료 ==="
```

## 📊 **데이터 품질 검증**

### **1. 기본 검증**

#### **데이터 구조 검증**
```python
# validate_synthetic_data.py
import json
import sys
sys.path.append('synthetic_data_generator/scripts')

from data_adapter import SyntheticDataAdapter

def validate_data(file_path):
    adapter = SyntheticDataAdapter(file_path)
    
    try:
        # 데이터 로드
        original_data = adapter.load_synthetic_data()
        print(f"✅ 원본 데이터 로드 성공: {len(original_data)}개 시퀀스")
        
        # 형식 변환
        converted_data = adapter.convert_to_main_format()
        print(f"✅ 데이터 변환 성공: {len(converted_data)}개 활동")
        
        # 훈련 데이터 준비 테스트
        train_loader, val_loader = adapter.prepare_training_data()
        print(f"✅ 훈련 데이터 준비 성공")
        print(f"   - 훈련 배치: {len(train_loader)}")
        print(f"   - 검증 배치: {len(val_loader)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 검증 실패: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        validate_data(sys.argv[1])
    else:
        print("사용법: python validate_synthetic_data.py <합성데이터파일>")
```

#### **실행**
```bash
python validate_synthetic_data.py synthetic_data_generator/output/synthetic_data.json
```

### **2. 고급 검증**

#### **데이터 통계 분석**
```python
# analyze_synthetic_data.py
import json
from collections import Counter

def analyze_synthetic_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 'training_data' in data:
        training_data = data['training_data']
        print(f"훈련 데이터 구조: {len(training_data)}개 시퀀스")
        
        # 시퀀스별 윈도우 수 분석
        window_counts = []
        activity_types = []
        
        for seq in training_data:
            if 'activity_sequences' in seq:
                activities = seq['activity_sequences']
                for activity in activities:
                    window_count = len(activity.get('all_windows', []))
                    window_counts.append(window_count)
                    activity_types.append(activity.get('activity_type', 'unknown'))
        
        print(f"윈도우 수 통계:")
        print(f"  - 평균: {sum(window_counts) / len(window_counts):.1f}")
        print(f"  - 최소: {min(window_counts)}")
        print(f"  - 최대: {max(window_counts)}")
        
        print(f"활동 타입 분포:")
        type_counts = Counter(activity_types)
        for activity_type, count in type_counts.most_common():
            print(f"  - {activity_type}: {count} ({count/len(activity_types)*100:.1f}%)")
    
    else:
        print("훈련 데이터 구조를 찾을 수 없습니다.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_synthetic_data(sys.argv[1])
    else:
        print("사용법: python analyze_synthetic_data.py <합성데이터파일>")
```

## 🚨 **문제 해결 및 디버깅**

### **1. 일반적인 오류들**

#### **Import 오류**
```bash
# 오류: ModuleNotFoundError: No module named 'backend'
# 해결: 경로 설정 확인
python -c "
import sys
sys.path.append('..')
from backend.core.data_collector import WindowInfo
print('Import 성공')
"
```

#### **데이터 형식 불일치**
```bash
# 오류: KeyError: 'all_windows'
# 해결: 데이터 구조 확인
python -c "
import json
with open('synthetic_data.json', 'r') as f:
    data = json.load(f)
print('데이터 키:', list(data.keys()) if isinstance(data, dict) else '리스트')
"
```

#### **메모리 부족**
```bash
# 오류: MemoryError
# 해결: 시퀀스 수 줄이기
python scripts/generate_data.py --num-sequences 500
```

### **2. 디버깅 도구**

#### **로깅 레벨 조정**
```python
# scripts/data_adapter.py에서
import logging
logging.basicConfig(level=logging.DEBUG)  # 상세한 로그 출력
```

#### **단계별 실행**
```bash
# 각 단계를 개별적으로 실행하여 문제 지점 파악
python -c "from synthetic_data_generator.scripts.data_adapter import SyntheticDataAdapter; print('Import 성공')"
```

## 📈 **성능 최적화 팁**

### **1. 데이터 생성 최적화**
- **시퀀스 수**: 초기에는 1000개, 점진적으로 증가
- **시나리오 선택**: 실제 사용 패턴과 유사한 것 우선
- **겹침 설정**: `overlap_seconds` 조정으로 다양성 확보

### **2. 메모리 사용량 최적화**
- **배치 크기**: 시스템 메모리에 맞게 조정
- **점진적 생성**: 대용량 데이터를 여러 파일로 분할
- **가비지 컬렉션**: 주기적인 메모리 정리

### **3. 훈련 효율성 향상**
- **데이터 검증**: 생성된 데이터 품질 사전 확인
- **시나리오 균형**: 각 시나리오별 균등한 분배
- **정규화**: 윈도우 위치/크기 정규화

## 🔮 **향후 개선 방향**

### **1. 자동화**
- **자동 통합**: CI/CD 파이프라인에 통합
- **스케줄링**: 정기적인 합성 데이터 생성 및 통합
- **모니터링**: 데이터 품질 자동 모니터링

### **2. 고급 기능**
- **적응형 생성**: 훈련 결과에 따른 데이터 생성 조정
- **실시간 통합**: 훈련 중 실시간 데이터 추가
- **품질 평가**: 자동화된 데이터 품질 평가 시스템

---

## 📞 **지원 및 문의**

연결 과정에서 문제가 발생하면 다음을 확인하세요:

1. **로그 파일**: `synthetic_data_generator/logs/` 디렉토리 확인
2. **경로 설정**: `sys.path` 설정 및 상대 경로 확인
3. **데이터 구조**: JSON 파일의 구조 및 형식 검증
4. **메모리 사용량**: 시스템 리소스 모니터링

**🎯 합성 데이터로 Zonemaker AI의 성능을 극대화하세요!**
