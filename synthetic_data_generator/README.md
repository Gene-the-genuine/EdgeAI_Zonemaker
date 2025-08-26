# 합성 윈도우 활동 데이터 생성기

```
synthetic_data_generator/
├── config/
│   └── data_config.py          # 데이터 생성 설정
├── scripts/
│   ├── synthetic_data_generator.py  # 핵심 데이터 생성기
│   ├── generate_data.py             # 데이터 생성 실행 스크립트
│   └── data_adapter.py              # 메인 프로젝트 연결 어댑터
├── data/                       # 입력 데이터 (필요시)
├── output/                     # 생성된 합성 데이터
├── logs/                       # 로그 파일
└── README.md                   # 이 파일
```

## 사용법

### 1단계: 합성 데이터 생성

#### 기본 사용법
```bash
# 테스트 모드 (5개 시퀀스)
python scripts/generate_data.py --test-mode

# 기본 생성 (1000개 시퀀스)
python scripts/generate_data.py

# 사용자 정의 설정
python scripts/generate_data.py \
    --num-sequences 5000 \
    --scenarios browsing document_work development \
    --output-dir custom_output
```

#### 명령행 옵션
- `--num-sequences`: 생성할 시퀀스 수 (기본값: 1000)
- `--scenarios`: 생성할 시나리오들 (기본값: 모든 시나리오)
- `--output-dir`: 출력 디렉토리 (기본값: output)
- `--filename`: 출력 파일명 (기본값: 자동 생성)
- `--test-mode`: 테스트 모드 (5개 시퀀스만 생성)

### 2단계: 메인 프로젝트와 연결

#### 데이터 형식 변환
```bash
# 합성 데이터를 메인 프로젝트 형식으로 변환
python scripts/data_adapter.py output/synthetic_window_activity_20250825_123456.json

# 사용자 정의 출력 파일명
python scripts/data_adapter.py output/synthetic_data.json --output-file my_training_data.json

# 훈련 데이터 준비 테스트
python scripts/data_adapter.py output/synthetic_data.json --test-training
```

#### 어댑터 옵션
- `synthetic_data_file`: 변환할 합성 데이터 파일 경로
- `--output-file`: 출력 파일명 (기본값: 자동 생성)
- `--test-training`: 훈련 데이터 준비 테스트 실행


### 데이터 형식

#### WindowInfo 구조
```python
@dataclass
class WindowInfo:
    handle: int                    # 윈도우 핸들
    title: str                     # 윈도우 제목
    class_name: str               # 윈도우 클래스명
    process_id: int               # 프로세스 ID
    process_name: str             # 프로세스명
    rect: Tuple[int, int, int, int]  # (left, top, right, bottom)
    is_visible: bool              # 보임 여부
    is_minimized: bool            # 최소화 여부
    is_maximized: bool            # 최대화 여부
    z_order: int                  # Z-순서
    timestamp: float              # 타임스탬프
```

#### UserActivity 구조
```python
@dataclass
class UserActivity:
    timestamp: float              # 타임스탬프
    active_window: Optional[WindowInfo]  # 활성 윈도우
    all_windows: List[WindowInfo]        # 모든 윈도우
    mouse_position: Tuple[int, int]      # 마우스 위치
    keyboard_active: bool                # 키보드 활동 여부
    activity_type: str                   # 활동 타입
```

## 설정 커스터마이징

### 데이터 생성 설정 수정

`config/data_config.py` 파일에서 다음 설정들을 조정할 수 있습니다:

#### 기본 설정
```python
DATA_CONFIG = {
    'sequence_length': 30,           # 30초 시퀀스
    'sample_interval': 0.1,          # 0.1초 샘플링
    'max_windows': 20,               # 최대 윈도우 수
    'screen_width': 1920,            # 화면 너비
    'screen_height': 1080,           # 화면 높이
    'num_sequences': 1000,           # 생성할 시퀀스 수
    'overlap_seconds': 15,           # 시퀀스 간 겹침
}
```

#### 윈도우 설정
```python
'min_window_width': 300,            # 최소 윈도우 너비
'max_window_width': 800,            # 최대 윈도우 너비
'min_window_height': 200,           # 최소 윈도우 높이
'max_window_height': 600,           # 최대 윈도우 높이
```

#### 활동 패턴 가중치
```python
'activity_weights': {
    'idle': 0.6,                    # 60% - 대기
    'click': 0.2,                   # 20% - 클릭
    'keyboard': 0.15,               # 15% - 키보드
    'window_change': 0.05           # 5% - 윈도우 변경
}
```

### 새로운 윈도우 타입 추가

```python
WINDOW_TYPES = {
    'new_category': {
        'titles': ['New App 1', 'New App 2'],
        'classes': ['NewAppClass1', 'NewAppClass2'],
        'processes': ['newapp1.exe', 'newapp2.exe'],
        'typical_sizes': [(600, 400), (800, 600)]
    }
}
```

### 새로운 시나리오 추가

```python
SCENARIO_CONFIGS = {
    'new_scenario': {
        'description': '새로운 작업 시나리오',
        'num_windows': 4,
        'activity_pattern': 'focused_work',
        'window_movement': 'low',
        'duration_multiplier': 1.0
    }
}
```

## 메인 프로젝트와의 통합

### 데이터 흐름

```
1. 합성 데이터 생성
   ↓
2. 데이터 형식 변환 (SyntheticDataAdapter)
   ↓
3. 메인 프로젝트 data/ 디렉토리에 저장
   ↓
4. 기존 훈련기에서 사용
```