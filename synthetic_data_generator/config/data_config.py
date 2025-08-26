# config/data_config.py 재설계

# 데이터 생성 기본 설정
DATA_CONFIG = {
    # 시퀀스 설정
    'sequence_length': 30,           # 30초 시퀀스
    'sample_interval': 0.1,          # 0.1초 샘플링
    'max_windows': 6,                # 최대 윈도우 수 (6개 프로그램에 맞춤)
    
    # 화면 해상도
    'screen_width': 1920,
    'screen_height': 1080,
    
    # 데이터 생성 설정
    'num_sequences': 1000,           # 생성할 시퀀스 수
    'overlap_seconds': 5,            # 시퀀스 간 겹침 (초) - 15초에서 5초로 축소
    
    # 윈도우 설정
    'min_window_width': 400,
    'max_window_width': 1200,
    'min_window_height': 300,
    'max_window_height': 800,
    
    # 활동 패턴 가중치 (개발자 작업에 최적화)
    'activity_weights': {
        'idle': 0.4,           # 40% - 코드 작성 중 대기
        'keyboard': 0.35,      # 35% - 코드 타이핑
        'click': 0.2,          # 20% - UI 클릭
        'window_change': 0.05  # 5% - 윈도우 전환
    }
}

# 6개 프로그램만 사용하도록 축소
WINDOW_TYPES = {
    'development': {
        'titles': [
            'Visual Studio',
            'Visual Studio Code'
        ],
        'classes': [
            'VisualStudioWindowClass',
            'Chrome_WidgetWin_1'    # VS Code
        ],
        'processes': [
            'devenv.exe',            # Visual Studio
            'Code.exe'               # VS Code
        ],
        'typical_sizes': [
            (1200, 800),   # IDE 기본 크기
            (1400, 900),   # IDE 확장 크기
            (1600, 1000),  # IDE 전체 화면
            (1000, 700)    # IDE 컴팩트 크기
        ]
    },
    
    'utility': {
        'titles': [
            'File Explorer',
            'Task Manager',
            'Command Prompt'
        ],
        'classes': [
            'CabinetWClass',
            'TaskManagerWindow',
            'ConsoleWindowClass'
        ],
        'processes': [
            'explorer.exe',
            'taskmgr.exe',
            'cmd.exe'
        ],
        'typical_sizes': [
            (800, 600),     # 유틸리티 기본 크기
            (1000, 700),    # 유틸리티 확장 크기
            (600, 400),     # 유틸리티 컴팩트 크기
            (900, 650)      # 유틸리티 중간 크기
        ]
    },
    
    'document': {
        'titles': [
            'Sticky Notes'
        ],
        'classes': [
            'Sticky_Notes_App'     # 스티키 메모
        ],
        'processes': [
            'StickyNot.exe'        # 스티키 메모
        ],
        'typical_sizes': [
            (300, 200),     # 스티키 메모 기본 크기
            (400, 300),     # 스티키 메모 확장 크기
            (250, 150),     # 스티키 메모 컴팩트 크기
            (350, 250)      # 스티키 메모 중간 크기
        ]
    }
}

# 단일 시나리오: 개발자 작업 환경
SCENARIO_CONFIGS = {
    'developer_workspace': {
        'description': '개발자 작업 환경 - IDE 중심 멀티태스킹',
        'num_windows': 6,           # IDE 2개 + 유틸리티 2개 + 문서 2개
        'activity_pattern': 'development_focused',
        'window_movement': 'moderate',  # IDE는 자주 이동하지 않음
        'duration_multiplier': 1.5,    # 개발 작업은 오래 지속
        'window_distribution': {
            'development': 0.4,     # 40% - IDE (2-3개)
            'utility': 0.3,         # 30% - 유틸리티 (2개)
            'document': 0.3         # 30% - 문서 (1-2개)
        }
    }
}

# 출력 설정
OUTPUT_CONFIG = {
    'base_dir': 'output',
    'filename_prefix': 'developer_workspace_synthetic',
    'format': 'json',
    'compression': False,
    'metadata': True
}