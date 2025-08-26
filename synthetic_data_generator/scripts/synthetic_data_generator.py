"""
합성 윈도우 활동 데이터 생성기
"""

import json
import time
import random
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import sys

# 상위 디렉토리의 backend 모듈을 import하기 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.data_config import DATA_CONFIG, WINDOW_TYPES, SCENARIO_CONFIGS, OUTPUT_CONFIG

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SyntheticWindowInfo:
    """합성 윈도우 정보"""
    handle: int
    title: str
    class_name: str
    process_id: int
    process_name: str
    rect: Tuple[int, int, int, int]  # (left, top, right, bottom)
    is_visible: bool
    is_minimized: bool
    is_maximized: bool
    z_order: int
    timestamp: float

@dataclass
class SyntheticUserActivity:
    """합성 사용자 활동 정보"""
    timestamp: float
    active_window: Optional[SyntheticWindowInfo]
    all_windows: List[SyntheticWindowInfo]
    mouse_position: Tuple[int, int]
    keyboard_active: bool
    activity_type: str

class SyntheticDataGenerator:
    """합성 윈도우 활동 데이터 생성기"""
    
    def __init__(self, config: Dict = None):
        self.config = config or DATA_CONFIG
        self.window_handle_counter = 1000
        
        # 시드 설정으로 재현 가능한 데이터 생성
        random.seed(42)
        
        logger.info("합성 데이터 생성기 초기화 완료")
    
    def generate_window_info(self, 
                           window_type: str = None,
                           timestamp: float = None,
                           position: Tuple[int, int] = None,
                           size: Tuple[int, int] = None) -> SyntheticWindowInfo:
        """합성 윈도우 정보 생성"""
        
        if timestamp is None:
            timestamp = time.time()
        
        # 윈도우 타입 선택
        if window_type is None:
            window_type = random.choice(list(WINDOW_TYPES.keys()))
        
        window_config = WINDOW_TYPES[window_type]
        
        # 랜덤 속성 선택
        title = random.choice(window_config['titles'])
        class_name = random.choice(window_config['classes'])
        process_name = random.choice(window_config['processes'])
        
        # 위치 및 크기 설정
        if position is None:
            x = random.randint(0, self.config['screen_width'] - 400)
            y = random.randint(0, self.config['screen_height'] - 300)
        else:
            x, y = position
        
        if size is None:
            # 윈도우 타입별 전형적인 크기 사용
            typical_sizes = window_config['typical_sizes']
            w, h = random.choice(typical_sizes)
        else:
            w, h = size
        
        # 화면 경계 내로 조정
        x = max(0, min(x, self.config['screen_width'] - w))
        y = max(0, min(y, self.config['screen_height'] - h))
        
        rect = (x, y, x + w, y + h)
        
        # 윈도우 상태 설정
        is_visible = random.choices([True, False], weights=[0.9, 0.1])[0]
        is_minimized = random.choices([False, True], weights=[0.95, 0.05])[0]
        is_maximized = random.choices([False, True], weights=[0.9, 0.1])[0]
        
        # Z-order 설정
        z_order = random.randint(0, 10)
        
        window_info = SyntheticWindowInfo(
            handle=self.window_handle_counter,
            title=title,
            class_name=class_name,
            process_id=random.randint(1000, 9999),
            process_name=process_name,
            rect=rect,
            is_visible=is_visible,
            is_minimized=is_minimized,
            is_maximized=is_maximized,
            z_order=z_order,
            timestamp=timestamp
        )
        
        self.window_handle_counter += 1
        return window_info
    
    # def generate_window_pool(self, scenario: str) -> List[SyntheticWindowInfo]:
    #     """시나리오별 윈도우 풀 생성"""
        
    #     scenario_config = SCENARIO_CONFIGS[scenario]
    #     num_windows = scenario_config['num_windows']
        
    #     windows = []
    #     window_types = list(WINDOW_TYPES.keys())
        
    #     for i in range(num_windows):
    #         # 시나리오에 맞는 윈도우 타입 선택
    #         if scenario == 'browsing':
    #             window_type = 'browser'
    #         elif scenario == 'document_work':
    #             window_type = random.choice(['document', 'utility'])
    #         elif scenario == 'development':
    #             window_type = random.choice(['development', 'browser', 'utility'])
    #         else:  # multitasking
    #             window_type = random.choice(window_types)
            
    #         window = self.generate_window_info(window_type=window_type)
    #         windows.append(window)
        
    #     return windows

    def generate_window_pool(self, scenario: str) -> List[SyntheticWindowInfo]:
        """시나리오별 윈도우 풀 생성 - 단일 시나리오 최적화"""
        
        scenario_config = SCENARIO_CONFIGS[scenario]
        num_windows = scenario_config['num_windows']
        distribution = scenario_config['window_distribution']
        
        windows = []
        
        # 개발 도구 윈도우 생성 (40%)
        dev_count = int(num_windows * distribution['development'])
        for i in range(dev_count):
            window = self.generate_window_info(window_type='development')
            windows.append(window)
        
        # 유틸리티 윈도우 생성 (30%)
        util_count = int(num_windows * distribution['utility'])
        for i in range(util_count):
            window = self.generate_window_info(window_type='utility')
            windows.append(window)
        
        # 문서 윈도우 생성 (30%)
        doc_count = int(num_windows * distribution['document'])
        for i in range(doc_count):
            window = self.generate_window_info(window_type='document')
            windows.append(window)
        
        # 남은 윈도우는 개발 도구로 채움
        remaining = num_windows - len(windows)
        for i in range(remaining):
            window = self.generate_window_info(window_type='development')
            windows.append(window)
        
        return windows
    
    def simulate_window_movement(self, 
                                windows: List[SyntheticWindowInfo], 
                                timestamp: float,
                                movement_intensity: str) -> List[SyntheticWindowInfo]:
        """윈도우 이동 시뮬레이션"""
        
        updated_windows = []
        
        for window in windows:
            # 이동 확률 설정
            if movement_intensity == 'low':
                move_prob = 0.1
                max_move = 20
            elif movement_intensity == 'moderate':
                move_prob = 0.3
                max_move = 40
            elif movement_intensity == 'high':
                move_prob = 0.5
                max_move = 60
            else:  # very_high
                move_prob = 0.7
                max_move = 80
            
            if random.random() < move_prob:
                # 위치 변화
                dx = random.randint(-max_move, max_move)
                dy = random.randint(-max_move, max_move)
                
                # 크기 변화
                dw = random.randint(-30, 30)
                dh = random.randint(-20, 20)
                
                # 새로운 위치 및 크기 계산
                new_x = max(0, min(self.config['screen_width'] - (window.rect[2] - window.rect[0] + dw), 
                                   window.rect[0] + dx))
                new_y = max(0, min(self.config['screen_height'] - (window.rect[3] - window.rect[1] + dh), 
                                   window.rect[1] + dy))
                new_w = max(200, min(1000, window.rect[2] - window.rect[0] + dw))
                new_h = max(150, min(800, window.rect[3] - window.rect[1] + dh))
                
                new_rect = (new_x, new_y, new_x + new_w, new_y + new_h)
                
                # 윈도우 정보 업데이트 (안전한 방법)
                updated_window = SyntheticWindowInfo(
                    handle=window.handle,
                    title=window.title,
                    class_name=window.class_name,
                    process_id=window.process_id,
                    process_name=window.process_name,
                    rect=new_rect,
                    is_visible=window.is_visible,
                    is_minimized=window.is_minimized,
                    is_maximized=window.is_maximized,
                    z_order=window.z_order,
                    timestamp=timestamp
                )
                updated_windows.append(updated_window)
            else:
                # 타임스탬프만 업데이트 (안전한 방법)
                updated_window = SyntheticWindowInfo(
                    handle=window.handle,
                    title=window.title,
                    class_name=window.class_name,
                    process_id=window.process_id,
                    process_name=window.process_name,
                    rect=window.rect,
                    is_visible=window.is_visible,
                    is_minimized=window.is_minimized,
                    is_maximized=window.is_maximized,
                    z_order=window.z_order,
                    timestamp=timestamp
                )
                updated_windows.append(updated_window)
        
        return updated_windows
    
    # def generate_activity_sequence(self, 
    #                               scenario: str,
    #                               duration: int = None,
    #                               sample_interval: float = None) -> List[SyntheticUserActivity]:
    #     """시나리오별 활동 시퀀스 생성"""
        
    #     if duration is None:
    #         duration = self.config['sequence_length']
    #     if sample_interval is None:
    #         sample_interval = self.config['sample_interval']
        
    #     scenario_config = SCENARIO_CONFIGS[scenario]
    #     duration_multiplier = scenario_config['duration_multiplier']
    #     adjusted_duration = int(duration * duration_multiplier)
        
    #     logger.info(f"시나리오 '{scenario}' 활동 시퀀스 생성 시작 (지속시간: {adjusted_duration}초)")
        
    #     activities = []
    #     start_time = time.time()
        
    #     # 윈도우 풀 생성
    #     windows = self.generate_window_pool(scenario)
        
    #     # 시퀀스 생성
    #     for i in range(int(adjusted_duration / sample_interval)):
    #         timestamp = start_time + i * sample_interval
            
    #         # 윈도우 이동 시뮬레이션
    #         if i > 0:
    #             windows = self.simulate_window_movement(
    #                 windows, timestamp, scenario_config['window_movement']
    #             )
            
    #         # 활동 타입 결정
    #         activity_type = random.choices(
    #             list(self.config['activity_weights'].keys()),
    #             weights=list(self.config['activity_weights'].values())
    #         )[0]
            
    #         # 마우스 위치 설정
    #         if activity_type == 'click':
    #             # 클릭 활동일 때는 윈도우 내부에 위치
    #             active_window = random.choice(windows)
    #             x = random.randint(active_window.rect[0] + 50, active_window.rect[2] - 50)
    #             y = random.randint(active_window.rect[1] + 50, active_window.rect[3] - 50)
    #         else:
    #             # 일반적인 마우스 위치
    #             x = random.randint(100, self.config['screen_width'] - 100)
    #             y = random.randint(100, self.config['screen_height'] - 100)
            
    #         # 활성 윈도우 결정
    #         if activity_type == 'window_change':
    #             # 윈도우 변경 시 랜덤 선택
    #             active_window = random.choice(windows)
    #         else:
    #             # 마우스 위치와 가장 가까운 윈도우
    #             active_window = min(windows, 
    #                               key=lambda w: abs(w.rect[0] - x) + abs(w.rect[1] - y))
            
    #         # 키보드 활동 여부
    #         keyboard_active = activity_type == 'keyboard'
            
    #         activity = SyntheticUserActivity(
    #             timestamp=timestamp,
    #             active_window=active_window,
    #             all_windows=windows.copy(),
    #             mouse_position=(x, y),
    #             keyboard_active=keyboard_active,
    #             activity_type=activity_type
    #         )
            
    #         activities.append(activity)
        
    #     logger.info(f"시나리오 '{scenario}' 활동 시퀀스 생성 완료: {len(activities)}개 샘플")
    #     return activities

    def generate_activity_sequence(self, 
                              scenario: str,
                              duration: int = None,
                              sample_interval: float = None) -> List[SyntheticUserActivity]:
        """개발자 작업 환경에 최적화된 활동 시퀀스 생성"""
        
        if duration is None:
            duration = self.config['sequence_length']
        if sample_interval is None:
            sample_interval = self.config['sample_interval']
        
        scenario_config = SCENARIO_CONFIGS[scenario]
        duration_multiplier = scenario_config['duration_multiplier']
        adjusted_duration = int(duration * duration_multiplier)
        
        logger.info(f"개발자 작업 환경 시퀀스 생성 시작 (지속시간: {adjusted_duration}초)")
        
        activities = []
        start_time = time.time()
        
        # 윈도우 풀 생성
        windows = self.generate_window_pool(scenario)
        
        # 개발자 작업 패턴 시뮬레이션
        for i in range(int(adjusted_duration / sample_interval)):
            timestamp = start_time + i * sample_interval
            
            # 윈도우 이동 시뮬레이션 (개발 환경에서는 적게 이동)
            if i > 0:
                windows = self.simulate_window_movement(
                    windows, timestamp, scenario_config['window_movement']
                )
            
            # 개발자 활동 패턴 결정
            if i < adjusted_duration * 0.3:  # 초기 30%: 설정 및 준비
                activity_type = random.choices(
                    ['window_change', 'click', 'idle'],
                    weights=[0.4, 0.4, 0.2]
                )[0]
            elif i < adjusted_duration * 0.8:  # 중간 50%: 집중 코딩
                activity_type = random.choices(
                    ['keyboard', 'idle', 'click'],
                    weights=[0.5, 0.4, 0.1]
                )[0]
            else:  # 마지막 20%: 검토 및 정리
                activity_type = random.choices(
                    ['click', 'window_change', 'idle'],
                    weights=[0.4, 0.3, 0.3]
                )[0]
            
            # 마우스 위치 설정 (개발 환경에 최적화)
            if activity_type == 'click':
                # IDE 내부 클릭 (코드 영역, 툴바 등)
                ide_windows = [w for w in windows if w.process_name in ['devenv.exe', 'Code.exe']]
                if ide_windows:
                    active_window = random.choice(ide_windows)
                    x = random.randint(active_window.rect[0] + 100, active_window.rect[2] - 100)
                    y = random.randint(active_window.rect[1] + 50, active_window.rect[3] - 50)
                else:
                    x = random.randint(100, self.config['screen_width'] - 100)
                    y = random.randint(100, self.config['screen_height'] - 100)
            else:
                x = random.randint(100, self.config['screen_width'] - 100)
                y = random.randint(100, self.config['screen_height'] - 100)
            
            # 활성 윈도우 결정 (개발 도구 우선)
            if activity_type == 'window_change':
                # 개발 도구를 더 자주 활성화
                dev_windows = [w for w in windows if w.process_name in ['devenv.exe', 'Code.exe']]
                if dev_windows and random.random() < 0.7:  # 70% 확률로 개발 도구 선택
                    active_window = random.choice(dev_windows)
                else:
                    active_window = random.choice(windows)
            else:
                # 마우스 위치와 가장 가까운 윈도우
                active_window = min(windows, 
                                key=lambda w: abs(w.rect[0] - x) + abs(w.rect[1] - y))
            
            # 키보드 활동 여부
            keyboard_active = activity_type == 'keyboard'
            
            activity = SyntheticUserActivity(
                timestamp=timestamp,
                active_window=active_window,
                all_windows=windows.copy(),
                mouse_position=(x, y),
                keyboard_active=keyboard_active,
                activity_type=activity_type
            )
            
            activities.append(activity)
        
        logger.info(f"개발자 작업 환경 시퀀스 생성 완료: {len(activities)}개 샘플")
        return activities
    
    def generate_training_sequences(self, 
                                   num_sequences: int = None,
                                   scenarios: List[str] = None) -> List[Tuple]:
        """훈련용 시퀀스 데이터 생성 - 정확한 수의 시퀀스 생성"""
        
        if num_sequences is None:
            num_sequences = self.config['num_sequences']
        if scenarios is None:
            scenarios = list(SCENARIO_CONFIGS.keys())
        
        logger.info(f"훈련용 시퀀스 데이터 생성 시작: {num_sequences}개 시퀀스")
        
        training_data = []
        overlap_seconds = self.config['overlap_seconds']
        
        for seq_id in range(num_sequences):
            # 시나리오 선택 (균등 분포)
            scenario = scenarios[seq_id % len(scenarios)]
            
            # 활동 시퀀스 생성
            activities = self.generate_activity_sequence(scenario)
            
            # 하나의 활동 시퀀스에서 하나의 훈련 시퀀스만 생성
            # overlap은 시퀀스 간의 연속성을 위한 것이므로 과도한 중복 방지
            if len(activities) >= self.config['sequence_length']:
                # 첫 번째 시퀀스만 사용 (overlap으로 인한 중복 방지)
                sequence = activities[:self.config['sequence_length']]
                
                # 윈도우 시퀀스
                window_seq = [activity.all_windows for activity in sequence]
                
                # 다음 시퀀스의 윈도우 위치를 목표로 설정
                if len(activities) > self.config['sequence_length']:
                    next_activity = activities[self.config['sequence_length']]
                    target_pos = []
                    
                    for window in next_activity.all_windows[:self.config['max_windows']]:
                        x, y = window.rect[0], window.rect[1]
                        w = window.rect[2] - window.rect[0]
                        h = window.rect[3] - window.rect[1]
                        target_pos.append((x, y, w, h))
                    
                    # 최대 윈도우 수로 패딩
                    while len(target_pos) < self.config['max_windows']:
                        target_pos.append((0, 0, 100, 100))
                else:
                    # 마지막 시퀀스인 경우 현재 윈도우 위치를 목표로 사용
                    last_activity = activities[-1]
                    target_pos = []
                    
                    for window in last_activity.all_windows[:self.config['max_windows']]:
                        x, y = window.rect[0], window.rect[1]
                        w = window.rect[2] - window.rect[0]
                        h = window.rect[3] - window.rect[1]
                        target_pos.append((x, y, w, h))
                    
                    # 최대 윈도우 수로 패딩
                    while len(target_pos) < self.config['max_windows']:
                        target_pos.append((0, 0, 100, 100))
                
                training_data.append((window_seq, sequence, target_pos))
        
        logger.info(f"훈련용 시퀀스 데이터 생성 완료: {len(training_data)}개 시퀀스")
        return training_data
    
    def save_training_data(self, 
                          training_data: List[Tuple], 
                          filename: str = None,
                          output_dir: str = None) -> str:
        """훈련 데이터를 JSON 파일로 저장"""
        
        if output_dir is None:
            output_dir = OUTPUT_CONFIG['base_dir']
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{OUTPUT_CONFIG['filename_prefix']}_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            # 데이터를 JSON 직렬화 가능한 형태로 변환
            export_data = []
            
            for window_seq, activity_seq, target_pos in training_data:
                export_item = {
                    'window_sequences': [[asdict(w) for w in seq] for seq in window_seq],
                    'activity_sequences': [asdict(a) for a in activity_seq],
                    'target_positions': target_pos
                }
                export_data.append(export_item)
            
            # 메타데이터 추가
            if OUTPUT_CONFIG['metadata']:
                metadata = {
                    'generation_info': {
                        'timestamp': datetime.now().isoformat(),
                        'generator_version': '1.0.0',
                        'config': self.config,
                        'num_sequences': len(training_data)
                    },
                    'data_structure': {
                        'sequence_length': self.config['sequence_length'],
                        'max_windows': self.config['max_windows'],
                        'sample_interval': self.config['sample_interval']
                    }
                }
                
                final_data = {
                    'metadata': metadata,
                    'training_data': export_data
                }
            else:
                final_data = export_data
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"훈련 데이터 저장 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"훈련 데이터 저장 실패: {e}")
            raise

if __name__ == "__main__":
    # 테스트 실행
    generator = SyntheticDataGenerator()
    
    # 간단한 테스트
    test_activities = generator.generate_activity_sequence('browsing', duration=10)
    print(f"테스트 시퀀스 생성 완료: {len(test_activities)}개 샘플")
    
    # 훈련 데이터 생성 테스트
    test_training_data = generator.generate_training_sequences(num_sequences=5)
    print(f"테스트 훈련 데이터 생성 완료: {len(test_training_data)}개 시퀀스")
