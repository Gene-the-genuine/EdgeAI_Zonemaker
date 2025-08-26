"""
합성 데이터를 메인 프로젝트 훈련기와 연결하는 어댑터
"""

import json
import os
import sys
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# 상위 디렉토리의 backend 모듈을 import하기 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.data_collector import UserActivity, WindowInfo
from backend.ml.trainer import ModelTrainer, prepare_training_data

logger = logging.getLogger(__name__)

class SyntheticDataAdapter:
    """합성 데이터를 메인 프로젝트 형식으로 변환하는 어댑터"""
    
    def __init__(self, synthetic_data_file: str):
        self.synthetic_data_file = synthetic_data_file
        self.original_data = None
        self.converted_data = None
        
        logger.info(f"합성 데이터 어댑터 초기화: {synthetic_data_file}")
    
    def load_synthetic_data(self) -> Dict:
        """합성 데이터 파일 로드"""
        try:
            with open(self.synthetic_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.original_data = data
            logger.info(f"합성 데이터 로드 완료: {self.synthetic_data_file}")
            
            # 메타데이터 확인
            if 'metadata' in data:
                metadata = data['metadata']
                logger.info(f"데이터 정보: {metadata['generation_info']['num_sequences']}개 시퀀스")
                logger.info(f"생성 시간: {metadata['generation_info']['timestamp']}")
            
            return data
            
        except Exception as e:
            logger.error(f"합성 데이터 로드 실패: {e}")
            raise
    
    def convert_to_main_format(self) -> List[UserActivity]:
        """합성 데이터를 메인 프로젝트 형식으로 변환"""
        
        if not self.original_data:
            raise ValueError("먼저 합성 데이터를 로드해야 합니다.")
        
        # 데이터 구조 확인
        if 'training_data' in self.original_data:
            training_data = self.original_data['training_data']
        else:
            training_data = self.original_data
        
        logger.info("데이터 형식 변환 시작...")
        
        converted_activities = []
        
        for seq_idx, sequence_data in enumerate(training_data):
            if 'activity_sequences' in sequence_data:
                # 새로운 형식 (training_data)
                activity_sequences = sequence_data['activity_sequences']
            else:
                # 기존 형식 (직접 UserActivity 리스트)
                activity_sequences = sequence_data
            
            for activity_data in activity_sequences:
                try:
                    # WindowInfo 객체들 변환
                    all_windows = []
                    for window_data in activity_data.get('all_windows', []):
                        window = WindowInfo(
                            handle=window_data.get('handle', 0),
                            title=window_data.get('title', ''),
                            class_name=window_data.get('class_name', ''),
                            process_id=window_data.get('process_id', 0),
                            process_name=window_data.get('process_name', ''),
                            rect=tuple(window_data.get('rect', [0, 0, 100, 100])),
                            is_visible=bool(window_data.get('is_visible', True)),
                            is_minimized=bool(window_data.get('is_minimized', False)),
                            is_maximized=bool(window_data.get('is_maximized', False)),
                            z_order=window_data.get('z_order', 0),
                            timestamp=float(window_data.get('timestamp', 0.0))
                        )
                        all_windows.append(window)
                    
                    # active_window 변환
                    active_window = None
                    if activity_data.get('active_window'):
                        active_data = activity_data['active_window']
                        active_window = WindowInfo(
                            handle=active_data.get('handle', 0),
                            title=active_data.get('title', ''),
                            class_name=active_data.get('class_name', ''),
                            process_id=active_data.get('process_id', 0),
                            process_name=active_data.get('process_name', ''),
                            rect=tuple(active_data.get('rect', [0, 0, 100, 100])),
                            is_visible=bool(active_data.get('is_visible', True)),
                            is_minimized=bool(active_data.get('is_minimized', False)),
                            is_maximized=bool(active_data.get('is_maximized', False)),
                            z_order=active_data.get('z_order', 0),
                            timestamp=float(active_data.get('timestamp', 0.0))
                        )
                    
                    # UserActivity 객체 생성
                    activity = UserActivity(
                        timestamp=float(activity_data.get('timestamp', 0.0)),
                        active_window=active_window,
                        all_windows=all_windows,
                        mouse_position=tuple(activity_data.get('mouse_position', [0, 0])),
                        keyboard_active=bool(activity_data.get('keyboard_active', False)),
                        activity_type=activity_data.get('activity_type', 'idle')
                    )
                    
                    converted_activities.append(activity)
                    
                except Exception as e:
                    logger.warning(f"활동 데이터 변환 실패 (시퀀스 {seq_idx}): {e}")
                    continue
        
        self.converted_data = converted_activities
        logger.info(f"데이터 형식 변환 완료: {len(converted_activities)}개 활동")
        
        return converted_activities
    
    def save_converted_data(self, output_file: str = None) -> str:
        """변환된 데이터를 메인 프로젝트 형식으로 저장"""
        
        if not self.converted_data:
            raise ValueError("먼저 데이터를 변환해야 합니다.")
        
        if not output_file:
            base_name = os.path.splitext(os.path.basename(self.synthetic_data_file))[0]
            output_file = f"converted_{base_name}.json"
        
        # 메인 프로젝트의 data 디렉토리에 저장
        main_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        output_path = os.path.join(main_data_dir, output_file)
        
        try:
            # 데이터를 JSON 직렬화 가능한 형태로 변환
            data_to_save = []
            for activity in self.converted_data:
                activity_dict = {
                    'timestamp': activity.timestamp,
                    'active_window': None,
                    'all_windows': [],
                    'mouse_position': activity.mouse_position,
                    'keyboard_active': activity.keyboard_active,
                    'activity_type': activity.activity_type
                }
                
                # active_window 변환
                if activity.active_window:
                    activity_dict['active_window'] = {
                        'handle': activity.active_window.handle,
                        'title': activity.active_window.title,
                        'class_name': activity.active_window.class_name,
                        'process_id': activity.active_window.process_id,
                        'process_name': activity.active_window.process_name,
                        'rect': list(activity.active_window.rect),
                        'is_visible': activity.active_window.is_visible,
                        'is_minimized': activity.active_window.is_minimized,
                        'is_maximized': activity.active_window.is_maximized,
                        'z_order': activity.active_window.z_order,
                        'timestamp': activity.active_window.timestamp
                    }
                
                # all_windows 변환
                for window in activity.all_windows:
                    window_dict = {
                        'handle': window.handle,
                        'title': window.title,
                        'class_name': window.class_name,
                        'process_id': window.process_id,
                        'process_name': window.process_name,
                        'rect': list(window.rect),
                        'is_visible': window.is_visible,
                        'is_minimized': window.is_minimized,
                        'is_maximized': window.is_maximized,
                        'z_order': window.z_order,
                        'timestamp': window.timestamp
                    }
                    activity_dict['all_windows'].append(window_dict)
                
                data_to_save.append(activity_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
            logger.info(f"변환된 데이터 저장 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"변환된 데이터 저장 실패: {e}")
            raise
    
    def prepare_training_data(self, 
                            train_split: float = 0.8,
                            batch_size: int = 8,
                            sequence_length: int = 30) -> Tuple:
        """훈련 데이터 준비 (메인 프로젝트 훈련기와 직접 연결)"""
        
        if not self.converted_data:
            raise ValueError("먼저 데이터를 변환해야 합니다.")
        
        logger.info("훈련 데이터 준비 시작...")
        
        try:
            # 메인 프로젝트의 prepare_training_data 함수 사용
            train_loader, val_loader = prepare_training_data(
                data_file=None,  # 이미 메모리에 있음
                train_split=train_split,
                batch_size=batch_size,
                sequence_length=sequence_length
            )
            
            logger.info("훈련 데이터 준비 완료")
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"훈련 데이터 준비 실패: {e}")
            raise

def main():
    """메인 함수 - 데이터 변환 및 연결 테스트"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="합성 데이터 어댑터")
    parser.add_argument('synthetic_data_file', help='합성 데이터 파일 경로')
    parser.add_argument('--output-file', help='출력 파일명')
    parser.add_argument('--test-training', action='store_true', help='훈련 데이터 준비 테스트')
    
    args = parser.parse_args()
    
    try:
        # 어댑터 생성
        adapter = SyntheticDataAdapter(args.synthetic_data_file)
        
        # 합성 데이터 로드
        adapter.load_synthetic_data()
        
        # 메인 프로젝트 형식으로 변환
        converted_data = adapter.convert_to_main_format()
        
        # 변환된 데이터 저장
        output_path = adapter.save_converted_data(args.output_file)
        
        logger.info("=" * 60)
        logger.info("데이터 변환 완료!")
        logger.info("=" * 60)
        logger.info(f"원본 파일: {args.synthetic_data_file}")
        logger.info(f"변환된 파일: {output_path}")
        logger.info(f"활동 수: {len(converted_data)}")
        logger.info("=" * 60)
        
        # 훈련 데이터 준비 테스트
        if args.test_training:
            logger.info("훈련 데이터 준비 테스트 시작...")
            try:
                train_loader, val_loader = adapter.prepare_training_data()
                logger.info("훈련 데이터 준비 테스트 성공!")
                logger.info(f"훈련 배치 수: {len(train_loader)}")
                logger.info(f"검증 배치 수: {len(val_loader)}")
            except Exception as e:
                logger.error(f"훈련 데이터 준비 테스트 실패: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"데이터 변환 실패: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
