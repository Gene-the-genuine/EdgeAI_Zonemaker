
#!/usr/bin/env python3
"""
Zonemaker AI - AI 기반 윈도우 배열 최적화 시스템
실행 스크립트
"""
from typing import List, Dict, Optional, Tuple
import argparse
import sys
import os
import time
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_backend():
    """백엔드 API 서버 실행"""
    try:
        from backend.api.main import app
        import uvicorn
        
        logger.info("백엔드 API 서버 시작...")
        uvicorn.run(
            app, 
            host="127.0.0.1", 
            port=8000, 
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"백엔드 실행 실패: {e}")
        return False
    return True

def run_data_collection(duration: int = 30):
    """데이터 수집 실행 (30초 관찰)"""
    try:
        from backend.core.data_collector import DataCollector
        
        logger.info(f"데이터 수집 테스트 시작 (지속 시간: {duration}초)")
        
        collector = DataCollector()
        
        # 30초간 데이터 수집
        activities = collector.collect_data_sample(duration)
        
        if activities:
            # 데이터 저장
            filename = f"data/window_activity_data_{int(time.time())}.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            collector.save_data(activities, filename)
            
            logger.info(f"데이터 수집 완료: {len(activities)}개 샘플, 저장: {filename}")
            return True
        else:
            logger.error("데이터 수집 실패: 샘플이 수집되지 않았습니다.")
            return False
            
    except Exception as e:
        logger.error(f"데이터 수집 실패: {e}")
        return False

# def run_training(data_file: str = None, epochs: int = 50):
#     """모델 훈련 실행"""
#     try:
#         from backend.ml.model import create_model
#         from backend.ml.trainer import ModelTrainer
        
#         logger.info("모델 훈련 시작...")
        
#         # 모델 생성
#         model = create_model()
#         logger.info("모델 생성 완료")
        
#         # 훈련기 생성
#         trainer = ModelTrainer(model)
        
#         # 훈련 데이터 준비
#         if not data_file:
#             # 최신 데이터 파일 찾기
#             data_dir = Path("data")
#             if data_dir.exists():
#                 data_files = list(data_dir.glob("window_activity_data_*.json"))
#                 if data_files:
#                     data_file = str(max(data_files, key=lambda x: x.stat().st_mtime))
#                     logger.info(f"자동으로 데이터 파일 선택: {data_file}")
#                 else:
#                     logger.error("훈련할 데이터 파일을 찾을 수 없습니다.")
#                     return False
#             else:
#                 logger.error("데이터 디렉토리가 존재하지 않습니다.")
#                 return False
        
#         # 훈련 데이터 준비
#         train_loader, val_loader = trainer.prepare_training_data(
#             data_file, 
#             train_split=0.8, 
#             batch_size=4,  # 메모리 절약을 위해 작은 배치 크기
#             sequence_length=30
#         )
        
#         # 훈련 실행
#         save_dir = "data/models"
#         os.makedirs(save_dir, exist_ok=True)
        
#         training_history = trainer.train(
#             train_loader, 
#             val_loader, 
#             num_epochs=epochs,
#             patience=15,
#             save_dir=save_dir
#         )
        
#         logger.info(f"훈련 완료! 최종 손실: {training_history['val_loss'][-1]:.4f}")
#         return True
        
#     except Exception as e:
#         logger.error(f"모델 훈련 실패: {e}")
#         return False

### 수정된 부분 (trainer.py에 추가적으로 리팩토링하면 좋을 듯.)

def run_training(data_file: str = None, epochs: int = 50):
    """모델 훈련 실행 - 합성 데이터 지원 추가"""
    try:
        from backend.ml.model import create_model
        from backend.ml.trainer import ModelTrainer
        
        logger.info("모델 훈련 시작...")
        
        # 모델 생성
        model = create_model()
        logger.info("모델 생성 완료")
        
        # 훈련기 생성
        trainer = ModelTrainer(model)
        
        # 훈련 데이터 준비
        if not data_file:
            # 최신 데이터 파일 찾기 (합성 데이터 포함)
            data_dir = Path("data")
            if data_dir.exists():
                # 합성 데이터 우선 검색
                synthetic_files = list(data_dir.glob("converted_*_synthetic_*.json"))
                regular_files = list(data_dir.glob("window_activity_data_*.json"))
                
                if synthetic_files:
                    data_file = str(max(synthetic_files, key=lambda x: x.stat().st_mtime))
                    logger.info(f"합성 데이터 파일 자동 선택: {data_file}")
                elif regular_files:
                    data_file = str(max(regular_files, key=lambda x: x.stat().st_mtime))
                    logger.info(f"일반 데이터 파일 자동 선택: {data_file}")
                else:
                    logger.error("훈련할 데이터 파일을 찾을 수 없습니다.")
                    return False
            else:
                logger.error("데이터 디렉토리가 존재하지 않습니다.")
                return False
        
        # 데이터 파일 타입 확인 및 적절한 처리 방법 선택
        if "synthetic" in data_file or "converted" in data_file:
            logger.info("합성 데이터 파일 감지 - 합성 데이터 어댑터 사용")
            # 합성 데이터 어댑터를 통한 데이터 준비
            train_loader, val_loader = prepare_synthetic_training_data(
                data_file, 
                train_split=0.8, 
                batch_size=4,  # 메모리 절약을 위해 작은 배치 크기
                sequence_length=30
            )
        else:
            logger.info("일반 데이터 파일 - 표준 데이터 로더 사용")
            # 기존 방식으로 데이터 준비
            train_loader, val_loader = trainer.prepare_training_data(
                data_file, 
                train_split=0.8, 
                batch_size=4,
                sequence_length=30
            )
        
        # 훈련 실행
        save_dir = "data/models"
        os.makedirs(save_dir, exist_ok=True)
        
        training_history = trainer.train(
            train_loader, 
            val_loader, 
            num_epochs=epochs,
            patience=15,
            save_dir=save_dir
        )
        
        logger.info(f"훈련 완료! 최종 손실: {training_history['val_loss'][-1]:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"모델 훈련 실패: {e}")
        return False

def prepare_synthetic_training_data(data_file: str,
                                  train_split: float = 0.8,
                                  batch_size: int = 4,
                                  sequence_length: int = 30):
    """합성 데이터를 위한 훈련 데이터 준비"""
    try:
        import json
        from backend.ml.trainer import WindowActivityDataset, custom_collate_fn
        from backend.core.data_collector import UserActivity, WindowInfo
        from torch.utils.data import DataLoader
        
        logger.info(f"합성 데이터 파일 로드: {data_file}")
        
        def load_data(filename: str) -> List[UserActivity]:
            """JSON 파일에서 데이터 로드"""
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # JSON 데이터를 UserActivity 객체로 변환
                activities = []
                for item in data:
                    # WindowInfo 객체들 복원
                    active_window = None
                    if item.get('active_window'):
                        active_window = WindowInfo(**item['active_window'])
                    
                    all_windows = [WindowInfo(**w) for w in item.get('all_windows', [])]
                    
                    activity = UserActivity(
                        timestamp=item['timestamp'],
                        active_window=active_window,
                        all_windows=all_windows,
                        mouse_position=tuple(item['mouse_position']),
                        keyboard_active=item['keyboard_active'],
                        activity_type=item['activity_type']
                    )
                    activities.append(activity)
                
                logger.info(f"데이터 로드 완료: {filename} ({len(activities)}개 샘플)")
                return activities
                
            except Exception:
                logger.error(f"데이터 로드 실패")
                return []

        # # 합성 데이터 파일 로드
        # with open(data_file, 'r', encoding='utf-8') as f:
        #     data = json.load(f)

        activities = load_data(data_file)
        
        # # 합성 데이터 구조 확인 및 변환
        # if 'training_data' in data:
        #     # 합성 데이터 어댑터 형식
        #     activities = data['training_data']
        #     logger.info(f"합성 데이터 로드 완료: {len(activities)}개 활동")
        # else:
        #     # 직접 변환된 형식
        #     activities = data
        #     logger.info(f"변환된 데이터 로드 완료: {len(activities)}개 활동")
        
        if not activities:
            raise ValueError("활동 데이터가 비어있습니다.")
        
        # 시퀀스 단위로 그룹화
        window_sequences, activity_sequences, target_positions = prepare_synthetic_sequences(
            activities, sequence_length
        )
        
        logger.info(f"시퀀스 변환 완료: 윈도우 {len(window_sequences)}개, "
                    f"활동 {len(activity_sequences)}개, 타겟 {len(target_positions)}개")
        
        if not window_sequences:
            raise ValueError("시퀀스 데이터가 생성되지 않았습니다.")
        
        # 훈련/검증 분할
        num_samples = len(window_sequences)
        train_size = int(num_samples * train_split)
        
        logger.info(f"데이터 분할: 전체 {num_samples}개, 훈련 {train_size}개, 검증 {num_samples - train_size}개")
        
        train_windows = window_sequences[:train_size]
        train_activities = activity_sequences[:train_size]
        train_targets = target_positions[:train_size]
        
        val_windows = window_sequences[train_size:]
        val_activities = activity_sequences[train_size:]
        val_targets = target_positions[train_size:]
        
        # 데이터셋 생성
        train_dataset = WindowActivityDataset(
            train_windows, train_activities, train_targets, sequence_length
        )
        val_dataset = WindowActivityDataset(
            val_windows, val_activities, val_targets, sequence_length
        )
        
        # 데이터로더 생성 (커스텀 collate_fn 사용)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Windows에서 안정성을 위해 0으로 설정
            collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        logger.info(f"합성 데이터 훈련 준비 완료: 훈련 {len(train_dataset)}개, 검증 {len(val_dataset)}개")
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"합성 데이터 훈련 준비 실패: {e}")
        raise

def prepare_synthetic_sequences(activities, sequence_length: int):
    """합성 데이터를 시퀀스로 변환"""
    from backend.core.data_collector import UserActivity, WindowInfo
    
    window_sequences = []
    activity_sequences = []
    target_positions = []
    
    # 시퀀스 단위로 그룹화
    for i in range(0, len(activities), sequence_length):
        sequence_activities = activities[i:i + sequence_length]
        
        if len(sequence_activities) < sequence_length:
            continue  # 완전한 시퀀스가 아닌 경우 건너뛰기
        
        # 윈도우 시퀀스
        window_seq = []
        for activity in sequence_activities:
            if hasattr(activity, 'all_windows'):
                # UserActivity 객체인 경우
                window_seq.append(activity.all_windows)
            elif isinstance(activity, dict) and 'all_windows' in activity:
                # 딕셔너리인 경우
                window_seq.append(activity['all_windows'])
            else:
                logger.warning(f"활동 데이터 형식 오류: {type(activity)}")
                continue
        
        # 다음 시퀀스의 윈도우 위치를 목표로 설정
        target_pos = []
        if i + sequence_length < len(activities):
            # 다음 시퀀스가 있는 경우: 실제 다음 윈도우 위치 사용
            next_activity = activities[i + sequence_length]
            if hasattr(next_activity, 'all_windows'):
                windows = next_activity.all_windows
            elif isinstance(next_activity, dict) and 'all_windows' in next_activity:
                windows = next_activity['all_windows']
            else:
                windows = []
            
            for w in windows:
                if hasattr(w, 'rect'):
                    rect = w.rect
                elif isinstance(w, dict) and 'rect' in w:
                    rect = w['rect']
                else:
                    continue
                
                target_pos.append((rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]))
        else:
            # 마지막 시퀀스인 경우: 현재 윈도우 위치를 목표로 사용
            last_activity = sequence_activities[-1]
            if hasattr(last_activity, 'all_windows'):
                windows = last_activity.all_windows
            elif isinstance(last_activity, dict) and 'all_windows' in last_activity:
                windows = last_activity['all_windows']
            else:
                windows = []
            
            for w in windows:
                if hasattr(w, 'rect'):
                    rect = w.rect
                elif isinstance(w, dict) and 'rect' in w:
                    rect = w['rect']
                else:
                    continue
                
                target_pos.append((rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]))
        
        # 최대 윈도우 수에 맞춰 패딩 (모델의 max_windows = 20)
        max_windows = 20
        while len(target_pos) < max_windows:
            target_pos.append((0, 0, 100, 100))  # 기본값
        target_pos = target_pos[:max_windows]
        
        # 시퀀스 추가
        window_sequences.append(window_seq)
        activity_sequences.append(sequence_activities)
        target_positions.append(target_pos)
    
    logger.info(f"합성 시퀀스 준비 완료: {len(window_sequences)}개 시퀀스")
    return window_sequences, activity_sequences, target_positions

### 수정된 부분 여기까지.

def run_inference(model_path: str = None, duration: int = 60):
    """실시간 추론 실행"""
    try:
        from backend.ml.inference import RealTimeInferenceEngine
        
        logger.info("실시간 추론 엔진 시작...")
        
        # 모델 경로 설정
        if not model_path:
            # 최신 모델 파일 찾기
            models_dir = Path("data/models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.pth"))
                if model_files:
                    model_path = str(max(model_files, key=lambda x: x.stat().st_mtime))
                    logger.info(f"자동으로 모델 파일 선택: {model_path}")
                else:
                    logger.error("추론할 모델 파일을 찾을 수 없습니다.")
                    return False
            else:
                logger.error("모델 디렉토리가 존재하지 않습니다.")
                return False
        
        # 추론 엔진 생성
        engine = RealTimeInferenceEngine(model_path, prediction_interval=1.0)
        
        # 추론 시작
        if engine.start_inference():
            logger.info(f"추론 엔진이 {duration}초간 실행됩니다...")
            time.sleep(duration)
            
            # 추론 중지
            try:
                engine.stop_inference()
                
                # 상태 및 통계 출력 (안전하게)
                try:
                    status = engine.get_inference_status()
                    logger.info(f"추론 완료 - 상태: {status}")
                except Exception as status_error:
                    logger.warning(f"상태 확인 실패: {status_error}")
                
                # 로그 저장 (안전하게)
                try:
                    engine.save_inference_log()
                except Exception as log_error:
                    logger.warning(f"로그 저장 실패: {log_error}")
                
                return True
                
            except Exception as stop_error:
                logger.error(f"추론 엔진 중지 중 오류: {stop_error}")
                return False
        else:
            logger.error("추론 엔진 시작 실패")
            return False
            
    except Exception as e:
        logger.error(f"실시간 추론 실패: {e}")
        return False

def run_npu_conversion(model_path: str = None):
    """NPU 변환 실행"""
    try:
        from backend.ml.npu_converter import NPUConverter
        
        logger.info("NPU 변환 시작...")
        
        # 모델 경로 설정
        if not model_path:
            # 최신 모델 파일 찾기
            models_dir = Path("data/models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.pth"))
                if model_files:
                    model_path = str(max(model_files, key=lambda x: x.stat().st_mtime))
                    logger.info(f"자동으로 모델 파일 선택: {model_path}")
                else:
                    logger.error("변환할 모델 파일을 찾을 수 없습니다.")
                    return False
            else:
                logger.error("모델 디렉토리가 존재하지 않습니다.")
                return False
        
        # NPU 변환기 생성
        converter = NPUConverter(model_path)
        
        # NPU 변환 실행
        success = converter.convert_to_npu()
        
        if success:
            # 벤치마크 실행
            benchmark_results = converter.benchmark_model()
            logger.info(f"NPU 변환 완료! 벤치마크 결과: {benchmark_results}")
            return True
        else:
            logger.error("NPU 변환 실패")
            return False
            
    except Exception as e:
        logger.error(f"NPU 변환 실패: {e}")
        return False

def run_frontend():
    """프론트엔드 GUI 실행"""
    try:
        from frontend.main import MainWindow
        from PySide6.QtWidgets import QApplication
        
        logger.info("프론트엔드 GUI 시작...")
        
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"프론트엔드 실행 실패: {e}")
        return False

def run_all():
    """전체 시스템 실행 (백엔드 + 프론트엔드)"""
    try:
        import threading
        import time
        
        logger.info("전체 시스템 실행 시작...")
        
        # 백엔드를 별도 스레드에서 실행
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # 백엔드 시작 대기
        time.sleep(3)
        
        # 프론트엔드 실행
        run_frontend()
        
    except Exception as e:
        logger.error(f"전체 시스템 실행 실패: {e}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Zonemaker AI - AI 기반 윈도우 배열 최적화 시스템"
    )
    
    parser.add_argument(
        '--mode', 
        choices=['backend', 'data-collect', 'train', 'inference', 'npu-convert', 'frontend', 'all'],
        default='all',
        help='실행 모드 선택'
    )
    
    parser.add_argument(
        '--duration', 
        type=int, 
        default=30,
        help='데이터 수집 또는 추론 지속 시간 (초)'
    )
    
    parser.add_argument(
        '--data-file', 
        type=str,
        help='훈련 또는 추론에 사용할 데이터 파일 경로'
    )
    
    parser.add_argument(
        '--model-path', 
        type=str,
        help='훈련, 추론 또는 NPU 변환에 사용할 모델 파일 경로'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='훈련 에포크 수'
    )
    
    args = parser.parse_args()
    
    # 프로젝트 정보 출력
    print("=" * 60)
    print("Zonemaker AI - AI 기반 윈도우 배열 최적화 시스템")
    print("=" * 60)
    print(f"실행 모드: {args.mode}")
    print(f"프로젝트 루트: {project_root}")
    print("=" * 60)
    
    # 모드별 실행
    success = False
    
    try:
        if args.mode == 'backend':
            success = run_backend()
        elif args.mode == 'data-collect':
            success = run_data_collection(args.duration)
        elif args.mode == 'train':
            success = run_training(args.data_file, args.epochs)
        elif args.mode == 'inference':
            success = run_inference(args.model_path, args.duration)
        elif args.mode == 'npu-convert':
            success = run_npu_conversion(args.model_path)
        elif args.mode == 'frontend':
            success = run_frontend()
        elif args.mode == 'all':
            success = run_all()
        elif args.mode == 'demo':
            success = run_demo(args.duration)
        elif args.mode == 'competition-demo':
            success = run_competition_demo()
        else:
            logger.error(f"알 수 없는 실행 모드: {args.mode}")
            return False
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        return True
    except Exception as e:
        logger.error(f"실행 중 예상치 못한 오류: {e}")
        return False
    
    if success:
        logger.info(f"{args.mode} 모드 실행 완료")
    else:
        logger.error(f"{args.mode} 모드 실행 실패")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)