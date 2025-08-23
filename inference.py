import torch
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import json
import os
from datetime import datetime

# from .model import ZonemakerViT, DataProcessor, ModelManager
# from ..core.window_manager import WindowManager
from ml.model import ZonemakerViT, DataProcessor, ModelManager
from core.window_manager import WindowManager

class InferenceEngine:
    """실시간 추론 엔진"""
    
    def __init__(self, 
                 model_path: str,
                 models_dir: str = "data/models"):
        self.model_manager = ModelManager(models_dir)
        self.window_manager = WindowManager()
        self.processor = DataProcessor()
        
        # 모델 로드
        self.model, self.metadata = self.model_manager.load_model(model_path)
        self.model.eval()
        
        # 성능 모니터링
        self.inference_times = []
        self.is_running = False
        
        print(f"🚀 추론 엔진 초기화 완료")
        print(f"📊 모델: {model_path}")
        print(f"�� 모델 크기: {self.model.get_model_size_mb():.2f} MB")
    
    def start_inference_loop(self, 
                           data_collector,
                           interval: float = 1.0,
                           max_duration: Optional[int] = None):
        """추론 루프 시작"""
        self.is_running = True
        start_time = time.time()
        
        print(f"🔄 추론 루프 시작 (간격: {interval}초)")
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # 데이터 수집
                window_data = data_collector.collect_window_data()
                
                if window_data:
                    # 추론 실행
                    predictions = self.infer_window_positions(window_data)
                    
                    # 윈도우 배열 적용
                    if predictions is not None:
                        self.apply_predictions(window_data, predictions)
                    
                    # 성능 측정
                    inference_time = time.time() - loop_start
                    self.inference_times.append(inference_time)
                    
                    # 성능 출력
                    if len(self.inference_times) % 10 == 0:  # 10회마다 출력
                        avg_time = np.mean(self.inference_times[-10:])
                        print(f"📊 평균 추론 시간: {avg_time*1000:.1f}ms")
                    
                    # 시간 제약 확인
                    if inference_time > 0.5:
                        print(f"⚠️ 추론 시간 초과: {inference_time*1000:.1f}ms (목표: 500ms)")
                
                # 지속 시간 확인
                if max_duration and (time.time() - start_time) > max_duration:
                    print(f"⏰ 최대 실행 시간 도달: {max_duration}초")
                    break
                
                # 다음 추론까지 대기
                elapsed = time.time() - loop_start
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중지됨")
        except Exception as e:
            print(f"❌ 추론 루프 오류: {e}")
        finally:
            self.is_running = False
            self.print_performance_summary()
    
    def stop_inference_loop(self):
        """추론 루프 중지"""
        self.is_running = False
        print("⏹️ 추론 루프 중지 요청됨")
    
    def infer_window_positions(self, window_data: List[dict]) -> Optional[np.ndarray]:
        """윈도우 위치 추론"""
        try:
            # 데이터 전처리
            input_tensor = self.processor.normalize_features(window_data)
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # 추론 실행
            with torch.no_grad():
                start_time = time.time()
                predictions = self.model(input_tensor)
                inference_time = time.time() - start_time
                
                # 성능 기록
                self.inference_times.append(inference_time)
            
            # 결과 후처리
            predictions_np = predictions.numpy()
            denormalized = self.processor.denormalize_predictions(predictions)
            
            # 화면 경계 검사 및 조정
            adjusted_predictions = self._adjust_predictions_to_screen(denormalized)
            
            return adjusted_predictions
            
        except Exception as e:
            print(f"❌ 추론 실패: {e}")
            return None
    
    def _adjust_predictions_to_screen(self, predictions: np.ndarray) -> np.ndarray:
        """예측 결과를 화면 경계 내로 조정"""
        screen_width, screen_height = 1920, 1080
        
        adjusted = predictions.copy()
        
        for i in range(len(adjusted)):
            # x, y, w, h 순서
            x, y, w, h = adjusted[i]
            
            # 너비와 높이 제한
            w = max(100, min(w, screen_width))
            h = max(100, min(h, screen_height))
            
            # 위치 조정 (화면 밖으로 나가지 않도록)
            x = max(0, min(x, screen_width - w))
            y = max(0, min(y, screen_height - h))
            
            adjusted[i] = [x, y, w, h]
        
        return adjusted
    
    def apply_predictions(self, window_data: List[dict], predictions: np.ndarray):
        """예측 결과를 실제 윈도우에 적용"""
        try:
            # 윈도우 정보 준비
            windows = []
            for window in window_data:
                window_info = {
                    'hwnd': self._get_window_handle(window),
                    'title': window.get('program_id', 'Unknown'),
                    'rect': (window.get('x', 0), window.get('y', 0), 
                            window.get('w', 800), window.get('h', 600))
                }
                windows.append(window_info)
            
            # 윈도우 배열 적용
            if len(windows) == len(predictions):
                success = self.window_manager.arrange_windows_ml(windows, predictions)
                if success:
                    print(f"✅ 윈도우 배열 적용 완료: {len(windows)}개 윈도우")
                else:
                    print("❌ 윈도우 배열 적용 실패")
            else:
                print(f"⚠️ 윈도우 수({len(windows)})와 예측 수({len(predictions)}) 불일치")
        
        except Exception as e:
            print(f"❌ 윈도우 배열 적용 오류: {e}")
    
    def _get_window_handle(self, window_data: dict) -> int:
        """윈도우 데이터에서 핸들 추출"""
        # 실제 구현에서는 프로그램 ID를 기반으로 윈도우 핸들을 찾아야 함
        # 현재는 더미 값 반환
        return hash(window_data.get('program_id', 'unknown')) % 10000
    
    def print_performance_summary(self):
        """성능 요약 출력"""
        if not self.inference_times:
            print("📊 성능 데이터가 없습니다.")
            return
        
        times_ms = [t * 1000 for t in self.inference_times]
        
        print("\n📊 성능 요약")
        print(f"총 추론 횟수: {len(self.inference_times)}")
        print(f"평균 추론 시간: {np.mean(times_ms):.1f}ms")
        print(f"최소 추론 시간: {np.min(times_ms):.1f}ms")
        print(f"최대 추론 시간: {np.max(times_ms):.1f}ms")
        print(f"표준편차: {np.std(times_ms):.1f}ms")
        
        # 목표 성능 달성률
        target_time = 500  # 500ms
        achieved_count = sum(1 for t in times_ms if t <= target_time)
        achievement_rate = (achieved_count / len(times_ms)) * 100
        
        print(f"목표 달성률 (≤{target_time}ms): {achievement_rate:.1f}%")
        
        # 성능 데이터 저장
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'total_inferences': len(self.inference_times),
            'average_time_ms': float(np.mean(times_ms)),
            'min_time_ms': float(np.min(times_ms)),
            'max_time_ms': float(np.max(times_ms)),
            'std_time_ms': float(np.std(times_ms)),
            'achievement_rate': float(achievement_rate),
            'inference_times_ms': times_ms
        }
        
        performance_file = f"data/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(performance_file), exist_ok=True)
        
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump(performance_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 성능 데이터 저장: {performance_file}")

class RealTimeInference:
    """실시간 추론을 위한 고수준 인터페이스"""
    
    def __init__(self, model_path: str):
        self.inference_engine = InferenceEngine(model_path)
    
    def start(self, data_collector, duration: Optional[int] = None):
        """실시간 추론 시작"""
        print("�� Zonemaker AI 실시간 추론 시작")
        print("�� 중지하려면 Ctrl+C를 누르세요")
        
        try:
            self.inference_engine.start_inference_loop(
                data_collector=data_collector,
                interval=1.0,
                max_duration=duration
            )
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중지됨")
        finally:
            print("👋 Zonemaker AI 종료")

if __name__ == "__main__":
    # 테스트 실행
    print("🧪 추론 엔진 테스트 시작")
    
    # 모델 파일 확인
    models_dir = "data/models"
    if not os.path.exists(models_dir):
        print(f"❌ 모델 디렉토리가 없습니다: {models_dir}")
        print("💡 먼저 모델 학습을 실행해주세요.")
    else:
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if not model_files:
            print("❌ 학습된 모델이 없습니다.")
            print("💡 먼저 모델 학습을 실행해주세요.")
        else:
            print(f"📁 발견된 모델: {model_files}")
            # 첫 번째 모델로 테스트
            model_path = model_files[0]
            print(f"🧪 테스트 모델: {model_path}")
            
            try:
                inference_engine = InferenceEngine(model_path)
                print("✅ 추론 엔진 초기화 성공")
            except Exception as e:
                print(f"❌ 추론 엔진 초기화 실패: {e}")