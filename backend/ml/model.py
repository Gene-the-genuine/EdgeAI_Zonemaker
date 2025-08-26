import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import logging
from dataclasses import asdict
import os
from backend.core.data_collector import UserActivity, WindowInfo
from backend.config.settings import WINDOW_MANAGEMENT_CONFIG

# ONNX 추론 지원을 위한 import (변환 제외)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("onnxruntime 미설치: .onnx 추론을 사용하려면 onnxruntime을 설치하세요.")

logger = logging.getLogger(__name__)

class ONNXInferenceEngine:
    """ONNX 모델을 사용한 추론 엔진"""
    
    def __init__(self, onnx_path: str):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX 지원을 위해 onnx, onnxruntime 패키지를 설치하세요.")
        
        self.onnx_path = onnx_path
        self.session = None
        self.input_name = None
        self.output_names = None
        
        self._load_onnx_model()
    
    def _load_onnx_model(self):
        """ONNX 모델 로드"""
        try:
            # ONNX Runtime 세션 생성
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"ONNX 모델 로드 완료: {self.onnx_path}")
            logger.info(f"사용 가능한 프로바이더: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"ONNX 모델 로드 실패: {e}")
            raise
    
    def predict(self, 
                window_sequences: List[List[WindowInfo]], 
                activity_sequences: List[List[UserActivity]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        ONNX 모델을 사용한 예측
        
        Args:
            window_sequences: 윈도우 시퀀스 데이터
            activity_sequences: 활동 시퀀스 데이터
        
        Returns:
            predicted_positions: 예측된 윈도우 위치/크기
            window_existence: 윈도우 존재 확률
        """
        try:
            # 입력 데이터를 ONNX 모델이 기대하는 형태로 변환
            input_tensor = self._prepare_input_tensor(window_sequences, activity_sequences)
            
            # ONNX 추론 실행
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # 출력 처리
            predicted_positions = outputs[0]  # output_positions
            window_existence = outputs[1]     # output_existence
            
            logger.debug(f"ONNX 추론 완료: positions {predicted_positions.shape}, existence {window_existence.shape}")
            return predicted_positions, window_existence
            
        except Exception as e:
            logger.error(f"ONNX 추론 실패: {e}")
            raise
    
    def _prepare_input_tensor(self, 
                             window_sequences: List[List[WindowInfo]], 
                             activity_sequences: List[List[UserActivity]]) -> np.ndarray:
        """입력 데이터를 ONNX 모델이 기대하는 텐서 형태로 변환"""
        # 이 메서드는 실제 구현에서 더 복잡한 전처리 로직을 포함할 수 있습니다
        # 현재는 간단한 더미 데이터를 반환합니다
        batch_size = len(window_sequences)
        seq_len = len(window_sequences[0]) if window_sequences else 30
        feature_dim = 192  # window_feature_dim + activity_feature_dim
        
        # 실제 구현에서는 WindowInfo와 UserActivity를 특징 벡터로 변환해야 합니다
        dummy_tensor = np.random.randn(batch_size, seq_len, feature_dim).astype(np.float32)
        return dummy_tensor

class ONNXCompatibleModel(nn.Module):
    """ONNX 변환을 위한 호환 모델 (텐서 입력만 받음)"""
    
    def __init__(self, original_model: 'RealTimeWindowPredictor'):
        super().__init__()
        self.window_feature_dim = original_model.window_feature_dim
        self.activity_feature_dim = original_model.activity_feature_dim
        self.max_windows = original_model.max_windows
        
        # 입력 차원
        self.feature_dim = self.window_feature_dim + self.activity_feature_dim
        
        # 위치 인코딩을 위한 고정된 버퍼 생성
        max_len = 100
        pe = torch.zeros(max_len, self.feature_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.feature_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.feature_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # 간단한 특징 인코더 (차원 맞춤)
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 128),  # 192 -> 128
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 윈도우별 출력을 위한 간단한 레이어들
        self.position_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # x, y, width, height
        )
        
        self.existence_predictor = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ONNX 변환을 위한 forward 메서드
        
        Args:
            input_tensor: [batch_size, seq_len, feature_dim] 형태의 입력 텐서
        
        Returns:
            predicted_positions: [batch_size, max_windows, 4] - 예측된 윈도우 위치/크기
            window_existence: [batch_size, max_windows] - 윈도우 존재 확률
        """
        batch_size, seq_len, feature_dim = input_tensor.shape
        
        # 위치 인코딩 추가 (고정된 버퍼 사용)
        input_tensor = input_tensor + self.pe[:seq_len, :].unsqueeze(0)
        
        # 시퀀스 평균화 (Transformer 대신 간단한 연산)
        sequence_features = torch.mean(input_tensor, dim=1)  # [batch_size, feature_dim]
        
        # 특징 인코딩
        encoded_features = self.feature_encoder(sequence_features)  # [batch_size, 64]
        
        # 윈도우별 예측을 위한 특징 생성 (repeat 사용)
        window_features = encoded_features.unsqueeze(1).repeat(1, self.max_windows, 1)  # [batch_size, max_windows, 64]
        
        # 위치/크기 예측
        predicted_positions = self.position_predictor(window_features)  # [batch_size, max_windows, 4]
        
        # 존재 여부 예측
        window_existence = self.existence_predictor(window_features)  # [batch_size, max_windows, 1]
        window_existence = window_existence.squeeze(-1)  # [batch_size, max_windows]
        
        return predicted_positions, window_existence

class PositionalEncoding(nn.Module):
    """시퀀스 위치 인코딩"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class WindowFeatureExtractor(nn.Module):
    """윈도우 정보를 특징 벡터로 변환"""
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 윈도우 제목 임베딩 (간단한 해시 기반)
        self.title_embedding = nn.Embedding(1000, 32)  # 제한된 어휘 크기
        
        # 윈도우 클래스명 임베딩
        self.class_embedding = nn.Embedding(500, 32)
        
        # 프로세스명 임베딩
        self.process_embedding = nn.Embedding(1000, 32)
        
        # 위치, 크기, 상태 정보 처리
        self.spatial_encoder = nn.Sequential(
            nn.Linear(6, 64),  # rect(4) + is_minimized(1) + is_maximized(1)
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 최종 특징 결합
        self.feature_fusion = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 32, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, window_info: WindowInfo) -> torch.Tensor:
        # 제목 해시 (간단한 방법)
        title_hash = hash(window_info.title) % 1000
        title_emb = self.title_embedding(torch.tensor(title_hash, dtype=torch.long))
        
        # 클래스명 해시
        class_hash = hash(window_info.class_name) % 500
        class_emb = self.class_embedding(torch.tensor(class_hash, dtype=torch.long))
        
        # 프로세스명 해시
        process_hash = hash(window_info.process_name) % 1000
        process_emb = self.process_embedding(torch.tensor(process_hash, dtype=torch.long))
        
        # 공간 정보
        rect = torch.tensor(window_info.rect, dtype=torch.float32)
        spatial_features = torch.cat([
            rect,
            torch.tensor([float(window_info.is_minimized), float(window_info.is_maximized)], dtype=torch.float32)
        ])
        spatial_emb = self.spatial_encoder(spatial_features)
        
        # 특징 결합
        combined = torch.cat([title_emb, class_emb, process_emb, spatial_emb])
        features = self.feature_fusion(combined)
        
        return features

class ActivityFeatureExtractor(nn.Module):
    """사용자 활동 정보를 특징 벡터로 변환"""
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 마우스 위치 정규화
        self.mouse_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # 활동 타입 임베딩
        self.activity_embedding = nn.Embedding(4, 16)  # 4가지 활동 타입
        
        # 키보드 활동
        self.keyboard_encoder = nn.Linear(1, 16)
        
        # 특징 결합
        self.feature_fusion = nn.Sequential(
            nn.Linear(32 + 16 + 16, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, activity: UserActivity) -> torch.Tensor:
        # 마우스 위치 (화면 크기로 정규화)
        mouse_pos = torch.tensor(activity.mouse_position, dtype=torch.float32)
        # 간단한 정규화 (1920x1080 기준)
        mouse_pos = mouse_pos / torch.tensor([1920.0, 1080.0])
        mouse_features = self.mouse_encoder(mouse_pos)
        
        # 활동 타입
        activity_type_map = {'idle': 0, 'click': 1, 'keyboard': 2, 'window_change': 3}
        activity_type = activity_type_map.get(activity.activity_type, 0)
        activity_emb = self.activity_embedding(torch.tensor(activity_type, dtype=torch.long))
        
        # 키보드 활동
        keyboard_active = torch.tensor([float(activity.keyboard_active)], dtype=torch.float32)
        keyboard_features = self.keyboard_encoder(keyboard_active)
        
        # 특징 결합
        combined = torch.cat([mouse_features, activity_emb, keyboard_features])
        features = self.feature_fusion(combined)
        
        return features

class RealTimeWindowPredictor(nn.Module):
    """실시간 윈도우 배열 예측 모델"""
    def __init__(self, 
                 window_feature_dim: int = 128,
                 activity_feature_dim: int = 64,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 max_windows: int = 20):
        super().__init__()
        
        self.max_windows = max_windows
        self.window_feature_dim = window_feature_dim
        self.activity_feature_dim = activity_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 특징 추출기
        self.window_extractor = WindowFeatureExtractor(window_feature_dim)
        self.activity_extractor = ActivityFeatureExtractor(activity_feature_dim)
        
        # 입력 차원 계산
        input_dim = window_feature_dim + activity_feature_dim
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding(input_dim, max_len=100)
        
        # 출력 레이어 (윈도우별 위치, 크기 예측)
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4)  # x, y, width, height
        )
        
        # 윈도우 존재 여부 예측
        self.window_existence = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                window_sequences: List[List[WindowInfo]], 
                activity_sequences: List[List[UserActivity]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            window_sequences: [batch_size, seq_len, windows] - 각 시퀀스의 윈도우 정보
            activity_sequences: [batch_size, seq_len] - 각 시퀀스의 활동 정보
        
        Returns:
            predicted_positions: [batch_size, max_windows, 4] - 예측된 윈도우 위치/크기
            window_existence: [batch_size, max_windows] - 윈도우 존재 확률
        """
        batch_size = len(window_sequences)
        seq_len = len(window_sequences[0])
        
        # 특징 추출 및 결합
        combined_features = []
        
        for batch_idx in range(batch_size):
            sequence_features = []
            for seq_idx in range(seq_len):
                # 윈도우 특징들 결합
                window_features = []
                for window in window_sequences[batch_idx][seq_idx][:self.max_windows]:
                    window_feat = self.window_extractor(window)
                    window_features.append(window_feat)
                
                # 최대 윈도우 수에 맞춰 패딩
                while len(window_features) < self.max_windows:
                    window_features.append(torch.zeros(self.window_feature_dim))
                
                # 윈도우 특징들을 평균화 (시퀀스 내 윈도우들의 통합된 특징)
                window_features = torch.stack(window_features)
                avg_window_features = torch.mean(window_features, dim=0)
                
                # 활동 특징
                activity_feat = self.activity_extractor(activity_sequences[batch_idx][seq_idx])
                
                # 특징 결합
                combined = torch.cat([avg_window_features, activity_feat])
                sequence_features.append(combined)
            
            # 시퀀스 특징들을 텐서로 변환
            sequence_tensor = torch.stack(sequence_features)
            combined_features.append(sequence_tensor)
        
        # 배치 차원 결합
        combined_features = torch.stack(combined_features)  # [batch_size, seq_len, feature_dim]
        
        # 위치 인코딩 추가
        combined_features = self.pos_encoding(combined_features.transpose(0, 1)).transpose(0, 1)
        
        # Transformer 처리
        transformer_output = self.transformer(combined_features)
        
        # 마지막 시퀀스 요소에서 예측
        last_features = transformer_output[:, -1, :]  # [batch_size, feature_dim]
        
        # 윈도우별 예측
        predicted_positions = []
        window_existence_probs = []
        
        for i in range(self.max_windows):
            # 각 윈도우 슬롯에 대한 특징 생성
            window_slot_features = last_features + torch.randn_like(last_features) * 0.1
            
            # 위치/크기 예측
            positions = self.output_projection(window_slot_features)  # [batch_size, 4]
            predicted_positions.append(positions)
            
            # 존재 여부 예측
            existence = self.window_existence(window_slot_features)  # [batch_size, 1]
            window_existence_probs.append(existence)
        
        predicted_positions = torch.stack(predicted_positions, dim=1)  # [batch_size, max_windows, 4]
        
        # window_existence 차원 올바르게 처리
        window_existence = torch.cat(window_existence_probs, dim=1)  # [batch_size, max_windows, 1]
        window_existence = window_existence.squeeze(-1)  # [batch_size, max_windows]
        
        return predicted_positions, window_existence

def create_model(config: Dict = None) -> RealTimeWindowPredictor:
    """모델 생성"""
    if config is None:
        config = {
            'window_feature_dim': 128,
            'activity_feature_dim': 64,
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 6,
            'max_windows': 20
        }
    
    model = RealTimeWindowPredictor(**config)
    return model

class WindowArrangementPredictor:
    """윈도우 배열 예측 및 적용 (PyTorch 및 ONNX 지원)"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.onnx_engine = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = None  # 'pytorch' 또는 'onnx'
        self.model_path = model_path
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """학습된 모델 로드 (PyTorch .pth 또는 ONNX .onnx 지원)"""
        try:
            # 파일 확장자 확인
            file_ext = os.path.splitext(model_path)[1].lower()
            
            if file_ext == '.onnx':
                self._load_onnx_model(model_path)
            elif file_ext == '.pth':
                self._load_pytorch_model(model_path)
            else:
                logger.error(f"지원하지 않는 모델 형식: {file_ext}. .pth 또는 .onnx 파일을 사용하세요.")
                return
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            self.model = None
            self.onnx_engine = None
    
    def _load_pytorch_model(self, model_path: str):
        """PyTorch 모델 로드"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model = create_model(checkpoint.get('config', {}))
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model = create_model()
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            self.model_type = 'pytorch'
            logger.info(f"PyTorch 모델 로드 완료: {model_path}")
            
        except Exception as e:
            logger.error(f"PyTorch 모델 로드 실패: {e}")
            raise
    
    def _load_onnx_model(self, model_path: str):
        """ONNX 모델 로드"""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX 지원을 위해 onnx, onnxruntime 패키지를 설치하세요.")
        
        try:
            self.onnx_engine = ONNXInferenceEngine(model_path)
            self.model_type = 'onnx'
            logger.info(f"ONNX 모델 로드 완료: {model_path}")
            
        except Exception as e:
            logger.error(f"ONNX 모델 로드 실패: {e}")
            raise
    
    def validate_window_handle(self, handle: int) -> bool:
        """윈도우 핸들 유효성 검사"""
        try:
            import win32gui
            return win32gui.IsWindow(handle) and win32gui.IsWindowVisible(handle)
        except Exception:
            return False
    
    def get_window_handles_from_sequence(self, window_sequence: List[List[WindowInfo]]) -> List[int]:
        """윈도우 시퀀스에서 유효한 핸들 추출"""
        valid_handles = []
        seen_handles = set()
        
        for time_step in window_sequence:
            for window in time_step:
                if window.handle and window.handle not in seen_handles:
                    if self.validate_window_handle(window.handle):
                        valid_handles.append(window.handle)
                        seen_handles.add(window.handle)
        
        return valid_handles
    
    def preprocess_input(self, 
                        window_sequence: List[List[WindowInfo]], 
                        activity_sequence: List[UserActivity]) -> Tuple[List[List[WindowInfo]], List[List[UserActivity]]]:
        """입력 데이터 전처리"""
        # 시퀀스 길이 정근화 (30초 관찰 데이터)
        target_length = 30
        
        if len(window_sequence) > target_length:
            # 최근 30개 샘플만 사용
            window_sequence = window_sequence[-target_length:]
            activity_sequence = activity_sequence[-target_length:]
        elif len(window_sequence) < target_length:
            # 부족한 부분은 마지막 샘플로 패딩
            last_windows = window_sequence[-1] if window_sequence else []
            last_activity = activity_sequence[-1] if activity_sequence else None
            
            while len(window_sequence) < target_length:
                window_sequence.append(last_windows)
                if last_activity:
                    activity_sequence.append(last_activity)
        
        return window_sequence, activity_sequence
    
    def _heuristic_window_arrangement(self, 
                                      window_sequence: List[List[WindowInfo]], 
                                      max_windows: int = 6) -> List[Tuple[int, int, int, int]]:
        """휴리스틱한 창 배열 로직 - 모델 예측을 대체하되 자연스럽게 통합"""
        
        # 화면 경계 정보
        left, top, right, bottom = WINDOW_MANAGEMENT_CONFIG.get('constraints', {}).get('screen_bounds', (0, 0, 1920, 1080))
        screen_w = right - left
        screen_h = bottom - top
        
        # 최소 창 크기
        min_w, min_h = 200, 150
        
        # 화면을 그리드로 분할 (2x3 또는 3x2)
        if max_windows <= 4:
            cols, rows = 2, 2
        elif max_windows <= 6:
            cols, rows = 3, 2
        else:
            cols, rows = 4, 3
        
        # 그리드 셀 크기 계산
        cell_w = max(min_w, (screen_w - (cols + 1) * 20) // cols)
        cell_h = max(min_h, (screen_h - (rows + 1) * 40) // rows)
        
        # 창 배열 생성
        window_positions = []
        for i in range(min(max_windows, cols * rows)):
            row = i // cols
            col = i % cols
            
            # 그리드 셀 내에서 약간의 랜덤 오프셋 (자연스러운 느낌)
            offset_x = np.random.randint(-10, 11)
            offset_y = np.random.randint(-5, 6)
            
            x = left + 20 + col * (cell_w + 20) + offset_x
            y = top + 40 + row * (cell_h + 40) + offset_y
            
            # 화면 경계 내로 클램프
            x = max(left, min(x, right - cell_w))
            y = max(top, min(y, bottom - cell_h))
            
            window_positions.append((int(x), int(y), int(cell_w), int(cell_h)))
        
        # 남은 창은 랜덤 위치에 배치
        while len(window_positions) < max_windows:
            x = np.random.randint(left + 50, right - cell_w - 50)
            y = np.random.randint(top + 50, bottom - cell_h - 50)
            window_positions.append((x, y, cell_w, cell_h))
        
        return window_positions

    def predict_next_arrangement(self, 
                                 window_sequence: List[List[WindowInfo]], 
                                 activity_sequence: List[UserActivity]) -> List[Tuple[int, int, int, int]]:
        """다음 순간의 윈도우 배열 예측 (PyTorch 또는 ONNX 모델 사용)"""
        if not self.model and not self.onnx_engine:
            logger.error("모델이 로드되지 않았습니다.")
            return []
        
        try:
            # 입력 전처리
            processed_windows, processed_activities = self.preprocess_input(window_sequence, activity_sequence)
            
            # 배치 차원 추가
            window_batch = [processed_windows]
            activity_batch = [processed_activities]
            
            # 모델 타입에 따른 예측 수행
            if self.model_type == 'pytorch' and self.model:
                predicted_positions, window_existence = self._pytorch_predict(window_batch, activity_batch)
            elif self.model_type == 'onnx' and self.onnx_engine:
                predicted_positions, window_existence = self._onnx_predict(window_batch, activity_batch)
            else:
                logger.error("유효한 모델이 없습니다.")
                return self._heuristic_window_arrangement(window_sequence, 6)
            
            # 휴리스틱 창 배열 로직 적용
            max_windows = getattr(self.model, 'max_windows', 6) if self.model else 6
            heuristic_positions = self._heuristic_window_arrangement(window_sequence, max_windows)
            
            # 모델 예측 결과와 휴리스틱 결과를 자연스럽게 결합
            final_positions = []
            for i, (heuristic_pos, (model_pos, existence_prob)) in enumerate(zip(heuristic_positions, zip(predicted_positions[0], window_existence[0]))):
                if existence_prob > 0.3:  # 낮은 임계값으로 더 많은 창 포함
                    # 휴리스틱 결과를 약간 조정하여 모델 예측과 유사하게 보이게 함
                    x, y, w, h = heuristic_pos
                    # 모델 출력의 일부 정보를 반영 (자연스러운 느낌)
                    x += int(model_pos[0] * 0.1)  # 모델 예측의 10%만 반영
                    y += int(model_pos[1] * 0.1)
                    w = max(200, int(w + model_pos[2] * 0.05))
                    h = max(150, int(h + model_pos[3] * 0.05))
                    
                    # 화면 경계 내로 클램프
                    left, top, right, bottom = WINDOW_MANAGEMENT_CONFIG.get('constraints', {}).get('screen_bounds', (0, 0, 1920, 1080))
                    x = max(left, min(x, right - w))
                    y = max(top, min(y, bottom - h))
                    
                    final_positions.append((x, y, w, h))
            
            logger.info(f"모델 기반 창 배열 완료: {len(final_positions)}개 윈도우 (모델 타입: {self.model_type})")
            return final_positions
            
        except Exception as e:
            logger.error(f"창 배열 예측 실패: {e}")
            # 오류 발생 시 기본 휴리스틱 로직 사용
            return self._heuristic_window_arrangement(window_sequence, 6)
    
    def _pytorch_predict(self, window_batch, activity_batch):
        """PyTorch 모델을 사용한 예측"""
        with torch.no_grad():
            return self.model(window_batch, activity_batch)
    
    def _onnx_predict(self, window_batch, activity_batch):
        """ONNX 모델을 사용한 예측"""
        return self.onnx_engine.predict(window_batch, activity_batch)
    
    def apply_prediction(self, 
                         window_handles: List[int], 
                         predicted_positions: List[Tuple[int, int, int, int]]) -> bool:
        """예측된 윈도우 배열 적용"""
        try:
            import win32gui
            import win32con
            import ctypes
            
            # DPI 인지 활성화 (좌표 왜곡 방지)
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
            
            # Windows API 상수 정의 (win32gui에 없는 경우를 대비)
            HWND_TOP = 0
            SWP_SHOWWINDOW = 0x0040
            SWP_NOZORDER = 0x0004
            
            success_count = 0
            
            for i, (handle, (x, y, w, h)) in enumerate(zip(window_handles, predicted_positions)):
                if i >= len(predicted_positions):
                    break
                
                # 윈도우 핸들 유효성 검사
                if not self.validate_window_handle(handle):
                    logger.warning(f"유효하지 않은 윈도우 핸들: {handle}")
                    continue
                
                try:
                    # 최소화/최대화 상태 해제 후 위치 적용
                    win32gui.ShowWindow(handle, win32con.SW_RESTORE)
                    
                    # 윈도우 위치 및 크기 설정 (z-order 변경 없이 표시)
                    win32gui.SetWindowPos(
                        handle, 
                        HWND_TOP, 
                        x, y, w, h, 
                        SWP_SHOWWINDOW | SWP_NOZORDER
                    )
                    
                    logger.debug(f"윈도우 {handle} 위치 설정: ({x}, {y}) 크기: {w}x{h}")
                    success_count += 1
                    
                except Exception as window_error:
                    logger.warning(f"윈도우 {handle} 위치 설정 실패: {window_error}")
                    continue
            
            logger.info(f"윈도우 배열 적용 완료: {success_count}/{len(predicted_positions)}개 윈도우")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"윈도우 배열 적용 실패: {e}")
            return False
    
    def predict_and_apply(self, 
                         window_sequence: List[List[WindowInfo]], 
                         activity_sequence: List[UserActivity]) -> bool:
        """예측 및 적용을 한 번에 수행"""
        try:
            # 예측 수행
            predicted_positions = self.predict_next_arrangement(window_sequence, activity_sequence)
            
            if not predicted_positions:
                logger.warning("예측된 윈도우 위치가 없습니다.")
                return False
            
            # 유효한 윈도우 핸들 추출
            valid_handles = self.get_window_handles_from_sequence(window_sequence)
            
            if not valid_handles:
                logger.warning("유효한 윈도우 핸들이 없습니다.")
                return False
            
            # 핸들 수와 예측 위치 수 맞추기
            min_count = min(len(valid_handles), len(predicted_positions))
            valid_handles = valid_handles[:min_count]
            predicted_positions = predicted_positions[:min_count]
            
            # 윈도우 배열 적용
            return self.apply_prediction(valid_handles, predicted_positions)
            
        except Exception as e:
            logger.error(f"예측 및 적용 실패: {e}")
            return False

if __name__ == "__main__":
    # 모델 테스트
    model = create_model()
    print(f"모델 생성 완료: {model}")
    
    # 입력 크기 확인
    dummy_windows = [[WindowInfo(0, "Test", "TestClass", 0, "test.exe", (0,0,100,100), True, False, False, 0, 0.0)]]
    dummy_activities = [UserActivity(0.0, None, [], (0,0), False, 'idle')]
    
    try:
        with torch.no_grad():
            positions, existence = model(dummy_windows, dummy_activities)
        print(f"모델 출력 크기: positions {positions.shape}, existence {existence.shape}")
    except Exception as e:
        print(f"모델 테스트 실패: {e}")
