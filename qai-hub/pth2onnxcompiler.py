import qai_hub as hub
import torch
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.model import create_model, ONNXCompatibleModel

def convert_pth_to_onnx(pth_path: str, output_path: str = None):
    """
    Zonemaker AI 모델을 ONNX로 변환
    
    Args:
        pth_path: PyTorch 체크포인트 파일 경로
        output_path: ONNX 출력 파일 경로 (None이면 자동 생성)
    """
    try:
        print(f"PyTorch 모델 로드 중: {pth_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(pth_path, map_location='cpu')
        
        # 모델 생성 및 가중치 로드
        if 'model_state_dict' in checkpoint:
            config = checkpoint.get('config', {})
            original_model = create_model(config)
            original_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            original_model = create_model()
            original_model.load_state_dict(checkpoint)
        
        original_model.eval()
        print(f"원본 모델 로드 완료: {type(original_model).__name__}")
        
        # ONNX 변환용 호환 모델 생성
        onnx_model = ONNXCompatibleModel(original_model)
        onnx_model.eval()
        print("ONNX 호환 모델 생성 완료")
        
        # 입력 형태 설정 (Zonemaker AI 모델에 맞게)
        batch_size = 1
        seq_len = 30  # 30초 시퀀스
        max_windows = getattr(onnx_model, 'max_windows', 20)
        
        # 모델의 입력 차원 계산
        window_feature_dim = getattr(onnx_model, 'window_feature_dim', 128)
        activity_feature_dim = getattr(onnx_model, 'activity_feature_dim', 64)
        input_dim = window_feature_dim + activity_feature_dim
        
        # 입력 텐서 생성 (시퀀스 길이 x 특징 차원)
        input_shape = (batch_size, seq_len, input_dim)
        example_input = torch.randn(input_shape, dtype=torch.float32)
        
        print(f"입력 형태: {input_shape}")
        print(f"모델 설정: window_feature_dim={window_feature_dim}, "
              f"activity_feature_dim={activity_feature_dim}, max_windows={max_windows}")
        
        # 모델 테스트 (정상 작동 확인)
        print("모델 테스트 실행 중...")
        with torch.no_grad():
            test_output = onnx_model(example_input)
            print(f"테스트 출력 형태: positions {test_output[0].shape}, existence {test_output[1].shape}")
        
        # JIT 트레이싱 시도 (QAI Hub 요구사항)
        print("JIT 트레이싱 시도 중...")
        traced_model = None
        
        try:
            # 모델을 eval 모드로 설정하고 그래디언트 비활성화
            onnx_model.eval()
            
            # 모델의 모든 서브모듈을 eval 모드로 설정
            for module in onnx_model.modules():
                module.eval()
            
            with torch.no_grad():
                # 더미 입력으로 여러 번 테스트하여 안정성 확인
                print("모델 안정성 테스트 중...")
                for i in range(3):
                    test_output = onnx_model(example_input)
                    print(f"테스트 {i+1}: positions {test_output[0].shape}, existence {test_output[1].shape}")
                
                # JIT 트레이싱 실행
                print("JIT 트레이싱 실행 중...")
                traced_model = torch.jit.trace(onnx_model, example_input)
                print("JIT 트레이싱 완료")
                
                # 트레이스된 모델 테스트
                with torch.no_grad():
                    traced_output = traced_model(example_input)
                    print(f"트레이스된 모델 출력 형태: positions {traced_output[0].shape}, existence {traced_output[1].shape}")
                    
        except Exception as trace_error:
            print(f"JIT 트레이싱 실패: {trace_error}")
            print(f"오류 타입: {type(trace_error).__name__}")
            import traceback
            traceback.print_exc()
            print("원본 모델 사용")
            traced_model = onnx_model
        
        # QAI Hub 컴파일 작업 제출
        print("QAI Hub 컴파일 작업 제출 중...")
        
        # input_specs를 QAI Hub가 기대하는 정확한 형태로 생성
        # Dict[str, Tuple[Tuple[int, ...], str]] 형태
        # dtype에서 'torch.' 접두사 제거
        dtype_str = str(example_input.dtype).replace('torch.', '')
        input_specs = {
            'input': (tuple(example_input.shape), dtype_str)
        }
        
        compile_job = hub.submit_compile_job(
            model=traced_model,
            device=hub.Device("Snapdragon X Elite CRD"),
            input_specs=input_specs,
            options="--target_runtime onnx",  # opset 옵션 제거
        )
        
        print("컴파일 작업 제출 완료")
        
        # 컴파일 완료 대기
        print("컴파일 완료 대기 중...")
        compile_job.wait()
        
        # 컴파일 결과 확인
        try:
            job_status = compile_job.get_status()
            if job_status == "FAILED":
                print(f"컴파일 실패: {job_status}")
                print("컴파일 로그 확인 중...")
                try:
                    logs = compile_job.get_logs()
                    print("컴파일 로그:")
                    print(logs)
                except Exception as log_error:
                    print(f"로그 확인 실패: {log_error}")
                return None
            elif job_status == "SUCCESS":
                print("컴파일 성공!")
            else:
                print(f"컴파일 상태: {job_status}")
        except Exception as status_error:
            print(f"상태 확인 실패: {status_error}")
            print("상태 확인 건너뛰고 계속 진행...")
        
        # 타겟 모델 가져오기
        target_model = compile_job.get_target_model()
        if target_model is None:
            print("컴파일된 모델이 None입니다. 컴파일이 실패했을 수 있습니다.")
            return None
        
        print("컴파일된 모델 획득")
        
        # 프로파일링 작업 (선택사항)
        if target_model is not None:
            print("프로파일링 작업 제출 중...")
            try:
                profile_job = hub.submit_profile_job(
                    model=target_model,
                    device=hub.Device("Snapdragon X Elite CRD"),
                )
                profile_job.wait()
                print("프로파일링 완료")
            except Exception as profile_error:
                print(f"프로파일링 실패: {profile_error}")
        else:
            print("프로파일링 건너뛰기 (유효한 모델 없음)")
        
        # 추론 테스트
        if target_model is not None:
            print("추론 테스트 실행 중...")
            try:
                inference_job = hub.submit_inference_job(
                    model=target_model,
                    device=hub.Device("Snapdragon X Elite CRD"),
                    inputs={'input': example_input.numpy()},  # numpy 배열로 변환
                )
                inference_job.wait()
                
                # 출력 데이터 다운로드
                on_device_output = inference_job.download_output_data()
                print(f"추론 출력: {list(on_device_output.keys())}")
            except Exception as inference_error:
                print(f"추론 테스트 실패: {inference_error}")
        else:
            print("추론 테스트 건너뛰기 (유효한 모델 없음)")
        
        # 모델 다운로드
        if target_model is not None:
            if output_path is None:
                # 입력 파일명에서 확장자만 변경
                base_name = os.path.splitext(os.path.basename(pth_path))[0]
                output_path = f"{base_name}.onnx"
            
            print(f"ONNX 모델 다운로드 중: {output_path}")
            try:
                target_model.download(output_path)
                print(f"변환 완료! ONNX 모델: {output_path}")
                return output_path
            except Exception as download_error:
                print(f"모델 다운로드 실패: {download_error}")
                return None
        else:
            print("모델 다운로드 건너뛰기 (유효한 모델 없음)")
            return None
        
    except Exception as e:
        print(f"변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Zonemaker AI PyTorch 모델을 ONNX로 변환')
    parser.add_argument('pth_path', help='PyTorch 체크포인트 파일 경로')
    parser.add_argument('--output', '-o', help='ONNX 출력 파일 경로 (선택사항)')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    # 입력 파일 존재 확인
    if not os.path.exists(args.pth_path):
        print(f"오류: 체크포인트 파일이 존재하지 않습니다: {args.pth_path}")
        sys.exit(1)
    
    # 출력 디렉토리 생성
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"출력 디렉토리 생성: {output_dir}")
    
    # 변환 실행
    print("=== Zonemaker AI PyTorch → ONNX 변환 시작 ===")
    result = convert_pth_to_onnx(args.pth_path, args.output)
    
    if result:
        print(f"\n✅ 변환 성공: {result}")
        print("\n사용법:")
        print(f"from backend.ml.model import WindowArrangementPredictor")
        print(f"predictor = WindowArrangementPredictor('{result}')")
        print(f"positions = predictor.predict_next_arrangement(window_seq, activity_seq)")
        sys.exit(0)
    else:
        print("\n❌ 변환 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()
