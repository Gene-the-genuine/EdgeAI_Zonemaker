"""
합성 데이터 생성 실행 스크립트
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# 상위 디렉토리의 스크립트를 import하기 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.synthetic_data_generator import SyntheticDataGenerator
from config.data_config import SCENARIO_CONFIGS

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/data_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="합성 윈도우 활동 데이터 생성기")
    
    parser.add_argument(
        '--num-sequences',
        type=int,
        default=1000,
        help='생성할 시퀀스 수 (기본값: 1000)'
    )
    
    parser.add_argument(
        '--scenarios',
        nargs='+',
        choices=list(SCENARIO_CONFIGS.keys()),
        default=list(SCENARIO_CONFIGS.keys()),
        help='생성할 시나리오들 (기본값: 모든 시나리오)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='출력 디렉토리 (기본값: output)'
    )
    
    parser.add_argument(
        '--filename',
        type=str,
        default=None,
        help='출력 파일명 (기본값: 자동 생성)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='테스트 모드 (5개 시퀀스만 생성)'
    )
    
    args = parser.parse_args()
    
    # 로그 디렉토리 생성
    os.makedirs('logs', exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("합성 윈도우 활동 데이터 생성 시작")
    logger.info("=" * 60)
    logger.info(f"시퀀스 수: {args.num_sequences}")
    logger.info(f"시나리오: {', '.join(args.scenarios)}")
    logger.info(f"출력 디렉토리: {args.output_dir}")
    logger.info("=" * 60)
    
    try:
        # 데이터 생성기 초기화
        generator = SyntheticDataGenerator()
        
        # 테스트 모드 확인
        if args.test_mode:
            logger.info("테스트 모드로 실행 (5개 시퀀스)")
            num_sequences = 5
        else:
            num_sequences = args.num_sequences
        
        # 훈련 데이터 생성
        logger.info("훈련 데이터 생성 시작...")
        training_data = generator.generate_training_sequences(
            num_sequences=num_sequences,
            scenarios=args.scenarios
        )
        
        # 데이터 저장
        logger.info("데이터 저장 시작...")
        output_file = generator.save_training_data(
            training_data=training_data,
            filename=args.filename,
            output_dir=args.output_dir
        )
        
        # 결과 요약
        logger.info("=" * 60)
        logger.info("데이터 생성 완료!")
        logger.info("=" * 60)
        logger.info(f"생성된 시퀀스 수: {len(training_data)}")
        logger.info(f"출력 파일: {output_file}")
        logger.info(f"파일 크기: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        # 시나리오별 통계
        scenario_stats = {}
        for scenario in args.scenarios:
            count = len([seq for seq in training_data if seq[0][0][0].title in 
                        [w.title for w in generator.generate_window_pool(scenario)]])
            scenario_stats[scenario] = count
        
        logger.info("시나리오별 통계:")
        for scenario, count in scenario_stats.items():
            logger.info(f"  {scenario}: {count}개 시퀀스")
        
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"데이터 생성 실패: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
