from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import asyncio
import json
import os
from datetime import datetime

# 상대 경로 임포트를 위한 설정
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_collector import DataCollector
from core.workstation import WorkstationManager
from core.window_manager import WindowManager
from ml.trainer import ModelTrainer, prepare_training_data
from ml.inference import RealTimeInference
from ml.npu_converter import NPUConverter
from ml.model import ZonemakerViT, DataProcessor

# FastAPI 앱 생성
app = FastAPI(
    title="Zonemaker AI API",
    description="Edge AI Developers Hackathon - 윈도우 배열 AI 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
data_collector = None
workstation_manager = None
window_manager = None
inference_engine = None
training_task = None

# Pydantic 모델들
class WorkstationCreate(BaseModel):
    name: str
    description: str = ""
    programs: List[str] = []

class WorkstationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    programs: Optional[List[str]] = None

class TrainingRequest(BaseModel):
    workstation_id: str
    duration_minutes: int = 10
    epochs: int = 100

class InferenceRequest(BaseModel):
    workstation_id: str
    duration_seconds: Optional[int] = None

class DataCollectionRequest(BaseModel):
    duration_seconds: int = 600

# 초기화
@app.on_event("startup")
async def startup_event():
    global data_collector, workstation_manager, window_manager
    
    print("�� Zonemaker AI API 서버 시작")
    
    # 핵심 컴포넌트 초기화
    data_collector = DataCollector()
    workstation_manager = WorkstationManager()
    window_manager = WindowManager()
    
    print("✅ 핵심 컴포넌트 초기화 완료")

# 상태 확인
@app.get("/")
async def root():
    return {
        "message": "Zonemaker AI API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "data_collector": data_collector is not None,
            "workstation_manager": workstation_manager is not None,
            "window_manager": window_manager is not None
        },
        "timestamp": datetime.now().isoformat()
    }

# Workstation 관리 API
@app.get("/workstations")
async def list_workstations():
    """모든 Workstation 목록 반환"""
    try:
        workstations = workstation_manager.list_workstations()
        return {
            "success": True,
            "workstations": [
                {
                    "id": ws.id,
                    "name": ws.name,
                    "description": ws.description,
                    "programs": ws.programs,
                    "is_trained": ws.is_trained,
                    "created_at": ws.created_at,
                    "last_used": ws.last_used
                }
                for ws in workstations
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workstation 목록 조회 실패: {str(e)}")

@app.post("/workstations")
async def create_workstation(workstation: WorkstationCreate):
    """새로운 Workstation 생성"""
    try:
        ws = workstation_manager.create_workstation(
            name=workstation.name,
            description=workstation.description,
            programs=workstation.programs
        )
        
        return {
            "success": True,
            "workstation": {
                "id": ws.id,
                "name": ws.name,
                "description": ws.description,
                "programs": ws.programs,
                "is_trained": ws.is_trained,
                "created_at": ws.created_at
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workstation 생성 실패: {str(e)}")

@app.get("/workstations/{workstation_id}")
async def get_workstation(workstation_id: str):
    """특정 Workstation 조회"""
    try:
        ws = workstation_manager.get_workstation(workstation_id)
        if not ws:
            raise HTTPException(status_code=404, detail="Workstation을 찾을 수 없습니다")
        
        return {
            "success": True,
            "workstation": {
                "id": ws.id,
                "name": ws.name,
                "description": ws.description,
                "programs": ws.programs,
                "is_trained": ws.is_trained,
                "created_at": ws.created_at,
                "last_used": ws.last_used
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workstation 조회 실패: {str(e)}")

@app.put("/workstations/{workstation_id}")
async def update_workstation(workstation_id: str, update_data: WorkstationUpdate):
    """Workstation 정보 업데이트"""
    try:
        update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
        
        if workstation_manager.update_workstation(workstation_id, **update_dict):
            return {"success": True, "message": "Workstation 업데이트 완료"}
        else:
            raise HTTPException(status_code=404, detail="Workstation을 찾을 수 없습니다")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workstation 업데이트 실패: {str(e)}")

@app.delete("/workstations/{workstation_id}")
async def delete_workstation(workstation_id: str):
    """Workstation 삭제"""
    try:
        if workstation_manager.delete_workstation(workstation_id):
            return {"success": True, "message": "Workstation 삭제 완료"}
        else:
            raise HTTPException(status_code=404, detail="Workstation을 찾을 수 없습니다")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workstation 삭제 실패: {str(e)}")

# 데이터 수집 API
@app.post("/collect-data")
async def start_data_collection(request: DataCollectionRequest):
    """데이터 수집 시작"""
    try:
        global data_collector
        
        if data_collector.is_collecting:
            return {
                "success": False,
                "message": "이미 데이터 수집이 진행 중입니다"
            }
        
        # 백그라운드에서 데이터 수집 시작
        def collect_data():
            data_collector.start_collection(duration=request.duration_seconds)
        
        # 별도 스레드에서 실행
        import threading
        collection_thread = threading.Thread(target=collect_data)
        collection_thread.start()
        
        return {
            "success": True,
            "message": f"데이터 수집 시작 (지속시간: {request.duration_seconds}초)",
            "estimated_completion": datetime.now().timestamp() + request.duration_seconds
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 수집 시작 실패: {str(e)}")

@app.get("/collect-data/status")
async def get_collection_status():
    """데이터 수집 상태 확인"""
    try:
        global data_collector
        
        if not data_collector:
            return {"success": False, "message": "데이터 수집기가 초기화되지 않았습니다"}
        
        return {
            "success": True,
            "is_collecting": data_collector.is_collecting,
            "collection_start_time": data_collector.collection_start_time,
            "elapsed_time": (datetime.now().timestamp() - data_collector.collection_start_time) if data_collector.collection_start_time else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"수집 상태 확인 실패: {str(e)}")

@app.post("/collect-data/stop")
async def stop_data_collection():
    """데이터 수집 중지"""
    try:
        global data_collector
        
        if data_collector:
            data_collector.stop_collection()
            return {"success": True, "message": "데이터 수집 중지됨"}
        else:
            return {"success": False, "message": "데이터 수집기가 초기화되지 않았습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 수집 중지 실패: {str(e)}")

# 모델 학습 API
@app.post("/train-model")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """모델 학습 시작"""
    try:
        global training_task
        
        if training_task and not training_task.done():
            return {
                "success": False,
                "message": "이미 학습이 진행 중입니다"
            }
        
        # Workstation 확인
        ws = workstation_manager.get_workstation(request.workstation_id)
        if not ws:
            raise HTTPException(status_code=404, detail="Workstation을 찾을 수 없습니다")
        
        # 백그라운드에서 학습 실행
        def train_model():
            try:
                # 데이터 준비
                train_loader, val_loader = prepare_training_data()
                
                # 모델 생성
                model = ZonemakerViT()
                processor = DataProcessor()
                
                # 학습기 생성 및 학습 실행
                trainer = ModelTrainer(model, processor)
                history = trainer.train(train_loader, val_loader, epochs=request.epochs)
                
                # NPU 변환
                converter = NPUConverter()
                output_dir = f"data/models/workstation_{request.workstation_id}"
                results = converter.convert_model_pipeline(model, output_dir, f"ws_{request.workstation_id}")
                
                # Workstation 상태 업데이트
                if 'npu_optimized' in results:
                    model_path = results['npu_optimized']
                else:
                    model_path = results.get('onnx', '')
                
                workstation_manager.mark_as_trained(request.workstation_id, model_path)
                
                print(f"✅ Workstation {request.workstation_id} 학습 완료")
                
            except Exception as e:
                print(f"❌ 학습 실패: {e}")
        
        # 백그라운드 태스크로 실행
        background_tasks.add_task(train_model)
        
        return {
            "success": True,
            "message": f"모델 학습 시작 (에포크: {request.epochs})",
            "workstation_name": ws.name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 학습 시작 실패: {str(e)}")

@app.get("/train-model/status")
async def get_training_status():
    """모델 학습 상태 확인"""
    global training_task
    
    if training_task and not training_task.done():
        return {
            "success": True,
            "is_training": True,
            "status": "학습 진행 중"
        }
    else:
        return {
            "success": True,
            "is_training": False,
            "status": "학습 대기 중"
        }

# 추론 API
@app.post("/start-inference")
async def start_inference(request: InferenceRequest):
    """실시간 추론 시작"""
    try:
        global inference_engine
        
        if inference_engine and inference_engine.inference_engine.is_running:
            return {
                "success": False,
                "message": "이미 추론이 실행 중입니다"
            }
        
        # Workstation 확인
        ws = workstation_manager.get_workstation(request.workstation_id)
        if not ws:
            raise HTTPException(status_code=404, detail="Workstation을 찾을 수 없습니다")
        
        if not ws.is_trained:
            raise HTTPException(status_code=400, detail="학습되지 않은 Workstation입니다")
        
        # 추론 엔진 생성 및 시작
        model_path = ws.model_path
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail="모델 파일을 찾을 수 없습니다")
        
        inference_engine = RealTimeInference(model_path)
        
        # 백그라운드에서 추론 시작
        def run_inference():
            inference_engine.start(data_collector, duration=request.duration_seconds)
        
        import threading
        inference_thread = threading.Thread(target=run_inference)
        inference_thread.start()
        
        return {
            "success": True,
            "message": "실시간 추론 시작",
            "workstation_name": ws.name,
            "estimated_duration": request.duration_seconds or "무제한"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 시작 실패: {str(e)}")

@app.post("/stop-inference")
async def stop_inference():
    """실시간 추론 중지"""
    try:
        global inference_engine
        
        if inference_engine:
            inference_engine.inference_engine.stop_inference_loop()
            return {"success": True, "message": "추론 중지 요청됨"}
        else:
            return {"success": False, "message": "실행 중인 추론이 없습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중지 실패: {str(e)}")

@app.get("/inference/status")
async def get_inference_status():
    """추론 상태 확인"""
    global inference_engine
    
    if inference_engine and inference_engine.inference_engine.is_running:
        return {
            "success": True,
            "is_running": True,
            "status": "추론 실행 중",
            "inference_count": len(inference_engine.inference_engine.inference_times)
        }
    else:
        return {
            "success": True,
            "is_running": False,
            "status": "추론 대기 중"
        }

# 윈도우 관리 API
@app.get("/windows")
async def get_windows():
    """현재 윈도우 목록 반환"""
    try:
        global window_manager
        
        if not window_manager:
            raise HTTPException(status_code=500, detail="윈도우 관리자가 초기화되지 않았습니다")
        
        windows = window_manager.get_all_windows()
        
        return {
            "success": True,
            "windows": [
                {
                    "title": w.get('title', 'Unknown'),
                    "x": w.get('x', 0),
                    "y": w.get('y', 0),
                    "width": w.get('width', 0),
                    "height": w.get('height', 0)
                }
                for w in windows
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"윈도우 목록 조회 실패: {str(e)}")

@app.post("/windows/arrange")
async def arrange_windows(layout: str = "grid"):
    """윈도우 배열"""
    try:
        global window_manager
        
        if not window_manager:
            raise HTTPException(status_code=500, detail="윈도우 관리자가 초기화되지 않았습니다")
        
        windows = window_manager.get_all_windows()
        
        if layout == "grid":
            success = window_manager.arrange_windows_grid(windows, "2x2")
        elif layout == "horizontal":
            success = window_manager.arrange_windows_grid(windows, "horizontal")
        elif layout == "vertical":
            success = window_manager.arrange_windows_grid(windows, "vertical")
        else:
            success = window_manager.arrange_windows_grid(windows)
        
        if success:
            return {"success": True, "message": f"윈도우 배열 완료 ({layout})"}
        else:
            return {"success": False, "message": "윈도우 배열 실패"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"윈도우 배열 실패: {str(e)}")

# 시스템 정보 API
@app.get("/system/info")
async def get_system_info():
    """시스템 정보 반환"""
    try:
        import psutil
        
        return {
            "success": True,
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "python_version": sys.version,
                "platform": sys.platform
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시스템 정보 조회 실패: {str(e)}")

if __name__ == "__main__":
    print("�� Zonemaker AI API 서버 시작 중...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )