import win32gui
import win32api
import win32con
import win32process
import win32hook
import time
import psutil
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import json
import os
from datetime import datetime
import threading

@dataclass
class WindowData:
    """윈도우 데이터 구조체"""
    timestamp: float
    program_id: str
    is_active: bool
    is_minimized: bool
    is_maximized: bool
    click_count: int
    keystroke_count: int
    resize_w_count: int
    resize_h_count: int
    scroll_down_count: int
    scroll_up_count: int
    x: int
    y: int
    w: int
    h: int
    z_order: int

class DataCollector:
    """Windows API를 활용한 실시간 데이터 수집기"""
    
    def __init__(self, save_dir: str = "data/raw"):
        self.window_states = {}
        self.event_counts = {}
        self.save_dir = save_dir
        self.is_collecting = False
        self.collection_start_time = None
        
        # 데이터 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 이벤트 후킹 설정
        self.setup_event_hooks()

        # 이벤트 후킹 관련(08230222추가)
        self.hooks = []
        self.event_lock = threading.Lock()
        self.current_active_window = None
    
    def setup_event_hooks(self):
        """마우스/키보드 이벤트 후킹 설정"""
        # TODO: 실제 이벤트 후킹 구현
        # 현재는 시뮬레이션 데이터 사용
        pass
    
    def start_collection(self, duration: int = 600):
        """데이터 수집 시작 (기본 10분)"""
        self.is_collecting = True
        self.collection_start_time = time.time()
        print(f"�� 데이터 수집 시작 (지속시간: {duration}초)")
        
        collected_data = []
        start_time = time.time()
        
        while self.is_collecting and (time.time() - start_time) < duration:
            # 현재 윈도우 데이터 수집
            window_data = self.collect_window_data()
            collected_data.extend(window_data)
            
            # 1초 대기
            time.sleep(1)
            
            # 진행률 표시
            elapsed = time.time() - start_time
            progress = (elapsed / duration) * 100
            print(f"📊 수집 진행률: {progress:.1f}% ({elapsed:.0f}s/{duration}s)")
        
        self.is_collecting = False
        print("✅ 데이터 수집 완료")
        
        # 데이터 저장
        self.save_collected_data(collected_data)
        return collected_data
    
    def stop_collection(self):
        """데이터 수집 중지"""
        self.is_collecting = False
        print("⏹️ 데이터 수집 중지")
    
    def collect_window_data(self) -> List[WindowData]:
        """현재 활성 윈도우들의 데이터를 수집"""
        windows_data = []
        
        def enum_windows_callback(hwnd, windows_data):
            if win32gui.IsWindowVisible(hwnd):
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                    x, y, w, h = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
                    
                    # 프로그램 정보 추출
                    program_id = self._get_program_id(hwnd)
                    
                    # 이벤트 카운트 가져오기
                    event_counts = self.event_counts.get(program_id, {
                        'clicks': 0, 'keystrokes': 0, 'resize_w': 0, 'resize_h': 0,
                        'scroll_down': 0, 'scroll_up': 0
                    })

                    # 다른 방법으로 최대화 상태 확인
                    def is_window_maximized(hwnd):
                        """윈도우가 최대화되었는지 확인"""
                        try:
                            # WS_MAXIMIZE 스타일 확인
                            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
                            return bool(style & win32con.WS_MAXIMIZE)
                        except:
                            return False

                    window_data = WindowData(
                        timestamp=time.time(),
                        program_id=program_id,
                        is_active=win32gui.GetForegroundWindow() == hwnd,
                        is_minimized=win32gui.IsIconic(hwnd),
                        is_maximized=is_window_maximized(hwnd),
                        click_count=event_counts['clicks'],
                        keystroke_count=event_counts['keystrokes'],
                        resize_w_count=event_counts['resize_w'],
                        resize_h_count=event_counts['resize_h'],
                        scroll_down_count=event_counts['scroll_down'],
                        scroll_up_count=event_counts['scroll_up'],
                        x=x, y=y, w=w, h=h,
                        z_order=self._get_z_order(hwnd)
                    )
                    windows_data.append(window_data)
                except Exception as e:
                    print(f"⚠️ 윈도우 데이터 수집 오류: {e}")
        
        try:
            win32gui.EnumWindows(enum_windows_callback, windows_data)
        except Exception as e:
            print(f"❌ 윈도우 열거 오류: {e}")
            # 시뮬레이션 데이터 반환
            return self._generate_simulation_data()
        
        return windows_data
    
    def _get_program_id(self, hwnd) -> str:
        """윈도우 핸들로부터 프로그램 ID 추출"""
        try:
            _, pid = win32gui.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid)
            return process.name()
        except:
            return f"unknown_{hwnd}"
    
    def _get_z_order(self, hwnd) -> int:
        """윈도우의 Z-order 반환"""
        return hwnd % 1000
    
    def _generate_simulation_data(self) -> List[WindowData]:
        """시뮬레이션 데이터 생성 (테스트용)"""
        programs = ["chrome.exe", "notepad.exe", "explorer.exe"]
        data = []
        
        for i, program in enumerate(programs):
            window_data = WindowData(
                timestamp=time.time(),
                program_id=program,
                is_active=i == 0,
                is_minimized=False,
                is_maximized=False,
                click_count=np.random.randint(0, 10),
                keystroke_count=np.random.randint(0, 50),
                resize_w_count=np.random.randint(0, 5),
                resize_h_count=np.random.randint(0, 5),
                scroll_down_count=np.random.randint(0, 20),
                scroll_up_count=np.random.randint(0, 20),
                x=i * 200, y=100, w=800, h=600,
                z_order=i
            )
            data.append(window_data)
        
        return data
    
    def save_collected_data(self, data: List[WindowData]):
        """수집된 데이터를 JSON 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"window_data_{timestamp}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        # 데이터를 JSON 직렬화 가능한 형태로 변환
        json_data = []
        for item in data:
            json_data.append({
                'timestamp': item.timestamp,
                'program_id': item.program_id,
                'is_active': item.is_active,
                'is_minimized': item.is_minimized,
                'is_maximized': item.is_maximized,
                'click_count': item.click_count,
                'keystroke_count': item.keystroke_count,
                'resize_w_count': item.resize_w_count,
                'resize_h_count': item.resize_h_count,
                'scroll_down_count': item.scroll_down_count,
                'scroll_up_count': item.scroll_up_count,
                'x': item.x, 'y': item.y, 'w': item.w, 'h': item.h,
                'z_order': item.z_order
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 데이터 저장 완료: {filepath}")
        print(f"�� 총 {len(data)}개 윈도우 데이터 수집됨")

if __name__ == "__main__":
    # 테스트 실행
    collector = DataCollector()
    print("🧪 데이터 수집기 테스트 시작")
    
    # 10초간 데이터 수집 테스트
    test_data = collector.start_collection(duration=10)
    print(f"테스트 완료: {len(test_data)}개 데이터 수집")