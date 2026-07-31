"""opencv2_rps_classic — 학습된 모델 0바이트로 만든 가위바위보 인식기.

Part 1 3자 비교의 세 번째 다리. YOLO(딥러닝 탐지)·MediaPipe(온디바이스 회귀)와
같은 문제를 **전통 영상처리만으로** 푼다.
"""

from .gesture import classify
from .judge import AIMoveProvider, judge
from .pipeline import Detection, detect
from .types import Gesture, HandShape, Result

__all__ = [
    "AIMoveProvider",
    "Detection",
    "Gesture",
    "HandShape",
    "Result",
    "classify",
    "detect",
    "judge",
]
