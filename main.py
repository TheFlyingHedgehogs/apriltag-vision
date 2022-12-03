import dataclasses
import math
import cv2
import pupil_apriltags as apriltag
import numpy as np
import pickle as pkl
from picamera2 import Picamera2
import time
from networktables import NetworkTables
from libcamera import controls

@dataclasses.dataclass
class Calibration:
    mtx: np.ndarray
    dist: np.ndarray


@dataclasses.dataclass
class PinholeCalibration(Calibration):
    pass


@dataclasses.dataclass
class FisheyeCalibration(Calibration):
    pass


class TargetFinder:
    def __init__(self, calibration: Calibration, width=0.1524):
        self.calibration = calibration
        self.detector = apriltag.Detector(families="tag36h11")

        # Set up real-world coordinates of the target corners
        self.object_points = np.array([
            [-width/2, -width/2, 0.0],  # top left
            [width/2,  -width/2, 0.0],  # top right
            [width/2,   width/2, 0.0],  # bottom right
            [-width/2,  width/2, 0.0],  # bottom left
        ])
        self.identity = np.eye(3)
        self.identity_dist = np.zeros(calibration.dist.shape)

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(gray)
        vecs = []

        # loop over the AprilTag detection results
        for r in results:
            if isinstance(self.calibration, PinholeCalibration):
                image_points = cv2.undistortPoints(
                    np.array(r.corners),
                    self.calibration.mtx, self.calibration.dist,
                    P=self.identity)
            elif isinstance(self.calibration, FisheyeCalibration):
                points = np.expand_dims(np.array(r.corners, dtype=np.float64), axis=-2)
                image_points = cv2.fisheye.undistortPoints(
                    points,
                    self.calibration.mtx, self.calibration.dist,
                    P=self.identity)
            else:
                raise AttributeError("Invalid calibration type!")

            ret, rvec, tvec = cv2.solvePnP(self.object_points, image_points, self.identity, self.identity_dist)

            if ret:
                vecs.append((rvec, tvec))

        return vecs


mtx, dist = pkl.load(open("calib-all-picam", "rb"))
finder = TargetFinder(PinholeCalibration(mtx, dist))

picam2 = Picamera2()
video_config = picam2.create_video_configuration({"size": (1296, 972)})
picam2.set_controls({"AeExposureMode": controls.AeExposureModeEnum.Short})
picam2.set_controls({"AnalogueGain": 100})
picam2.configure(video_config)
picam2.start()

time.sleep(2)

start = time.perf_counter()
frames = 0

NetworkTables.initialize(server="10.28.98.2")
sd = NetworkTables.getTable("SmartDashboard")

while True:
    img = picam2.capture_array("main")
    v = finder.detect(img)
    for (r, t) in v:
        sd.putNumber("VisionX", t[0])
        sd.putNumber("VisionY", t[1])
        sd.putNumber("VisionZ", t[2])
        NetworkTables.flush()
        # print(t)
        # print(f"{math.sqrt(t[0] ** 2 + t[1] ** 2 + t[2] ** 2) * 39.370079} in")
    frames += 1
    if frames >= 100:
        now = time.perf_counter()
        print(f"avg fps: {frames / (now - start)}")
        start = now
        frames = 0
