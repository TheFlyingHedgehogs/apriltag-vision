import glob
import math
import pprint

import cv2
import apriltag
import numpy as np
import pickle as pkl

# image = cv2.imread("images/1.png")
width = 0.1651  # m

options = apriltag.DetectorOptions(families="tag36h11")  # , refine_edges=True, refine_pose=True
detector = apriltag.Detector(options)

object_points = np.array([
    [-width/2, -width/2, 0.0],  # top left
    [width/2,  -width/2, 0.0],  # top right
    [width/2,   width/2, 0.0],  # bottom right
    [-width/2,  width/2, 0.0],  # bottom left
])

# focal_length = (1920 / 36) * 50
#
# mtx = np.array([[focal_length,          0.0, 960],
#                 [0.0,          focal_length, 540],
#                 [0.0,                   0.0, 1.0]])
# dist = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])

identity = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

mtx, dist = pkl.load(open("calib-fisheye", "rb"))

identity_dist = np.zeros(dist.shape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)
    vecs = []

    # loop over the AprilTag detection results
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        # (ptA, ptB, ptC, ptD) = r.corners

        points = np.expand_dims(np.array(r.corners, dtype=np.float64), axis=-2)

        image_points = cv2.fisheye.undistortPoints(points, mtx, dist, P=identity)
        # image_points = cv2.undistortPoints(np.array(r.corners), mtx, dist, P=identity)
        # print(image_points)
        # corners2 = cv2.cornerSubPix(gray, np.array([
        #     [int(r.corners[0][0]), int(r.corners[0][1])],
        #     [int(r.corners[1][0]), int(r.corners[1][1])],
        #     [int(r.corners[2][0]), int(r.corners[2][1])],
        #     [int(r.corners[3][0]), int(r.corners[3][1])]
        # ], dtype=np.float32), (11, 11), (-1, -1), criteria)
    # print(imagepoints)

        # pprint.pprint(detector.detection_pose(r, (focal_length, focal_length, 1920/2, 1080/2), tag_size=width))

        ret, rvec, tvec = cv2.solvePnP(object_points, image_points, identity, identity_dist)

        # cv2.drawFrameAxes(image, mtx, dist, rvec, tvec, width)

        # pprint.pprint(tvec)

        if ret:
            vecs.append((rvec, tvec))

        # new_img = cv2.resize(image, (1920 * 4, 1080 * 4))
        #
        # for item in r.corners:
        #     cv2.drawMarker(new_img, (int(item[0] * 4), int(item[1] * 4)), (255, 0, 0))
        #     # cv2.drawMarker(new_img, (int(int(item[0]) * 4), int(int(item[1]) * 4)), (255, 0, 0))
        # cv2.imwrite("/tmp/aaa.png", new_img)
        #
        # ptB = (int(ptB[0]), int(ptB[1]))
        # ptC = (int(ptC[0]), int(ptC[1]))
        # ptD = (int(ptD[0]), int(ptD[1]))
        # ptA = (int(ptA[0]), int(ptA[1]))
        # # draw the bounding box of the AprilTag detection
        # cv2.line(image, ptA, ptB, (0, 255, 0), 1)
        # cv2.line(image, ptB, ptC, (0, 255, 0), 1)
        # cv2.line(image, ptC, ptD, (0, 255, 0), 1)
        # cv2.line(image, ptD, ptA, (0, 255, 0), 1)
        # # draw the center (x, y)-coordinates of the AprilTag
        # (cX, cY) = (int(r.center[0]), int(r.center[1]))
        # cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # # draw the tag family on the image
        # tagFamily = r.tag_family.decode("utf-8")
        # cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # print("[INFO] tag family: {}".format(tagFamily))
    # show the output image after AprilTag detection
    # cv2.imshow("Image", cv2.resize(image, (1920//2, 1080//2)))
    # cv2.waitKey(0)
    return vecs


for i, path in enumerate(sorted(glob.glob("images/one-to-nine-fisheye/*"))):
    dst = int(path.removesuffix(".png").removeprefix("images/one-to-nine-fisheye/")) * (8 / 128) + 1
    gotten = detect(cv2.imread(path))[0][1]
    found = math.sqrt(gotten[0]**2 + gotten[1]**2 + gotten[2]**2)
    # print(f"dst: {dst:<4} error: {abs(dst - found) * 100:<19} cm")
    print(f"{i},{dst},{found}")
