import cv2
# import cscore as cs
import numpy as np
import math
from time import sleep
import glob

import array as arr
# # import socket
#
from networktables import NetworkTables

cam = cv2.VideoCapture(1)
cam2 = cv2.VideoCapture(0)

x = 0
y = 0
xdist = 0
w = 0
h = 0
i = 0
counterr = 0
total = 0
area = 0
xavg = 0
pairs = []

irl_coords = np.array([
    # Left target
    (-1, -1, 0.0),  # bottom right
    (-1, -2, 0.0),  # bottom left
    (-1, 1, 0.0), # top right
    (-1, 2, 0.0),  # top left

     # Right target
     (1, -1, 0.0), # bottom left
     (1, 2, 0.0),  # top left
     (1, -2, 0.0), # bottom right
     (1, 1, 0.0),  # top right left vri
])

# irl_coords = arr.array(
#     # Left target
#     [-5.938, 2.938, 0.0], # top left
#     [-4.063, 2.375, 0.0], # top right
#     [-5.438, -2.938, 0.0], # bottom left
#     [-7.375, -2.500, 0.0], # bottom right
#
#     # Right target
#     [3.938, 2.375, 0.0], # top left
#     [5.875, 2.875, 0.0], # top right
#     [7.313, -2.500, 0.0], # bottom left
#     [5.375, -2.938, 0.0], # bottom right
# )
# distortion = np.array([[-0.033708085477096146, 0.10006827139699476, -1.9867941524887844e-05, 0.0018171172937285097, 0.045931410659223106]])
# camera_matrix = np.array([[539.8601217880448, 0.0, 308.9882002976487], [0.0, 538.8789089693274, 242.75908795191646], [0.0, 0.0, 1.0]])
distortion = np.array([[-0.1280467033953502, 0.3705425722116351, -0.004970563347086823, 0.002196848240443673, -0.260619649793084]])
camera_matrix = np.array([[518.150498915968, 0.0, 309.42576909954715], [0.0, 518.3932939237726, 238.3598974513517], [0.0, 0.0, 1.0]])


# def partition(lst, low, high):
#     i = low - 1
#     pivot = lst[high]
#     for j in range(low, high):
#         if cv2.contourArea(lst[j]) <= cv2.contourArea(pivot):
#             i += 1
#             lst[i], lst[j] = lst[j], lst[i]
#     lst[i + 1], lst[high] = lst[high], lst[i + 1]
#     return i + 1
#
# def quick_sort(lst, low, high):
#     if low < high:
#         pi = partition(lst, low, high)
#         quick_sort(lst, low, pi - 1)
#         quick_sort(lst, pi + 1, high)
#
# def sort(list):
#     quick_sort(list, 0, len(list) - 1)
#     return list
#
CONTOUR_MIN_AREA = 35 #20014.5
MIN_ASPECT_RATIO = 0.1 #0.1
MAX_ASPECT_RATIO = 10 #1
BOX_RADIUS = 10 #5
IMAGE_WIDTH = 640 #320
IMAGE_HEIGHT = 480 #240

def contourfilter(contours, img, minasp, maxasp, minarea):
    # This filters out noise based off of aspect ratio and area
    gudContours = []
    minx = 0
    maxx = 0
    miny = 0
    maxy = 0
    for contour in contours:
        # This checks the area of the contours to see if it should be drawn or not
        if cv2.contourArea(contour) > minarea:
            x, y, width, height = cv2.boundingRect(contour)
            aspectRatio = float(width) / height

            # Checks the aspect ratio to see if it fits within the required min and max
            if minasp < aspectRatio < maxasp:
                gudContours.append(contour)
                # rect = cv2.minAreaRect(contour)
                # box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
                # box = np.int0(box)
                x, y, w, h = cv2.boundingRect(contour)
                minx = x - 5
                miny = y - 5
                maxx = x + w + 5
                maxy = y + h + 5
                # print(minx, maxx, miny, maxy)
                cv2.rectangle(img, (minx, miny), (maxx, maxy), (255, 0, 0), 2)
                # draw = cv2.drawContours(img,[box],0,(0,0,255),2)
                # toDraw = cv2.drawContours(draw, gudContours, -1, (0, 0, 255), 3)
                # cv2.rectangle(toDraw, (x, y), (x + width, y + height), (0, 255, 0),2)
                # (x,y),radius = cv2.minEnclosingCircle(contour)
                # center = (int(x),int(y))
                # radius = int(radius)14.5
                # cv2.circle(img,center,radius,(0,255,0),2)
    return gudContours, minx, maxx, miny, maxy

def getcentervalues(contours):
    # grab center average of the contour list.
    # for loop to append all center values to x and y lists
    # grab average value of the lists.
    i = 0
    centerXavg = 0
    centerYavg = 0
    if len(contours) > 0:
        while (i < len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            centerX = x + int(w / 2)
            centerY = y + int(h / 2)
            centerXavg += centerX
            centerYavg += centerY
            i += 1
        centerXavg /= len(contours)
        centerYavg /= len(contours)
    return (centerXavg, centerYavg)

def getEllipseRotation(image, cnt):
    try:
        # Gets rotated bounding ellipse of contour
        ellipse = cv2.fitEllipse(cnt)
        centerE = ellipse[0]
        # Gets rotation of ellipse; same as rotation of contour
        rotation = ellipse[2]
        # Gets width and height of rotated ellipse
        widthE = ellipse[1][0]
        heightE = ellipse[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translateRotation(rotation, widthE, heightE)

        # cv2.ellipse(image, ellipse, (23, 184, 80), 3)
        return rotation
    except:
        # Gets rotated bounding rectangle of contour
        rect = cv2.minAreaRect(cnt)
        # Creates box around that rectangle
        box = cv2.boxPoints(rect)
        # Not exactly sure
        box = np.int0(box)
        # Gets center of rotated rectangle
        center = rect[0]
        # Gets rotation of rectangle; same as rotation of contour
        rotation = rect[2]
        # Gets width and height of rotated rectangle
        width = rect[1][0]
        height = rect[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translateRotation(rotation, width, height)
        return rotation

def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)

def findDimensions(img, cntr):
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
    rect = cv2.minAreaRect(cntr)
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    area = cv2.contourArea(cntr)
    # for i in range(0, 3):
    #     cv2.circle(img, (box[i][0], box[i][1]), 10, (colors[i]), 3)
    # draw = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    #print(box)

    point1 = box[0]
    point2 = box[1]
    point3 = box[2]
    #print(point1, point2, point3)
    distance1 = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    distance2 = math.sqrt((point3[0] - point2[0])**2 + (point3[1] - point2[1])**2)
    width = min(distance1, distance2)
    height = max(distance1, distance2)
    #print(int(width), int(height))
    #print(height)
    return (width, height, int(point1[0]), int(point1[1]), int(point3[0]), int(point3[1]), area, box)

def findPairs(img, contours, isPnP):
    contourArray = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contourArray.append([contour, x, y, w, h])
    contourArray = sorted(contourArray, key=lambda x: x[1])
    pairs = []
    for i in range(len(contours) - 1):
        cntr1 = contourArray[i][0]
        cntr2 = contourArray[i + 1][0]
        angle1 = getEllipseRotation(img, cntr1)
        angle2 = getEllipseRotation(img, cntr2)
        if(np.sign(angle1) != np.sign(angle2) and angle1 < 0):
            x1, y1 = getcentervalues(cntr1)
            x2, y2 = getcentervalues(cntr2)
            xdist = distanceToX(cntr1, cntr2)
            ydist = distanceToY(cntr1, cntr2)
            w1,h1,p1x1,p1y1,p2x1,p2y1, a1, b1 = findDimensions(img, cntr1)
            w2,h2,p1x2,p1y2,p2x2,p2y2, a2, b2 = findDimensions(img, cntr2)
            # print(w1,h1, w2, h2)
            xavg = int((x1 + x2) / 2)
            yavg = int((y1 + y2) / 2)
            if(a1 > a2):
                isLeft = True
            else:
                isLeft = False
            if isPnP:
                x, y, z, dist, imgc = get3DCoordinates(b1, b2, yavg, img)
            else:
                x = 0
                y = 0
                z = 0
                dist = 0
            # print("x %d" % x)
            # print("y %d" % y)
            # print("z %d" % z)
            xtarget1 = contourArray[i][1]
            ytarget12 = contourArray[i + 1][2]
            ytarget11 = contourArray[i][2]
            ytarget1 = min(ytarget11, ytarget12)
            xtarget2 = contourArray[i + 1][1] + contourArray[i + 1][3]
            ytarget21 = contourArray[i + 1][2] + contourArray[i + 1][4]
            ytarget22 = contourArray[i][2] + contourArray[i][4]
            ytarget2 = max(ytarget21, ytarget22)
            width = xtarget2 - xtarget1
            height = max(ytarget2, ytarget21) - ytarget1
            ratio = width/height
            if((abs(p1y2 - p1y1) + abs(p2y2-p2y1)) > 6):
                cv2.rectangle(img, (xtarget1, ytarget1), (xtarget2, max(ytarget2, ytarget21)), (0, 0, 255), 2)
            else:
                cv2.rectangle(img, (xtarget1, ytarget1), (xtarget2, max(ytarget2, ytarget21)), (0, 255, 0), 2)
            cv2.circle(img, (int(xavg), int(yavg)), 5, (0, 255, 0))
            pairs.append([cntr1, cntr2, xavg, yavg, xdist[0], ydist[0], xtarget1, ytarget1, xtarget2, ytarget2, ratio, width, dist, isLeft, x, y, z])

            # cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 5)
    return pairs

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img
axis = np.float32([[-3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
irlcoords = np.float32([[0,0,0],[0,2,0],[-5.5,0,0],[-5.5,2,0]])
irlcoords2 = np.float32([[-4,0,0],[-5.37709,-5.32481,0],[4,0,0],[5.37709,-5.32481,0]])
def get3DCoordinates(box1, box2, centerY, img):
    box1 = sorted(box1, key=lambda x: x[1])
    # print(img_coords)
    # img_coords[:, 0] -= 160
    # img_coords[:, 1] -= centerY
    # img_coords[:, 1] *= -1
    box2 = sorted(box2, key=lambda x: x[1])
    img_coords = np.float32([box1[1], box1[3], box2[1], box2[3]])
    # print(img_coords)
    ret, rvecs, tvecs = cv2.solvePnP(irlcoords2, img_coords, camera_matrix, distortion, True)
    # print(rvecs)
    # print(tvecs)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, distortion)
    x = tvecs[0][0]
    y = tvecs[1][0]
    z = tvecs[2][0]
    dist = math.sqrt(x*x + z*z)
    actdist = math.sqrt(y*y + dist*dist)
    print(actdist)
    draw(img, img_coords, imgpts)
    return (x, y, z, actdist, img_coords)

def distanceToY(one, two):
    x1, y1, w1, h1 = cv2.boundingRect(one)

    x2, y2, w2, h2 = cv2.boundingRect(two)

    centerOneX = x1 + int(w1 / 2)
    centerTwoX = x2 + int(w2 / 2)
    centerOneY = y1 + int(h1 / 2)
    centerTwoY = y2 + int(h2 / 2)



    # print ("Distance To X - ", max(centerOneY, centerTwoY) - min(centerOneY, centerTwoY))
    dist = max(centerOneY, centerTwoY) - min(centerOneY, centerTwoY)
    return (dist, centerOneX, centerTwoX, centerOneY, centerTwoY)

def distanceToX(one, two):
    x1, y1, w1, h1 = cv2.boundingRect(one)

    x2, y2, w2, h2 = cv2.boundingRect(two)

    centerOneX = x1 + int(w1 / 2)
    centerTwoX = x2 + int(w2 / 2)
    centerOneY = y1 + int(h1 / 2)
    centerTwoY = y2 + int(h2 / 2)

    # print ("Distance To X - ", max(centerOneX, centerTwoX) - min(centerOneX, centerTwoX))
    dist = max(centerOneX, centerTwoX) - min(centerOneX, centerTwoX)
    return (dist, centerOneX, centerTwoX, centerOneY, centerTwoY)
# cam = cv2.VideoCapture('incremental.avi')
def grabFeed():
    ca = visionCam.getDouble(0.0)
    if (ca == 0.0):
        ret, capture = cam.read()
    elif (ca == 1.0):
        ret, capture = cam2.read()
    else:
        visionCam.setDouble(0.0)
    # ret, capture = cam.read()
    # ret, capture = cam.read()
    return capture
def filter(img, H_LOW, S_LOW, L_LOW, H_HIGH, S_HIGH, L_HIGH):
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 70, 90, 70
    thresh = cv2.inRange(hsl, (H_LOW, L_LOW, S_LOW), (H_HIGH, L_HIGH, S_HIGH))
    return thresh

# out = cv2.VideoWriter('incremental.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))
def displayImage(img, i):
    font = cv2.FONT_HERSHEY_COMPLEX
    i = str(i)
    cv2.putText(img, i, (30, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Team3482 OpenCV", img)
    # font = cv2.FONT_HERSHEY_COMPLEX
    # camera.putFrame(img)
    # out.write(img)

def displayImage(img):
    # font = cv2.FONT_HERSHEY_COMPLEX
    # i = str(i)
    # cv2.putText(img, i, (30, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Team3482 OpenCV", img)
    # font = cv2.FONT_HERSHEY_COMPLEX
    # camera.putFrame(img)
    # out.write(img)

def pipeline1(j):
    capture = grabFeed()

    displayImage(j)


def pipeline2(j):
    # x1 = 0
    # x2 = 0
    # y1 = 0
    # y2 = 0
    NetworkTables.flush()
    initial = cv2.getTickCount()
    capture = grabFeed()
    thresh = filter(capture, H_LOW.getDouble(47.5), S_LOW.getDouble(67.5), L_LOW.getDouble(70), H_HIGH.getDouble(72.5), 255, 255)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gudContours, mix, max, miy, may = contourfilter(contours, capture, 0.1, 10, 75)
    contoursNew = cv2.drawContours(capture, gudContours, -1, (240, 0, 0), 3)
    # # hls = cv2.cvtColor(thresh, cv2.COLOR_HLS2BGR)
    # # gray = cv2.cvtColor(hls, cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((5,5), np.uint8)
    # erosion = cv2.erode(thresh, kernel, iterations = 1)
    # gray = np.float32(erosion)
    # corners = cv2.goodFeaturesToTrack(gray, 8, 0.01, 10, useHarrisDetector=True)
    # print(mix, max, miy, may)
    # for corner in corners:
    #     x, y = corner.ravel()
    #     if(mix <= x <= max and miy <= y <= may):
    #         cv2.circle(contoursNew, (x, y), 5, (0, 0, 255), -1)
    # sleep(0.01)
    displayImage(contoursNew)

def pipeline3(i):
    capture = grabFeed()
    # gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    # # Find the chess board corners
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    # # If found, add object points, image points (after refining them)
    # if ret == True:
    #     corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    #     # Draw and display the corners
    #     cv2.drawChessboardCorners(capture, (7, 6), corners2, ret)
    displayImage(capture)

def pipeline4(i):
    capture = grabFeed()
    # capture1 = grabFeed()
    thresh = filter(capture, 47.5, 67.5, 70, 72.5,
                     255, 255)

    kernel = np.ones((4, 4), np.float32)/25
    thresh = cv2.filter2D(thresh, -1, kernel)
    points = []


    #
    # bgr = cv2.cvtColor(thresh, cv2.COLOR_HLS2BGR)
    # gray = cv2.cvtColor(points, cv2.COLOR_BGR2GRAY)
    #
    #
    corner = cv2.goodFeaturesToTrack(thresh, 8, .075, 8, useHarrisDetector=True)
    corner = np.int0(corner)
    # #
    for i in corner:
        x, y = i.ravel()
        cv2.circle(capture, (x, y), 5, (0, 0, 255), -1)
        points.append([x, y])

    # dst = cv2.cornerHarris(thresh, 8, 3, 0.1, 1, 0)
    #
    # dst = cv2.dilate(dst, None)
    # #
    # capture[dst > 0.3 * dst.max()] = [255, 255, 255]
    # points = np.unravel_index(dst.argmax(), dst.shape)
    # print(points)
    # gray[dst > 0.01 * dst.max()] = [255, 255, 255]
    # #
    #


    if len(points) == 8:
        print(points)

    displayImage(capture)
def pipeline5(i):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('1.png')
    img = grabFeed()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8, 6), corners2, ret)
        # capture =
    cv2.imshow('img', img)
    # coordinates = np.unravel_index(dst.argmax(), dst.shape)
    # coordinates = [coordinates[1], coordinates[0]]
    # print(coordinates)




    # displayImage(cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY), i)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
def pipeline6(i):
    capture = grabFeed()
    gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Find the rotation and translation vectors.
        ransacisajackass, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, camera_matrix, distortion)
        # project 3D points to image plane
        print("RVec: " + str(rvecs))
        print("TVec: " + str(tvecs))
        print("Inliers: " + str(inliers))
        print("ransacisajackass: " + str(ransacisajackass))
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, distortion)
        x = tvecs[0][0]
        y = tvecs[1][0]
        z = tvecs[2][0]
        print("x: " + str(x) + " y: " + str(y) + " z: " + str(z))
        dist = math.sqrt(x*x + z*z)
        actdist = math.sqrt(y*y + dist*dist)
        print("dist: "+ str(actdist))
        capture = draw(capture, corners2, imgpts)
    displayImage(capture)


def redo():
    j = 0
    while(True):
        # pipe = int(pipelines.getDouble(1.0))
        pipe = 3
        if(pipe == 1):
            pipeline1(j)
        elif(pipe == 2):
            pipeline2(j)
        elif(pipe == 3):
            pipeline3(j)
        elif(pipe == 4):
            pipeline4(j)
        elif (pipe == 5):
            pipeline5(j)
            j=1
        elif(pipe == 6):
            pipeline6(j)
        else:
            pipe = 1
        # if (ticker.getDouble(0.0) == 0.0):
        #     ticker.setDouble(1.0)
        # else:
        #     ticker.setDouble(0.0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        j += 1
