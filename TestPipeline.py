import cv2
import numpy as np

SRC_IMAGE = "WPILib Robot Vision Images\BlueGoal-084in-Center.jpg"
H_LOW = 60
S_LOW = 67.5
L_LOW = 70
H_HIGH = 100
S_HIGH = 255
L_HIGH = 255
cap = cv2.VideoCapture(0)
if(cap.isOpened() == False):
    cap.open()
def grabFeed():
    ret, capture = cap.read()
    print(ret)
    # cam.open(CV_CAP_DSHOW);
    # cam.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
    # cam.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    # cam.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
    return capture

def grabImage():
    return cv2.imread(SRC_IMAGE)

def display(img):
    cv2.imshow('image', img)
    test = cap.get(cv2.CAP_PROP_POS_MSEC)
    ratio = cap.get(cv2.CAP_PROP_POS_AVI_RATIO)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    saturation = cap.get(cv2.CAP_PROP_SATURATION)
    hue = cap.get(cv2.CAP_PROP_HUE)
    gain = cap.get(cv2.CAP_PROP_GAIN)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    print("Test: ", test)
    print("Ratio: ", ratio)
    print("Frame Rate: ", frame_rate)
    print("Height: ", height)
    print("Width: ", width)
    print("Brightness: ", brightness)
    print("Contrast: ", contrast)
    print("Saturation: ", saturation)
    print("Hue: ", hue)
    print("Gain: ", gain)
    print("Exposure: ", exposure)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def filter(img, h_low, s_low, l_low, h_high, s_high, l_high):
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    thresh = cv2.inRange(hsl, (h_low, l_low, s_low), (h_high, l_high, s_high))
    return thresh

def contourfilter(contours, img, minasp, maxasp, minarea):
    # This filters out noise based off of aspect ratio and area
    gudContours = []
    for contour in contours:
        # This checks the area of the contours to see if it should be drawn or not
        if cv2.contourArea(contour) > minarea:
            x, y, width, height = cv2.boundingRect(contour)
            aspectRatio = float(width) / height
            print("one")
            # Checks the aspect ratio to see if it fits within the required mqqqqqqqin and max
            if minasp < aspectRatio < maxasp:
                print("two")
                gudContours.append(contour)
                cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
                box = np.int0(box)
                x, y, w, h = cv2.boundingRect(contour)
                # print(minx, maxx, miny, maxy)
                # draw = cv2.drawContours(img,[box],0,(0,0,255),2)
                # toDraw = cv2.drawContours(draw, gudContours, -1, (0, 0, 255), 3)
                # cv2.rectangle(toDraw, (x, y), (x + width, y + height), (0, 255, 0),2)
                # (x,y),radius = cv2.minEnclosingCircle(contour)
                # center = (int(x),int(y))
                # radius = int(radius)14.5
                # cv2.circle(img,center,radius,(0,255,0),2)
    return gudContours

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

distortion = np.array([[-0.1280467033953502, 0.3705425722116351, -0.004970563347086823, 0.002196848240443673, -0.260619649793084]])
camera_matrix = np.array([[518.150498915968, 0.0, 309.42576909954715], [0.0, 518.3932939237726, 238.3598974513517], [0.0, 0.0, 1.0]])
irl_coords = []
def get3DCoordinates(corners):
    ret, rvecs, tvecs = cv2.solvePnP(irl_coords, corners, camera_matrix, distortion, True)
    return rvecs, tvecs
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, distortion)
    x = tvecs[0][0]
    y = tvecs[1][0]
    z = tvecs[2][0]
    dist = math.sqrt(x * x + z * z)
    actdist = math.sqrt(y * y + dist * dist)
    print(actdist)
    draw(img, img_coords, imgpts)

def pipeline():
    feed = grabFeed()
    display(feed)



    # img = grabImage()
    # filtered = filter(img, H_LOW, S_LOW, L_LOW, H_HIGH, S_HIGH, L_HIGH)
    # im2, contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # target = contourfilter(contours, filtered, 0.1, 5.0, 250)
    # # print(str(target))
    # corners = []
    # for cnt in target:
    #
    #     approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    #
    #     # draws boundary of contours.
    #     cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
    #
    #     # Used to flatted the array containing
    #     # the co-ordinates of the vertices.
    #     n = approx.ravel()
    #     i = 0
    #
    #     for j in n:
    #         if (i % 2 == 0):
    #             x = n[i]
    #             y = n[i + 1]
    #             cv2.circle(img, (x, y), 3, (255, 0, 0), 3)
    #             corners.append((x, y))
    #             # String containing the co-ordinates.
    #             string = str(x) + " " + str(y)
    #             # if (i == 0):
    #             #     # text on topmost co-ordinate.
    #             #     cv2.putText(img, "Arrow tip", (x, y),
    #             #                 cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    #             # else:
    #             #     # text on remaining co-ordinates.
    #             #     cv2.putText(img, string, (x, y),
    #             #                 cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
    #         i = i + 1
    # cv2.drawContours(img, target, -1, (255, 0, 0), 2)
    # corners = sorted(corners, key=lambda x: x[1])
    # print(corners)
    # display(img)




while(True):
    pipeline()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()