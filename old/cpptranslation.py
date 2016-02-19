#Direct Translation from C++

import cv2
import numpy as np


def dist(v1, v2):
    return (v1[0] - v2[0])**2 + (v1[1] - v2[1])**2

def circleFromPoints(point1, point2, p3):
    offset = point2[0] ** 2 + point2[1] ** 2
    bc = (point1[0] ** 2 + point1[1] ** 2 - offset) / 2.0
    cd = (offset - p3[0]**2 - p3[1]**2)/2.0
    det = (point1[0] - point2[0]) * (point2[1] - p3[1]) - (point2[0] - p3[0]) * (point1[1] - point2[1])
    TOL = 0.0000001
    if abs(det) < TOL:
        print("POINTS TOO CLOSE")
        return (0, 0), 0

    idet = 1/det
    centerx = (bc * (point2[1] - p3[1]) - cd * (point1[1] - point2[1])) * idet
    centery = (cd * (point1[0] - point2[0]) - bc * (point2[0] - p3[0])) * idet
    radius = (pow(point2[0] - centerx, 2) + pow(point2[1] - centery, 2)) ** 0.5
    return (centerx,centery), radius

def main():
    palm_centers = []
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    bg.setNMixtures(3)

    backgroundFrame = 100

    for x in range(0, 50):
        ret, frame_unflipped = cam.read()


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    while True:
        ret, frame_unflipped = cam.read()
        if ret:

            full_frame = cv2.flip(frame_unflipped, 1)
            #full_frame = cv2.GaussianBlur(full_frame, (111, 111), 1)



            width = full_frame.shape[1]
            height = full_frame.shape[0]
            halfwidth = int(width*0.6)
            halfheight = int(height*0.7)

            cv2.rectangle(full_frame, pt1=(halfwidth-1, 0), pt2=(width, halfheight), color=[255,0,0], thickness=1, lineType=8)


            frame = full_frame[:halfheight, halfwidth:]



            back = frame

            if backgroundFrame > 0:
                fore = bg.apply(frame)
                backgroundFrame -= 1
            else:
                fore = bg.apply(frame, learningRate=0)
            back = bg.getBackgroundImage()
            fore = cv2.erode(fore, np.zeros((3, 3)))
            _, contours, hierarchy, = cv2.findContours(fore, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for i in range(len(contours)):
                contour1 = contours[i]

                if cv2.contourArea(contour1) >= 5000:
                    tcontours = list()
                    tcontours.append(contours[i])
                    cv2.drawContours(frame, tcontours, -1, (0,0,255), 2)

                    hull = cv2.convexHull(tcontours[0], returnPoints=False)

                    cv2.drawContours(frame, [tcontours[0][x] for x in hull], -1, (0,255,0), 2)

                    rect = cv2.minAreaRect(tcontours[0])

                    if len(hull) > 0:
                        rect_points = [(int(rect[0][0]), int(rect[0][1])),
                                       (int(rect[1][0]), int(rect[1][1])),
                                       ]

                        #cv2.rectangle(frame, pt1=rect_points[0], pt2=rect_points[1], color=[255,0,0], thickness=1, lineType=8)
                        rough_palm_center = np.array([0, 0])

                        defects = cv2.convexityDefects(tcontours[0], hull)
                        defects = np.array([x[0] for x in defects])


                        if len(defects) >= 3:
                            palm_points = []
                            for j in range(len(defects)):
                                startidx = defects[j][0]
                                ptStart = tcontours[0][startidx]

                                endidx = defects[j][1]
                                ptEnd = tcontours[0][endidx]

                                faridx = defects[j][2]
                                ptfar = tcontours[0][faridx]

                                rough_palm_center += ptStart[0] + ptEnd[0] + ptfar[0]

                                palm_points.append(ptfar[0])
                                palm_points.append(ptStart[0])
                                palm_points.append(ptEnd[0])
                            rough_palm_center[0] /= len(defects)*3
                            rough_palm_center[1] /= len(defects)*3
                            distvec = []
                            for i in range(len(palm_points)):
                                distvec.append([dist(rough_palm_center, palm_points[i]), i])
                            distvec.sort()

                            soln_circle = [0, 0]
                            for i in range(0, len(distvec)-2):
                                p1 = palm_points[distvec[i+0][1]]
                                p2 = palm_points[distvec[i+1][1]]
                                p3 = palm_points[distvec[i+2][1]]
                                soln_circle = circleFromPoints(p1, p2, p3)
                                if soln_circle[1] != 0:
                                    break
                            palm_centers.append(soln_circle)
                            if len(palm_centers) > 20:
                                palm_centers = palm_centers[1:]

                            palm_center = np.array([0,0], dtype=np.float64)
                            radius = 0
                            for i in range(0, len(palm_centers)):
                                palm_center += palm_centers[i][0]
                                radius += palm_centers[i][1]
                            palm_center[0] /= len(palm_centers)
                            palm_center[1] /= len(palm_centers)

                            radius /= len(palm_centers)

                            cv2.circle(frame, tuple(palm_center.astype(int)), 5, (144, 144, 255), 3)
                            #cv2.circle(frame, tuple(palm_center.astype(int)), int(radius), (144, 144, 255), 2)

                            no_of_fingers = 0

                            finger_points = []
                            inner_points = []
                            inner_lines = []
                            outer_lines = []
                            inter_lines = []
                            for j in range(len(defects)):
                                startidx = defects[j][0]
                                ptStart = tcontours[0][startidx]

                                endidx = defects[j][1]
                                ptEnd = tcontours[0][endidx]

                                faridx = defects[j][2]
                                ptFar = tcontours[0][faridx]

                                Xdist=(dist(palm_center, ptFar[0]))**0.5
                                Ydist=(dist(palm_center, ptStart[0]))**0.5
                                length=(dist(ptFar[0], ptStart[0]))**0.5


                                retLength = (dist(ptEnd[0], ptFar[0]))**0.5

                                #cv2.line(frame, tuple(ptEnd[0]), tuple(ptFar[0]), (0, 255, 0), 1)


                                no_of_fingers += 1

                                if Ydist >= 0.4*radius and 3*radius >= length >= 10 and retLength >= 10 and max(length, retLength)/min(length, retLength) >= 0.8:
                                    if min(Xdist, Ydist)/max(Xdist, Ydist) <= 0.8:
                                        a, b = [Xdist, Ydist]
                                        if min(a, 1.3*radius) >= b >= 0.1*radius or min(b, 1.3*radius) >= a >= 0.1*radius:
                                            if ptfar[0][1] < palm_center[1]:
                                                finger_points.append(tuple(ptEnd[0]))
                                                finger_points.append(tuple(ptStart[0]))
                                                inner_points.append(tuple(ptFar[0]))

                                                inner_lines.append((tuple(ptEnd[0]),tuple(ptFar[0])))
                                                outer_lines.append((tuple(ptStart[0]),tuple(ptFar[0])))
                                                inter_lines.append((tuple(ptStart[0]),tuple(ptEnd[0])))

                                                no_of_fingers += 1


                            finger_points = merge_finger_points(finger_points)
                            cv2.circle(frame, tuple(palm_center.astype(int)), 5, (144, 144, 255), 3)





                            for line in inner_lines + outer_lines + inter_lines:
                                cv2.line(frame, line[0], line[1], (0, 0, 255), 1)
                                pass




                            for point in finger_points:
                                cv2.circle(frame, point, 3, (0, 255, 0), 3)
                            for point in inner_points:
                                cv2.circle(frame, point, 3, (255, 0, 0), 3)


                            #no_of_fingers = min(5, no_of_fingers)
                            #print("NO OF FINGERS: ", no_of_fingers)

            if backgroundFrame > 0:
                cv2.putText(frame, "Recording Background", (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1, cv2.LINE_AA);
            cv2.imshow("Window", full_frame)
            cv2.imshow("Background", back)
            ch = cv2.waitKey(1)
            if ch == 1048603:
                break
            elif ch > 0:
                bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
                bg.setNMixtures(3)

                backgroundFrame = 100

def merge_finger_points(finger_points):
    for point1 in finger_points:
        close_points = []
        for point2 in finger_points:
            if dist(point1, point2)**0.5 < 40 and point1!=point2:
                close_points.append(point2)
        if len(close_points) > 0:
            close_points.append(point1)
            avg_point = np.mean(np.array(close_points), axis=0).astype(int)
            new_points = [x for x in finger_points if x!=point1] + [list(avg_point)]
            return merge_finger_points(new_points)

    return [tuple(x) for x in finger_points]












if __name__ == "__main__":
    main()
