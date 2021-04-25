#!/usr/bin/python3

import itertools
import sys
import rospy
import dlib
import os
import cv2
import numpy as np
import tf2_geometry_msgs
import tf2_ros
from pprint import pprint
#import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose, Point, Twist
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class color_localizer:

    def __init__(self):
        self.found_rings = []
        self.found_cylinders = []

        rospy.init_node('color_localizer', anonymous=True)
        
        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()
        self.m_arr = MarkerArray()
        self.nM = 0
        # Subscribe to the image and/or depth topic
        # self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        # self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher('ring_markers', MarkerArray, queue_size=1000)
        self.pic_pub = rospy.Publisher('face_im', Image, queue_size=1000)
        #self.points_pub = rospy.Publisher('/our_pub1/chat1', Point, queue_size=1000)
        # self.twist_pub = rospy.Publisher('/our_pub1/chat1', Twist, queue_size=1000)

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        
        #notri hranimo "priblizen center slike" pod katerim je mnzoica 100 ter normala stene na kateri je
        #key so stevilke znotraj imamo pa prvo povprecje vseh tock in nato se vse tocke (np.array) shranjene v seznamu
        # self.detected_pos_fin = {}
        # self.detected_norm_fin = {}
        # self.entries = 0
        # self.range = 0.2
    
    def chk_ring(self,de,h1,h2,w1,w2,cent):
        #depoth im je for some reason 480x640
        cmb_me = []
        if(h1[0]>=0 and h1[1]>=0 and h1[0]<640 and h1[1]<480 and de[h1[1],h1[0]]):
            print(de[h1[1],h1[0]])
            cmb_me.append(de[h1[1],h1[0]])
        
        if(h2[0]>=0 and h2[1]>=0 and h2[0]<640 and h2[1]<480 and de[h2[1],h2[0]]):
            cmb_me.append(de[h2[1],h2[0]])
        
        if(w1[0]>=0 and w1[1]>=0 and w1[0]<640 and w1[1]<480 and de[w1[1],w1[0]]):
            cmb_me.append(de[w1[1],w1[0]])
        
        if(w2[0]>=0 and w2[1]>=0 and w2[0]<640 and w2[1]<480 and de[w2[1],w2[0]]):
            cmb_me.append(de[w2[1],w2[0]])
        if len(cmb_me)<3:
            #print("garbo")
            return None
        points = []
        for i in itertools.combinations(cmb_me,3):
            #preverimo ce so vse 3 točke manj kto 20cm narazen(to je največji možn diameter kroga
            good = True
            for j in itertools.combinations(i,2):
                if np.abs(j[0] - j[1])>0.2:
                    good = False
                    break
            if good:
                #print("we good")
                avgdist = sum(i)/3
                #hardcodana natancnost ker zadeva zna mal zajebavat mogoč obstaja bolj robusten način
                if(avgdist>3):
                    continue
                #print(de.shape)
                if de[cent[1],cent[0]] > avgdist+0.15:
                    print("center je dlje od povprečja točk")
                    #MOŽNO DA JE TREBA ZAMENAT!!
                    # for z in i:
                    #     print(f"\t{z}")
                    return (cent[1],cent[0],avgdist)
        return None

    def get_pose(self,xin,yin,dist, depth_im,objectType):
        # Calculate the position of the detected ellipse
        print(dist)
        k_f = 525 # kinect focal length in pixels

        xin = -(xin - depth_im.shape[1]/2)

        angle_to_target = np.arctan2(xin,k_f)

        # Get the angles in the base_link relative coordinate system
        x,y = dist*np.cos(angle_to_target), dist*np.sin(angle_to_target)


        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = 0
        point_s.point.z = x
        point_s.header.frame_id = "camera_rgb_optical_frame"
        point_s.header.stamp = rospy.Time(0)

        # Get the point in the "map" coordinate system
        point_world = self.tf_buf.transform(point_s, "map")

        # print(f"\nlocal point")
        # print(f"\tx: {point_s.point.x}")
        # print(f"\ty: {point_s.point.y}")
        # print(f"\tz: {point_s.point.z}")
        # print(f"world point")
        # print(f"\tx: {point_world.point.x}")
        # print(f"\ty: {point_world.point.y}")
        # print(f"\tz: {point_world.point.z}")


        allDist = []
        if objectType == "ring":
            for elipse in self.found_rings:
                pointsDist = np.sqrt((elipse["x"]-point_world.point.x)**2+(elipse["y"]-point_world.point.y)**2)
                if pointsDist < 0.5 or dist>2:
                    return
                allDist.append(pointsDist)
            # allDist = np.array(allDist)
            self.found_rings.append({"x":point_world.point.x, "y":point_world.point.y})
        elif objectType == "cylinder":
            for cylinder in self.found_cylinders:
                pointsDist = np.sqrt((cylinder["x"]-point_world.point.x)**2+(cylinder["y"]-point_world.point.y)**2)
                if pointsDist < 0.5 or dist>2:
                    return
                allDist.append(pointsDist)
            # allDist = np.array(allDist)
            self.found_cylinders.append({"x":point_world.point.x, "y":point_world.point.y})


        pprint(sorted(allDist))


        # Create a Pose object with the same position
        pose = Pose()
        pose.position.x = point_world.point.x
        pose.position.y = point_world.point.y
        pose.position.z = point_world.point.z

        # Create a marker used for visualization
        self.nM += 1
        marker = Marker()
        marker.header.stamp = point_world.header.stamp
        marker.header.frame_id = point_world.header.frame_id
        marker.pose = pose
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False
        marker.lifetime = rospy.Duration.from_sec(0)
        marker.id = self.nM
        marker.scale = Vector3(0.1, 0.1, 0.1)
        marker.color = ColorRGBA(0, 1, 0, 1)
        self.m_arr.markers.append(marker)

        self.markers_pub.publish(self.m_arr)

    #dela as intended!
    def calc_pnts(self, el):
        #width
        h1 = np.array([0,el[1][1]/2])
        h2 = np.array([0,-el[1][1]/2])
        w1 = np.array([el[1][0]/2,0])
        w2 = np.array([-el[1][0]/2,0])


        c = np.cos(np.radians(el[2]))
        s = np.sin(np.radians(el[2]))
        rot = np.array([[c,-s],[s,c]])


        h1 = el[0] + rot.dot(h1)
        h2 = el[0] + rot.dot(h2)
        w1 = el[0] + rot.dot(w1)
        w2 = el[0] + rot.dot(w2)
        return h1,h2,w1,w2
    
    def find_elipses_first(self,image,depth_image,stamp,grayBGR_toDrawOn):

        minValue = np.nanmin(depth_image)
        maxValue = np.nanmax(depth_image)
        depth_im = depth_image
        temp_mask = ~np.isfinite(depth_image)
        nanToValue = maxValue*1.25
        depth_image[temp_mask] = nanToValue
        depth_image = depth_image - minValue
        depth_image = depth_image * 255/(nanToValue - minValue)
        depth_image = np.round(depth_image)
        depth_image = depth_image.astype(np.uint8)
        # print(depth_image)
        # print(np.min(depth_image))
        # print(np.max(depth_image))
        # print(depth_image.shape)

        frame = image.copy()

        imagesToTest = [
                        frame[:,:,0],
                        frame[:,:,1],
                        frame[:,:,2],
                        cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[:,:,0],
                        cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[:,:,1],
                        cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[:,:,2],
                        depth_image
                        ]

        # Set the dimensions of the image
        dims = frame.shape

        # Tranform image to gayscale
        gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(gray1,cv2.COLOR_GRAY2BGR)
        # gray1 = cv2.cvtColor(depth_image,cv2.COLOR_GRAY2BGR)

        # return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # gray[:,:,1] = 255
        # gray[:,:,2] = 255
        # sat = gray[:,:,1] < 128
        # gray[sat] = [0,0,0]
        # return cv2.cvtColor(gray,cv2.COLOR_HSV2BGR)
        # return cv2.cvtColor(gray[:,:,2],cv2.COLOR_GRAY2BGR)

        for i in range(len(imagesToTest)):
            gray = imagesToTest[i]

            # Do histogram equlization
            img = cv2.equalizeHist(gray)
            
            threshDict = {}
            # Binarize the image - original
            # ret, thresh = cv2.threshold(img, 50, 255, 0)
            ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
            threshDict["normal"] = thresh

            # binary + otsu
            ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshDict["binaryOtsu"] = thresh

            # gauss + binary + otsu
            img2 = cv2.GaussianBlur(img,(5,5),2)
            ret, thresh = cv2.threshold(img2, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshDict["gaussBinaryOtsu"] = thresh
            
            # adaptive mean threshold
            thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,2)
            threshDict["adaptiveMean"] = thresh
            
            # adaptive gaussian threshold
            thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
            threshDict["adaptiveGauss"] = thresh

            # t = threshDict["normal"]
            # pprint(t)
            t = None
            for i,key in enumerate(threshDict):
                if i==0:
                    t = threshDict[key]
                else:
                    part1 = i/(i+1)
                    part2 = 1-part1
                    t = cv2.addWeighted(t,part1, threshDict[key],part2, 0)
            
            thresh = t

            thresh = cv2.Canny(gray,100,200)

            #return cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)




            kernel = np.ones((5,5), "uint8")

            # threshDict["adaptiveGauss"] = cv2.erode(threshDict["adaptiveGauss"], kernel)
            #return cv2.cvtColor(cv2.bitwise_not(threshDict["adaptiveGauss"]),cv2.COLOR_GRAY2RGB)

            # Extract contours
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Example how to draw the contours
            # cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

            # Fit elipses to all extracted contours
            elps = []
            for cnt in contours:
                #     print cnt
                #     print cnt.shape
                if cnt.shape[0] >= 20:
                    ellipse = cv2.fitEllipse(cnt)
                    elps.append(ellipse)

            # Find two elipses with same centers
            candidates = []
            for n in range(len(elps)):
                for m in range(n + 1, len(elps)):
                    e1 = elps[n]
                    e2 = elps[m]
                    dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                    avg_cent = (int((e1[0][0] + e2[0][0])/2), int((e1[0][1] + e2[0][1])/2))
                    #             print dist
                    #preverimo da sta res 2 razlicne elipse in en vec za malo različnih elips
                    if dist < 5 and np.abs(e1[1][0]-e2[1][0])>1 and np.abs(e1[1][1]-e2[1][1]) > 1:
                        #print()
                        conf = True
                        #precerimo da ne dodajamo vec kot en isti par elips
                        for d in candidates:
                            dis = np.sqrt(((avg_cent[0] - d[2][0]) ** 2 + (avg_cent[1] - d[2][1]) ** 2))
                            if dis < 5:
                                conf = False
                        if conf:
                            candidates.append((e1,e2,avg_cent))
            ##print(len(candidates))
            #rabimo trimat tiste ki so si izredno blizu
            
            for c in candidates:
                #print("candidates found",)
                # the centers of the ellipses
                e1 = c[0]
                e2 = c[1]

                h1,h2,w1,w2 = self.calc_pnts(e1)
                h11,h21,w11,w21 = self.calc_pnts(e2)
                h11= np.round((h11+h1)/2).astype(int)
                h21= np.round((h21+h2)/2).astype(int)
                w11= np.round((w11+w1)/2).astype(int)
                w21= np.round((w21+w2)/2).astype(int)
                # drawing the ellipses on the image
                cv2.ellipse(grayBGR_toDrawOn, e1, (0, 0, 255), 2)
                cv2.ellipse(grayBGR_toDrawOn, e2, (0, 0, 255), 2)

                
                cv2.circle(grayBGR_toDrawOn,tuple( c[2]),1,(0,255,0),2)
                cv2.circle(grayBGR_toDrawOn,tuple( h11),1,(0,255,0),2)
                cv2.circle(grayBGR_toDrawOn,tuple( h21),1,(0,255,0),2)
                cv2.circle(grayBGR_toDrawOn,tuple( w11),1,(0,255,0),2)
                cv2.circle(grayBGR_toDrawOn,tuple( w21),1,(0,255,0),2)
                
                cntr_ring = self.chk_ring(depth_im,h11,h21,w11,w21,c[2])
                try:
                    ring_point = self.get_pose(cntr_ring[1],cntr_ring[0],cntr_ring[2],depth_im,"ring")
                    
                except Exception as e:
                    print(e)
            return grayBGR_toDrawOn

    def calc_rgb_distance_mask(self, image, rgb_value, minDistance):
        rangeImage = image - np.array(rgb_value)
        temp = np.reshape(rangeImage, (image.shape[0]*image.shape[1],3))
        temp = np.linalg.norm(temp,axis=1)
        temp = np.reshape(temp, (image.shape[0],image.shape[1]))

        temp = (temp / np.sqrt(255**2 + 255**2 + 255**2))*255
        temp = temp.astype(np.uint8)

        distance = temp.copy()

        toMask = temp < minDistance
        temp = (toMask * 255).astype(np.uint8)

        return temp, distance

    def calc_mask(self, mask, kernel, kernel1, operations):
        oper1 = operations[0]
        oper2 = operations[1]
        oper3 = operations[2]
        oper4 = operations[3]
        oper5 = operations[4]

        if oper1:
            mask = cv2.dilate(mask, kernel)
        if oper2:
            mask = cv2.erode(mask, kernel)
        if oper3:
            mask = cv2.erode(mask, kernel1)
        if oper4:
            mask = cv2.dilate(mask, kernel1)
        if oper5:
            mask = cv2.dilate(mask, kernel)
        return mask

    def find_color(self,image, depth_image, grayBGR_toDrawOn):
        '''
        hsvIm = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red_low = np.array([136, 87, 11], np.uint8)
        red_up = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvIm, red_low, red_up)

        red_low1 = np.array([0, 87, 11], np.uint8)
        red_up1 = np.array([18, 255, 255], np.uint8)
        red_mask1 = cv2.inRange(hsvIm, red_low1, red_up1)

        green_low = np.array([25,52,72], np.uint8)
        green_up = np.array([102, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsvIm, green_low, green_up)

        blue_low = np.array([94,80,2], np.uint8)
        blue_up = np.array([120, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvIm, blue_low, blue_up)


        black_low = np.array([0,0,0], np.uint8)
        black_up = np.array([180, 255, 60], np.uint8)
        black_mask = cv2.inRange(hsvIm, black_low, black_up)
        '''
        # red
        red_mask, red_distance = self.calc_rgb_distance_mask(image, [0,0,255], 100)

        # green
        green_mask, green_distance = self.calc_rgb_distance_mask(image, [0,255,0], 100)

        # blue
        blue_mask, blue_distance = self.calc_rgb_distance_mask(image, [255,0,0], 100)

        # yellow
        yellow_mask, yellow_distance = self.calc_rgb_distance_mask(image, [0,255,255], 100)

        # cyan
        cyan_mask, cyan_distance = self.calc_rgb_distance_mask(image, [255,255,0], 100)

        # black
        black_mask, black_distance = self.calc_rgb_distance_mask(image, [0,0,0], 60)

        # white
        white_mask, white_distance = self.calc_rgb_distance_mask(image, [255,255,255], 5)

        kernel = np.ones((30,30), "uint8")
        kernel1 = np.ones((7,7), "uint8")

        oper1 = True
        oper2 = True
        oper3 = False
        oper4 = False
        oper5 = False

        operations = [oper1, oper2, oper3, oper4, oper5]

        # red
        red_mask = self.calc_mask(mask=red_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_red = cv2.bitwise_and(image, image, mask = red_mask)

        # green
        green_mask = self.calc_mask(mask=green_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_green = cv2.bitwise_and(image, image, mask = green_mask)

        # blue
        blue_mask = self.calc_mask(mask=blue_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_blue = cv2.bitwise_and(image, image, mask = blue_mask)

        # yellow
        yellow_mask = self.calc_mask(mask=yellow_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_yellow = cv2.bitwise_and(image, image, mask = yellow_mask)

        # cyan
        cyan_mask = self.calc_mask(mask=cyan_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_cyan = cv2.bitwise_and(image, image, mask = cyan_mask)

        # black
        black_mask = self.calc_mask(mask=black_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_black = cv2.bitwise_and(image, image, mask = black_mask)

        # white
        white_mask = self.calc_mask(mask=white_mask,kernel=kernel,kernel1=kernel1,operations=operations)
        res_white = cv2.bitwise_and(image, image, mask = white_mask)



        final_mask = cv2.bitwise_or(red_mask,red_mask)
        final_mask = cv2.bitwise_or(final_mask,green_mask)
        final_mask = cv2.bitwise_or(final_mask,blue_mask)
        final_mask = cv2.bitwise_or(final_mask,yellow_mask)
        final_mask = cv2.bitwise_or(final_mask,cyan_mask)
        final_mask = cv2.bitwise_or(final_mask,black_mask)
        final_mask = cv2.bitwise_or(final_mask,white_mask)
        # final_mask = cv2.bitwise_and(final_mask,bitMaskForAnd)

        # final_mask = black_mask

        # return green_mask
        # return final_mask
        # return cv2.Canny(final_mask,100,200)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageGray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        masksList = [red_mask, green_mask, blue_mask, yellow_mask, cyan_mask, black_mask, white_mask]
        colorsForBorder = [ (0,0,255),
                            (0,255,0),
                            (255,0,0), 
                            (0,255,255), 
                            (255,255,0), 
                            (0,0,0),
                            (255,255,255)
                            ]
        colorDistanceList = [red_distance, green_distance, blue_distance, yellow_distance, cyan_distance, black_distance, white_distance]
        i = -1
        colorsDict = {  0:"red",
                        1:"green",
                        2:"blue",
                        3:"yellow",
                        4:"cyan",
                        5:"black",
                        6:"white"}
        for mask, borderColor, colorDistance in zip(masksList,colorsForBorder,colorDistanceList):
            i += 1
            contour, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for pic, cont in enumerate(contour):
                area = cv2.contourArea(cont)
                if(area > 300):
                    x, y, w, h = cv2.boundingRect(cont)
                    depth_cut = depth_image[y:(y+h),x:(x+w)]
                    image_cut = image[y:(y+h),x:(x+w)]
                    dist_cut = colorDistance[y:(y+h),x:(x+w)]
                    mask_cut = mask[y:(y+h),x:(x+w)]
                    # print(colorsDict[i])
                    # print(mask_cut.shape)
                    mask_row_sum = np.sum(mask_cut>0,1)
                    # print("\t",mask_row_sum.shape)
                    bestRowIndx = np.argmax(mask_row_sum)
                    mask_row = mask_cut[bestRowIndx,:]>0
                    depth_row = depth_cut[bestRowIndx,:]

                    tempColumnRow = depth_row*mask_row
                    tempColumnRow[(tempColumnRow==0)] = np.Inf
                    tempColumnRow[np.isnan(tempColumnRow)] = np.Inf
                    bestColumnIndx = np.argmin(tempColumnRow)

                    numOfCorrectColors = np.sum(mask_cut==255)
                    numOfAllColors = h*w
                    # print(f"{colorsDict[i]}\n\t{numOfCorrectColors}/{numOfAllColors} [{numOfCorrectColors/numOfAllColors}]")

                    if numOfCorrectColors/numOfAllColors < 0.5:
                        continue
                    
                    counter = 0
                    while True:
                        if (bestColumnIndx-counter) < 0:
                            counter -= 1
                            break
                        if (bestColumnIndx+counter) == len(mask_row):
                            counter -= 1
                            break 
                        if mask_row[bestColumnIndx-counter]==False or mask_row[bestColumnIndx+counter]==False or np.isinf(tempColumnRow[bestColumnIndx-counter])==True or np.isinf(tempColumnRow[bestColumnIndx+counter])==True:
                            counter -= 1
                            break
                        counter += 1
                    # print(f"\tcolumn indx: {bestColumnIndx}")
                    # print(tempColumnRow)
                    # print(mask_row)
                    # print(f"\tcounter: {counter}")

                    shift = np.floor((counter*2)/5).astype(int)
                    if shift < 1:
                        continue
                    
                    LL = depth_row[bestColumnIndx-shift*2]
                    L = depth_row[bestColumnIndx-shift]
                    C = depth_row[bestColumnIndx]
                    R = depth_row[bestColumnIndx+shift]
                    RR = depth_row[bestColumnIndx+shift*2]

                    LL_C = LL-C
                    L_C = L-C
                    C_R = R-C
                    C_RR = RR-C

                    if LL_C<=0 or L_C<=0 or C_R<=0 or C_RR<=0:
                        continue

                    if LL_C<L_C*3 or C_RR<C_R*3:
                        continue
                    
                    center_row_indx = np.floor(h/2).astype(int)
                    up_shif = np.floor(h/5).astype(int)
                    if up_shif<1:
                        continue
                    
                    rowIndx1 = bestRowIndx-up_shif
                    rowIndx2 = bestRowIndx
                    rowIndx3 = bestRowIndx+up_shif
                    if rowIndx1 < 0:
                        a = depth_cut[bestRowIndx+up_shif*2][bestColumnIndx]
                    else:
                        a = depth_cut[bestRowIndx-up_shif][bestColumnIndx]
                    
                    b = depth_cut[bestRowIndx][bestColumnIndx]
    
                    if rowIndx3 >= h :
                        c = depth_cut[bestRowIndx-up_shif*2][bestColumnIndx]
                    else:
                        c = depth_cut[bestRowIndx+up_shif][bestColumnIndx]
                    # print(colorsDict[i])
                    # print(a)
                    # print(f"\t{np.abs(a-b)}")
                    # print(b)
                    # print(f"\t{np.abs(b-c)}")
                    # print(c)
                    # print(f"\t{np.abs(a-c)}")
                    # print(a)
                    # print()
                    # if c > 2:
                    #     continue

                    distThreshold = 0.005
                    if np.abs(a-b)>distThreshold or np.abs(a-c)>distThreshold or np.abs(b-c)>distThreshold:
                        continue
                    
                    print(f"\tcylinder detected --> {colorsDict[i]}")
                    try:
                        self.get_pose((x+w/2),(y+h/2),C,depth_image,"cylinder")
                    except Exception as e:
                        print(e)
                    grayBGR_toDrawOn = cv2.rectangle(grayBGR_toDrawOn, (x, y),(x+w,y+h), borderColor ,2)
        
        """
        contour, hierarchy = cv2.findContours(red_mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, cont in enumerate(contour):
            area = cv2.contourArea(cont)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(cont)
                image = cv2.rectangle(image, (x, y),(x+w,y+h), (0,255,0),2)
        """
        return grayBGR_toDrawOn
        # return cv2.cvtColor(final_mask,cv2.COLOR_GRAY2BGR)

    def find_objects(self):
        #print('I got a new image!')

        # Get the next rgb and depth images that are posted from the camera
        try:
            rgb_image_message = rospy.wait_for_message("/camera/rgb/image_raw", Image)
        except Exception as e:
            print(e)
            return 0
        
        try:
            depth_image_message = rospy.wait_for_message("/camera/depth/image_raw", Image)
        except Exception as e:
            print(e)
            return 0
        


        # Convert the images into a OpenCV (numpy) format
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_message, "bgr8")
        except CvBridgeError as e:
            print(e)

        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_message, "32FC1")
        except CvBridgeError as e:
            print(e)

        grayImage = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2GRAY)
        grayImage = cv2.cvtColor(grayImage,cv2.COLOR_GRAY2BGR)
        markedImage = self.find_color(rgb_image, depth_image, grayImage)
        markedImage = self.find_elipses_first(rgb_image, depth_image,depth_image_message.header.stamp, grayImage)

        self.pic_pub.publish(CvBridge().cv2_to_imgmsg(markedImage, encoding="passthrough"))
        #self.markers_pub.publish(self.m_arr)

    def find_elipse(self,contours):
        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            #     print cnt
            #     print cnt.shape
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)
                elps.append(ellipse)


        # Find two elipses with same centers
        # candidates = []
        # for n in range(len(elps)):
        #     for m in range(n + 1, len(elps)):
        #         e1 = elps[n]
        #         e2 = elps[m]
        #         dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
        #         #             print dist
        #         if dist < 5:
        #             candidates.append((e1,e2))

        return elps

def main():

        color_finder = color_localizer()

        rate = rospy.Rate(1.25)
        while not rospy.is_shutdown():
            # print("hello!")
            color_finder.find_objects()
            #print("hello")
            #print("\n-----------\n")
            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
