import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import statistics
import math
import sys

global debug
debug = False

def edit_contrast_greyscale(img, k=0.04):
    image = np.copy(img) 
    for x in range(len(image)):
        for y in range(len(image[0])):
                image[x][y] = 255 / (1 + np.e ** (-k * (image[x][y]-122)))
    return image

def detect_peaks(imgpath):
    src = cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
    rows, cols = src.shape

    best_angle = 0
    best_verticals = [] # from x axis (columns)
    best_horizontals = [] # from y axis (rows)
    best_dx = []
    best_dy = []
    img = edit_contrast_greyscale(src, 0.04)
    

    for rot_angle in [-1, -0.5, 0, 0.5, 1]: 
        rot_matrix = cv.getRotationMatrix2D((cols/2, rows/2), rot_angle, 1) # rotate around the center, scale by 1
        img_rot = cv.warpAffine(img, rot_matrix, (cols,rows))

        mask = (img_rot < 160).astype(np.uint8) # Ideal threshold: < 160

        kernel = np.ones((5, 5), np.uint8)
        mask = cv.dilate(mask, kernel, iterations=2)
        hist_x = mask.sum(axis=0)
        hist_y = mask.sum(axis=1)

        title = f"Rotation {rot_angle}"
        fig = plt.figure(title)
        ((ax1, ax2), (d_ax1, d_ax2)) = fig.subplots(2, 2)
        ax1.bar(np.arange(mask.shape[1]), hist_x, width=1.0)
        x_peaks, _ = find_peaks(hist_x, width=(None, 100), prominence=500, height=4000, distance=1000) 
        ax1.plot(x_peaks, hist_x[x_peaks], "x")

        ax2.bar(np.arange(mask.shape[0]), hist_y, width=1.0)
        y_peaks, _ = find_peaks(hist_y, width=(None, 100), prominence=500, height=4000, distance=1000)
        ax2.plot(y_peaks, hist_y[y_peaks], "x")

        dx = 0.0001
        derivative_x = np.gradient(hist_x, dx)
        derivative_y = np.gradient(hist_y, dx)

        d_ax1.bar(np.arange(mask.shape[1]), derivative_x, width=1.0)
        dx_peaks, _ = find_peaks(derivative_x, height=0.5, distance=1000)
        d_ax1.plot(dx_peaks, derivative_x[dx_peaks], "x")

        d_ax2.bar(np.arange(mask.shape[0]), derivative_y, width=1.0)
        dy_peaks, _ = find_peaks(derivative_y, height=0.5, distance=1000)
        d_ax2.plot(dy_peaks, derivative_y[dy_peaks], "x")

        if debug:
            print(f"Rot {rot_angle}")
            cv.imshow("Rotated", img_rot)
            cv.waitKey(0)
            cv.destroyAllWindows()
            plt.imshow(mask, cmap="gray")
            plt.show()

        x_peaks = x_peaks.tolist()
        dx_peaks = dx_peaks.tolist()

        # Ensure x_peaks contains only sharp lines (from grid) by comparing with dx_peaks
        p = 0
        while p < len(x_peaks):
            steep = False
            for dpeak in dx_peaks:
                if abs(dpeak-x_peaks[p]) < 200: # if more than 200 pixels from its derivative peak (i.e. not steep enough for dp)
                    steep = True
            if not steep:
                x_peaks.pop(p)
                p -= 1
            p += 1
        p = 0

        y_peaks = y_peaks.tolist()
        dy_peaks = dy_peaks.tolist()

        # Ensure y_peaks contains only sharp lines (from grid) by comparing with dy_peaks
        while p < len(y_peaks):
            steep = False
            for dpeak in dy_peaks:
                if abs(dpeak-y_peaks[p]) < 200: 
                    steep = True
            if not steep:
                y_peaks.pop(p)
                p -= 1
            p += 1

        if len(x_peaks) + len(y_peaks) > len(best_verticals) + len(best_horizontals):
            best_angle = rot_angle
            best_verticals = x_peaks
            best_horizontals = y_peaks
            best_dx = dx_peaks
            best_dy = dy_peaks


    print("Identified grid lines.")
    if debug:
        print(f"best angle: {best_angle}")
        print(f"best verticals: {best_verticals}")
        print(f"best horizontals: {best_horizontals}")
        
        rot_matrix = cv.getRotationMatrix2D((cols/2, rows/2), best_angle, 1)
        img_debug = cv.warpAffine(img, rot_matrix, (cols,rows))

        cv.imshow("Debug", img_debug)
        cv.waitKey(0)
        cv.destroyAllWindows()
        plt.imshow(mask, cmap="gray")
        plt.show()



    print("VERTICALS") # x
    best_verticals = clean_distances(best_verticals, cols, best_dx) 

    print("HORIZONTALS") # y
    best_horizontals = clean_distances(best_horizontals, int(rows//1.1), best_dy)
    # Max for rows shouldn't be the very end, because that's well below the end of the grid

    if debug: 
        print(f"corrected verticals: {best_verticals}")
        print(f"corected horizontals: {best_horizontals}")
    
        # DISPLAY EDGE DETECTION
        rot_matrix = cv.getRotationMatrix2D((cols/2, rows/2), best_angle, 1) # rotate around the center, scale by 1
        img_out = cv.warpAffine(img, rot_matrix, (cols,rows))
        for y in best_horizontals:
            cv.line(img_out,(0, y),(cols, y),(0,0,0),50)
        for x in best_verticals:
            cv.line(img_out,(x, 0),(x, rows),(0,0,0),50)
        cv.imwrite("edgedetected.png", img_out)
        cv.imshow('Image with marked edges',img_out)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    return best_horizontals, best_verticals, rot_angle

def clean_distances(peaks, maximum, dpeaks):
    '''
    Cleans the edge detection to reflect the way the cap grids are expected to look
    '''
    # FIND AVG DISTANCES (in proper range)
    dists = []
    for p in range(len(dpeaks)-1):
        d = abs(dpeaks[p]-dpeaks[p+1])
        if d > 1000 and d < 1450:
            dists.append(d)
    if dists:
        avgdist = math.floor(statistics.mean(dists))
    else:
        avgdist = 1250 # default average distance between grid lines, used for generating missing ones

    while len(peaks) != 6:
        print(peaks)
        if len(peaks) > 6:
            if peaks[1] < 800: #2 peaks in the starting area 
                peaks.pop(0)
            elif peaks[len(peaks)-2] < maximum-800:
                peaks.pop(len(peaks)-1)
            else:
                peaks.pop(len(peaks)-1)
        elif len(peaks) < 6:
            inter = False
            i = 0        
            while i < len(peaks)-1: 
                if abs(peaks[i]-peaks[i+1]) > avgdist+250: # interpolate
                    # Option 1: insert a peak from dpeaks if one appears between peaks i and i+1
                    in_dpeaks = False
                    for dp in dpeaks:
                        if dp > peaks[i] and dp < peaks[i+1] and (abs(peaks[i]-dp) > 200) and (abs(peaks[i+1]-dp) > 200):
                            peaks.insert(i+1, dp)
                            in_dpeaks = True
                            continue
                    # Option 2: extrapolate a peak between the lines using the average distance between dpeaks
                    if not in_dpeaks:
                        peaks.insert(i+1, peaks[i]+avgdist)
                        inter = True
                        continue
                i+= 1
            if not inter: # Insert on the ends
                if peaks[0] > 800: # insert start line
                    peaks.insert(0, max(0, peaks[0]-avgdist))
                else:
                    peaks.insert(len(peaks), min(maximum, peaks[len(peaks)-1]+avgdist))
    return peaks

def hough_circles(imgpath):
    src = cv.imread(imgpath, cv.IMREAD_COLOR)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=300, maxRadius=425)
    circles = np.uint16(np.around(circles))
    if debug:
        print(circles)
        if circles is not None:
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(src, center, 1, (0, 100, 100), 30)
                # circle outline
                radius = i[2]
                cv.circle(src, center, radius, (255, 0, 255), 5)
        cv.imshow("detected circles", src)
        cv.waitKey(0)
    
    return circles

def split_image(path, imgpath, imgid, scalefactor, rows, columns, rotation, circles):
    img = cv.imread(imgpath, 1)
    height0, width0, _ = img.shape
    rot_matrix = cv.getRotationMatrix2D((width0/2, height0/2), rotation, 1) 
    img = cv.warpAffine(img, rot_matrix, (width0,height0))
    height1, width1, _ = img.shape
    if "1-25" in imgid:
        index = 1 
        imgname = imgid[0:imgid.find("1-25")]
    elif "26-50" in imgid:
        index = 26
        imgname = imgid[0:imgid.find("26-50")]
    else:
        index = 1
        imgname = imgid[0:imgid.find(".")]+" "
    print(f"SPLITTING {imgid}")
    print(columns)
    print(rows)
    for c in range(len(columns)-1):
        for r in range(len(rows)-1):
            y1 = rows[r]
            y2 = rows[r+1]
            x1 = columns[c]
            x2 = columns[c+1]

            nocap = True
            for circle in circles[0, :]:
                x = circle[0]
                y = circle[1]
                if x >= x1 and x <= x2 and y >= y1 and y <= y2: # Circle within the grid space
                    radius = circle[2] + 75 # add a small buffer to make sure the cap doesn't get cut off

                    ystart = y-radius
                    yend = y+radius
                    xstart = x-radius
                    xend = x+radius
                    if ystart < 0: # Ensures the crop doesn't fall outside the image
                        ystart = 0 
                    if yend >= height1:
                        yend = height1-1
                    if xstart < 0:
                        xstart = 0
                    if xend >= width1:
                        xend = width1-1
                    crop_img = img[ystart:yend, xstart:xend]

                    print(f"rad: {radius}")
                    print(f"y: {y}, {ystart}:{yend}")
                    print(f"x: {x}, {xstart}:{xend}")
                    nocap = False
                    break
            if nocap:
                crop_img = img[y1:y2, x1:x2] # NOTE: This acts as a BACK-UP for TESTING (to ensure proper numbering)
                print(str(index)+ " - NOCAP")
            if not nocap: 
                print(str(index)+ " cap") 
            # Scale image to 800x800
            newsize = 800
            height, width, _ = crop_img.shape
            print(f"height: {height}, width: {width}")
            if height > width: # use width, cut height
                print("by width")
                scale = newsize/width
                scalewidth = newsize 
                scaleheight = math.floor(height*scale)
                croplength = abs(math.floor((scaleheight-scalewidth)/2))
                scaled_img = cv.resize(crop_img, (scalewidth, scaleheight), interpolation=cv.INTER_AREA)
                squared_img = scaled_img[croplength:scalewidth+croplength, 0:scalewidth]
                h, w, _ = squared_img.shape
                if not (h == newsize and w == newsize):
                    squared_img = cv.resize(squared_img, (newsize, newsize), interpolation=cv.INTER_AREA)
                print(f"scale: {scale}, scaleheight: {scaleheight}, croplength: {croplength}")
                print(f"[{croplength}:{scalewidth+croplength}, 0:{scalewidth}]")
            else: # use height, cut width
                print("by height")
                scale = newsize/height
                scalewidth = math.floor(width*scale)
                scaleheight = newsize 
                croplength = abs(math.floor((scalewidth-scaleheight)/2))
                scaled_img = cv.resize(crop_img, (scalewidth, scaleheight), interpolation=cv.INTER_AREA)
                squared_img = scaled_img[0:scaleheight, croplength:scaleheight+croplength]
                h, w, _ = squared_img.shape
                if not (h == newsize and w == newsize):
                    squared_img = cv.resize(squared_img, (newsize, newsize), interpolation=cv.INTER_AREA)
                print(f"scale: {scale}, scalewidth: {scaleheight}, croplength: {croplength}")
                print(f"[0:{scalewidth}, {croplength}:{scalewidth+croplength}]")
            cv.imwrite(r'{}/{}{}.png'.format(path,imgname,index),squared_img)
            index += 1
            print("---")

def main(argv):
    path = "" 
    for i in range(1, len(sys.argv)):
        path += sys.argv[i] + " "
    path = path[0:len(path)-1]
    for x in os.listdir(path):
        imgpath = f'{path}/{x}'
        print(imgpath)
        if  x.endswith(".jpg") or x.endswith(".png"): 
            outpath = r'{}-cap'.format(path)
            if not os.path.exists(outpath):
                os.mkdir(outpath)
            rows, columns, rotation = detect_peaks(imgpath)
            circles = hough_circles(imgpath)
            split_image(outpath, imgpath, x, 100, rows, columns, rotation, circles)
    print("Finished!")

main(0) # to run, type: > python3 main.py ../../Lithium Fecundity/Replicate 1_s/04-29