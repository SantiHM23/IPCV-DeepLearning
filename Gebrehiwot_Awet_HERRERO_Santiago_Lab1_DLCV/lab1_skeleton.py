#https://docs.opencv.org/3.4.1/d6/d00/tutorial_py_root.html

from __future__ import print_function

import argparse
import numpy as np
import cv2

import matplotlib.pyplot as plt


def computeMSE(prev, curr):
    mse = 0
    #TODO
    prev = prev.astype(np.int32)
    curr = curr.astype(np.int32)
    N, M = prev.shape[:2]
    inter = curr - prev
    inter = np.power(inter,2)
    mse = (1.0/(N*M))* np.sum(inter)
    return mse

def computePSNR(mse):
    psnr = 0
    #TODO
    div = (255^2)/(mse)
    psnr = 10* np.log10(div)
    #print(psnr)
    return psnr

def computeEntropy(img):
    ent = 0
    #TODO
    N, M = img.shape[:2]
    #conter = 0
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #hist,bins = np.histogram(img,256,[0,256])
    probs = hist/(N*M)
    ent = - np.sum(probs*np.log2(probs+0.0001))
    #print (ent)
    return ent

def computeErrorImage(im1, im2):
    res = im1
    #TODO
    #im1 = im1.astype(np.int32)
    #im2 = im2.astype(np.int32)
    diff = im2 - im1
    white = np.ones_like(im1)*255
    z = np.zeros_like(im1)
    max_im = np.maximum(z, diff+128)
    res = np.minimum(white, max_im)
    return res

def computeOpticalFlow1(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(curr, prev, flow=None, pyr_scale=0.5, levels=3, winsize=20, iterations=15, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow

def computeCompensatedFrame(prev, flow):
    h, w = flow.shape[:2]
    map = flow.copy()
    map[:,:,0] += np.arange(w)
    map[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(prev, map, None, cv2.INTER_LINEAR)
    return res


def computeGME(flow):
    src = np.zeros_like(flow)
    h, w = flow.shape[:2]
    c = np.array([w/2, h/2])
    src[:,:,0] += np.arange(w)
    src[:,:,1] += np.arange(h)[:,np.newaxis]
    src -= c;
    srcPts = src.reshape((h*w, 2))

    #TODO
    # You should use
    # - cv2.findHomography
    #   see https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
    # - cv2.perspectiveTransform
    #   see https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7
    gme = flow  #change this...
    dst = src + flow
    dstPts = dst.reshape((h*w, 2))
    H, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC)
    gme = cv2.perspectiveTransform(src, H) - src
    return gme

def computeGMEError(flow, gme):
    #TODO
    # since the flow has a size of h,w,2 b/c of reshape. for ploting we can only access the image matrix as folows
    flow = flow[:,:,0]
    gme = gme[:,:,0]
    err = np.abs(flow - gme)
    return err

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Read video file')
    parser.add_argument('video', help='input video filename')
    parser.add_argument('deltaT', help='input deltaT between frames', type=int)

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    if (cap.isOpened() == False):
        print("ERROR: unable to open video: "+args.video)
        quit()

    deltaT=args.deltaT

    previousFrames=[]
    frameNumbers = []
    mses = []
    psnrs = []
    mse0s = []
    psnr0s = []
    ents = []
    entEs = []

    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if (i==60):

            cv2.imwrite('frame.png', gray)
            cv2.imwrite('flow.png', draw_flow(gray, flow))
            cv2.imwrite('gme.png', draw_flow(gray, gme))
            cv2.imwrite('gme_error.png', gmeError)
            cv2.imwrite('imErr0.png', imErr0)
            cv2.imwrite('imErr.png', imErr)
            #break
           
        if (ret==False):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (len(previousFrames) >= deltaT):
            prev = previousFrames.pop(0)

            flow = computeOpticalFlow1(prev, gray)

            compensatedFrame = computeCompensatedFrame(prev, flow)

            #cv2.imshow('compensated', compensatedFrame)

            imErr0 = computeErrorImage(prev, gray)
            imErr = computeErrorImage(compensatedFrame, gray)

            cv2.imshow('imErr0', imErr0)
            cv2.imshow('imErr', imErr)


            mse0 = computeMSE(prev, gray)
            psnr0 = computePSNR(mse0)
            mse = computeMSE(compensatedFrame, gray)
            psnr = computePSNR(mse)
            ent = computeEntropy(gray)
            entE = computeEntropy(imErr)

            frameNumbers.append(i)
            mses.append(mse)
            psnrs.append(psnr)
            mse0s.append(mse0)
            psnr0s.append(psnr0)
            ents.append(ent)
            entEs.append(entE)


            gme = computeGME(flow)

            gmeError = computeGMEError(flow, gme)
            
            gmeError = gmeError / gmeError.max() #normalizes data in range 0 - 255
            #print(gmeError.max())
            #print(gmeError.min())
            gmeError = 255 * gmeError
            gmeError = gmeError.astype(np.uint8)
            
            cv2.imshow('flow', draw_flow(gray, flow))
            cv2.imshow('gme', draw_flow(gray, gme))
            cv2.imshow('gmeError', gmeError)
            
            #print(gray.dtype)
            #print(gmeError.dtype)

        previousFrames.append(gray.copy())
        i+=1

        cv2.imshow('frame', gray)

        cv2.waitKey(1)

    plt.figure()
    plt.plot(frameNumbers, mse0s, label='MSE0')
    plt.plot(frameNumbers, mses, label='MSE')
    plt.xlabel('frames')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE0 vs MSE')
    plt.savefig("mse.png")
    plt.show()
    
    plt.figure()
    plt.plot(frameNumbers, ents, label='Entropy')
    plt.plot(frameNumbers, entEs, label='EntropyE')
    plt.xlabel('frames')
    plt.ylabel('Entropy')
    plt.legend()
    plt.title('Entropy vs EntropyE')
    plt.savefig("entropy.png")
    plt.show()
    
    plt.figure()
    plt.plot(frameNumbers, psnr0s, label='PSNR0')
    plt.plot(frameNumbers, psnrs, label='PSNR')
    plt.xlabel('frames')
    plt.ylabel('PSNR')
    plt.legend()
    plt.title('PSNR0 vs PSNR')
    plt.savefig("psnr.png")
    plt.show()


    cap.release()
    cv2.destroyAllWindows()
    
