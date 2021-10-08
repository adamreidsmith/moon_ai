import cv2
import os
import numpy as np
import pickle

dot_batch = lambda v1, v2: np.sum(v1*v2, axis=-1)
norm_sq_batch = lambda v: dot_batch(v,v)
norm_batch = lambda v: np.sqrt(norm_sq_batch(v))

red = np.empty((1170, 750, 3))
red[:] = np.array([6, 0, 251])

blue = np.empty((1170, 750, 3))
blue[:] = np.array([255, 0, 0])

green = np.empty((1170, 750, 3))
green[:] = np.array([8, 255, 34])

color_base = {'r':red, 'g':green, 'b':blue}

def isolate_circ(im, color=None):
    thres = {'r':26, 'g':41, 'b':39}
    if color is not None:
        base = color_base[color]
        diff = np.abs(im - base)
        circs = np.where(norm_batch(diff) < thres[color], 255, 0)
        return circs.astype('uint8')
    
    circ_imgs = []
    for color in ['r', 'g', 'b']:
        base = color_base[color]
        diff = np.abs(im - base)
        circs = np.where(norm_batch(diff) < thres[color], 255, 0)
        circ_imgs.append(circs.astype('uint8'))
    return circ_imgs


def getCircCoords(circ_im, draw=False, n=0):
    contours, _ = cv2.findContours(circ_im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []
    for cont in contours:
        M = cv2.moments(cont)
        center = (M['m10']/M['m00'], M['m01']/M['m00'])
        centers.append(center)
    
    if draw:
        output = cv2.cvtColor(circ_im, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output, contours, -1, (0,255,0), 3)
        cv2.imwrite('output%i.png' % n, output)
    
    return centers


dim = (18,11)

def getHoldHist(coords):
    hold_hist = np.zeros(dim, dtype='int')
    for (x, y) in coords:
        row = int((y-65)/61)
        col = int((x-57)/61)
        hold_hist[row, col] += 1
    return hold_hist
    

master_dict = {}

image_dir = './problems/'
image_names = os.listdir(image_dir)
n_probs = len(image_names)

for i, im_name in enumerate(image_names):
    if '.png' not in im_name:
        continue
    im = cv2.imread(image_dir + im_name)
    
    # red, green, then blue
    circ_ims = isolate_circ(im)
    
    coords = [getCircCoords(circ) for circ in circ_ims]
    
    start_hist = getHoldHist(coords[1])
    inter_hist = getHoldHist(coords[2])
    fin_hist = getHoldHist(coords[0])
    all_hist = start_hist + inter_hist + fin_hist
    
    name_split = im_name.split('~')
    grade = name_split[-1][:-4]
    
    master_dict.update({name_split[0] : (grade, start_hist, inter_hist, fin_hist, all_hist)})

    if i % 100 == 0:
        print('%.2f percent complete. Analyzed %i out of %i problems.' % (i/n_probs*100, i, n_probs))

with open('problems.pkl', 'wb') as f:
    pickle.dump(master_dict, f)
