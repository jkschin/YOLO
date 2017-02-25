import cv2

img = cv2.imread('/home/jkschin/code/Tiny-YOLO/sample.jpg')
f = open('bboxes/bboxes.txt')

counter = 0
for line in f:
    coords = [float(i) for i in line.split()]
    scaled_coords = [coords[0]*1080, coords[1]*1920, coords[2]*1080, coords[3]*1920]
    # img = cv2.rectangle(img, scaled_coords[0:2], scaled_coords[2:], (0, 0, 255), 2)
    print scaled_coords
    counter += 1
    if counter == 3:
        break
