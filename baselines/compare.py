import os

directory = 'last-layer-car-sample-baseline'
darknet = open(os.path.join(directory, 'darknet.txt'))
tf = open(os.path.join(directory, 'tf-maxpool-ul-pad.txt'))

counter = 0
total = 0
epsilon = 0.0001
while True:
    d_val = darknet.readline()
    t_val = tf.readline()
    if d_val == '' and t_val == '':
        break
    d_val = float(d_val.split()[1])
    t_val = float(t_val.split()[1])
    diff = abs(d_val - t_val)
    total += diff
    if diff >= epsilon:
        print counter, diff
    counter += 1
print "Average error: ", total/counter

