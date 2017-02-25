
f = open('last-layer-car-sample-baseline/darknet.txt')

base = 4
interval = 85
counter = 0
for l in f:
    if counter == base:
        prob = float(l.split()[1])
        if prob > 0.5:
            print counter, prob
        base += interval
    counter += 1
