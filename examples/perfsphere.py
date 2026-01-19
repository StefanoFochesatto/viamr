import numpy as np
import matplotlib.pyplot as plt
import csv

path = "./"

def fname(method):
    return path + "sphere_" + method + ".csv"

def floatvals(method, field):
    with open(fname(method), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return np.array([float(row[field]) for row in reader], dtype=np.float64)

def intvals(method, field):
    with open(fname(method), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return np.array([int(row[field]) for row in reader], dtype=np.int32)

ms0 = 10.0
fs0 = 14.0

methods = ["uni", "udobr", "avm", "nsv"]
markers = ["k+", "ko", "k*", "ko"]
fcolors = ["k", "k", "k", "w"]

measure = ["ENORM", "JACCARD", "HAUSDORFF"]  # also "ENORMPREF"?
ylabels = [r"$||u-u_h||_2$", "active set Jaccard distance", "free boundary Hausdorff distance"]

for k in range(3):
    plt.figure()
    for j in range(len(methods)):
        meth = methods[j]
        if measure[k] == "JACCARD":
            vals = 1.0 - floatvals(meth, measure[k])
        else:
            vals = floatvals(meth, measure[k])
        plt.loglog(intvals(meth, "NE"), vals, markers[j], ms=ms0, mfc=fcolors[j], label=meth)
    plt.grid(True)
    plt.xlabel("elements", fontsize=fs0+2.0)
    plt.ylabel(ylabels[k], fontsize=fs0+2.0)
    plt.legend(fontsize=fs0, loc='lower left')
    outname = f"perf_{measure[k]}.png"
    print(f"writing {outname} ...")
    plt.savefig(outname, bbox_inches="tight")
