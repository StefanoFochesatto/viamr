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

plt.figure()
meth = "udobr"
plt.loglog(intvals(meth, "NE"), floatvals(meth, "ENORM"), 'ko', ms=ms0, label=meth)
#plt.loglog(intvals(meth, "NE"), floatvals(meth, "ENORMPREF"), 'ks', ms=ms0-1, label=r"$||u-\tilde u_h||_2$")
meth = "nsv"
plt.loglog(intvals(meth, "NE"), floatvals(meth, "ENORM"), 'ks', ms=ms0+2, mfc='w', label=meth)
uni = "uni"
plt.loglog(intvals(meth, "NE"), floatvals(meth, "ENORM"), 'k+', ms=ms0+2, mfc='w', label=meth)
plt.grid(True)
plt.xlabel("elements", fontsize=fs0+2.0)
plt.ylabel(r"$||u-u_h||_2$", fontsize=fs0+2.0)
plt.legend(fontsize=fs0, loc='lower left')
outname = "perf_enorm.png"
print(f"writing {outname} ...")
plt.savefig(outname, bbox_inches="tight")

plt.figure()
meth = "udobr"
plt.loglog(intvals(meth, "NE"), 1.0 - floatvals(meth, "JACCARD"), 'ko', ms=ms0, label=meth)
meth = "nsv"
plt.loglog(intvals(meth, "NE"), 1.0 - floatvals(meth, "JACCARD"), 'ks', ms=ms0+2, mfc='w', label=meth)
uni = "uni"
plt.loglog(intvals(meth, "NE"), 1.0 - floatvals(meth, "JACCARD"), 'k+', ms=ms0+2, mfc='w', label=meth)
plt.grid(True)
plt.ylabel("active set Jaccard distance", fontsize=fs0)
plt.xlabel("elements", fontsize=fs0+2.0)
plt.legend()
outname = "perf_jacc.png"
print(f"writing {outname} ...")
plt.savefig(outname, bbox_inches="tight")
