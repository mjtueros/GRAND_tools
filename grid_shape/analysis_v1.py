import matplotlib.pyplot as plt
import numpy as np
import os


class Event:
    def __init__(self,f1,f2, step, name):
        p2px, p2py, p2pz, p2ptot = np.loadtxt(f1)
        self.p2px = p2px
        self.p2py = p2py
        self.p2pz = p2pz
        self.p2ptot = p2ptot
        # JobName,Primary,Energy,Zenith,Azimuth,XmaxDistance,SlantXmax,RandomCore[0],RandomCore[1],RandomAzimuth,HadronicModel
        A = open(f2).readlines()[0]
        A = A.strip().split()
        self.jobname = A[0]
        self.primary = A[1]
        self.energy = np.float32(A[2])
        self.zenith = np.float32(A[3])
        self.azimuth = np.float32(A[4])
        self.xmax_distance = np.float32(A[5])
        self.slant_xmax = np.float32(A[6])
        self.random_core0 = np.float32(A[7])
        self.random_core1 = np.float32(A[8])
        self.random_azimuth = np.float32(A[9])
        self.hadronic_model = A[10]
        self.step = np.float32(step)
        self.name = name
        self.init_layout()

    def init_layout(self):
        if "hexhex" in self.name:
            self.layout = "hexhex"
        elif "rect" in self.name:
            self.layout = "rect"
        else:
            self.layout = "unknown"

    def is_triggered(self, threshold):
        return self.p2ptot > threshold

path = "/Users/benoitl/Documents/GRAND/InterpolationOutputExample/"

ev_list = []

for subdir in os.listdir(path):
    if os.path.isdir(os.path.join(path, subdir)):
        list_fn = os.listdir(os.path.join(path, subdir))        
        for fn in list_fn:
            if fn[-6:] == "P2Pdat":
                f1 = os.path.join(path, subdir, fn)
                f2 = os.path.join(path, subdir, fn+'.showerinfo')
                step  = fn.split("_")[-2]

                ev_list.append(Event(f1, f2, step, fn))

thresold = 50
# is_triggered_list = [sum(ev.is_triggered(75)) for ev in ev_list if "voltage" in ev.name]

for ev in ev_list:
    if "voltage" in ev.name:
        ev.num_triggered = sum(ev.is_triggered(thresold))

# A = [(ev.num_triggered, ev.energy, ev.step, ev.primary, ev.layout, ev.zenith) for  ev in ev_list if "voltage" in ev.name]

A = [(ev.num_triggered, ev.energy, ev.step, ev.layout, ev.zenith) for  ev in ev_list if "voltage" in ev.name and ev.primary == "Proton"]

A_rect = [
    (ev.num_triggered, ev.energy, ev.step, ev.zenith) for  ev in ev_list
    if "voltage" in ev.name
    and ev.primary == "Proton"
    and ev.layout == 'rect'
]
A_hexhex = [
    (ev.num_triggered, ev.energy, ev.step, ev.zenith) for  ev in ev_list
    if "voltage" in ev.name
    and ev.primary == "Proton"
    and ev.layout == 'hexhex'
]

A_rect = np.array(A_rect)
A_hexhex = np.array(A_hexhex)

i_rect_low = np.where(A_rect[:,3] <= 9.500000e+1)[0]
i_rect_high = np.where(A_rect[:,3] > 9.500000e+1)[0]
i_hexhex_low = np.where(A_hexhex[:,3] <= 9.500000e+1)[0]
i_hexhex_high = np.where(A_hexhex[:,3] > 9.500000e+1)[0]

plt.figure(1)
plt.clf()
plt.plot(A_rect[i_rect_low,2], A_rect[i_rect_low,0], 'ro', label="rect - low zenith")
plt.plot(A_rect[i_rect_high,2], A_rect[i_rect_high,0], 'rv', label="rect - high zenith")

plt.plot(A_hexhex[i_hexhex_low,2], A_hexhex[i_hexhex_low,0], 'bo', label="hexhex - low zenith")
plt.plot(A_hexhex[i_hexhex_high,2], A_hexhex[i_hexhex_high,0], 'bv', label="hexhex - high zenith")


plt.xlabel("step [m]")
plt.ylabel("N triggred")
plt.legend(loc=0)
plt.show()

# ev1 = Event(f1,f2)

