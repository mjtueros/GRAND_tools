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

#path = "/Users/kotera/BROQUE/Data_GRAND/Matias/InterpolationOutputExample/"
path = "/Users/kotera/BROQUE/Data_GRAND/Matias/StshpLibrary02/"

ev_list = []
count = 0


for subdir in os.listdir(path)[0:25000]:
    if os.path.isdir(os.path.join(path, subdir)):
        list_fn = os.listdir(os.path.join(path, subdir))        
        for fn in list_fn:
            if fn[-6:] == "P2Pdat":
                f1 = os.path.join(path, subdir, fn)
                f2 = os.path.join(path, subdir, fn+'.showerinfo')
                step  = fn.split("_")[-2]

                ev_list.append(Event(f1, f2, step, fn))

    count += 1 
    if(count % 100 == 0):
        print("Event #{} done".format(count))

threshold = 30
# is_triggered_list = [sum(ev.is_triggered(75)) for ev in ev_list if "voltage" in ev.name]

for ev in ev_list:
    if "voltage" in ev.name:
        ev.num_triggered = sum(ev.is_triggered(threshold))

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


enerbins = np.unique(A_rect[:,1])
#zenbins = np.unique(A_rect[:,3])
zenbins = [94,100,105,110,120,131]
stepbins = np.unique(A_rect[:,2])

meanNtrig_ener = []
varNtrig_ener = []

for istep, step in enumerate(stepbins):
    meanNtrig_step = []
    varNtrig_step = []

    for iener, ener in enumerate(enerbins):
        meanNtrig_zen = []
        varNtrig_zen = []
    
        for izen in range(0, len(zenbins)-1):
            ind = np.where((A_rect[:,1] == ener) * (A_rect[:,2] == step) 
                * (A_rect[:,3] >= zenbins[izen]) * (A_rect[:,3] < zenbins[izen+1]))
            meanNtrig_zen.append(np.mean(A_rect[ind[0],0]))
            varNtrig_zen.append(np.var(A_rect[ind[0],0]))

        meanNtrig_step.append(meanNtrig_zen)
        varNtrig_step.append(varNtrig_zen)

    meanNtrig_ener.append(meanNtrig_step)
    varNtrig_ener.append(varNtrig_step)

meanNtrig_ener = np.array(meanNtrig_ener)
varNtrig_ener = np.array(varNtrig_ener)


#plt.hist2d(A_rect[:,1],A_rect[:,0])
#plt.tight_layout()
#plt.show()
#plt.ylabel('N triggered')
#plt.xlabel('energy [EeV]')
# ev1 = Event(f1,f2)


sym_list = ['.','o','v','*','s']


for istep, step in enumerate(stepbins):
    plt.figure(istep) 
    plt.clf()
    for izen in range(0, len(zenbins)-1):
        #plt.plot(enerbins, meanNtrig_ener[istep], sym_list[istep], 
         #    label='step = %d m'%(np.int32(step)))
        plt.errorbar(enerbins, meanNtrig_ener[istep,:,izen], yerr=sqrt(varNtrig_ener[istep,:,izen]), 
            fmt=sym_list[izen], capsize=2, label='%4.0f > zen >%4.0f deg'%(180-zenbins[izen], 180-zenbins[izen+1]))
    plt.yscale('log')
    plt.ylabel('N triggered')
    plt.xlabel('energy [EeV]')
    plt.title('step = %d m'%(np.int32(step)))
    plt.legend(loc=4)
    plt.show()

 #        plt.errorbar(enerbins, meanNtrig_ener[istep,:,izen], yerr=sqrt(varNtrig_ener[istep,:,izen]), 
  #          fmt='.', capsize=2, label='step = %d m'%(np.int32(step)))
 

for izen in range(0, len(zenbins)-1):
    plt.figure(izen+4) 
    plt.clf()
    for istep, step in enumerate(stepbins):
                #plt.plot(enerbins, meanNtrig_ener[istep], sym_list[istep], 
                #    label='step = %d m'%(np.int32(step)))
        plt.errorbar(enerbins, meanNtrig_ener[istep,:,izen], yerr=sqrt(varNtrig_ener[istep,:,izen]), 
            fmt=sym_list[istep], capsize=2, label='step = %d m'%(np.int32(step)))
    plt.yscale('log')
    plt.ylabel('N triggered')
    plt.xlabel('energy [EeV]')
    plt.title('%4.0f > zenith >%4.0f deg'%(180-zenbins[izen], 180-zenbins[izen+1]))
    plt.legend(loc=4)
    plt.show()

 