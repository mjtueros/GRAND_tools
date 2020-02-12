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

    def is_triggered1(self, threshold):
        return self.p2ptot > threshold


# read files

#path = "/Users/kotera/BROQUE/Data_GRAND/Matias/InterpolationOutputExample/"
path = "/Users/kotera/BROQUE/Data_GRAND/Matias/StshpLibrary02/"

ev_list = []
count = 0

for subdir in os.listdir(path)[0:1000]:
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

# select triggered antennas and events
threshold = 75
# is_triggered_list = [sum(ev.is_triggered(75)) for ev in ev_list if "voltage" in ev.name]

for ev in ev_list:
    if "voltage" in ev.name:
        ev.num_triggered = sum(ev.is_triggered1(threshold))
        ev.is_triggered2 = (ev.num_triggered > 5)

# A = [(ev.num_triggered, ev.energy, ev.step, ev.primary, ev.layout, ev.zenith) for  ev in ev_list if "voltage" in ev.name]

# make array with each event information 
A = [(ev.num_triggered, ev.energy, ev.step, ev.layout, ev.zenith) for  ev in ev_list if "voltage" in ev.name and ev.primary == "Proton"]

A_rect = [
    (ev.num_triggered, ev.energy, ev.step, ev.zenith, ev.is_triggered2) for  ev in ev_list
    if "voltage" in ev.name
    and ev.primary == "Proton"
    and ev.layout == 'rect'
]
A_hexhex = [
    (ev.num_triggered, ev.energy, ev.step, ev.zenith, ev.is_triggered2) for  ev in ev_list
    if "voltage" in ev.name
    and ev.primary == "Proton"
    and ev.layout == 'hexhex'
]

A_rect = np.array(A_rect)
A_hexhex = np.array(A_hexhex)

# calculate mean and variance of triggered antenna numbers in each zenith angle and energy bins 
enerbins = np.unique(A_rect[:,1])
zenbins = 180-np.unique(A_rect[:,3])
#zenbins = [94,100,105,110,120,131]
stepbins = np.unique(A_rect[:,2])

meanNtrig_ener = []
varNtrig_ener = []

for istep, step in enumerate(stepbins):
    meanNtrig_step = []
    varNtrig_step = []

    for iener, ener in enumerate(enerbins):
        meanNtrig_zen = []
        varNtrig_zen = []
    
        #for izen in range(0, len(zenbins)-1):
        for izen, zen in enumerate(zenbins):
            ind = np.where((A_rect[:,1] == ener) * (A_rect[:,2] == step)
                * (A_rect[:,3] == 180-zen) * (A_rect[:,0] > 0))
                #* (A_rect[:,3] >= zenbins[izen]) * (A_rect[:,3] < zenbins[izen+1]))

            if sum(A_rect[:,0]) == 0:
                ind[0] = -1

            meanNtrig_zen.append(np.mean(A_rect[ind[0],0]))
            varNtrig_zen.append(np.var(A_rect[ind[0],0]))

        meanNtrig_step.append(meanNtrig_zen)
        varNtrig_step.append(varNtrig_zen)

    meanNtrig_ener.append(meanNtrig_step)
    varNtrig_ener.append(varNtrig_step)

meanNtrig_ener = np.array(meanNtrig_ener)
varNtrig_ener = np.array(varNtrig_ener)


# calculate mean and variance of triggered event numbers in each zenith angle and energy bins 

Ntrig2_ener = []

for istep, step in enumerate(stepbins):
    Ntrig2_step = []

    for iener, ener in enumerate(enerbins):
        Ntrig2_zen = []
    
        #for izen in range(0, len(zenbins)-1):
        for izen, zen in enumerate(zenbins):
            ind = np.where((A_rect[:,1] == ener) * (A_rect[:,2] == step) 
                #* (A_rect[:,3] >= zenbins[izen]) * (A_rect[:,3] < zenbins[izen+1]))
                * (A_rect[:,3] == zen))
            Ntrig2_zen.append(sum(A_rect[ind[0],4])/size(A_rect[ind[0],4]))

        Ntrig2_step.append(Ntrig2_zen)

    Ntrig2_ener.append(Ntrig2_step)

Ntrig2_ener = np.array(Ntrig2_ener)



# plot Ntriggered antennas vs energies for fixed steps
sym_list = ['.','o','v','*','s']
myc = ['0','0.20','0.4','0.6','0.8']

for istep, step in enumerate(stepbins):
    plt.figure(istep) 
    plt.clf()
    for izen in range(0, len(zenbins)-1):
        plt.errorbar(enerbins, meanNtrig_ener[istep,:,izen], yerr=sqrt(varNtrig_ener[istep,:,izen]), 
            fmt=sym_list[izen], capsize=2, alpha=0.7, label='%4.0f > zen >%4.0f deg'%(180-zenbins[izen], 180-zenbins[izen+1]))
        #plt.errorbar(enerbins, Ntrig2_ener[istep,:,izen], 
        #    fmt=sym_list[izen], capsize=2)
    plt.yscale('log')
    plt.ylabel('N triggered antennas')
    plt.xlabel('energy [EeV]')
    plt.title('hex, step = %d m'%(np.int32(step)))
    plt.legend(loc=4)
    plt.show()
 
# plot Ntriggered antennas vs energies for fixed zenith angles
for izen in range(0, len(zenbins)-1):
    plt.figure(izen+4) 
    plt.clf()
    for istep, step in enumerate(stepbins):
        plt.errorbar(enerbins, meanNtrig_ener[istep,:,izen], yerr=sqrt(varNtrig_ener[istep,:,izen]), 
            fmt=sym_list[istep], capsize=2, alpha=0.7, label='step = %d m'%(np.int32(step)))
        #plt.errorbar(enerbins, Ntrig2_ener[istep,:,izen],  
         #   fmt=sym_list[istep], capsize=2, alpha=0.7)
    plt.yscale('log')
    plt.ylabel('N triggered antennas')
    plt.xlabel('energy [EeV]')
    plt.title('hex, %4.0f > zenith >%4.0f deg'%(180-zenbins[izen], 180-zenbins[izen+1]))
    plt.legend(loc=4)
    plt.show()

 # plot Ntriggered antennas vs zenith angles for fixed energies 
 for iener, ener in enumerate(enerbins):
    plt.figure(iener) 
    plt.clf()
    for istep, step in enumerate(stepbins):
        plt.errorbar(zenbins, meanNtrig_ener[istep,iener,:], yerr=sqrt(varNtrig_ener[istep,iener,:]), 
            fmt=sym_list[istep], capsize=2, alpha=0.7, label='step = %d m'%(np.int32(step)))
        #plt.errorbar(enerbins, Ntrig2_ener[istep,:,izen], 
        #    fmt=sym_list[izen], capsize=2)
    plt.yscale('log')
    plt.ylabel('N triggered antennas')
    plt.xlabel('zenith [deg]')
    plt.title('rect, E = %4.3f EeV'%(ener))
    plt.legend(loc=4)
    plt.show()
    plt.savefig('/Users/kotera/BROQUE/Plots_GRAND/Ntrig_vs_zen_E%4.3f_rect.png'%(ener))



# plot Ntriggered events vs energies for fixed step size and zenith angles

for istep, step in enumerate(stepbins):
    plt.figure(istep) 
    plt.clf()
    for izen in range(0, len(zenbins)-1):
        plt.errorbar(enerbins, Ntrig2_ener[istep,:,izen], 
            fmt=sym_list[izen], capsize=2, alpha=0.7, label='%4.0f > zen >%4.0f deg'%(180-zenbins[izen], 180-zenbins[izen+1]))
    plt.yscale('log')
    plt.ylabel('N triggered events')
    plt.xlabel('energy [EeV]')
    plt.title('hex, step = %d m'%(np.int32(step)))
    plt.legend(loc=4)
    plt.show()
    plt.savefig('/Users/kotera/BROQUE/Plots_GRAND/Ntrigev_vs_energy_step%d_rect_30muV.png'%(np.int32(step)))

for izen in range(0, len(zenbins)-1):
    plt.figure(izen+4) 
    plt.clf()
    for istep, step in enumerate(stepbins):
        plt.errorbar(enerbins, Ntrig2_ener[istep,:,izen],  
            fmt=sym_list[istep], capsize=2, alpha=0.7, label='step = %d m'%(np.int32(step)))
    plt.yscale('log')
    plt.ylabel('N triggered events')
    plt.xlabel('energy [EeV]')
    plt.title('hex, %4.0f > zenith >%4.0f deg'%(180-zenbins[izen], 180-zenbins[izen+1]))
    plt.legend(loc=4)
    plt.show()
    plt.savefig('/Users/kotera/BROQUE/Plots_GRAND/Ntrigev_vs_energy_z%4.1f_rect_30muV.png'%(180-zenbins[izen+1]))

# plot Ntriggered events vs zenith angles for fixed energies 
for iener, ener in enumerate(enerbins):
    plt.figure(iener) 
    plt.clf()
    for istep, step in enumerate(stepbins):
        plt.errorbar(zenbins, Ntrig2_ener[istep,iener,:],  
            fmt=sym_list[istep], capsize=2, alpha=0.7, label='step = %d m'%(np.int32(step)))
        plt.yscale('log')
    plt.ylabel('N triggered antennas')
    plt.xlabel('zenith [deg]')
    plt.title('rect, E = %4.3f EeV'%(ener))
    plt.legend(loc=4)
    plt.show()
    plt.savefig('/Users/kotera/BROQUE/Plots_GRAND/Ntrigev_vs_zen_E%4.3f_rect.png'%(ener))
