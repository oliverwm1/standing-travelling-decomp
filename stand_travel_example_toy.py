################################################################################
# This script shows how to use the functions in the wnfreq_routines package by
# applying them to some artificial test data.
#
# Oliver Watt-Meyer
# September 2015
################################################################################

import wnfreq_routines_2_3 as wnfreq
import numpy as np
from matplotlib import pyplot as plt

# plot longitude versus time filled contour plot
def plot_hovmuller(x,y,z,cons_auto=True,cons=0,my_cmap=plt.get_cmap(name='RdBu_r'),my_xlabel='Longitude [$^{\circ}$E]',my_ylabel='',my_title='',plot_colorbar=True):
    if cons_auto:
        C=plt.contourf(x,y,z,7,cmap=my_cmap)
    else:
        C=plt.contourf(x,y,z,cons,cmap=my_cmap)
    plt.title(my_title)
    plt.xlabel(my_xlabel)
    plt.ylabel(my_ylabel)
    plt.xticks([0,60,120,180,240,300])
    plt.ylim([0,len(y)-1])

    if plot_colorbar:
        cbar=plt.colorbar(orientation='horizontal')
        cbar.ax.set_xticklabels(C.levels,rotation='vertical',text='%f.0')

# contours for hovmoller
contours=np.arange(-2,2.1,0.5)

# number of wavenumbers for which to calculate/plot spectrum
wn_plot=3
plot_freq_cutoff=1

# create x/y grid
T=151                    # length of timeseries (in days)
t=np.arange(0,T)         # time axis
x=np.arange(0,360,1.5)   # longitude axis (in degrees of longitude)
N=len(x)                 # length of longitude axis

xx,tt=np.meshgrid(x,t)   # 2D array of longitudes and times (each will be of shape TxN)

# generate some articial test data

# standing wave data
freq_S=1./25.0			# 1/days
amp_S=1.0
phase_S_x=85			# degrees longitude
phase_S_t=25.16			# days
k_S=2.0					# wavenumber

data_S=0.5*amp_S*(np.cos(k_S*np.deg2rad(xx-phase_S_x)-2*np.pi*freq_S*(tt-phase_S_t))+np.cos(k_S*np.deg2rad(xx-phase_S_x)+2*np.pi*freq_S*(tt-phase_S_t)))

# propagating wave data
freq_P=-1./25			# 1/days
amp_P=1.0
phase_P=80.0			# degrees longitude
phase_P2=50.0			# days
k_P=2.0					# wavenumber

data_P=amp_P*np.cos(k_P*np.deg2rad(xx-phase_P)-2*np.pi*freq_P*tt)

# sum to get "total" data
data_T=data_S+data_P


# plot original data
plt.figure(1,figsize=(10,6))
plt.subplot(131)
plot_hovmuller(x,t,data_T,my_ylabel='time [days]',my_title='Total data')

plt.subplot(132)
plot_hovmuller(x,t,data_S,my_title='Standing part of data')

plt.subplot(133)
plot_hovmuller(x,t,data_P,my_title='Traveling part of data')

plt.tight_layout()


# compute wavenumber-frequency spectrum, and standing/traveling decomposition
(wnfreq_total,wnfreq_standing,wnfreq_travelling)=wnfreq.calc_wnfreq_spectrum(data_T,wn_plot)

# compute absolute value squared, and covariance term
var_total=abs(wnfreq_total)**2
var_standing=abs(wnfreq_standing)**2
var_travelling=abs(wnfreq_travelling)**2
cov_standing_travelling=2*abs(wnfreq_standing)*abs(wnfreq_travelling)

# plot wavenumber-frequency spectrum
vertical_scale=max([var_total[:T/2-plot_freq_cutoff+1,:].max(),var_total[T/2+plot_freq_cutoff:,:].max()])

wnfreq.plot_wnfreq_spectrum_lineplots(var_total,2,plot_freq_cutoff,vertical_scale,leg_label='Total',plot_xlim=8)
wnfreq.plot_wnfreq_spectrum_lineplots(var_standing,2,plot_freq_cutoff,vertical_scale,my_linestyle='-r',leg_label='Standing',plot_xlim=8,second_plot=True)
wnfreq.plot_wnfreq_spectrum_lineplots(var_travelling,2,plot_freq_cutoff,vertical_scale,my_linestyle='-b',leg_label='Traveling',plot_xlim=8,second_plot=True)
wnfreq.plot_wnfreq_spectrum_lineplots(cov_standing_travelling,2,plot_freq_cutoff,vertical_scale,my_linestyle='-g',leg_label='Covar',plot_xlim=8,second_plot=True)

plt.legend()
plt.tight_layout()

plt.show()
