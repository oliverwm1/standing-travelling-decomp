######################################################################################
# This script shows how to use the functions in the stand_travel_routines package by #
# applying them to some ERA-Interim reanalysis data.                                 #
#                                                                                    #   
# Oliver Watt-Meyer                                                                  #
# September 2015                                                                     #
######################################################################################

import stand_travel_routines as wnfreq
import numpy as np
from matplotlib import pyplot as plt
import netCDF4 as nc4

# plot longitude versus time filled contour plot
def plot_hovmuller(x,y,z,cons_auto=True,cons=0,my_cmap=plt.get_cmap(name='RdBu_r'),my_xlabel='Longitude [$^{\circ}$E]',my_ylabel='',my_title='',plot_colorbar=True):
    if cons_auto:
        C=plt.contourf(x,y,z,7,cmap=my_cmap)
    else:
        C=plt.contourf(x,y,z,cons,cmap=my_cmap)
    plt.title(my_title)
    plt.xlabel(my_xlabel)
    plt.ylabel(my_ylabel)
    #plt.xticks([0,60,120,180,240,300])
    plt.xticks(range(-180,180,60))
    plt.ylim([0,len(y)-1])

    if plot_colorbar:
        cbar=plt.colorbar(orientation='horizontal')
        cbar.ax.set_xticklabels(C.levels,rotation='vertical',text='%f.0')

# contours for hovmoller
contours=np.arange(-500,501,100)
fig_num=1

wn_plot_max=6 # number of wavenumbers for which to calculate/plot spectrum
wn_plot_realspace=1 # wavenumber for which to plot real space travelling/standing components
wn_str='Wave-'+str(wn_plot_realspace)

plot_freq_cutoff=0
smooth_amount=2
plot_freq_cutoff_smooth=2

lat_plot_list=[30,60] # latitudes at which to plot spectra

g=9.81 # for conversion from geopotential to geopotential height

# load reanalysis data
filename='z_anom_500hPa_NDJFM_1979-1980_ERAInterim.nc'
var_name='dclimRem_Z_GDS0_ISBL'
nc = nc4.Dataset(filename)
data = nc.variables[var_name][:]
data /= g # convert from geopotential to geopotential height
lon = nc.variables['g0_lon_3'][:]
lat = nc.variables['g0_lat_2'][:]
plev = nc.variables['lv_ISBL1'][:]
time = nc.variables['time'][:]
nc.close()

T=len(time)
N=len(lon)

# remove plev dimension since it has length 1
data = np.squeeze(data)

# compute wavenumber-frequency spectrum, and standing/traveling decomposition
(wnfreq_total,wnfreq_standing,wnfreq_travelling)=wnfreq.calc_wnfreq_spectrum(data,wn_plot_max)

for lat_plot in lat_plot_list:
    # find indices for particular level and latitude we want to plot
    lat_ind=np.argmin(abs(lat-lat_plot))

    loc_label='500hPa'+str(lat_plot)+'N'

    # compute absolute value squared, and covariance term (only for desired latitude)
    var_total=abs(wnfreq_total[:,lat_ind,:])**2
    var_standing=abs(wnfreq_standing[:,lat_ind,:])**2
    var_travelling=abs(wnfreq_travelling[:,lat_ind,:])**2
    cov_standing_travelling=2*abs(wnfreq_standing[:,lat_ind,:])*abs(wnfreq_travelling[:,lat_ind,:])

    # plot wavenumber-frequency spectrum and traveling/standing decomposition
    vertical_scale=max([var_total[:T/2-plot_freq_cutoff+1,:].max(),var_total[T/2+plot_freq_cutoff:,:].max()])

    fig=plt.figure(fig_num,figsize=(8,6))
    plt.suptitle(loc_label+', raw power spectra')
    wnfreq.plot_wnfreq_spectrum_lineplots(var_total,fig,plot_freq_cutoff,vertical_scale,leg_label='Total',plot_xlim=5)
    wnfreq.plot_wnfreq_spectrum_lineplots(var_standing,fig,plot_freq_cutoff,vertical_scale,my_linestyle='-r',leg_label='Standing',plot_xlim=5,second_plot=True)
    wnfreq.plot_wnfreq_spectrum_lineplots(var_travelling,fig,plot_freq_cutoff,vertical_scale,my_linestyle='-b',leg_label='Traveling',plot_xlim=5,second_plot=True)
    wnfreq.plot_wnfreq_spectrum_lineplots(cov_standing_travelling,fig,plot_freq_cutoff,vertical_scale,my_linestyle='-g',leg_label='Covar',plot_xlim=5,second_plot=True)

    plt.legend()
    plt.tight_layout(rect=(0,0,1,0.97))
    fig_num+=1

    # compute smoothed spectra
    var_total_smoothed=wnfreq.smooth_wnfreq_spectrum_gaussian(var_total,smooth_amount)
    var_standing_smoothed=wnfreq.smooth_wnfreq_spectrum_gaussian(var_standing,smooth_amount)
    var_travelling_smoothed=wnfreq.smooth_wnfreq_spectrum_gaussian(var_travelling,smooth_amount)
    cov_standing_travelling_smoothed=wnfreq.smooth_wnfreq_spectrum_gaussian(cov_standing_travelling,smooth_amount)

    # plot smoothed spectra
    vertical_scale=max([var_total_smoothed[:T/2-plot_freq_cutoff_smooth+1,:].max(),var_total_smoothed[T/2+plot_freq_cutoff_smooth:,:].max()])

    fig=plt.figure(fig_num,figsize=(8,6))
    plt.suptitle(loc_label+', smoothed power spectra')
    wnfreq.plot_wnfreq_spectrum_lineplots(var_total_smoothed,fig,plot_freq_cutoff_smooth,vertical_scale,leg_label='Total',plot_xlim=5)
    wnfreq.plot_wnfreq_spectrum_lineplots(var_standing_smoothed,fig,plot_freq_cutoff_smooth,vertical_scale,my_linestyle='-r',leg_label='Standing',plot_xlim=5,second_plot=True)
    wnfreq.plot_wnfreq_spectrum_lineplots(var_travelling_smoothed,fig,plot_freq_cutoff_smooth,vertical_scale,my_linestyle='-b',leg_label='Traveling',plot_xlim=5,second_plot=True)
    wnfreq.plot_wnfreq_spectrum_lineplots(cov_standing_travelling_smoothed,fig,plot_freq_cutoff_smooth,vertical_scale,my_linestyle='-g',leg_label='Covar',plot_xlim=5,second_plot=True)

    plt.legend()
    plt.tight_layout(rect=(0,0,1,0.97))
    fig_num+=1

    # compute inverted data and standing and travelling signals for wave wn_plot_realspace
    data_lev_invert_total=wnfreq.invert_wnfreq_spectrum(wnfreq_total[:,lat_ind,:],wn_plot_realspace,wn_plot_realspace,N)
    data_lev_invert_travelling=wnfreq.invert_wnfreq_spectrum(wnfreq_travelling[:,lat_ind,:],wn_plot_realspace,wn_plot_realspace,N)
    data_lev_invert_standing=wnfreq.invert_wnfreq_spectrum(wnfreq_standing[:,lat_ind,:],wn_plot_realspace,wn_plot_realspace,N)

    # plot inverted data for wave wn_plot_realspace
    plt.figure(fig_num,figsize=(10,6))
    fig_num+=1
    plt.suptitle('Inverted wave-1 and standing and travelling components at 500hPa and '+str(lat_plot)+'N.')

    plt.subplot(131)
    plot_hovmuller(lon,range(T),data_lev_invert_total,my_ylabel='time [days]',my_title=wn_str+' Total')

    plt.subplot(132)
    plot_hovmuller(lon,range(T),data_lev_invert_standing,my_title=wn_str+' Standing')

    plt.subplot(133)
    plot_hovmuller(lon,range(T),data_lev_invert_travelling,my_title=wn_str+' Travelling')


plt.show()
