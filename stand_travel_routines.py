#################################################################################
# This file includes a variety of functions which implement the                 #
# standing-travelling wave decomposition of Watt-Meyer and Kushner (JAS, 2015). #
#                                                                               #
# Oliver Watt-Meyer                                                             #
# January 2017                                                                  #
#################################################################################

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.cm import get_cmap

def calc_wnfreq_spectrum(data):
    """Compute total, standing and travelling spectra of input data.

    Compute the spectral analysis of Watt-Meyer and Kushner (JAS, 2015).

    Parameters
    ----------
    data : ndarray, where first dimension is time, and last dimension is longitude
		Input data for spectral analysis.
    wn_max : int
        Largest wavenumber to keep in output.

    Returns
    -------
    fft2_trans : complex ndarray, same shape as data except final axis has length wn_max+1
        2D Fourier coefficients (over time and longitude) of input data. Note
		the coefficients are stored such that the zero-frequency component is at
		the centre of the frequency axis, which has the same length as the input
		time axis. The wavenumbers go from 0 to wn_max.
    fft2_standing : as above
        The standing component of the Fourier coefficients. Stored as above.
    fft2_travelling : as above
        The travelling component of the Fourier coefficients. Stored as above.
    """
    data_shape=np.shape(data)

    # calculate real 2D Fourier transform on time and longitude axes (i.e. assuming
    # input data is real)
    fft2_data=np.fft.rfft2(data,axes=(0,len(data_shape)-1))

    # rearrange data so freq=0 point is at centre of time dimension
    fft2_trans=np.fft.fftshift(fft2_data,axes=0)

    # if the data had an even number of timesteps, remove and store the first Fourier coefficient,
    # which corresponds to the Nyquist frequency and doesn't have a corresponding opposite-sign frequency.
    if data.shape[0]%2==0:
		fft2_nyquist=fft2_trans[0:1,...]
		fft2_trans=fft2_trans[1:,...]

    # compute magnitudes and phases of fourier coefficients
    fft2_trans_abs=np.abs(fft2_trans)
    fft2_trans_phases=np.angle(fft2_trans)

    fft2_standing_abs=np.minimum(fft2_trans_abs,fft2_trans_abs[::-1,...]) # minimum of eastward and westward parts of spectrum
    fft2_travelling_abs=fft2_trans_abs-fft2_standing_abs # remainder (i.e. travelling part) computed based on amplitude

    # reconstruct standing and travelling parts of spectrum with original phase information
    fft2_standing=fft2_standing_abs*(np.cos(fft2_trans_phases)+1j*np.sin(fft2_trans_phases))
    fft2_travelling=fft2_travelling_abs*(np.cos(fft2_trans_phases)+1j*np.sin(fft2_trans_phases))

    # if data had an even number of timesteps, append back on the Nyquist frequency component
    if data.shape[0]%2==0:
		fft2_trans=np.append(fft2_nyquist,fft2_trans,axis=0)
		fft2_standing=np.append(np.zeros(np.shape(fft2_nyquist),dtype='complex'),fft2_standing,axis=0)
		fft2_travelling=np.append(fft2_nyquist,fft2_travelling,axis=0)

    return (fft2_trans,fft2_standing,fft2_travelling)


def invert_wnfreq_spectrum(fft_coeffs_in,wn_min,wn_max):
    """Compute real space inverse of fft_coeffs_in, summed over wavenumber wn_min to wn_max.

    Invert fft_coeffs_in to real space and time. The wavenumbers used for the inversion are
	restricted to be between wn_min and wn_max, inclusive. The data is inverted onto a grid
    with N points in longitude.

    Parameters
    ----------
    fft_coeffs_in : complex ndarray of Fourier coefficients
	Input Fourier coefficients assumed to be stored in same way as output
	of calc_wnfreq_spectra
    wn_min : int, Minimum wavenumber for which to compute inverse.
    wn_max : int,  Maximum wavenumer for which to compute inverse.

    Returns
    -------
    data_out : ndarray, same shape as input except final axis has length N
	Real space output data.
    """

    fft_coeffs_shape=np.shape(fft_coeffs_in)
    fft_coeffs_copy=np.copy(fft_coeffs_in)

    # set all coefficients for wavenumbers outside wn_min to wn_max to zero
    fft_coeffs_copy[...,:wn_min]=0.0
    fft_coeffs_copy[...,wn_max+1:]=0.0

    # put coefficients back in "standard" fft order
    fft_coeffs_in=np.fft.ifftshift(fft_coeffs_copy,axes=0)

    # compute inverse fourier transform
    data_out=np.fft.irfft2(fft_coeffs_in,axes=(0,len(fft_coeffs_shape)-1))

    return data_out


def smooth_wnfreq_spectrum_gaussian(fft_coeffs_in,smooth_amount):
    """Smooth Fourier coefficients over frequency.

    Smooth the fourier coefficients over frequency with a Gaussian filter of
    width smooth_amount.

    Parameters
    ----------
    fft_coeffs_in : complex ndarray, first dimension must be frequency
	Input Fourier coefficients.
    smooth_amount : float
	Amount to smooth the Fourier coefficients, in units of the frequency index.

    Returns
    -------
    fft_coeffs_out : complex ndarray, same shape as input
	Smoothed Fourier coefficients.

    """
    fft_coeffs_shape=np.shape(fft_coeffs_in)
    num_dim=len(fft_coeffs_shape)
    T=fft_coeffs_shape[0]

    fft_coeffs_shape2=np.copy(fft_coeffs_shape)
    fft_coeffs_shape2[0]=1

    smoother_shape=[T]+[1]*(num_dim-1)

    fft_coeffs_out=np.zeros(fft_coeffs_shape)

    for t in range(T):
        smoother=np.reshape(np.exp(-((np.arange(T)-t)/float(smooth_amount))**2),smoother_shape)
        smoother/=sum(smoother)

	fft_coeffs_out[t,...]=np.sum(fft_coeffs_in*np.tile(smoother,fft_coeffs_shape2),axis=0)

    return fft_coeffs_out


def plot_wnfreq_spectrum_lineplots(fft_coeffs,fig,freq_cutoff,scale_factor,plot_xlim=2.0,my_ylabel='Wavenumber',my_linestyle='-k',my_linewidth=1.0,second_plot=False,leg_label='',data_freq=1.0):
    """Plot wavenumber-frequency spectrum as series of lineplots

    Plots one line of power versus frequency for each wavenumber in the
    inputted coefficients.

    Parameters
    ----------
    fft_coeffs : real 2D array, with dimensions (freq x wavenumber)
		Input coefficients for plotting. Note is assumed the input coefficients
		are real, i.e. the power spectrum has already been computed.
	fig : the figure object in which plotting will be done
    freq_cutoff : int
        Integer representing how many frequency indices away from the zero-
		frequency to start plotting the spectrum.
    scale_factor : float
        The value by which all input data scaled for plotting.


    """
    T=np.size(fft_coeffs,axis=0)
    freq=np.fft.fftshift(np.fft.fftfreq(T))*data_freq # data_freq is the number of data points per day (=1 for daily data, =4 for 6hourly data)

    # if there is an even number of frequencies, remove the Nyquist frequency so that frequencies are symmetric about zero.
    if T%2==0:
        T-=1
        freq=freq[1:]
        fft_coeffs=fft_coeffs[1:,...]

    plot_wavenumbers=np.size(fft_coeffs,axis=1)

    # invert order of frequency axis of coefficients, so plotting westward on left, eastward on right
    fft_coeffs=fft_coeffs[::-1,:]

    ax1 = fig.add_subplot(111)

    for wn in range(1,plot_wavenumbers+1):
        if wn==plot_wavenumbers:
            ax1.plot(freq[:T/2+1-freq_cutoff],wn+fft_coeffs[:T/2-freq_cutoff+1,wn-1]/scale_factor,my_linestyle,linewidth=my_linewidth,label=leg_label)
            ax1.plot(freq[T/2+freq_cutoff:],wn+fft_coeffs[T/2+freq_cutoff:,wn-1]/scale_factor,my_linestyle,linewidth=my_linewidth)
        else:
            ax1.plot(freq[:T/2+1-freq_cutoff],wn+fft_coeffs[:T/2-freq_cutoff+1,wn-1]/scale_factor,my_linestyle,linewidth=my_linewidth)
            ax1.plot(freq[T/2+freq_cutoff:],wn+fft_coeffs[T/2+freq_cutoff:,wn-1]/scale_factor,my_linestyle,linewidth=my_linewidth)

    if not(second_plot):
        my_xticks=np.array([-1./4.,-1./5.,-1./10.0,-1./20.0,-1./50.0,0.0,1./50.0,1./20.0,1./10.0,1./5.,1./4.])
        my_xticks_freq_labels=['0.25','0.2','0.1','0.05','','0','','0.05','0.1','0.2','0.25']
        my_xticks_period_labels=[4,5,10,20,50,'',50,20,10,5,4]

        plt.xticks(my_xticks,my_xticks_freq_labels)
        ax1.set_xlim([-1./plot_xlim,1./plot_xlim])
        cur_xticks=ax1.get_xticks()
        ax1.set_yticks(range(1,plot_wavenumbers+1))
        ax1.set_ylabel(my_ylabel,fontsize=11)

        my_yticks_labels_1=[]
        my_yticks_labels_2=[]
        for wn in range(plot_wavenumbers):
            my_yticks_labels_1.append('Wave-'+str(wn))

        plt.yticks(np.arange(1.5,plot_wavenumbers+1,1),my_yticks_labels_1,rotation='vertical',va='bottom')

        ax1.set_ylim([1,plot_wavenumbers+1])
        ax1.set_xlabel('Westward     Freq. [days$^{-1}$]     Eastward',labelpad=-.05)

        ax2=ax1.twiny()
        plt.yticks(range(1,plot_wavenumbers+1,1))
        
        plt.xticks(my_xticks,my_xticks_period_labels)

        ax2.set_xlim([-1./plot_xlim,1./plot_xlim])
        ax2.set_xlabel('Westward      Period [days]      Eastward')

        ax1.grid(axis='y')
        ax2.grid(axis='x')

def transform_spectrum_to_WK99(fft_coeffs):
	T=np.size(fft_coeffs,axis=0)
	num_wn=np.size(fft_coeffs,axis=1)

	# rearrange coefficients
	if T%2==0:
		fft_coeffs_WK99=np.zeros((2*(num_wn-1)+1,T/2))
		fft_coeffs_WK99[:num_wn-1,:]=np.transpose(fft_coeffs[T/2:,1:][:,::-1])
		fft_coeffs_WK99[num_wn-1,:]=np.transpose(fft_coeffs[T/2:,0])
		fft_coeffs_WK99[num_wn:,:]=np.transpose(fft_coeffs[1:T/2+1,1:][::-1,:])
	else:
		#T-=1
		#fft_coeffs=fft_coeffs[1:,:]
		fft_coeffs_WK99=np.zeros((2*(num_wn-1)+1,T/2))
		fft_coeffs_WK99[:num_wn-1,:]=np.transpose(fft_coeffs[T/2:-1,1:][:,::-1])
		fft_coeffs_WK99[num_wn-1,:]=np.transpose(fft_coeffs[T/2:-1,0])
		fft_coeffs_WK99[num_wn:,:]=np.transpose(fft_coeffs[1:T/2+1,1:][::-1,:])

	fft_coeffs_WK99=np.transpose(fft_coeffs_WK99)

	return fft_coeffs_WK99

def plot_wnfreq_spectrum_contourf_WK99(fft_coeffs,data_freq=1,log_scale=False,cons=np.arange(0,1.01,.1),plot_style='contourf',plot_axis='None'):
	"""Plot wavenumber-frequency spectrum as filled contour plot, but with wavenumber on x-axis
	in the style of Wheeler and Kiladis (1999).

	Plot filled contour plot of spectral power as function of wavenumber and frequency.

	Parameters
	----------
	fft_coeffs : real 2D array, with dimensions (freq x wavenumber)
	Input coefficients for plotting. Note it is assumed the input coefficients
	are real, i.e. the power spectrum has already been computed.

	"""

	# first modify fft_coeffs array so that it spans -max_wavenumber to +max_wavenumber and 0 to max_freq
	fft_coeffs_WK99=transform_spectrum_to_WK99(fft_coeffs)

	# generate frequency coordinate
	T=np.size(fft_coeffs,axis=0)
	freq=np.fft.fftshift(np.fft.fftfreq(T,d=data_freq))
	freq=freq[T/2:]
	
	if T%2!=0:
		freq=freq[:-1]

	# generate wavenumber coordinate
	num_wn=np.size(fft_coeffs,axis=1)
	wn=np.arange(-num_wn+1,num_wn)

	if plot_axis=='None':
		if plot_style=='contourf':
			C=plt.contourf(wn,freq,fft_coeffs_WK99,cons,cmap=get_cmap('Greys'))
		elif plot_style=='contour':
			C=plt.contour(wn,freq,fft_coeffs_WK99,cons,colors='k')
	else:
		if plot_style=='contourf':
			C=plot_axis.contourf(wn,freq,fft_coeffs_WK99,cons,cmap=get_cmap('Greys'))
		elif plot_style=='contour':
			C=plot_axis.contour(wn,freq,fft_coeffs_WK99,cons,colors='k')

	return C
