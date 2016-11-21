import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

def calc_wnfreq_spectrum(data,wn_max):
    """Compute total, standing and travelling spectra of input data.

    Compute the spectral analysis of Watt-Meyer and Kushner (2015).

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

    # calculate 2D Fourier transform on time and longitude axes
    fft2_data=np.fft.fft2(data,axes=(0,len(data_shape)-1))

    # rearrange data so freq=0 point is at centre of time dimension, and only keep wn_max wavenumbers
    fft2_trans=transform_wnfreq_spectrum(fft2_data,wn_max)

    # if the data had an even number of timesteps, remove the last Fourier coefficient, which corresponds
    # to the Nyquist frequency and doesn't have a corresponding opposite-sign frequency.
    if data.shape[0]%2==0:
		fft2_nyquist=fft2_trans[data_shape[0]-1:data_shape[0],...]
		fft2_trans=fft2_trans[:-1,...]

    # compute absolute value and phases of fourier coefficients
    fft2_trans_abs=abs(fft2_trans)
    fft2_trans_phases=np.angle(fft2_trans)

    fft2_standing_abs=np.minimum(fft2_trans_abs,fft2_trans_abs[::-1,...]) # minimum of eastward and westward parts of spectrum
    fft2_travelling_abs=fft2_trans_abs-fft2_standing_abs # remainder (i.e. travelling part) computed based on amplitude

    # reconstruct standing and travelling parts of spectrum with original phase information
    fft2_standing=fft2_standing_abs*(np.cos(fft2_trans_phases)+1j*np.sin(fft2_trans_phases))
    fft2_travelling=fft2_travelling_abs*(np.cos(fft2_trans_phases)+1j*np.sin(fft2_trans_phases))

    # if data had an even number of timesteps, append back on the Nyquist frequency component
    if data.shape[0]%2==0:
		fft2_trans=np.append(fft2_trans,fft2_nyquist,axis=0)
		fft2_standing=np.append(fft2_standing,np.zeros(np.shape(fft2_nyquist),dtype='complex'),axis=0)
		fft2_travelling=np.append(fft2_travelling,fft2_nyquist,axis=0)

    return (fft2_trans,fft2_standing,fft2_travelling)


def invert_wnfreq_spectrum(fft_coeffs_in,wn_min,wn_max,N,tol=1e-6):
    """Compute real space inverse of fft_coeffs_in, summed over wavenumber wn_min to wn_max.

    Invert fft_coeffs_in to real space and time. The wavenumbers are restricted
    to be between wn_min and wn_max, inclusive. The data is inverted onto a grid
    with N points in longitude. Note that only unsmoothed Fourier coefficients
    should be used when inverting.

    Parameters
    ----------
    fft_coeffs_in : complex ndarray of Fourier coefficients
	Input Fourier coefficients assumed to be stored in same way as output
	of calc_wnfreq_spectra
    wn_min : int, Minimum wavenumber for which to compute inverse.
    wn_max : int,  Maximum wavenumer for which to compute inverse.
    N : int
	Number of points of longitude to invert data onto

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

    # put coefficients back in "standard" fft order, fill with zeros in order
    # to have N wavenumbers
    fft_coeffs_in=untransform_wnfreq_spectrum(fft_coeffs_copy,N)

    # compute inverse fourier transform
    data_out=np.fft.ifft2(fft_coeffs_in,axes=(0,len(fft_coeffs_shape)-1))

    # check that inverted data is real to within some tolerance
    assert abs(np.imag(data_out)).max()<tol

    # return real part of data
    return np.real(data_out)


def transform_wnfreq_spectrum(fft_coeffs_in,wn_max):
    """Rearrange Fourier coefficients and limit to wn_max wavenumbers.

    Shift Fourier coefficients such that the frequency axis has the
    zero-frequency component at the center, and goes from positive (i.e.
    eastward) frequencies to negative (i.e. westward). Also limit the
    wavenumbers of wn_max, and store them starting from 0 to wn_max.

    Parameters
    ----------
    fft_coeffs_in : complex ndarray, where first dimension is frequency, and last dimension is wavenumber
		Input Fourier coefficients ordered following standard output of fft2.
    wn_max : int
		Largest wavenumber to keep in output.

    Returns
    -------
    fft2_coeffs_out : complex ndarray, same dimensions as input except final dimensional has length wn_max+1
		Rearranged coefficients, limited to wn_max wavenumbers.

    """
    fft_coeffs_out=fft_coeffs_in[...,0:wn_max+1] # keep up to wn_max wavenumbers (including wave-0, i.e. zonal mean)
    fft_coeffs_out=np.fft.fftshift(fft_coeffs_out,axes=0) # shift frequency axis
    fft_coeffs_out=fft_coeffs_out[::-1,...] # invert order of frequency components (so westward first, then eastward)

    return fft_coeffs_out

def transform_wnfreq_spectrum_to_WK99(fft_coeffs):
    plot_wavenumbers=np.size(fft_coeffs,axis=1)
    T=np.size(fft_coeffs,axis=0)

    # if there is an even number of frequencies, remove the Nyquist frequency so that frequencies are symmetric about zero.
    if T%2==0:
	T-=1
	fft_coeffs=fft_coeffs[:-1,...]

    fft_coeffs_new=np.zeros((2*plot_wavenumbers-1,(T-1)/2))
    fft_coeffs_new[:plot_wavenumbers-1,:]=np.transpose(fft_coeffs[:(T-1)/2,1:][::-1,::-1])
    fft_coeffs_new[plot_wavenumbers:,:]=np.transpose(fft_coeffs[(T-1)/2+1:,1:])
    fft_coeffs_new[plot_wavenumbers-1,:]=np.transpose(fft_coeffs[(T-1)/2+1:,0])

    return fft_coeffs_new

def untransform_wnfreq_spectrum(fft_coeffs_in,N):
    """Rearrange Fourier coefficients and pad to N wavenumbers.

    Performs the opposite operations of transform_wnfreq_spectrum to make data
    with limited number of wavenumbers and frequency centered on 0 into the
    appropriate form for ifft2.

    Parameters
    ----------
    fft_coeffs_in : complex ndarray, where first dimension is frequency, and last dimension is wavenumber
		Input Fourier coefficients ordered following output of
		transform_wnfreq_spectrum.
    N : int
		Number of points of longitude for which to calculate realspace data.

    Returns
    -------
    fft2_coeffs_out : complex ndarray, same dimensions as input except final dimension has length N
		Rearranged coefficients, padded to N wavenumbers.

    """
    fft_coeffs_shape=np.shape(fft_coeffs_in)
    num_dim=len(fft_coeffs_shape)
    wn=fft_coeffs_shape[-1] # number of wavenumbers in input data
    T=fft_coeffs_shape[0] # number of frequencies in input data

    # create output coefficient array with N wavenumbers
    fft_coeffs_out_shape=np.array(fft_coeffs_shape)
    fft_coeffs_out_shape[num_dim-1]=N
    fft_coeffs_out=np.zeros(fft_coeffs_out_shape,dtype='complex')

    fft_coeffs_in=fft_coeffs_in[::-1,...] # reverse order of frequencies

    fft_coeffs_out[...,0:wn]=fft_coeffs_in
    if T%2==0:
	fft_coeffs_out[0,...,-1:-wn:-1]=np.conj(fft_coeffs_in[0,...,1:])
	fft_coeffs_out[1:,...,-1:-wn:-1]=np.conj(fft_coeffs_in[-1:0:-1,...,1:])
    else:
	fft_coeffs_out[...,-1:-wn:-1]=np.conj(fft_coeffs_in[::-1,...][...,1:])

    fft_coeffs_out=np.fft.ifftshift(fft_coeffs_out,axes=0) # shift frequencies

    return fft_coeffs_out

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


def plot_wnfreq_spectrum_lineplots(fft_coeffs,fig_num,freq_cutoff,scale_factor,my_figsize=(8,6),title='',plot_xlim=2.0,my_ylabel='Wavenumber',my_linestyle='-k',my_linewidth=1.0,second_plot=False,leg_label='',fig_label='',data_freq=1.0):
    """Plot wavenumber-frequency spectrum as series of lineplots

    Plots one line of power versus frequency for each wavenumber in the
    inputted coefficients.

    Parameters
    ----------
    fft_coeffs : real 2D array, with dimensions (freq x wavenumber)
	Input coefficients for plotting. Note is assumed the input coefficients
	are real, i.e. the power spectrum has already been computed.
    fig_num : int
        The figure number in which to plot the data.
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
	fft_coeffs=fft_coeffs[:-1,...]

    plot_wavenumbers=np.size(fft_coeffs,axis=1)

    fig=plt.figure(fig_num,figsize=my_figsize)

    if not(second_plot) and title!='':
        plt.suptitle(title+'. (scale='+str(round(scale_factor,1))+')')

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

	ax2.text(-1./plot_xlim+0.011,plot_wavenumbers+0.8,fig_label,backgroundcolor='white',va='top')


def plot_wnfreq_spectrum_contourf(fft_coeffs,fig_num,freq_cutoff,scale_factor,my_figsize=(8,6),title='',plot_xlim=2.0,my_ylabel='Wavenumber',fig_label='',log_scale=False):
    """Plot wavenumber-frequency spectrum as filled contour plot

    Plot filled contour plot of spectral power as function of wavenumber and frequency.

    Parameters
    ----------
    fft_coeffs : real 2D array, with dimensions (freq x wavenumber)
	Input coefficients for plotting. Note is assumed the input coefficients
	are real, i.e. the power spectrum has already been computed.
    fig_num : int
        The figure number in which to plot the data.
    freq_cutoff : int
        Integer representing how many frequency indices away from the zero-
	frequency to start plotting the spectrum.
    scale_factor : float
        The value by which all input data scaled for plotting.


    """
    T=np.size(fft_coeffs,axis=0)
    freq=np.fft.fftshift(np.fft.fftfreq(T))

    # if there is an even number of frequencies, remove the Nyquist frequency so that frequencies are symmetric about zero.
    if T%2==0:
	T-=1
	freq=freq[1:]
	fft_coeffs=fft_coeffs[:-1,...]

    plot_wavenumbers=np.size(fft_coeffs,axis=1)

    fig=plt.figure(fig_num,figsize=my_figsize)

    if title!='':
        plt.suptitle(title+'. (scale='+str(round(scale_factor,1))+')')

    lev_exp = np.arange(np.floor(np.log10(fft_coeffs.min())-1), np.ceil(np.log10(fft_coeffs.max())+1),0.5)
    levs = np.power(10, lev_exp)

    ax1 = fig.add_subplot(111)
    ax1.contourf(freq[:T/2+1-freq_cutoff],range(1,plot_wavenumbers+1),np.transpose(fft_coeffs[:T/2-freq_cutoff+1,:]),levs,cmap=plt.get_cmap('Greys'),norm=colors.LogNorm())
    ax1.contourf(freq[T/2+freq_cutoff:],range(1,plot_wavenumbers+1),np.transpose(fft_coeffs[T/2+freq_cutoff:,:]),levs,cmap=plt.get_cmap('Greys'),norm=colors.LogNorm())

    my_xticks=np.array([-1./4.,-1./5.,-1./10.0,-1./20.0,-1./50.0,0.0,1./50.0,1./20.0,1./10.0,1./5.,1./4.])
    my_xticks_freq_labels=['0.25','0.2','0.1','0.05','','0','','0.05','0.1','0.2','0.25']
    my_xticks_period_labels=[4,5,10,20,50,'',50,20,10,5,4]

    plt.xticks(my_xticks,my_xticks_freq_labels)
    ax1.set_xlim([-1./plot_xlim,1./plot_xlim])
    cur_xticks=ax1.get_xticks()
    ax1.set_yticks(range(1,plot_wavenumbers))
    ax1.set_ylabel(my_ylabel,fontsize=11)

    ax1.set_ylim([1,plot_wavenumbers])
    ax1.set_xlabel('Westward       Freq. [days$^{-1}$]       Eastward',labelpad=-.05)

    ax2=ax1.twiny()
    plt.yticks(range(1,plot_wavenumbers,1))

    plt.xticks(my_xticks,my_xticks_period_labels)

    ax2.set_xlim([-1./plot_xlim,1./plot_xlim])
    ax2.set_xlabel('Westward        Period [days]        Eastward')

    ax1.grid(axis='y')
    ax2.grid(axis='x')

    ax2.text(-1./plot_xlim+0.011,plot_wavenumbers-0.2,fig_label,backgroundcolor='white',va='top')


def plot_wnfreq_spectrum_contourf_WK99(fft_coeffs,fig_num,scale_factor,data_freq=1,my_figsize=(8,6),title='',plot_xlim=2.0,my_ylabel='Wavenumber',fig_label='',log_scale=False,cons=np.arange(0.7,2,.1),con_to_mark=1.1):
    """Plot wavenumber-frequency spectrum as filled contour plot

    Plot filled contour plot of spectral power as function of wavenumber and frequency.
    Plot as in Wheeler and Kiladis (1999).

    Parameters
    ----------
    fft_coeffs : real 2D array, with dimensions (freq x wavenumber)
	Input coefficients for plotting. Note is assumed the input coefficients
	are real, i.e. the power spectrum has already been computed.
    fig_num : int
        The figure number in which to plot the data.
    freq_cutoff : int
        Integer representing how many frequency indices away from the zero-
	frequency to start plotting the spectrum.
    scale_factor : float
        The value by which all input data scaled for plotting.


    """
    T=np.size(fft_coeffs,axis=1)
    freq=np.fft.fftshift(np.fft.fftfreq(2*T,d=data_freq))

    plot_wavenumbers=np.size(fft_coeffs,axis=0)

    if title!='':
        plt.title(title)
        #plt.title(title+'. (scale='+str(round(scale_factor,1))+')')

    #ax1 = fig.add_subplot(111)
    if log_scale:
	lev_exp = np.arange(7.25,10.6,0.25)
	levs = np.power(10, lev_exp)
	#C=plt.contourf(range(-(plot_wavenumbers-1)/2,(plot_wavenumbers-1)/2+1),freq[(len(freq)-1)/2+1:],1e-14+np.transpose(fft_coeffs),levs,cmap=plt.get_cmap('Greys'),norm=colors.LogNorm())
	C=plt.contourf(range(-(plot_wavenumbers-1)/2,(plot_wavenumbers-1)/2+1),freq[(len(freq)-1)/2+1:],np.log10(1e-14+np.transpose(fft_coeffs)),lev_exp,cmap=plt.get_cmap('Greys'))
	#plt.contour(range(-(plot_wavenumbers-1)/2,(plot_wavenumbers-1)/2+1),freq[(len(freq)-1)/2+1:],1e-14+np.transpose(fft_coeffs),levs,colors='k')
    else:
	C=plt.contourf(range(-(plot_wavenumbers-1)/2,(plot_wavenumbers-1)/2+1),freq[(len(freq)-1)/2+1:],1e-14+np.transpose(fft_coeffs),cons,cmap=plt.get_cmap('Greys'))
	plt.contour(range(-(plot_wavenumbers-1)/2,(plot_wavenumbers-1)/2+1),freq[(len(freq)-1)/2+1:],1e-14+np.transpose(fft_coeffs),[con_to_mark],colors='r',linewidths=2)


    plt.xticks(range(-(plot_wavenumbers-1)/2,(plot_wavenumbers-1)/2+1,2),fontsize=9)
    plt.yticks(fontsize=9)
    #plt.xlabel('Westward                  Wavenumber                  Eastward')
    #plt.ylim([0,freq[(T-1)/2+1:].max()])
    plt.ylim([0,0.8])
    #plt.ylabel('Frequency [day**-1]')

    plt.grid(axis='y',color='slategray')
    plt.grid(axis='x',color='slategray')

    #plt.text(-plot_wavenumbers+0.5,freq.max()-.03,fig_label,backgroundcolor='white',va='top')

    return C

	
def calc_dispersion_relations_WK99(wavenumbers,n,H):
	g=9.81 # acceleration due to gravity
	a=6.371e6 # radius of earth
	Omega=2*np.pi/(24.0*60*60)
	beta=2.0*Omega/a
	c=np.sqrt(g*H)

	wavenumbers_dim=wavenumbers/(a)

	disp_freq=np.zeros((3,len(wavenumbers)))
	if n==0:
		w0_ind=np.argmin(abs(wavenumbers))

		for w,wavenumber in enumerate(wavenumbers_dim):
			disp_freq[:,w]=np.roots([1/(c*beta),0.0,(-c/beta*wavenumber**2-2*n-1),c*wavenumber])

		# fix cross-overs MRG/Kelvin wave solution
		cross_ind=np.argmin(abs(disp_freq[1,w0_ind:]-disp_freq[2,w0_ind:]))+w0_ind
		if disp_freq[1,cross_ind+1]>disp_freq[2,cross_ind+1]:
			#print wavenumbers[cross_ind]
			#print '2 is greater after cross'
			disp_freq[1,cross_ind:],disp_freq[2,cross_ind:]=disp_freq[2,cross_ind:],disp_freq[1,cross_ind:]
			# set last solution as Kelvin wave solution
			disp_freq[2,:]=-c*wavenumbers_dim
		else:
			#print wavenumbers[cross_ind]
			#print '2 is not greater after cross'
			disp_freq[1,cross_ind:],disp_freq[2,cross_ind:]=disp_freq[2,cross_ind:],disp_freq[1,cross_ind:]
			# set last solution as Kelvin wave solution
			disp_freq[2,:]=-c*wavenumbers_dim
	else:
		for w,wavenumber in enumerate(wavenumbers_dim):
			disp_freq[:,w]=np.roots([1/(c*beta),0.0,(-c/beta*wavenumber**2-2*n-1),c*wavenumber])

	return disp_freq*24*60*60/(2*np.pi) # convert to days^-1

