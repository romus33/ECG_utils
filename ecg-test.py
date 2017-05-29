from time import sleep
import sys
import argparse

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np 
import scipy as scipy
from scipy.interpolate import interp1d
from scipy import signal
import math
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz
import scipy.sparse as sparse
from scipy.sparse.linalg import (inv, spsolve)
from scipy.signal import medfilt
from scipy.ndimage import filters
from peakdetect import peakdetect

import xmltodict
#http://www.nehalemlabs.net/prototype/blog/2013/04/09/an-introduction-to-smoothing-time-series-in-python-part-ii-wiener-filter-and-smoothing-splines/
#https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html

def filter128Hz(y):
# ECG-tools by G. Clifford http://www.robots.ox.ac.uk/~gari/CODE/ECGtools/ecgBag/
    denominator = 1;
    numerator_lp =[
    -7.836313016322635e-3,
    -1.896972756320189e-2,
    -5.175330833579545e-3,
     1.828346656167859e-2,
    -6.018872954550802e-4,
    -1.897562067220059e-2,
     2.024098477556695e-2,
     2.007552016297064e-2,
    -4.029205517855476e-2,
     3.980639054552623e-3,
     7.022378021831376e-2,
    -5.937364415063083e-2,
    -8.822946558472169e-2,
     3.032906360635707e-1,
     5.996608499295303e-1,
     3.032906360635707e-1,
    -8.822946558472169e-2,
    -5.937364415063083e-2,
     7.022378021831376e-2,
     3.980639054552623e-3,
    -4.029205517855476e-2,
     2.007552016297064e-2,
     2.024098477556695e-2,
    -1.897562067220059e-2,
    -6.018872954550802e-4,
     1.828346656167859e-2,
    -5.175330833579545e-3,
    -1.896972756320189e-2,
    -7.836313016322635e-3
    ];


    numerator_hp = [ 
    -5.272437301928293e-4,
    -1.270087347478445e-4,
    -1.421348135287156e-4,
    -1.582961734171513e-4,
    -1.755764587237310e-4,
    -1.939486402411108e-4,
    -2.135296230579367e-4,
    -2.342715794110092e-4,
    -2.563093703085422e-4,
    -2.796160181433441e-4,
    -3.043014698229107e-4,
    -3.303453248658040e-4,
    -3.578455028268245e-4,
    -3.867214105939904e-4,
    -4.170888923036170e-4,
    -4.488811733969728e-4,
    -4.822063006316578e-4,
    -5.170197946280047e-4,
    -5.535253395139226e-4,
    -5.916625148888152e-4,
    -6.316044529097936e-4,
    -6.732081327790902e-4,
    -7.165788660268917e-4,
    -7.614673946729495e-4,
    -8.081236729920874e-4,
    -8.564379420809926e-4,
    -9.069404054062480e-4,
    -9.592891108521169e-4,
    -1.013520022388740e-3,
    -1.068455050967581e-3,
    -1.126689269663744e-3,
    -1.186204883152881e-3,
    -1.247380114262856e-3,
    -1.310856912732174e-3,
    -1.375900461340159e-3,
    -1.443041461735466e-3,
    -1.511885029416557e-3,
    -1.582738786552247e-3,
    -1.655320583005337e-3,
    -1.729893353369970e-3,
    -1.806203460527855e-3,
    -1.884446397624664e-3,
    -1.964430802275773e-3,
    -2.046262955654651e-3,
    -2.129750428836313e-3,
    -2.215037813560021e-3,
    -2.301981811742379e-3,
    -2.390672229028491e-3,
    -2.480975908659032e-3,
    -2.572931105092679e-3,
    -2.666360748382136e-3,
    -2.761304979221734e-3,
    -2.857698150519032e-3,
    -2.955598535068774e-3,
    -3.054901001596308e-3,
    -3.155478224722803e-3,
    -3.257211616317062e-3,
    -3.360192620167190e-3,
    -3.464496103258388e-3,
    -3.569633821553856e-3,
    -3.675798007866096e-3,
    -3.783106411027402e-3,
    -3.891012640258272e-3,
    -3.999917962979201e-3,
    -4.109378741483071e-3,
    -4.219573587392301e-3,
    -4.330206242883826e-3,
    -4.441415505603127e-3,
    -4.552895068871517e-3,
    -4.664787674933710e-3,
    -4.776796783833356e-3,
    -4.888968546054471e-3,
    -5.001078813544399e-3,
    -5.113207893538720e-3,
    -5.225099919404130e-3,
    -5.336816346211259e-3,
    -5.448125259639512e-3,
    -5.559029938118835e-3,
    -5.669293181689946e-3,
    -5.778976225031504e-3,
    -5.887879122882245e-3,
    -5.996012432011148e-3,
    -6.103126173593918e-3,
    -6.209214132300607e-3,
    -6.314120595416707e-3,
    -6.417882343141420e-3,
    -6.520214636498100e-3,
    -6.621088290402421e-3,
    -6.720450076732404e-3,
    -6.818237948419261e-3,
    -6.914128204056832e-3,
    -7.008372785989206e-3,
    -7.100533958060398e-3,
    -7.190766902504709e-3,
    -7.278807347187491e-3,
    -7.364733068710562e-3,
    -7.448280274431376e-3,
    -7.529573080376522e-3,
    -7.608284813103398e-3,
    -7.684521116446521e-3,
    -7.758085897312433e-3,
    -7.829019629435470e-3,
    -7.897106543328992e-3,
    -7.962444021838374e-3,
    -8.024797816710869e-3,
    -8.084212682808753e-3,
    -8.140518300829559e-3,
    -8.193808564001160e-3,
    -8.243881936890022e-3,
    -8.290796858484446e-3,
    -8.334367833865763e-3,
    -8.374699271329103e-3,
    -8.411637628938252e-3,
    -8.445234427388959e-3,
    -8.475297677477069e-3,
    -8.501979217581199e-3,
    -8.525132171625144e-3,
    -8.544799261897136e-3,
    -8.560853979448457e-3,
    -8.573474502979061e-3,
    -8.582371323167616e-3,
    -8.587790488764461e-3,
     9.914104520892446e-01
    -8.587790488764461e-3,
    -8.582371323167616e-3,
    -8.573474502979061e-3,
    -8.560853979448457e-3,
    -8.544799261897136e-3,
    -8.525132171625144e-3,
    -8.501979217581199e-3,
    -8.475297677477069e-3,
    -8.445234427388959e-3,
    -8.411637628938252e-3,
    -8.374699271329103e-3,
    -8.334367833865763e-3,
    -8.290796858484446e-3,
    -8.243881936890022e-3,
    -8.193808564001160e-3,
    -8.140518300829559e-3,
    -8.084212682808753e-3,
    -8.024797816710869e-3,
    -7.962444021838374e-3,
    -7.897106543328992e-3,
    -7.829019629435470e-3,
    -7.758085897312433e-3,
    -7.684521116446521e-3,
    -7.608284813103398e-3,
    -7.529573080376522e-3,
    -7.448280274431376e-3,
    -7.364733068710562e-3,
    -7.278807347187491e-3,
    -7.190766902504709e-3,
    -7.100533958060398e-3,
    -7.008372785989206e-3,
    -6.914128204056832e-3,
    -6.818237948419261e-3,
    -6.720450076732404e-3,
    -6.621088290402421e-3,
    -6.520214636498100e-3,
    -6.417882343141420e-3,
    -6.314120595416707e-3,
    -6.209214132300607e-3,
    -6.103126173593918e-3,
    -5.996012432011148e-3,
    -5.887879122882245e-3,
    -5.778976225031504e-3,
    -5.669293181689946e-3,
    -5.559029938118835e-3,
    -5.448125259639512e-3,
    -5.336816346211259e-3,
    -5.225099919404130e-3,
    -5.113207893538720e-3,
    -5.001078813544399e-3,
    -4.888968546054471e-3,
    -4.776796783833356e-3,
    -4.664787674933710e-3,
    -4.552895068871517e-3,
    -4.441415505603127e-3,
    -4.330206242883826e-3,
    -4.219573587392301e-3,
    -4.109378741483071e-3,
    -3.999917962979201e-3,
    -3.891012640258272e-3,
    -3.783106411027402e-3,
    -3.675798007866096e-3,
    -3.569633821553856e-3,
    -3.464496103258388e-3,
    -3.360192620167190e-3,
    -3.257211616317062e-3,
    -3.155478224722803e-3,
    -3.054901001596308e-3,
    -2.955598535068774e-3,
    -2.857698150519032e-3,
    -2.761304979221734e-3,
    -2.666360748382136e-3,
    -2.572931105092679e-3,
    -2.480975908659032e-3,
    -2.390672229028491e-3,
    -2.301981811742379e-3,
    -2.215037813560021e-3,
    -2.129750428836313e-3,
    -2.046262955654651e-3,
    -1.964430802275773e-3,
    -1.884446397624664e-3,
    -1.806203460527855e-3,
    -1.729893353369970e-3,
    -1.655320583005337e-3,
    -1.582738786552247e-3,
    -1.511885029416557e-3,
    -1.443041461735466e-3,
    -1.375900461340159e-3,
    -1.310856912732174e-3,
    -1.247380114262856e-3,
    -1.186204883152881e-3,
    -1.126689269663744e-3,
    -1.068455050967581e-3,
    -1.013520022388740e-3,
    -9.592891108521169e-4,
    -9.069404054062480e-4,
    -8.564379420809926e-4,
    -8.081236729920874e-4,
    -7.614673946729495e-4,
    -7.165788660268917e-4,
    -6.732081327790902e-4,
    -6.316044529097936e-4,
    -5.916625148888152e-4,
    -5.535253395139226e-4,
    -5.170197946280047e-4,
    -4.822063006316578e-4,
    -4.488811733969728e-4,
    -4.170888923036170e-4,
    -3.867214105939904e-4,
    -3.578455028268245e-4,
    -3.303453248658040e-4,
    -3.043014698229107e-4,
    -2.796160181433441e-4,
    -2.563093703085422e-4,
    -2.342715794110092e-4,
    -2.135296230579367e-4,
    -1.939486402411108e-4,
    -1.755764587237310e-4,
    -1.582961734171513e-4,
    -1.421348135287156e-4,
    -1.270087347478445e-4,
    -5.272437301928293e-4
    ];

    # low pass filter
    aff_lp = filtfilt(numerator_lp,denominator,y);

    # Don't high pass filter it if you don't want to remove the baseline 
    #  fluctuations due to resp, BP? and electrode noise?
    filtdata = filtfilt(numerator_hp,denominator,aff_lp);
    return filtdata

def WienerFilter(y,mysize,noise):
    wi = wiener(y, mysize, noise)
    return wi

def gaussfilter(y,window,sigma):
    b = gaussian(window, sigma)
    #ga = filtfilt(b/b.sum(), [1.0], y)
    ga = filters.convolve1d(y, b/b.sum())
    return ga
#band filter
def highpass(y,fs,f1,N):
   #Noise cancelation(Filtering)
   #f1=0.5 #cuttoff low frequency to get rid of baseline wander
   #f2=45 #cuttoff frequency to discard high frequency noise
  # Wn=45 # cutt off based on fs
  # N = 3 #order of 3 less processing
   a,b = butter(N,f1*2/fs,btype='highpass') #bandpass filtering
   ecg = filtfilt(a,b,y)
   return ecg
def lowpass(y,fs,f1,N):
   #Noise cancelation(Filtering)
   #f1=0.5 #cuttoff low frequency to get rid of baseline wander
   #f2=45 #cuttoff frequency to discard high frequency noise
  # Wn=45 # cutt off based on fs
  # N = 3 #order of 3 less processing
   a,b = butter(N,f1*2/fs,btype='lowpass') #bandpass filtering
   ecg = filtfilt(a,b,y)
   return ecg
def bandreject(y,fs,f1,f2, N):
   #Noise cancelation(Filtering)
   #f1=0.5 #cuttoff low frequency to get rid of baseline wander
   #f2=45 #cuttoff frequency to discard high frequency noise
  # Wn=45 # cutt off based on fs
  # N = 3 #order of 3 less processing
   a,b = butter(N,[f1*2/fs,f2*2/fs],btype='bandstop') #bandpass filtering
   ecg = filtfilt(a,b,y)
   return ecg
def bandpass(y,fs,f1,f2, N):
   #Noise cancelation(Filtering)
   #f1=0.5 #cuttoff low frequency to get rid of baseline wander
   #f2=45 #cuttoff frequency to discard high frequency noise
  # Wn=45 # cutt off based on fs
  # N = 3 #order of 3 less processing
   a,b = butter(N,[f1*2/fs,f2*2/fs],btype='bandpass') #bandpass filtering
   ecg = filtfilt(a,b,y)
   #ecg=signal.lfilter(a,b,y)
   return ecg

def baseline_medfilt(Y,lam,filt):
 
  Ybase = medfilt(Y, np.int(filt)) # 51 should be large in comparison to your peak X-axis lengths and an odd number.
  return Ybase

#remove baseline https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf
#Asymmetric Least Squares Smoothing
# niter - Maximum number of iterations
# lam - 2nd derivative constraint
# p - Weighting of positive residuals
def baseline_als(y, lam, p, niter=100):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in xrange(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

#Baseline correction using adaptive iteratively reweighted penalized least squares
#Zhang et al Baseline correction using adaptive iteratively reweighted penalized least squares DOI:10.1039/B922045C, Analyst, 2010,135, 1138-1146 
def baseline_aPls(y, lam, ratio):
  N = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(N), 2))
  w = np.ones(N)
  while (1):
    W = sparse.spdiags(w, 0, N, N)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    d=y-z
    dn=d*(d<0)
    m=np.mean(dn)
    s = np.std(dn)
    wt = 1./ ( 1 + np.exp( 2* (d-(2*s-m))/s )) 
    if np.linalg.norm(w-wt)/np.linalg.norm(w)<ratio:
      break    
    w=wt 
  return z
  

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
# main() function


def pulse_count(xpd):
  i=0
  iii_old=0
  summ=0
  for iii in xpd:
        
        if (i>0): 
               summ=summ+(iii-iii_old)
               iii_old=iii
        else: 
               iii_old=iii
               i=1 
  return 60.0*summ/(np.float(len(xpd)))

def pulse_interval(xpd):
  return 60.0*np.diff(xpd)

def substract_bline(dat,lamda,ratio,tpe='apls'):
   baseline={'apls':baseline_aPls, 'als':baseline_als,'med':baseline_medfilt}
   bline=baseline[tpe](dat,lamda,ratio)
   return bline-dat,bline

def peaks_find(dat,dx,lookahead=200,delta=0,type='max'):
  peaks_max,peaks_min=peakdetect(dat,None, lookahead,delta)
  xpd=[]
  ybeat=[]
  if (type=='max'):
    for datapoint in peaks_max:
        xpd.append(datapoint[0]*dx)
        ybeat.append(datapoint[1])
  if (type=='min'):
    for datapoint in peaks_min:
        xpd.append(datapoint[0]*dx)
        ybeat.append(datapoint[1])
  return xpd,ybeat

#J.P.V. Madeiro Medical Engineering and Physics Volume 34, Issue 9, Pages 1236-1246 DOI: 10.1016/j.medengphy.2011.12.011
#QRS segmentation based on first-derivative, Hilbert and Wavelet Transforms
def QRS_detection(dat, widths):
  cwtmatr = signal.cwt(dat, signal.ricker, widths)
  difference_ecg=scipy.fftpack.diff(cwtmatr[3],order=1, period=1000)
  hilbert_ecg=scipy.fftpack.hilbert(difference_ecg)
  V=hilbert_ecg**2+difference_ecg**2
  V=V/V.max()
  return V

def get_spectrum(dat,dx):

  return signal.periodogram(dat, dx)
  #freqs = np.fft.fftfreq(dat.size, dx)
  #idx = np.argsort(freqs)
  #return 20*np.log10(np.abs(np.fft.fft(dat))/max(np.abs(np.fft.fft(dat)))),freqs,idx
  
def main():
#Parsin of configure file to the dict
  with open('config.xml') as fd:
    doc = xmltodict.parse(fd.read())

  # create parser
  parser = argparse.ArgumentParser(description="LDR serial")
  # add expected arguments
  parser.add_argument('--file', dest='file', required=True)
  # parse args
  args = parser.parse_args()
  #strPort = '/dev/tty.usbserial-A7006Yqh'
  filename = args.file
  print('reading from file%s...' % filename)
  # clean com-port
  data=[]
  x=[]
  y=[]
  for line in open(filename):
    columns = line.split('\t')
    if len(columns) >= 2:
        x.append(np.float(columns[0])/1000)
        y.append(np.float(columns[1]))
  #Interpolation input data to align data interval
  f2 = interp1d(x, y)
  xnew = np.linspace(0, max(x), len(x), endpoint=True)
  dx=max(x)/len(x)
 
  #Calculate moving average with 0.3s in both directions, then append do dataset
  #hrw = 0.02 #One-sided window size, as proportion of the sampling frequency
  fs = 1/dx #The example dataset was recorded at 1/dx
  draw_dict={}
  hist_dict={}
#Create graph 1
#  f,(ax1, ax2,ax3,ax4) = plt.subplots(4, sharex=False, sharey=False)
#Plot raw signal
#  ax1.plot(x,y,alpha=0.2,color='grey')
#Plot moving average signal
#  ax1.set_xlim(0, 15)
#  box = ax1.get_position()
#  ax1.legend(['Linear', 'Moving Average','baseline median filter','baseline ALS','baseline ApLS'],loc='upper center', bbox_to_anchor=(0.5, box.y0 + 3*box.height), fancybox=True, shadow=True, ncol=5)
#  box = ax2.get_position()
#  ax2.legend(['Susbtract ALSS','Substract medfilt','Substract ApLSS','average pulse'],loc='upper center', bbox_to_anchor=(0.5, box.height), fancybox=True, shadow=True, ncol=5)
#  ax2.set_xlim(0, 15) 
  print('Sampling rate %.1f Hz' %fs)
  draw_dict['Raw']=f2(xnew)

#Check in config moving average filter
  
  for kin in doc['root']:
   print kin
   if (kin=='Filters'): 
     print ('Filters is True')
     for i in doc['root'][kin]['object'] :
           print ("Apply %s filter named %s to %s data" %(i['@type'],i['@name'],i['data']))
           if (i['@type']=='MovingAverage'): 
                  
                  hrw=np.float(i['windowsize'])#One-sided window size, as proportion of the sampling frequency
                  draw_dict[i['@name']]=movingaverage(draw_dict[i['data']], window_size=(hrw*fs))
           if (i['@type']=='BandPass'):
                  f_l=np.float(i['LowFreq'])
                  f_h=np.float(i['HighFreq'])
                  f_order=np.int(i['Order'])
                  draw_dict[i['@name']]=bandpass(draw_dict[i['data']],fs,f_l,f_h,f_order)
           if (i['@type']=='BandReject'):
                  f_l=np.float(i['LowFreq'])
                  f_h=np.float(i['HighFreq'])
                  f_order=np.int(i['Order'])
                  draw_dict[i['@name']]=bandreject(draw_dict[i['data']],fs,f_l,f_h,f_order)
           if (i['@type']=='LowPass'):
                  f_l=np.float(i['Freq'])
                  f_order=np.int(i['Order'])
                  draw_dict[i['@name']]=lowpass(draw_dict[i['data']],fs,f_l,f_order)
           if (i['@type']=='HighPass'):
                  f_l=np.float(i['Freq'])
                  f_order=np.int(i['Order'])
                  draw_dict[i['@name']]=highpass(draw_dict[i['data']],fs,f_l,f_order)
           if (i['@type']=='Gauss'):
                  draw_dict[i['@name']]=gaussfilter(draw_dict[i['data']],np.float(i['window']),np.float(i['sigma']))        
           if (i['@type']=='Wiener'):
                  if (i['noise']=='None') : draw_dict[i['@name']]=WienerFilter(draw_dict[i['data']],np.int(i['window']),None)
                  else: draw_dict[i['@name']]=WienerFilter(draw_dict[i['data']],np.int(i['window']),np.float(i['noise']))    
           if (i['@type']=='Filter128Hz'):
                  draw_dict[i['@name']]=filter128Hz(draw_dict[i['data']])    

   if (kin=='Baseline'):
     print ('Baseline substraction is True')
     #name= doc['root']['Baseline']['object'][0]['@type']
     #y_b1=y_av
     for i in doc['root'][kin]['object']:
       print ("Substract background from %s by the %s method" %(i['data'],i['@name']))    
       y_b1,bgrn=substract_bline(draw_dict[i['data']],np.float(i['lambda']),np.float(i['ratio']),i['@type'])
       draw_dict[i['@name']]=y_b1
       draw_dict["%s_bg" %i['@name']]=bgrn
   #  for tp in name:
   #    
   #    y_b1,bgrn=substract_bline(y_av,lamda,ratio,tp)
   #    draw_dict["Baseline_%s" %tp]=[y_b1,bgrn]
   #    ax2.plot(xnew,y_b1)
 # ax1.plot(xnew,draw_dict['apls'])
   if (kin=='PeakDetect'):
     #Detect peaks
     print ('Peak detection is True')
     for j in doc['root'][kin]['object']:
               xpd,ybeat=peaks_find(draw_dict[j['data']],dx,np.int(j['interval']),np.float(j['threshold']),j['type'])
               bpm=pulse_count(xpd)
               p_interval=pulse_interval(xpd)
               bpm1=np.median(p_interval)
               draw_dict[j['@name']]=[xpd,ybeat]
               hist_dict[j['@name']]=p_interval
               #print "Average Heart Beat is: %.01f" %bpm #Round off to 1 decimal and print
               #print "Average Heart Beat from peak intervals is: %.01f BPM" %bpm1 #Round off to 1 decimal and print
     #ax2.scatter(xpd, ybeat, color='red') 
     #ECG peak distributions  
     #ax3.hist(p_interval,normed=True, bins=200)
     #ax3.set_xlim(0, 100)
   if (kin=='Spectrum'):
    print ('Plotting spectra is True')
    f21,ax5=plt.subplots(1, sharex=True, sharey=True)
    #ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel('frequency [Hz]')
    ax5.set_ylabel('|PSD [V**2/Hz]|')
    for i in doc['root'][kin]['object'] :
      #ps,freqs,idx=get_spectrum(draw_dict[i['data']],dx)
      #ax5.plot(freqs[idx], ps[idx],label=i['@name'])
      freqs,ps=get_spectrum(draw_dict[i['data']],fs)
      ax5.plot(freqs, ps,label=i['@name'])
    ax5.legend(loc='upper left')
 
   if (kin=='QRSDetect'):
     print ('QRS detection is True')
#Wavelet QRS analisis Madeira et all Med.Eng.phys., 34, 2012,1236
     for i in doc['root'][kin]['object'] :
              widths=[float(j) for j in (i['widths']['width'])]
              V=QRS_detection(draw_dict[i['data']], widths)
              xpd,ybeat1=peaks_find(V,dx,np.int(i['interval']),np.float(i['threshold']),type='max')
              p_interval=pulse_interval(xpd)
              bpm_w=np.median(p_interval)
              draw_dict["%s_wave" %i['@name']]=V
              draw_dict[i['@name']]=[xpd,ybeat1]
              hist_dict[i['@name']]=p_interval
              #print "Average Heart Beat (QRS find) is: %.01f" %bpm_w #Round off to 1 decimal and print
              #ybeat1=ybeat1/max(ybeat1)

#Show graphs  
   if (kin=='graph'):
             f0,ax1 = plt.subplots(1, sharex=False, sharey=False)
             f1,ax2=plt.subplots(1, sharex=False, sharey=False)
             f2,ax3=plt.subplots(1, sharex=False, sharey=False)

             graphp={'filtered':ax1.plot, 'substr':ax2.plot,'wavelet':ax3.plot}
             graphs={'filtered':ax1.scatter, 'substr':ax2.scatter,'wavelet':ax3.scatter}
             for i in doc['root'][kin]:
                          for j in i['plot']:
                               if (j['@type']=='line'):
                                      if (len(draw_dict[j['data']])==2): 
                                            graphp[i['@name']](draw_dict[j['data']][0],draw_dict[j['data']][1],label=j['label'],alpha=np.float(j['trans']))
                                            
                                      else: graphp[i['@name']](xnew,draw_dict[j['data']],label=j['label'],alpha=np.float(j['trans']))
                               if (j['@type']=='scatter'):
                                      if (len(draw_dict[j['data']])==2): graphs[i['@name']](draw_dict[j['data']][0],draw_dict[j['data']][1],label=j['label'],alpha=np.float(j['trans']))
                                      else: graphs[i['@name']](xnew,draw_dict[j['data']],label=j['label'],alpha=np.float(j['trans'])) 
             ax1.legend(loc='upper left')
             ax2.legend(loc='upper left')
             ax3.legend(loc='upper left')                                     
   if (kin=='hist'):
             f3,ax4= plt.subplots(1, sharex=False, sharey=False)
             for i in doc['root'][kin]['data']:
                    #print(i['#text'])
                    ax4.hist(hist_dict[(i['#text'])],normed=True, bins=100,label=i['@label'])
                    ax4.legend(loc='upper left')
                
  plt.show()

  

  

# call main
if __name__ == '__main__':
  main()
