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
import csv
import xmltodict
#http://www.nehalemlabs.net/prototype/blog/2013/04/09/an-introduction-to-smoothing-time-series-in-python-part-ii-wiener-filter-and-smoothing-splines/
#https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html


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
   return -bline+dat,bline

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
#Parsing of configure file to the dict


  # create parser
  parser = argparse.ArgumentParser(description="LDR serial")
  # add expected arguments
  parser.add_argument('-f','--file', dest='file', help='input data filename', required=True)
  parser.add_argument('-c', '--config', dest='config',help='configuration filename', default='config.xml')
  parser.add_argument('-d', '--divider', dest='divider',help='X scale divider', default='1000')
  # parse args
  args = parser.parse_args()
  filename = args.file
  configfile=args.config
  divider=np.float(args.divider)
  print('reading from file%s...' % filename)
  with open(args.config) as fd:
    doc = xmltodict.parse(fd.read())
  # clean com-port
  data=[]
  x=[]
  y=[]
  for line in open(filename):
    columns = line.split('\t')
    if len(columns) >= 2:
        x.append(np.float(columns[0])/divider)
        y.append(np.float(columns[1]))
  #Interpolation input data to align data interval
  dx=max(x)/len(x)
  fs = 1/dx #The example dataset was recorded at 1/dx
  print('Sampling rate %.1f Hz' %fs)
  draw_dict={}
  hist_dict={}
  try:
      f2 = interp1d(x, y)
      xnew = np.linspace(0, max(x), len(x), endpoint=True)
      draw_dict['Raw']=f2(xnew) 
      print xnew
  except:
      xnew=x
      draw_dict['Raw']=y      
#Check in config moving average filter
  
  for kin in doc['root']:
   print kin
   if (kin=='Filters'): 
     print ('Filters is True')
     d=doc['root'][kin]['object']
     if type(d) is not list: d=[d]
     for i in d:
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


   if (kin=='Baseline'):

     print ('Baseline substraction is True')
     d=doc['root'][kin]['object']
     if type(d) is not list: d=[d]
     for i in d:
       print i
       print ("Substract background from %s by the %s method" %(i['data'],i['@name']))    
       y_b1,bgrn=substract_bline(draw_dict[i['data']],np.float(i['lambda']),np.float(i['ratio']),i['@type'])
       draw_dict[i['@name']]=y_b1
       draw_dict["%s_bg" %i['@name']]=bgrn

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
   if (kin=='Spectrum'):
    print ('Plotting spectra is True')
    f21,ax5=plt.subplots(1, sharex=True, sharey=True)
    #ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel('frequency [Hz]')
    ax5.set_ylabel('|PSD [V**2/Hz]|')
    d=doc['root'][kin]['object']
    if type(d) is not list: d=[d]
    for i in d:
      freqs,ps=get_spectrum(draw_dict[i['data']],fs)
      ax5.plot(freqs, ps,label=i['@name'])
    ax5.legend(loc='upper left')
 
   if (kin=='QRSDetect'):
     print ('QRS detection is True')
#Wavelet QRS analisis Madeira et all Med.Eng.phys., 34, 2012,1236
     d=doc['root'][kin]['object']
     if type(d) is not list: d=[d]
     for i in d:
              widths=[float(j) for j in (i['widths']['width'])]
              V=QRS_detection(draw_dict[i['data']], widths)
              xpd,ybeat1=peaks_find(V,dx,np.int(i['interval']),np.float(i['threshold']),type='max')
              p_interval=pulse_interval(xpd)
              bpm_w=np.median(p_interval)
              draw_dict["%s_wave" %i['@name']]=V
              draw_dict[i['@name']]=[xpd,ybeat1]
              hist_dict[i['@name']]=p_interval
   if (kin=='graph'):

             f0,ax1 = plt.subplots(1, sharex=False, sharey=False)
             f0,ax2=plt.subplots(1, sharex=False, sharey=False)
             f0,ax3=plt.subplots(1, sharex=False, sharey=False)

             graphp={'filtered':ax1.plot, 'substr':ax2.plot,'wavelet':ax3.plot}
             graphs={'filtered':ax1.scatter, 'substr':ax2.scatter,'wavelet':ax3.scatter}
             d=doc['root'][kin]
             if type(d) is not list: d=[d]
             for i in d:
                          d=i['plot']
                          if type(d) is not list: d=[d]
                          for j in d:
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
             d=doc['root'][kin]['data']
             if type(d) is not list: d=[d]
             for i in d:
                    ax4.hist(hist_dict[(i['#text'])],normed=True, bins=100,label=i['@label'])
                    ax4.legend(loc='upper left')
    
           
  plt.show()
  del_index=[]
  new_dict={}
  for i in draw_dict: 
     print i
     if (len(draw_dict[i])!=2): new_dict[i]=draw_dict[i]
  new_dict['Aaw_x']=xnew

  with open('result.csv', 'wb') as file:
       writer = csv.writer(file, delimiter='\t')
       writer.writerow(new_dict.keys())
       for row in zip(*new_dict.values()):
          writer.writerow(list(row))
 

# call main
if __name__ == '__main__':
  main()
