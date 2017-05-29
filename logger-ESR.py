from time import sleep
import sys
import argparse

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial
from time import sleep
import sys, serial, argparse
import time
import datetime
import scipy.signal as signal

# plot class
class AnalogPlot:
  # constr
  def __init__(self, strPort,baudrate, ss10, ss11):
      # open serial port

      self.ser = serial.Serial(strPort, baudrate)
      format = "%a %b %d %H%M%S %Y"
      today = datetime.datetime.today()
      filen0='data-'+today.strftime(format)+'_0.lvm'
      filen1='data-'+today.strftime(format)+'_1.lvm'
      self.f0 = open(filen0,'w')
      self.f1 = open(filen1,'w')
      self.ax=[]
      self.ay=[]
      self.ax1=[]
      self.ay1=[]
      self.s10=ss10
      self.s11=ss11
      self.a0, = self.s10.plot([], [])
      self.a1, = self.s11.plot([], [])

  # add data

  def add(self, data):
      if len(data)>3:
                         
              
               self.ax.append(data[0]/1000)
               self.ay.append(data[1])                     
               self.ax1.append(data[2]/1000)
               self.ay1.append(data[3])
               self.f0.write('%f\t%f\n' % (data[0], data[1]))
               self.f1.write('%f\t%f\n' % (data[2], data[3]))
      else:
               self.ax.append(data[0]/1000)
               self.ay.append(data[1]) 
               self.f0.write('%f\t%f\n' % (data[0], data[1]))   
  # update plot
  def toFloatorNone(self, valu):
      import numpy as np
      try:
          return (np.float(valu)),
      except ValueError:
          return None,

  def update(self, frameNum):
      import numpy as np
      try:
          j=0

          while j<100:
               j=j+1
               while self.ser.inWaiting()==0: sleep(0.01)
               line = self.ser.readline()
               data = []
               for val in line.split('\t'):
                  data.append(np.float(val))
               #self.f.write(line)
               
               # print data
               self.add(data)
          if isinstance(self.ax, list):              
                    maxValX=max([i for i in self.ax])
                    minVal=min([i for i in self.ax])
                    self.s10.set_xlim(minVal, maxValX)
              
          if isinstance(self.ax1, list):            
                    maxVal=max([i for i in self.ax1])
                    minVal=min([i for i in self.ax1])
                    self.s11.set_xlim(minVal, maxVal)
          if isinstance(self.ay, list):            
                    maxVal=max([i for i in self.ay])
                    minVal=min([i for i in self.ay])
                    self.s10.set_ylim(minVal, maxVal)
          if isinstance(self.ay1, list):            
                    maxVal=max([i for i in self.ay1])
                    minVal=min([i for i in self.ay1])
                    self.s11.set_ylim(minVal, maxVal)
          self.a0.set_data(self.ax, self.ay)
          self.a1.set_data(self.ax1, self.ay1)

      except KeyboardInterrupt:
          print('exiting')
      
      return self.a0,self.a1

  # clean up
  def close(self):
      # close serial
      self.ser.flush()
      self.ser.close()
      self.f0.close() 
      self.f1.close()   

def _blit_draw(self, artists, bg_cache):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        # ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)

# main() function
def main():
  # create parser
  parser = argparse.ArgumentParser(description="LDR serial")
  # add expected arguments
  parser.add_argument('--port', dest='port', required=True)
  # parse args
  args = parser.parse_args()
  #strPort = '/dev/tty.usbserial-A7006Yqh'
  strPort = args.port
  print('reading from serial port %s...' % strPort)
  # clean com-port
  try:
      ser1 = serial.Serial(strPort, 1200)
      ser1.setDTR(False) # Drop DTR
      sleep(0.01)
      ser1.flush()
      ser1.setDTR(True)  # UP the DTR back
      ser1.close()
      sleep(0.5)
  except:
      print "No port"
  print('plotting data...')
  fig = plt.figure()
  s0 = fig.add_subplot(211)
  s1 = fig.add_subplot(212)
  s0.set_title("0 electrodes")
  s1.set_title("1 electrodes")
  s0.set_ylim(200, 600)
  s1.set_ylim(200, 600)
  s0.set_xlim(0, 1)
  s1.set_xlim(0, 1)
  s1.set_xlabel("time, s")
  s0.set_ylabel("Output voltage, channels")
  s1.set_ylabel("Output voltage, channels")
  animation.Animation._blit_draw = _blit_draw
  try:
       analogPlot = AnalogPlot(strPort, 57600,s0,s1)
       
       anim = animation.FuncAnimation(fig, analogPlot.update, interval=1,repeat=False)
      # show plot
       plt.show()
      # clean up
       analogPlot.close()
  except:
       sys.exit

  print('exiting.')
  

# call main
if __name__ == '__main__':
  main()
