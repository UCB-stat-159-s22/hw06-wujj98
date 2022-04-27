import matplotlib
matplotlib.use('Agg')
import numpy as np
import ligotools.readligo as rl
import ligotools.utils as ut
from scipy.interpolate import interp1d
import matplotlib.mlab as mlab
from os.path import exists
from os import remove
import h5py
from scipy import signal
from scipy.signal import filtfilt

def test_whiten():
  strain_H1 = rl.loaddata("data/H-H1_LOSC_4_V2-1126259446-32.hdf5")[0]
  time_H1 = rl.loaddata("data/H-H1_LOSC_4_V2-1126259446-32.hdf5")[1]
  Pxx_H1, freqs = mlab.psd(strain_H1, Fs = 4096, NFFT = 4*4096)
  psd_H1 = interp1d(freqs, Pxx_H1)
  dt = time_H1[1] - time_H1[0]
  strain_H1_whiten = ut.whiten(strain_H1,psd_H1,dt)
  assert len(strain_H1_whiten) == 131072
  
def test_write_wavfile():
  ut.write_wavfile("audio/test.wav", 10, np.array([1,0,-1,-1.3,-1.9,-0.5]))
  assert exists("audio/test.wav")
  remove("audio/test.wav")

def test_reqshift():
  assert np.sum(ut.reqshift(np.arange(0,100,0.5), 400, 4096)) == -2.842170943040401e-14
  
def test_plotting():
  time = rl.loaddata("data/H-H1_LOSC_4_V2-1126259446-32.hdf5")[1]
  fs = 4096
  NFFT = 4*fs
  psd_window = np.blackman(NFFT)
  NOVL = NFFT/2
  fn_template = 'GW150914_4_template.hdf5'
  f_template = h5py.File('data/'+fn_template, "r")
  template_p, template_c = f_template["template"][...]
  template = (template_p + template_c*1.j) 
  etime = time+16
  datafreq = np.fft.fftfreq(template.size)*fs
  df = np.abs(datafreq[1] - datafreq[0])
  try:   dwindow = signal.tukey(template.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
  except: dwindow = signal.blackman(template.size)          # Blackman window OK if Tukey is not available
  template_fft = np.fft.fft(template*dwindow) / fs
  data = rl.loaddata("data/H-H1_LOSC_4_V2-1126259446-32.hdf5")[0]
  data_psd, freqs = mlab.psd(data, Fs = 4096, NFFT = 4*4096, window=psd_window, noverlap=NOVL)
  data_fft = np.fft.fft(data*dwindow) / fs
  power_vec = np.interp(np.abs(datafreq), freqs, data_psd)
  optimal = data_fft * template_fft.conjugate() / power_vec
  optimal_time = 2*np.fft.ifft(optimal)*fs
  sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
  sigma = np.sqrt(np.abs(sigmasq))
  SNR_complex = optimal_time/sigma
  peaksample = int(data.size / 2)  # location of peak in the template
  SNR_complex = np.roll(SNR_complex,peaksample)
  SNR = abs(SNR_complex)
  indmax = np.argmax(SNR)
  timemax = time[indmax]
  SNRmax = SNR[indmax]
  d_eff = sigma / SNRmax
  horizon = sigma/8
  phase = np.angle(SNR_complex[indmax])
  offset = (indmax-peaksample)
  template_phaseshifted = np.real(template*np.exp(1j*phase))    # phase shift the template
  template_rolled = np.roll(template_phaseshifted,offset) / d_eff  # Apply time offset and scale amplitude
  template_whitened = ut.whiten(template_rolled,interp1d(freqs, data_psd),0.000244140625)  # whiten the template
  bb = np.array([0.00094662, 0, -0.00378646, 0, 0.00567969, 0, -0.00378646, 0, 0.00094662])
  ab = np.array([1, -6.86591878, 20.74750835, -36.05684101, 39.43035496, -27.79002252, 12.32845652, -3.14765894, 0.35412193])
  template_match = filtfilt(bb, ab, template_whitened) / 0.3542432515235823 
  pcolor='r'
  strain_H1 = rl.loaddata("data/H-H1_LOSC_4_V2-1126259446-32.hdf5")[0]
  time_H1 = rl.loaddata("data/H-H1_LOSC_4_V2-1126259446-32.hdf5")[1]
  Pxx_H1, freqs = mlab.psd(strain_H1, Fs = 4096, NFFT = 4*4096)
  psd_H1 = interp1d(freqs, Pxx_H1)
  dt = time_H1[1] - time_H1[0]
  strain_H1_whiten = ut.whiten(strain_H1,psd_H1,dt)
  strain_whitenbp = filtfilt(bb, ab, strain_H1_whiten) / 0.3542432515235823 
  template_H1 = template_match.copy()
  det = 'H1'
  eventname = 'GW150914'
  plottype = 'png'
  tevent = 1126259462.44
  ut.plotting(time, timemax, SNR, pcolor, det, eventname, plottype,
  tevent, strain_whitenbp, template_match, template_fft, datafreq,
  d_eff, freqs, data_psd, fs)
  assert exists("figures/" + eventname+"_"+det+"_matchfreq."+plottype)
  assert exists("figures/" + eventname+"_"+det+"_matchtime."+plottype)
  assert exists("figures/" + eventname+"_"+det+"_SNR."+plottype)
  remove("figures/" + eventname+"_"+det+"_matchfreq."+plottype)
  remove("figures/" + eventname+"_"+det+"_matchtime."+plottype)
  remove("figures/" + eventname+"_"+det+"_SNR."+plottype)