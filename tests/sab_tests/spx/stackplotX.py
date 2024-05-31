import gc
import multiprocessing as mp
import os
import pickle
import time
from copy import deepcopy
from functools import partial

import astropy.units as u
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import sunpy
from astropy.time import Time
from matplotlib.widgets import Slider, Button
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import pdb
from packaging import version as pversion
from scipy import signal
# from astropy import units as u
# import sunpy.map as smap
from scipy.interpolate import griddata
from tqdm import *

import DButil
import signal_utils as su
import stputils as stpu
from lineticks import LineTicks

# check sunpy version
sunpy1 = sunpy.version.major >= 1
if sunpy1:
    pass
else:
    pass

import sunpy.map

from sunpy.map.mapsequence import MapSequence

if pversion.parse(sunpy.version.version) >= pversion.parse('4.0'):
    from sunkit_image.coalignment import mapsequence_coalign_by_rotation
else:
    # deprecated in sunpy 4
    pass


if pversion.parse(sunpy.version.version) >= pversion.parse('2.1'):
    from aiapy.calibrate import register, update_pointing

    def aiaprep(sunpymap):
        m_updated_pointing = update_pointing(sunpymap)
        m_registered = register(m_updated_pointing)
        return m_registered
else:
    from sunpy.instr.aia import aiaprep


# warnings.filterwarnings('ignore')


def resettable(f):
    import copy

    def __init_and_copy__(self, *args, **kwargs):
        f(self, *args)
        self.__original_dict__ = copy.deepcopy(self.__dict__)

        def reset(o=self):
            o.__dict__ = o.__original_dict__

        self.reset = reset

    return __init_and_copy__


def b_filter(data, lowcut, highcut, fs, ix):
    x = data[ix]
    # y = butter_bandpass_filter(x, lowcut * fs, highcut * fs, fs, order=5)
    y = su.bandpass_filter(None, x, fs=fs, cutoff=[lowcut, highcut]) + 1.0
    return {'idx': ix, 'y': y}


def runningmean(data, window, mode, ix):
    '''

    :param data:
    :param window:
    :param ix:
    :param mode: available options are ratio and diff
    :return:
    '''
    x = data[ix]
    if mode.endswith('ratio'):
        return {'idx': ix, 'y': su.smooth(x, window[0]) / su.smooth(x, window[1])}
    elif mode.endswith('diff'):
        return {'idx': ix, 'y': su.smooth(x, window[0]) - su.smooth(x, window[1])}
    else:
        return {'idx': ix, 'y': su.smooth(x, window[0]) / su.smooth(x, window[1])}


def c_correlate(a, v, returnx=False):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) / np.std(v)
    if returnx:
        return np.arange(len(a)) - np.floor(len(a) / 2.0), np.correlate(a, v, mode='same')
    else:
        return np.correlate(a, v, mode='same')


def XCorrMap(data, refpix=[0, 0]):
    from tqdm import tqdm
    ny, nx, nt = data.shape
    ccmax = np.empty((ny, nx))
    ccmax[:] = np.nan
    ccpeak = np.empty((ny, nx))
    ccpeak[:] = np.nan

    lc_ref = data[refpix[0], refpix[1], :]
    for xidx in tqdm(range(0, ny - 1)):
        for yidx in range(0, nx - 1):
            lc = data[xidx, yidx, :]
            ccval = c_correlate(lc, lc_ref)
            if sum(lc) != 0:
                cmax = np.nanmax(ccval)
                cpeak = np.nanargmax(ccval) - (nt - 1) / 2
            else:
                cmax = 0
                cpeak = 0
            ccmax[xidx, yidx - 1] = cmax
            ccpeak[xidx, yidx - 1] = cpeak

    return {'ny': ny, 'nx': nx, 'nt': nt, 'ccmax': ccmax, 'ccpeak': ccpeak}


def XCorrStackplt(z, x, y, doxscale=True):
    '''
    get the cross correlation map along y axis
    :param z: data
    :param x: x axis
    :param y: y axis
    :return:
    '''
    from tqdm import tqdm
    from scipy.interpolate import splev, splrep
    if doxscale:
        xfit = np.linspace(x[0], x[-1], 10 * len(x) + 1)
        zfit = np.zeros((len(y), len(xfit)))
        for yidx1, yq in enumerate(y):
            xx = x
            yy = z[yidx1, :]
            s = len(yy)  # (len(yy) - np.sqrt(2 * len(yy)))*2
            tck = splrep(xx, yy, s=s)
            ys = splev(xfit, tck)
            zfit[yidx1, :] = ys
    else:
        xfit = x
        zfit = z
    ny, nxfit = zfit.shape
    ccpeak = np.empty((ny - 1, ny - 1))
    ccpeak[:] = np.nan
    ccmax = ccpeak.copy()
    ya = ccpeak.copy()
    yv = ccpeak.copy()
    yidxa = ccpeak.copy()
    yidxv = ccpeak.copy()
    for idx1 in tqdm(range(1, ny)):
        for idx2 in range(0, idx1):
            lightcurve1 = zfit[idx1, :]
            lightcurve2 = zfit[idx2, :]
            ccval = c_correlate(lightcurve1, lightcurve2)
            if sum(lightcurve1) != 0 and sum(lightcurve2) != 0:
                cmax = np.amax(ccval)
                cpeak = np.argmax(ccval) - (nxfit - 1) / 2
            else:
                cmax = 0
                cpeak = 0
            ccmax[idx2, idx1 - 1] = cmax
            ccpeak[idx2, idx1 - 1] = cpeak
            ya[idx2, idx1 - 1] = y[idx1 - 1]
            yv[idx2, idx1 - 1] = y[idx2]
            yidxa[idx2, idx1 - 1] = idx1 - 1
            yidxv[idx2, idx1 - 1] = idx2
            # if idx1 - 1 != idx2:
            #     ccmax[idx1 - 1, idx2] = cmax
            #     ccpeak[idx1 - 1, idx2] = cpeak
            #     ya[idx1 - 1, idx2] = y[idx2]
            #     yv[idx1 - 1, idx2] = y[idx1 - 1]
            #     yidxa[idx1 - 1, idx2] = idx2
            #     yidxv[idx1 - 1, idx2] = idx1 - 1

    return {'zfit': zfit, 'ccmax': ccmax, 'ccpeak': ccpeak, 'x': x, 'nx': len(x), 'xfit': xfit, 'nxfit': nxfit, 'y': y,
            'ny': ny, 'yv': yv, 'ya': ya,
            'yidxv': yidxv, 'yidxa': yidxa}


def FitSlit(xx, yy, cutwidth, cutang, cutlength, s=None, method='Polyfit', ascending=False):
    if len(xx) <= 3 or method == 'Polyfit':
        '''polynomial fit'''
        out = stpu.polyfit(xx, yy, cutlength, len(xx) - 1 if len(xx) <= 3 else 2, keepxorder=True)
        xs, ys, posangs = out['xs'], out['ys'], out['posangs']
    else:
        if method == 'Param_Spline':
            '''parametic spline fit'''
            out = stpu.paramspline(xx, yy, cutlength, s=s)
            xs, ys, posangs = out['xs'], out['ys'], out['posangs']
        else:
            '''spline fit'''
            out = stpu.spline(xx, yy, cutlength, s=s)
            xs, ys, posangs = out['xs'], out['ys'], out['posangs']
    if ascending and (method != 'Param_Spline' or len(xx) <= 3):
        if xs[-1] < xs[0]:
            xs, ys = xs[::-1], ys[::-1]
            posangs = posangs[::-1]
    dist = stpu.findDist(xs, ys)
    dists = np.cumsum(dist)
    posangs2 = posangs + np.pi / 2
    cutwidths = dists * np.tan(cutang) + cutwidth
    xs0 = xs - cutwidths / 2. * np.cos(posangs2)
    ys0 = ys - cutwidths / 2. * np.sin(posangs2)
    xs1 = xs + cutwidths / 2. * np.cos(posangs2)
    ys1 = ys + cutwidths / 2. * np.sin(posangs2)
    return {'xcen': xs, 'ycen': ys, 'xs0': xs0, 'ys0': ys0, 'xs1': xs1, 'ys1': ys1, 'cutwidth': cutwidths,
            'posangs': posangs, 'posangs2': posangs2,
            'dist': dists}


def MakeSlit(pointDF):
    pointDFtmp = pointDF
    xx = pointDFtmp.loc[:, 'xx'].values
    yy = pointDFtmp.loc[:, 'yy'].values
    if len(pointDFtmp.index) <= 1:
        cutslitplt = {'xcen': [], 'ycen': [], 'xs0': [], 'ys0': [], 'xs1': [], 'ys1': [], 'cutwidth': [], 'posangs': [],
                      'posangs2': [], 'dist': []}
    else:
        # if len(pointDFtmp.index) <= 3:
        cutslitplt = FitSlit(xx, yy, 10, 0.0, 200, method='Polyfit')
    return cutslitplt


def getimprofile(data, cutslit, xrange=None, yrange=None, get_peak=False, verbose=False):
    """
    Get values at a slice

    Inputs:
        data: input image data. Dimension: (ny, nx) or (ny, nx, nwv). nwv is the number of wavelengths/frequencies
        cutslit: cutslit generated from CutslitBuilder().cutslitplt
        xrange: [min(xs), max(xs)], where xs is the x coordinate values of the input image data.
                If None (default), assume pixel coordinate values in cutslit
        yrange: [min(ys), max(ys)], where ys is the y coordinate values of the input image data.
                If None (default), assume pixel coordinate values in cutslit
        get_peak: If True, return the peak of all pixels across the slit within the slit width.
                  If False (default), return the average value.
        verbose: If True, print out more details in command line. Default is False

    return:
        A dictionary of {'x': distance from min(cutslit['xcen']), min(cutslit['ycen'])
                         'y': value on the cut, the shape is (len(cutslit['xcen'], [nwv])}
    """
    # first, check the dimension of the input image data
    if data.ndim == 2:
        ndy, ndx = data.shape
        intens = np.zeros(len(cutslit['xcen']))
        if verbose:
            print("Input data cube is 2D, the dimension (ny, nx) is ({0:d}, {1:d})".format(ndy, ndx))
    elif data.ndim == 3:
        ndy, ndx, nwv = data.shape
        intens = np.zeros((len(cutslit['xcen']), nwv))
        if verbose:
            print("Input data cube is 3D, the dimension (ny, nx, nwv) is ({0:d}, {1:d}, {2:d})".format(ndy, ndx, nwv))

    num = len(cutslit['xcen'])
    if num < 2:
        print("The slice should have at least two anchoring points! Return -1")
        return -1
    else:
        if xrange is not None and yrange is not None:
            xs0 = (cutslit['xs0'] - xrange[0]) / (xrange[1] - xrange[0]) * ndx
            xs1 = (cutslit['xs1'] - xrange[0]) / (xrange[1] - xrange[0]) * ndx
            ys0 = (cutslit['ys0'] - yrange[0]) / (yrange[1] - yrange[0]) * ndy
            ys1 = (cutslit['ys1'] - yrange[0]) / (yrange[1] - yrange[0]) * ndy
        else:
            xs0 = cutslit['xs0']
            xs1 = cutslit['xs1']
            ys0 = cutslit['ys0']
            ys1 = cutslit['ys1']
        for ll in range(num):
            inten = stpu.improfile(data, [xs0[ll], xs1[ll]], [ys0[ll], ys1[ll]], interp='nearest')
            try:
                if get_peak:
                    intens[ll] = np.nanmax(inten, axis=0)
                else:
                    intens[ll] = np.nanmean(inten, axis=0)
            except:
                intens[ll] = np.nan
        intensdist = {'x': cutslit['dist'], 'y': intens}
        return intensdist


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='same')
    y = y[window_len - 1:-(window_len - 1)]
    return y


def grid(x, y, z, resX=20, resY=40):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(np.nanmin(x), np.nanmax(x), resX)
    yi = np.linspace(np.nanmin(y), np.nanmax(y), resY)
    Z = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z


def polyfit(x, y, length, deg):
    xs = np.linspace(x.min(), x.max(), length)
    z = np.polyfit(x=x, y=y, deg=deg)
    p = np.poly1d(z)
    ys = p(xs)
    rms = np.sqrt(np.sum((np.polyval(z, x) - y) ** 2) / len(x))
    return {'xs': xs, 'ys': ys, 'rms': rms}


class LightCurveBuilder:
    def __init__(self, stackplt, axes, scale=1.0, color='white'):
        self.stackplt = stackplt
        self.axes_dspec = axes
        self.lghtcurvline, = self.axes_dspec.plot([], [], color=color, ls='-', lw=1.0)
        self.lghtcurvlines = []
        self.clickedpoints, = self.axes_dspec.plot([], [], '+', color=color)
        self.scale = scale
        self.color = color
        self.xx = list(self.clickedpoints.get_xdata())
        self.yy = list(self.clickedpoints.get_ydata())
        self.cid = self.clickedpoints.figure.canvas.mpl_connect('button_press_event', self)
        ax_dspec_pos = self.axes_dspec.get_position().extents
        wbut = (ax_dspec_pos[2] - ax_dspec_pos[0]) * 0.5 / 4
        y0but = ax_dspec_pos[3] + 0.01
        hbut = 0.03
        self.bcut_curvsav = Button(plt.axes([ax_dspec_pos[2] - wbut, y0but, wbut, hbut]), 'Save')
        self.bcut_curvdel = Button(plt.axes([ax_dspec_pos[2] - wbut * 2.1, y0but, wbut, hbut]), 'Delete')
        self.lightcurves = []
        self.clickedpoints_text = []
        self.vmax = np.nanmax(self.stackplt['dspec'])
        self.vmin = np.nanmin(self.stackplt['dspec'])
        self.ymax = np.nanmax(self.stackplt['y'])
        self.ymin = np.nanmin(self.stackplt['y'])
        self.bcut_curvsav.on_clicked(self.save)
        self.bcut_curvdel.on_clicked(self.delete)

    def __call__(self, event):
        tmode = '{}'.format(self.clickedpoints.figure.canvas.toolbar.mode)
        if tmode == '':
            if event.inaxes != self.axes_dspec:
                return
            if event.button == 1:
                if len(self.xx) == 0:
                    self.xx.append(event.xdata)
                    self.yy.append(event.ydata)
                else:
                    self.xx = [event.xdata]
                    self.yy = [event.ydata]
            elif event.button == 3:
                if len(self.xx) > 0:
                    self.xx.pop()
                    self.yy.pop()
            self.clickedpoints.set_data(self.xx, self.yy)
            self.clickedpoints.figure.canvas.draw()
            self.update()
        else:
            if event.inaxes != self.axes_dspec:
                return
            if event.button == 1 or event.button == 3:
                self.clickedpoints.figure.canvas.toolbar.set_message('Uncheck toolbar button {} first!'.format(tmode))

    def update(self, mask=None):
        xx = np.array(self.xx, dtype=np.float64)
        yy = np.array(self.yy, dtype=np.float64)
        lightcurveplt = {'x': [], 'flux': [], 'fluxscale': None}
        if len(self.xx) > 0:
            lightcurveplt['x'] = (self.stackplt['x'][1:] + self.stackplt['x'][:-1]) / 2.0
            xdiff = np.abs(lightcurveplt['x'] - xx[0])
            xidx, = np.where(xdiff == np.nanmin(xdiff))
            ddist = np.abs(self.stackplt['y'] - yy[0])
            yidx, = np.where(ddist == np.nanmin(ddist))
            lightcurveplt['flux'] = ma.masked_invalid(self.stackplt['dspec'][yidx[0], :])
            lightcurveplt['fluxscale'] = lambda x: (x - x[xidx[0]]) / (self.vmax - self.vmin) * (
                    self.ymax - self.ymin) * 0.3 + self.stackplt['y'][
                                                       yidx[0]]
            x = lightcurveplt['x']
            flux = lightcurveplt['fluxscale'](lightcurveplt['flux'])
        else:
            x = []
            flux = []

        self.lightcurveplt = lightcurveplt

        self.lghtcurvline.set_data(x, flux)
        self.lghtcurvline.figure.canvas.draw()

    def save(self, event):
        if len(self.lightcurveplt['x']) > 0:
            if len(self.lightcurves) > 0:
                x0s = [lc['lghtcurv']['x'][0] for lc in self.lightcurves]
                if self.lightcurveplt['x'][0] >= x0s[-1]:
                    indx = len(self.lightcurves)
                else:
                    indx = next(x[0] for x in enumerate(x0s) if x[1] > self.lightcurveplt['x'][0])
            else:
                indx = len(self.lightcurves)
            self.lightcurves.insert(indx, {'x': self.clickedpoints.get_xdata(), 'y': self.clickedpoints.get_ydata(),
                                           'lghtcurv': self.lightcurveplt})
            self.xx = []
            self.yy = []
            self.clickedpoints.set_data(self.xx, self.yy)
            lghtcurvline, = self.axes_dspec.plot(self.lightcurves[indx]['lghtcurv']['x'],
                                                 self.lightcurves[indx]['lghtcurv']['fluxscale'](
                                                     self.lightcurves[indx]['lghtcurv']['flux']),
                                                 color=self.color, ls='-', lw=0.5)
            self.lghtcurvlines.insert(indx, lghtcurvline)
            self.update()
            # self.update_text()
            self.clickedpoints.figure.canvas.draw()

    def delete(self, event):
        if len(self.lghtcurvlines) > 0:
            self.lghtcurvlines[-1].remove()
            self.lghtcurvlines.pop()
            self.lightcurves.pop()
            self.clickedpoints.figure.canvas.draw()

    def delete_byindex(self, index):
        self.lghtcurvlines[index].remove()
        del self.lghtcurvlines[index]
        del self.lightcurves[index]
        # self.update_text()
        self.clickedpoints.figure.canvas.draw()

    def update_text(self):
        for tx in self.clickedpoints_text:
            tx.remove()
        self.clickedpoints_text = []
        for idx, lc in enumerate(self.lightcurves):
            text = self.axes_dspec.text(lc['lghtcurv']['x'][-1],
                                        lc['lghtcurv']['fluxscale'](lc['lghtcurv']['flux'][-1]),
                                        '{}'.format(idx), color=self.color, transform=self.axes_dspec.transData,
                                        ha='left', va='bottom')
            self.clickedpoints_text.append(text)

    def lightcurves_tofile(self, outfile=None, lightcurves=None):
        if not lightcurves:
            lightcurves = self.lightcurves
        with open('{}'.format(outfile), 'wb') as sf:
            pickle.dump(lightcurves, sf)

    def lightcurves_fromfile(self, infile, color=None):
        if color is not None:
            self.color = color
        with open('{}'.format(infile), 'rb') as sf:
            self.lightcurves = pickle.load(sf, encoding='latin1')
        for sl in self.lghtcurvlines:
            sl.remove()
        self.lghtcurvlines = []

        for tx in self.clickedpoints_text:
            tx.remove()
        self.clickedpoints_text = []
        for idx, lc in enumerate(self.lightcurves):
            lghtcurvline, = self.axes_dspec.plot(lc['lghtcurv']['x'],
                                                 lc['lghtcurv']['fluxscale'](lc['lghtcurv']['flux']),
                                                 color=self.color, ls='-', lw=0.5)
            self.lghtcurvlines.append(lghtcurvline)


class SpaceTimeSlitBuilder:
    def __init__(self, axes, dspec, cutlength=80, cutsmooth=10.0, scale=1.0, color='white'):
        if isinstance(axes, list):
            self.axes_dspec = axes[0]
            naxes = len(axes)
            if naxes >= 2:
                self.axes_speed = axes[1]
                self.speedline, = self.axes_speed.plot([], [], color=color, ls='-')
                self.speedlines = []
            else:
                self.axes_speed = None
            if naxes >= 3:
                self.axes_accel = axes[2]
                self.accelline, = self.axes_accel.plot([], [], color=color, ls='-')
                self.accellines = []
            else:
                self.axes_accel = None
        else:
            self.axes_dspec = axes
            self.axes_speed = None
            self.axes_accel = None
            self.speedlines = []
            self.accellines = []
        self.dspec = dspec
        self.clickedpoints, = self.axes_dspec.plot([], [], 'o', color=color)
        self.slitline, = self.axes_dspec.plot([], [], color=color, ls=':')
        self.cutlength = cutlength
        self.scale = scale
        self.color = color
        self.xx = list(self.clickedpoints.get_xdata())
        self.yy = list(self.clickedpoints.get_ydata())
        self.cid = self.clickedpoints.figure.canvas.mpl_connect('button_press_event', self)
        ax_dspec_pos = self.axes_dspec.get_position().extents
        wbut = (ax_dspec_pos[2] - ax_dspec_pos[0]) * 0.5 / 4
        y0but = ax_dspec_pos[3] + 0.01
        hbut = 0.03
        self.bcut_cutsav = Button(plt.axes([ax_dspec_pos[2] - wbut, y0but, wbut, hbut]), 'Save')
        self.bcut_cutdel = Button(plt.axes([ax_dspec_pos[2] - wbut * 2.1, y0but, wbut, hbut]), 'Delete')
        self.cutsmooth = cutsmooth
        self.spacetimeslits = []
        self.slitlines = []
        self.slitlines_text = []

        self.bcut_cutsav.on_clicked(self.save)
        self.bcut_cutdel.on_clicked(self.delete)

    def __call__(self, event):
        tmode = '{}'.format(self.clickedpoints.figure.canvas.toolbar.mode)
        if tmode == '':
            if event.inaxes != self.axes_dspec:
                return
            if event.button == 1:
                self.xx.append(event.xdata)
                self.yy.append(event.ydata)
            elif event.button == 3:
                if len(self.xx) > 0:
                    self.xx.pop()
                    self.yy.pop()
            self.clickedpoints.set_data(self.xx, self.yy)
            self.clickedpoints.figure.canvas.draw()
            self.update()
            if event.button == 2:
                self.xx.append(event.xdata)
                self.select_distance_along_a_slice(int(self.xx[-1]))
        else:
            if event.inaxes != self.axes_dspec:
                return
            if event.button == 1 or event.button == 3:
                self.clickedpoints.figure.canvas.toolbar.set_message('Uncheck toolbar button {} first!'.format(tmode))

    def FitSlit(self, xx, yy, cutlength, method='Polyfit', s=0, ascending=True):
        '''polynomial fit'''
        # xs = np.linspace(np.nanmin(xx), np.nanmax(xx), cutlength)
        # z = np.polyfit(x=xx, y=yy, deg=3)
        # p = np.poly1d(z)
        # ys = p(xs)
        if len(xx) <= 3 or method == 'Polyfit':
            '''polynomial fit'''
            out = stpu.polyfit(xx, yy, cutlength, len(xx) - 1 if len(xx) <= 3 else 2, keepxorder=True)
            xs, ys, posangs = out['xs'], out['ys'], out['posangs']
        else:
            if method == 'Param_Spline':
                '''parametic spline fit'''
                out = stpu.paramspline(xx, yy, cutlength, s=s)
                xs, ys, posangs = out['xs'], out['ys'], out['posangs']
            else:
                '''spline fit'''
                out = stpu.spline(xx, yy, cutlength, s=s)
                xs, ys, posangs = out['xs'], out['ys'], out['posangs']

        if not ascending and (method != 'Param_Spline' or len(xx) <= 3):
            xs, ys = xs[::-1], ys[::-1]
        dist = stpu.findDist(xs, ys)
        dists = np.cumsum(dist)
        return {'xcen': xs, 'ycen': ys, 'dist': dists}

    def update(self, mask=None):
        xx = np.array(self.xx, dtype=np.float64)
        yy = np.array(self.yy, dtype=np.float64)

        if len(self.xx) <= 1:
            spacetimeslitplt = {'xcen': [], 'ycen': [], 'dist': [], 'speed': [], 'accel': []}
        else:
            if len(self.xx) <= 3:
                spacetimeslitplt = self.FitSlit(xx, yy, self.cutlength, method='Polyfit')
            else:
                spacetimeslitplt = self.FitSlit(xx, yy, self.cutlength, s=len(xx) / 100.0 * self.cutsmooth,
                                                method='Param_Spline')
        self.spacetimeslitplt = spacetimeslitplt
        if len(self.xx) >= 2:
            edge_order = 1
            if len(self.xx) >= 3:
                edge_order = 2
            if self.axes_speed:
                spacetimeslitplt['speed'] = np.gradient(spacetimeslitplt['ycen'], spacetimeslitplt['xcen'],
                                                        edge_order=edge_order)
            if self.axes_accel:
                spacetimeslitplt['accel'] = np.gradient(spacetimeslitplt['speed'], spacetimeslitplt['xcen'],
                                                        edge_order=edge_order)
        else:
            spacetimeslitplt['speed'] = []
            spacetimeslitplt['accel'] = []
        if mask is None:
            x = spacetimeslitplt['xcen']
            dist = spacetimeslitplt['ycen']
            if self.axes_speed: speed = spacetimeslitplt['speed']
            if self.axes_accel: accel = spacetimeslitplt['accel']
        else:
            x = ma.masked_array(spacetimeslitplt['xcen'], mask)
            dist = ma.masked_array(spacetimeslitplt['ycen'], mask)
            if self.axes_speed: speed = ma.masked_array(spacetimeslitplt['speed'], mask)
            if self.axes_accel: accel = ma.masked_array(spacetimeslitplt['accel'], mask)
        self.slitline.set_data(x, dist)
        if self.axes_speed:
            self.speedline.set_data(x, speed)
            if len(speed) >= 2:
                vmax = np.nanmax(speed)
                vmin = np.nanmin(speed)
                for ll in self.spacetimeslits:
                    vmax_ = np.nanmax(ll['cutslit']['speed'])
                    if vmax < vmax_:
                        vmax = vmax_
                    vmin_ = np.nanmin(ll['cutslit']['speed'])
                    if vmin > vmin_:
                        vmin = vmin_
                vmargin = (vmax - vmin) * 0.1
                self.axes_speed.set_ylim(vmin - vmargin, vmax + vmargin)
        if self.axes_accel:
            self.accelline.set_data(x, accel)
            if len(accel) >= 2:
                vmax = np.nanmax(accel)
                vmin = np.nanmin(accel)
                for ll in self.spacetimeslits:
                    vmax_ = np.nanmax(ll['cutslit']['accel'])
                    if vmax < vmax_:
                        vmax = vmax_
                    vmin_ = np.nanmin(ll['cutslit']['accel'])
                    if vmin > vmin_:
                        vmin = vmin_
                vmargin = (vmax - vmin) * 0.1
                self.axes_accel.set_ylim(vmin - vmargin, vmax + vmargin)
        self.slitline.figure.canvas.draw()

    def save(self, event):
        if len(self.spacetimeslitplt['xcen']) > 0:
            if len(self.spacetimeslits) > 1:
                x0s = [cut['cutslit']['xcen'][0] for cut in self.spacetimeslits]

                if self.spacetimeslitplt['xcen'][0] >= x0s[-1]:
                    indx = len(self.spacetimeslits)
                else:
                    indx = next(x[0] for x in enumerate(x0s) if x[1] > self.spacetimeslitplt['xcen'][0])
            else:
                indx = len(self.spacetimeslits)
            self.spacetimeslits.insert(indx, {'x': self.clickedpoints.get_xdata(), 'y': self.clickedpoints.get_ydata(),
                                              'cutslit': self.spacetimeslitplt})
            self.xx = []
            self.yy = []
            self.clickedpoints.set_data(self.xx, self.yy)
            slitline, = self.axes_dspec.plot(self.spacetimeslits[indx]['cutslit']['xcen'],
                                             self.spacetimeslits[indx]['cutslit']['ycen'],
                                             color=self.color, ls=':')
            self.slitlines.insert(indx, slitline)
            if self.axes_speed:
                speedline, = self.axes_speed.plot(self.spacetimeslits[indx]['cutslit']['xcen'],
                                                  self.spacetimeslits[indx]['cutslit']['speed'], color=self.color,
                                                  ls='-')
                self.speedlines.insert(indx, speedline)
            if self.axes_accel:
                accelline, = self.axes_accel.plot(self.spacetimeslits[indx]['cutslit']['xcen'],
                                                  self.spacetimeslits[indx]['cutslit']['accel'], color=self.color,
                                                  ls='-')
                self.accellines.insert(indx, accelline)
            self.update()
            self.update_text()
            self.clickedpoints.figure.canvas.draw()

    def delete(self, event):
        if len(self.slitlines) > 0:
            self.slitlines[-1].remove()
            self.slitlines_text[-1].remove()
            self.slitlines.pop()
            self.slitlines_text.pop()
            self.spacetimeslits.pop()
            if self.axes_speed:
                self.speedlines[-1].remove()
                self.speedlines.pop()
            if self.axes_accel:
                self.accellines[-1].remove()
                self.accellines.pop()
            self.clickedpoints.figure.canvas.draw()

    def delete_byindex(self, index):
        self.slitlines[index].remove()
        del self.slitlines[index]
        del self.spacetimeslits[index]
        if self.axes_speed:
            del self.speedlines[index]
        if self.axes_accel:
            del self.accellines[index]
        self.update_text()
        self.clickedpoints.figure.canvas.draw()

    def update_text(self):
        for tx in self.slitlines_text:
            tx.remove()
        self.slitlines_text = []
        for idx, cut in enumerate(self.spacetimeslits):
            text = self.axes_dspec.text(cut['cutslit']['xcen'][-1], cut['cutslit']['ycen'][-1],
                                        '{}'.format(idx), color=self.color, transform=self.axes_dspec.transData,
                                        ha='left', va='bottom')
            self.slitlines_text.append(text)

    def select_distance_along_a_slice(self, ixx):
        'select points more accurately by doing it on a distance-flux plot'
        cur_time = self.dspec['x'][0] + (ixx/3600./24.)
        cur_idx = np.argmin(np.abs(self.dspec['x'] - cur_time))
        fig_sdas, axes_sdas = plt.subplots(nrows=1, ncols=1)
        axes_sdas.plot(self.dspec['y'][1:], self.dspec['dspec'][:,cur_idx])
        axes_sdas.set_xlabel(self.dspec['ytitle'])
        axes_sdas.set_ylabel('Flux')
        axes_sdas.set_title('Close to save the last choice')
        tmp_selected_distance_list=[]

        def sdas_on_click(event):
            if event.button == 1:
                tmp_selected_distance_list.append(event.xdata)
                axes_sdas.axvspan(tmp_selected_distance_list[-1],tmp_selected_distance_list[-1]+1,facecolor='r',edgecolor='r')
                axes_sdas.figure.canvas.draw()
            else:
                print('Only left clicking is supported')

        def sdas_close_figure(event):
            if len(tmp_selected_distance_list)> 1.e-9:
                self.yy.append(tmp_selected_distance_list[-1])
                self.clickedpoints.set_data(self.xx, self.yy)
                self.clickedpoints.figure.canvas.draw()
                self.update()
        plt.connect('button_press_event', sdas_on_click)
        plt.connect('close_event', sdas_close_figure)

    def spacetimeslits_tofile(self, outfile=None, spacetimeslits=None):
        if not spacetimeslits:
            spacetimeslits = self.spacetimeslits
        with open('{}'.format(outfile), 'wb') as sf:
            pickle.dump(spacetimeslits, sf)

    def spacetimeslits_fromfile(self, infile, color=None):
        if color is not None:
            self.color = color
        with open('{}'.format(infile), 'rb') as sf:
            self.spacetimeslits = pickle.load(sf, encoding='latin1')
        for tx in self.slitlines_text:
            tx.remove()
        self.slitlines_text = []
        for sl in self.slitlines:
            sl.remove()
        self.slitlines = []
        self.speedlines = []
        self.accellines = []
        for idx, cut in enumerate(self.spacetimeslits):
            text = self.axes_dspec.text(cut['cutslit']['xcen'][-1], cut['cutslit']['ycen'][-1],
                                        '{}'.format(idx), color=self.color, transform=self.axes_dspec.transData,
                                        ha='left', va='bottom')
            self.slitlines_text.append(text)
            slitline, = self.axes_dspec.plot(cut['cutslit']['xcen'], cut['cutslit']['ycen'], color=self.color, ls=':')
            self.slitlines.append(slitline)
            if self.axes_speed:
                speedline, = self.axes_speed.plot(cut['cutslit']['xcen'], cut['cutslit']['speed'], color=self.color,
                                                  ls='-')
                self.speedlines.append(speedline)
            if self.axes_accel:
                accelline, = self.axes_accel.plot(cut['cutslit']['xcen'], cut['cutslit']['accel'], color=self.color,
                                                  ls='-')
                self.accellines.append(accelline)


class CutslitBuilder:
    def __init__(self, axes, cutwidth=5.0, cutlength=150, cutang=0.0, cutsmooth=10.0, scale=1.0):
        self.axes = axes
        self.clickedpoints, = self.axes.plot([], [], 'o', color='white')
        self.slitline, = self.axes.plot([], [], color='white', ls='solid')
        self.slitline0, = self.axes.plot([], [], color='white', ls='dotted')
        self.slitline1, = self.axes.plot([], [], color='white', ls='dotted')
        self.cutlength = cutlength
        self.cutwidth = cutwidth
        self.cutang = cutang
        self.cutsmooth = cutsmooth
        self.scale = scale
        self.xx = list(self.clickedpoints.get_xdata())
        self.yy = list(self.clickedpoints.get_ydata())
        self.cid = self.clickedpoints.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        tmode = '{}'.format(self.clickedpoints.figure.canvas.toolbar.mode)
        if tmode == '':
            if event.inaxes != self.axes:
                return
            if event.button == 1:
                self.xx.append(event.xdata)
                self.yy.append(event.ydata)
            elif event.button == 3:
                if len(self.xx) > 0:
                    self.xx.pop()
                    self.yy.pop()
            self.clickedpoints.set_data(self.xx, self.yy)
            self.clickedpoints.figure.canvas.draw()
            self.update()
        else:
            if event.inaxes != self.axes:
                return
            if event.button == 1 or event.button == 3:
                self.clickedpoints.figure.canvas.toolbar.set_message('Uncheck toolbar button {} first!'.format(tmode))

    def update(self, mask=None):
        xx = np.array(self.xx, dtype=np.float64)
        yy = np.array(self.yy, dtype=np.float64)

        if len(self.xx) <= 1:
            cutslitplt = {'xcen': [], 'ycen': [], 'xs0': [], 'ys0': [], 'xs1': [], 'ys1': [], 'cutwidth': [],
                          'posangs': [], 'posangs2': [],
                          'dist': []}
        else:
            if len(self.xx) <= 3:
                cutslitplt = FitSlit(xx, yy, self.cutwidth * self.scale, self.cutang, 300, method='Polyfit')
            else:
                cutslitplt = FitSlit(xx, yy, self.cutwidth * self.scale, self.cutang, 300,
                                     s=len(xx) / 10.0 * self.cutsmooth, method='Param_Spline')
            self.cutlength = int(np.ceil(cutslitplt['dist'][-1] / self.scale) * 1.5)
            if len(self.xx) <= 3:
                cutslitplt = FitSlit(xx, yy, self.cutwidth * self.scale, self.cutang, self.cutlength, method='Polyfit')
            else:
                cutslitplt = FitSlit(xx, yy, self.cutwidth * self.scale, self.cutang, self.cutlength,
                                     s=len(xx) / 10.0 * self.cutsmooth, method='Param_Spline')
        self.cutslitplt = cutslitplt
        if mask is None:
            self.slitline.set_data(cutslitplt['xcen'], cutslitplt['ycen'])
            self.slitline0.set_data(cutslitplt['xs0'], cutslitplt['ys0'])
            self.slitline1.set_data(cutslitplt['xs1'], cutslitplt['ys1'])
        else:
            self.slitline.set_data(ma.masked_array(cutslitplt['xcen'], mask), ma.masked_array(cutslitplt['ycen'], mask))
            self.slitline0.set_data(ma.masked_array(cutslitplt['xs0'], mask), ma.masked_array(cutslitplt['ys0'], mask))
            self.slitline1.set_data(ma.masked_array(cutslitplt['xs1'], mask), ma.masked_array(cutslitplt['ys1'], mask))
        self.slitline.figure.canvas.draw()
        self.slitline0.figure.canvas.draw()
        self.slitline1.figure.canvas.draw()


class Stackplot:
    instrum_meta = {'SDO/AIA': {'scale': 0.6 * u.arcsec / u.pix}}
    # try to find predefined data directory, AIA_LVL1 takes precedence
    aia_lvl1 = os.getenv('AIA_LVL1')
    suncasadb = os.getenv('SUNCASADB')
    if aia_lvl1:
        print('Use ' + aia_lvl1 + ' as the file searching path')
        fitsdir = aia_lvl1
    else:
        if suncasadb:
            fitsdir = suncasadb + '/aiaBrowserData/Download/'
        else:
            print('Environmental variable for either AIA_LVL1 or SUNCASADB not defined')
            print('Use current path')
            fitsdir = './'
    mapseq = None
    mapseq_diff = None
    mapseq_plot = None
    cutslitbd = None
    stackplt = None
    trange = None
    wavelength = None
    fitsfile = None
    exptime_orig = None
    fov = None
    binpix = None
    dt_data = None
    divider_im = None
    divider_dspec = None
    sCutwdth = None
    sCutang = None
    fig_mapseq = None
    pixscale = None

    @resettable
    def __init__(self, infile=None):
        if infile:
            if isinstance(infile, MapSequence):
                self.mapseq = infile
                self.mapseq_info()
            else:
                self.mapseq_fromfile(infile)

    def get_plot_title(self, smap, title):
        titletext = ''
        if 'observatory' in title:
            titletext = titletext + ' {}'.format(smap.observatory)
        if 'detector' in title:
            titletext = titletext + ' {}'.format(smap.detector)
        if 'wavelength' in title:
            titletext = titletext + ' {}'.format(smap.wavelength)
        if 'time' in title:
            titletext = titletext + ' {}'.format(smap.meta['date-obs'])
        return titletext

    def plot_map(self, smap, dspec=None, diff=False, norm=None, cmap=None, SymLogNorm=False, linthresh=0.5,
                 returnImAx=False,
                 layout_vert=False, uni_cm=False, draw_limb=False, draw_grid=False, colortitle=None,
                 title=['observatory', 'detector', 'wavelength', 'time'], fov=fov,
                 *args, **kwargs):
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        def plot_limb(axes, smap):
            rsun = smap.rsun_obs
            phi = np.linspace(-180, 180, num=181) * u.deg
            x = np.cos(phi) * rsun
            y = np.sin(phi) * rsun
            axes.plot(x, y, color='w', linestyle='-')

        def plot_grid(axes, smap, grid_spacing=10. * u.deg):
            def hgs2hcc(rsun, lon, lat, B0, L0):
                lon_L0 = lon - L0
                x = rsun * np.cos(lat) * np.sin(lon)
                y = rsun * (np.sin(lat) * np.cos(B0) - np.cos(lat) * np.cos(lon_L0) * np.sin(B0))
                z = rsun * (np.sin(lat) * np.sin(B0) + np.cos(lat) * np.cos(lon_L0) * np.cos(B0))
                return x, y, z

            def hcc2hpc(x, y, z, dsun):
                d = np.sqrt(x ** 2 + y ** 2 + (dsun - z) ** 2)
                Tx = np.arctan2(x, dsun - z)
                Ty = np.arcsin(y / d)
                return Tx, Ty

            dsun = smap.dsun
            rsun = smap.rsun_meters

            b0 = smap.heliographic_latitude.to(u.deg)
            l0 = smap.heliographic_longitude.to(u.deg)
            hg_longitude_deg = np.linspace(-90, 90, num=91) * u.deg
            hg_latitude_deg = np.arange(0, 90, grid_spacing.to(u.deg).value)
            hg_latitude_deg = np.hstack([-hg_latitude_deg[1:][::-1], hg_latitude_deg]) * u.deg
            for lat in hg_latitude_deg:
                c = hgs2hcc(rsun, hg_longitude_deg, lat * np.ones(91), b0, l0)
                coords = hcc2hpc(c[0], c[1], c[2], dsun)
                axes.plot(coords[0].to(u.arcsec), coords[1].to(u.arcsec), color='w', linestyle=':')

            hg_longitude_deg = np.arange(0, 90, grid_spacing.to(u.deg).value)
            hg_longitude_deg = np.hstack([-hg_longitude_deg[1:][::-1], hg_longitude_deg]) * u.deg
            hg_latitude_deg = np.linspace(-90, 90, num=91) * u.deg

            for lon in hg_longitude_deg:
                c = hgs2hcc(rsun, lon * np.ones(91), hg_latitude_deg, b0, l0)
                coords = hcc2hpc(c[0], c[1], c[2], dsun)
                axes.plot(coords[0].to(u.arcsec), coords[1].to(u.arcsec), color='w', linestyle=':')

        try:
            clrange = DButil.sdo_aia_scale_dict(wavelength=smap.meta['wavelnth'])
        except:
            clrange = {'high': None, 'log': False, 'low': None}
        plt.clf()
        if dspec:
            if layout_vert:
                ax = plt.subplot(211)
            else:
                ax = plt.subplot(121)
        else:
            ax = plt.subplot()
        if 'vmin' in kwargs.keys():
            vmin = kwargs['vmin']
        else:
            vmin = clrange['low']
        if 'vmax' in kwargs.keys():
            vmax = kwargs['vmax']
        else:
            vmax = clrange['high']
        if uni_cm:
            norm = dspec['args']['norm']
        if norm is None:
            if diff:
                if SymLogNorm:
                    norm = colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
                else:
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                if clrange['log']:
                    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
                else:
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
        if not cmap:
            try:
                cmap = cm.get_cmap('sdoaia{}'.format(smap.meta['wavelnth']))
            except:
                cmap = 'gray_r'
        imshow_args = {'cmap': cmap, 'norm': norm, 'interpolation': 'nearest', 'origin': 'lower'}
        try:
            if smap.coordinate_system.x == 'HG':
                xlabel = 'Longitude [{lon}]'.format(lon=smap.spatial_units.x)
            else:
                xlabel = 'X-position [{xpos}]'.format(xpos=smap.spatial_units.x)
            if smap.coordinate_system.y == 'HG':
                ylabel = 'Latitude [{lat}]'.format(lat=smap.spatial_units.y)
            else:
                ylabel = 'Y-position [{ypos}]'.format(ypos=smap.spatial_units.y)
        except:
            if smap.coordinate_system.axis1 == 'HG':
                xlabel = 'Longitude [{lon}]'.format(lon=smap.spatial_units.axis1)
            else:
                xlabel = 'X-position [{xpos}]'.format(xpos=smap.spatial_units.axis1)
            if smap.coordinate_system.axis2 == 'HG':
                ylabel = 'Latitude [{lat}]'.format(lat=smap.spatial_units.axis2)
            else:
                ylabel = 'Y-position [{ypos}]'.format(ypos=smap.spatial_units.axis2)

        # try:
        #     smap.draw_limb()
        # except:
        #     pass
        #
        # try:
        #     smap.draw_grid()
        # except:
        #     pass
        if draw_limb:
            plot_limb(ax, smap)
        if draw_grid:
            if type(draw_grid) in [int, float]:
                plot_grid(ax, smap, draw_grid * u.deg)
            else:
                plot_grid(ax, smap, 10 * u.deg)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        imshow_args.update({'extent': stpu.get_map_corner_coord(smap).value})
        if smap.detector == 'HMI':
            im1 = ax.imshow(np.rot90(smap.data, 2), **imshow_args)
        else:
            im1 = ax.imshow(smap.data, **imshow_args)
        # ['observatory', 'detector', 'wavelength', 'time']
        titletext = self.get_plot_title(smap, title)
        ax.set_title(titletext)
        print(imshow_args['extent'])
        if fov:
            ax.set_xlim(fov[:2])
            ax.set_ylim(fov[2:])
        self.divider_im = make_axes_locatable(ax)
        cax = self.divider_im.append_axes('right', size='1.5%', pad=0.05)
        cax.tick_params(direction='in')
        if colortitle is None:
            colortitle = 'DN counts per second'
        plt.colorbar(im1, ax=ax, cax=cax, label=colortitle)
        ax.set_autoscale_on(False)
        if dspec:
            fig = plt.gcf()
            # if figsize:
            #     fig.set_size_inches(figsize)
            # else:
            #     fig.set_size_inches(13, 5)
            if layout_vert:
                ax2 = plt.subplot(212)
            else:
                ax2 = plt.subplot(122)
            im2 = plt.pcolormesh(dspec['x'], dspec['y'], dspec['dspec'], **dspec['args'])
            date_format = mdates.DateFormatter('%H:%M:%S')
            ax2.xaxis_date()
            ax2.xaxis.set_major_formatter(date_format)
            for xlabel in ax2.get_xmajorticklabels():
                xlabel.set_rotation(30)
                xlabel.set_horizontalalignment("right")
            ax2.yaxis.set_label_text(dspec['ytitle'])
            # self.divider_dspec = make_axes_locatable(ax2)
            # cax = self.divider_dspec.append_axes('right', size='1.5%', pad=0.05)
            # cax.tick_params(direction='in')
            # plt.colorbar(im2, ax=ax2, cax=cax, label=dspec['ctitle'])
            ax2.set_autoscale_on(False)
            if 'axvspan' in dspec.keys():
                vspan = ax2.axvspan(dspec['axvspan'][0], dspec['axvspan'][1], alpha=0.5, color='white')
            if 'xs' in dspec.keys() and 'ys' in dspec.keys():
                ax2.plot(dspec['xs'], dspec['ys'], '--', lw=2.0, alpha=0.7, c='black')
            if 'xlim' in dspec.keys():
                ax2.set_xlim(dspec['xlim'])
            if 'ylim' in dspec.keys():
                ax2.set_ylim(dspec['ylim'])
            if returnImAx:
                return ax, im1, ax2, im2, vspan
            else:
                return ax, ax2
        else:
            if returnImAx:
                return ax, im1
            else:
                return ax  # ax.autoscale(True, 'both', True)  # ax.autoscale_view(True, True, True)  # ax.relim(visible_only=True)

    def make_mapseq(self, trange, outfile=None, fov=None, wavelength='171', binpix=1, dt_data=1, derotate=False,
                    tosave=True, superpixel=False, aia_prep=False, mapinterp=False, overwrite=False, dtype=None,
                    normalize=True):
        if not overwrite:
            if outfile is not None:
                if os.path.exists(outfile):
                    return

        def map_interp(mapin, xycoord):
            from scipy import ndimage
            # from astropy.coordinates import SkyCoord
            pixelxy = mapin.world_to_pixel(xycoord)
            return ndimage.map_coordinates(mapin.data, [pixelxy.y.value, pixelxy.x.value], order=1)

        if isinstance(trange, list):
            if isinstance(trange[0], Time):
                trange = Time([trange[0], trange[-1]])
                fitsfile = stpu.readsdofile(datadir=self.fitsdir, wavelength=wavelength, trange=trange)
            else:
                fitsfile = trange
        elif isinstance(trange, Time):
            fitsfile = stpu.readsdofile(datadir=self.fitsdir, wavelength=wavelength, trange=trange)
        else:
            fitsfile = trange
            print(
                'Input trange format not recognized. trange can either be a file list or a timerange of astropy Time object')

        maplist = []
        self.exptime_orig = []
        print('Loading fits files....')
        for idx, ll in enumerate(tqdm(fitsfile[::dt_data])):
            if mapinterp:
                maptmp = sunpy.map.Map(ll)
                if type(maptmp) is list:
                    maptmp = maptmp[0]
                if idx == 0:
                    all_coord = sunpy.map.all_coordinates_from_map(maptmp)
                    meta0 = deepcopy(maptmp.meta)
                else:
                    # print('1',meta0['date-obs'])
                    meta0.update({'date-obs': maptmp.meta['date-obs']})
                    meta0.update({'date_obs': maptmp.meta['date_obs']})
                    meta0.update({'date_end': maptmp.meta['date_end']})
                    # print('2',meta0['date-obs'])
                    maptmp = sunpy.map.Map(map_interp(maptmp, all_coord), meta0)
                    # print('3',meta0['date-obs'])

            else:
                maptmp = sunpy.map.Map(ll)
                if type(maptmp) is list:
                    maptmp = maptmp[0]
            self.exptime_orig.append(maptmp.exposure_time.value)
            if dtype is not None:
                maptmp = sunpy.map.Map(maptmp.data.astype(dtype), maptmp.meta)
            if aia_prep:
                maptmp = aiaprep(maptmp)
            if fov:
                x0, x1, y0, y1 = fov
                try:
                    submaptmp = maptmp.submap(u.Quantity([x0 * u.arcsec, x1 * u.arcsec]),
                                              top_right=u.Quantity([y0 * u.arcsec, y1 * u.arcsec]))
                except:
                    from astropy.coordinates import SkyCoord
                    bl = SkyCoord(x0 * u.arcsec, y0 * u.arcsec, frame=maptmp.coordinate_frame)
                    tr = SkyCoord(x1 * u.arcsec, y1 * u.arcsec, frame=maptmp.coordinate_frame)
                    submaptmp = maptmp.submap(bl, top_right=tr)
            else:
                submaptmp = maptmp
            if superpixel:
                submaptmp = submaptmp.superpixel(u.Quantity((binpix * u.pix, binpix * u.pix)))
                data = submaptmp.data / float(binpix ** 2)
                submaptmp = sunpy.map.Map(data, submaptmp.meta)
            else:
                submaptmp = submaptmp.resample(u.Quantity(submaptmp.dimensions) / binpix)
            if submaptmp.detector == 'HMI':
                pass
            else:
                if normalize:
                    try:
                        submaptmp = DButil.normalize_aiamap(submaptmp)
                    except:
                        pass
            maplist.append(submaptmp)
        if derotate:
            mapseq = mapsequence_coalign_by_rotation(sunpy.map.Map(maplist, sequence=True))
        else:
            mapseq = sunpy.map.Map(maplist, sequence=True)
        trange = Time([mapseq[0].date, mapseq[-1].date])
        self.fitsfile = fitsfile
        self.dt_data = dt_data
        self.mapseq = mapseq
        self.exptime_orig = np.array(self.exptime_orig)
        self.mapseq_info()

        if tosave:
            if not outfile:
                outfile = 'mapseq_{0}_bin{3}_dtdata{4}_{1}_{2}.mapseq'.format(mapseq[0].meta['wavelnth'],
                                                                              trange[0].isot[:-4].replace(':', ''),
                                                                              trange[1].isot[:-4].replace(':', ''),
                                                                              binpix,
                                                                              dt_data)
            if overwrite:
                if os.path.exists(outfile):
                    os.system('rm -rf {}'.format(outfile))
            for ll in range(42):
                if os.path.exists(outfile):
                    if not os.path.exists(outfile + '_{}'.format(ll)):
                        outfile = outfile + '_{}'.format(ll)
            self.mapseq_tofile(outfile)
        gc.collect()

    def mapseq_fromfile(self, infile):
        t0 = time.time()
        with open(infile, 'rb') as sf:
            print('Loading mapseq....')
            tmp = pickle.load(sf, encoding='latin1')
            if isinstance(tmp, dict):
                isMapseq, islist = isinstance(tmp['mp'], MapSequence), isinstance(tmp['mp'], list)
                if not (isMapseq or islist):
                    print('Load failed. mapseq must be a instance of list or MapSequence')
                    return
                if isMapseq:
                    self.mapseq = tmp['mp']
                else:
                    self.mapseq = MapSequence(tmp['mp'])
                self.dt_data = tmp['dt_data']
                self.fitsfile = tmp['fitsfile']
                if 'exptime_orig' in tmp.keys():
                    self.exptime_orig = tmp['exptime_orig']
                else:
                    self.exptime_orig = []
            else:
                isMapseq, islist = isinstance(tmp, MapSequence), isinstance(tmp, list)
                if not (isMapseq or islist):
                    print('Load failed. mapseq must be a instance of list MapSequence')
                    return
                if isMapseq:
                    self.mapseq = tmp
                else:
                    self.mapseq = MapSequence(tmp)
            self.mapseq_info()
        print('It took {} to load the mapseq.'.format(time.time() - t0))

    def mapseq_tofile(self, outfile=None, mapseq=None):
        t0 = time.time()
        if not mapseq:
            mapseq = self.mapseq
        mp_info = self.mapseq_info(mapseq)
        if not outfile:
            outfile = 'mapseq_{0}_{1}_{2}.mapseq'.format(mapseq[0].meta['wavelnth'],
                                                         self.trange[0].isot[:-4].replace(':', ''),
                                                         self.trange[1].isot[:-4].replace(':', ''))
        with open(outfile, 'wb') as sf:
            print('Saving mapseq to {}'.format(outfile))
            pickle.dump({'mp': mapseq, 'trange': mp_info['trange'], 'fov': mp_info['fov'], 'binpix': mp_info['binpix'],
                         'dt_data': self.dt_data, 'fitsfile': self.fitsfile, 'exptime_orig': self.exptime_orig}, sf)
        print('It took {} to save the mapseq.'.format(time.time() - t0))

    def mapseq_drot(self):
        self.mapseq = mapsequence_coalign_by_rotation(self.mapseq)
        return self.mapseq

    def mapseq_resample(self, binpix=1):
        print('resampling mapseq.....')
        maplist = []
        for idx, ll in enumerate(tqdm(self.mapseq)):
            maplist.append(deepcopy(ll.resample(u.Quantity(ll.dimensions) / binpix)))
        self.mapseq = sunpy.map.Map(maplist, sequence=True)
        self.binpix *= binpix

    def mapseq_diff_denoise(self, log=False, vmax=None, vmin=None):
        datacube = self.mapseq.as_array().astype(float)
        if vmax is None:
            vmax = np.nanmax(datacube)
            if log:
                if vmax < 0:
                    vmax = 0
                else:
                    vmax = np.log10(vmax)
        if vmin is None:
            vmin = np.nanmin(datacube)
            if log:
                if vmin < 0:
                    vmin = 0
                else:
                    vmin = np.log10(vmin)

        datacube_diff = self.mapseq_diff.as_array().astype(float)

        if log:
            datacube[datacube < 10. ** vmin] = 10. ** vmin
            datacube[datacube > 10. ** vmax] = 10. ** vmax
            datacube_diff = datacube_diff * (np.log10(datacube) - vmin) / (vmax - vmin)
        else:
            datacube[datacube < vmin] = vmin
            datacube[datacube > vmax] = vmax
            datacube_diff = datacube_diff * (datacube - vmin) / (vmax - vmin)

        maplist = []
        for idx, ll in enumerate(tqdm(self.mapseq)):
            maplist.append(sunpy.map.Map(datacube_diff[:, :, idx], self.mapseq[idx].meta))
        mapseq_diff = sunpy.map.Map(maplist, sequence=True)
        self.mapseq_diff = mapseq_diff
        return mapseq_diff

    def mapseq_mkdiff(self, mode='rdiff', dt=36., medfilt=None, gaussfilt=None, bfilter=False, lowcut=1 / 10 / 60.,
                      highcut=1 / 1 / 60., window=[None, None], outfile=None, tosave=False, dtype=None):
        '''

        :param mode: accept modes: rdiff, rratio, bdiff, bratio, dtrend, dtrend_diff, dtrend_ratio
        :param dt: time difference in second between frames when [rdiff, rratio, bdiff, bratio] is invoked
        :param medfilt:
        :param gaussfilt:
        :param bfilter: do butter bandpass filter
        :param lowcut: low cutoff frequency in Hz
        :param highcut: high cutoff frequency in Hz
        :param outfile:
        :param tosave:
        :return:
        '''
        if dtype is None:
            dtype = np.float32
        self.mapseq_diff = None
        # modes = {0: 'rdiff', 1: 'rratio', 2: 'bdiff', 3: 'bratio'}
        maplist = []
        datacube = self.mapseq.as_array().astype(float)
        if gaussfilt:
            from scipy.ndimage import gaussian_filter
            print('gaussian filtering map.....')
            for idx, ll in enumerate(tqdm(self.mapseq)):
                datacube[:, :, idx] = gaussian_filter(datacube[:, :, idx], gaussfilt, mode='nearest')
        if medfilt:
            print('median filtering map.....')
            for idx, ll in enumerate(tqdm(self.mapseq)):
                datacube[:, :, idx] = signal.medfilt(datacube[:, :, idx], medfilt)
        print('making the diff mapseq.....')
        tplt = self.tplt.jd
        if mode in ['rdiff', 'rratio', 'bdiff', 'bratio']:
            for idx, ll in enumerate(tqdm(self.mapseq)):
                maplist.append(deepcopy(ll))
                tjd_ = tplt[idx]
                sidx = np.argmin(np.abs(tplt - (tjd_ - dt / 3600. / 24.)))
                # if idx - dt_frm < 0:
                #     sidx = 0
                # else:
                #     sidx = idx - dt_frm
                if mode == 'rdiff':
                    mapdata = datacube[:, :, idx] - datacube[:, :, sidx]
                    mapdata[np.isnan(mapdata)] = 0.0
                elif mode == 'rratio':
                    mapdata = datacube[:, :, idx].astype(float) / datacube[:, :, sidx].astype(float)
                    mapdata[np.isnan(mapdata)] = 1.0
                elif mode == 'bdiff':
                    mapdata = datacube[:, :, idx] - datacube[:, :, 0]
                elif mode == 'bratio':
                    mapdata = datacube[:, :, idx].astype(float) / datacube[:, :, 0].astype(float)
                maplist[idx] = sunpy.map.Map(mapdata.astype(dtype), maplist[idx].meta)
        elif mode.startswith('dtrend'):
            datacube_ft = np.zeros_like(datacube)
            ny, nx, nt = datacube_ft.shape
            ncpu = mp.cpu_count() - 1
            if window[0] is None:
                window[0] = 0
            if window[1] is None:
                window[1] = int(nt / 2)
            print('detrending the mapseq in time domain.....')
            for ly in tqdm(range(ny)):
                runningmean_partial = partial(runningmean, datacube[ly], window, mode)
                pool = mp.Pool(ncpu)
                res = pool.map(runningmean_partial, range(nx))
                pool.close()
                pool.join()
                for lx in range(nx):
                    datacube_ft[ly, lx] = res[lx]['y']

            maplist = []
            for idx, ll in enumerate(tqdm(self.mapseq)):
                maplist.append(sunpy.map.Map(datacube_ft[:, :, idx].astype(dtype), self.mapseq[idx].meta))
        else:
            print('diff mode not recognized. Accept modes: rdiff, rratio, bdiff, bratio, dtrend')
            return None
        mapseq_diff = sunpy.map.Map(maplist, sequence=True)

        if bfilter:
            datacube = mapseq_diff.as_array()
            datacube_ft = np.zeros_like(datacube)
            ny, nx, nt = datacube_ft.shape
            # fs = len(mapseq_diff) * 100.
            fs = 1. / (np.mean(np.diff(self.tplt.mjd)) * 24 * 3600)
            ncpu = mp.cpu_count() - 1
            print('filtering the mapseq in time domain.....')
            for ly in tqdm(range(ny)):
                b_filter_partial = partial(b_filter, datacube[ly], lowcut, highcut, fs)
                pool = mp.Pool(ncpu)
                res = pool.map(b_filter_partial, range(nx))
                pool.close()
                pool.join()
                for lx in range(nx):
                    datacube_ft[ly, lx] = res[lx]['y']

            maplist = []
            for idx, ll in enumerate(tqdm(mapseq_diff)):
                maplist.append(sunpy.map.Map(datacube_ft[:, :, idx].astype(dtype), mapseq_diff[idx].meta))
            mapseq_diff = sunpy.map.Map(maplist, sequence=True)

        if tosave:
            if not outfile:
                outfile = 'mapseq_{5}_{0}_bin{3}_dtdata{4}_{1}_{2}.mapseq'.format(self.mapseq[0].meta['wavelnth'],
                                                                                  self.trange[0].isot[:-4].replace(
                                                                                      ':', ''),
                                                                                  self.trange[1].isot[:-4].replace(
                                                                                      ':', ''),
                                                                                  self.binpix, self.dt_data,
                                                                                  mode)
            self.mapseq_tofile(outfile=outfile, mapseq=mapseq_diff)
        self.mapseq_diff = mapseq_diff
        return mapseq_diff

    def plot_mapseq(self, mapseq=None, hdr=False, norm=None, vmax=None, vmin=None, cmap=None, diff=False,
                    sav_img=False, out_dir=None, dpi=100, anim=False, silent=False, draw_limb=False, draw_grid=False,
                    colortitle=None, title=['observatory', 'detector', 'wavelength', 'time'], fov=[], fps=15):
        '''

        :param mapseq:
        :param hdr:
        :param vmax:
        :param vmin:
        :param diff:
        :param sav_img:
        :param out_dir:
        :param dpi:
        :param anim:
        :return:
        '''
        if mapseq:
            try:
                mapseq_plot = deepcopy(mapseq)
            except:
                mapseq_plot_list = [deepcopy(m) for m in mapseq]
                mapseq_plot = sunpy.map.Map(mapseq_plot_list, sequence=True)
        else:
            if diff:
                try:
                    mapseq_plot = deepcopy(self.mapseq_diff)
                except:
                    mapseq_plot_list = [deepcopy(m) for m in self.mapseq_diff]
                    mapseq_plot = sunpy.map.Map(mapseq_plot_list, sequence=True)
            else:
                try:
                    mapseq_plot = deepcopy(self.mapseq)
                except:
                    mapseq_plot_list = [deepcopy(m) for m in self.mapseq]
                    mapseq_plot = sunpy.map.Map(mapseq_plot_list, sequence=True)
        if mapseq_plot is None:
            print('No mapseq found. Load a mapseq first!')
            return
        if not isinstance(mapseq_plot, MapSequence):
            print('mapseq must be a instance of MapSequence')
            return
        if hdr:
            maplist = []
            for idx, smap in enumerate(tqdm(mapseq_plot)):
                if type(hdr) is bool:
                    smap = DButil.sdo_aia_scale_hdr(smap)
                else:
                    smap = DButil.sdo_aia_scale_hdr(smap, sigma=hdr)
                maplist.append(sunpy.map.Map(smap.data, mapseq_plot[idx].meta))
            mapseq_plot = sunpy.map.Map(maplist, sequence=True)
        if not diff:
            if mapseq_plot[0].detector == 'AIA':
                maplist = []
                for idx, smap in enumerate(tqdm(mapseq_plot)):
                    mapdata = mapseq_plot[idx].data
                    mapdata[np.where(smap.data < 1)] = 1
                    maplist.append(sunpy.map.Map(mapdata, mapseq_plot[idx].meta))
                mapseq_plot = sunpy.map.Map(maplist, sequence=True)
        self.mapseq_plot = mapseq_plot
        # sp = stackplot(parent_obj = self, mapseq = mapseq_plot)
        fig_mapseq = plt.figure()
        self.fig_mapseq = fig_mapseq
        try:
            if self.mapseq_plot[0].observatory == 'SDO':
                clrange = DButil.sdo_aia_scale_dict(mapseq_plot[0].meta['wavelnth'])
            else:
                clrange = {'high': None, 'log': False, 'low': None}
        except:
            clrange = {'high': None, 'log': False, 'low': None}
        if not vmax:
            vmax = clrange['high']
        if not vmin:
            vmin = clrange['low']
        if sav_img:
            if out_dir is None:
                out_dir = '../'
            else:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

            ax, im1 = self.plot_map(mapseq_plot[0], norm=norm, vmax=vmax, vmin=vmin, cmap=cmap, diff=diff,
                                    returnImAx=True,
                                    draw_limb=draw_limb, draw_grid=draw_grid, colortitle=colortitle, title=title,
                                    fov=fov)
            if anim:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.animation as animation
                nframe = len(mapseq_plot)

                def update_frame(num):
                    smap = mapseq_plot[int(num)]
                    im1.set_data(smap.data)
                    titletext = self.get_plot_title(smap, title)
                    ax.set_title(titletext)
                    fig_mapseq.canvas.draw()
                    return

                ani = animation.FuncAnimation(fig_mapseq, update_frame, nframe, interval=50, blit=False)

            if not silent:
                prompt = ''
                while not (prompt.lower() in ['y', 'n']):
                    # try:
                    #     input = raw_input
                    # except NameError:
                    #     pass
                    prompt = input('Satisfied with current FOV? [y/n]')
                if prompt.lower() == 'n':
                    return
            if anim:
                print('Saving movie to {}'.format(out_dir))
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
                ani.save('{0}/{2}{1}.mp4'.format(out_dir, mapseq_plot[0].meta['wavelnth'], mapseq_plot[0].detector),
                         writer=writer)
            else:
                plt.ioff()
                print('Saving images to {}'.format(out_dir))
                for smap in tqdm(mapseq_plot):
                    im1.set_data(smap.data)
                    if smap.meta.has_key('t_obs'):
                        tstr = smap.meta['t_obs']
                    else:
                        tstr = smap.meta['date-obs']
                    ax.set_title('{} {} {} {}'.format(smap.observatory, smap.detector, smap.wavelength, tstr))
                    t_map = Time(tstr)
                    fig_mapseq.canvas.draw()
                    fig_mapseq.savefig('{0}/{3}{1}-{2}.png'.format(out_dir, smap.meta['wavelnth'],
                                                                   t_map.iso.replace(' ', 'T').replace(':',
                                                                                                       '').replace('-',
                                                                                                                   '')[
                                                                   :-4],
                                                                   smap.detector), format='png', dpi=dpi)
                plt.ion()
        else:
            ax, im1 = self.plot_map(mapseq_plot[0], norm=norm, vmax=vmax, vmin=vmin, cmap=cmap, diff=diff,
                                    returnImAx=True,
                                    draw_limb=draw_limb, draw_grid=draw_grid, colortitle=colortitle, title=title,
                                    fov=fov)
            plt.subplots_adjust(bottom=0.10)
            dims = mapseq_plot[0].dimensions
            diagpix = int(np.sqrt(dims[0] ** 2 + dims[1] ** 2).value)
            axcolor = 'lightgoldenrodyellow'
            # axStackplt = plt.axes([0.8, 0.02, 0.10, 0.05], facecolor=axcolor)
            # bStackplt = Button(axStackplt, 'StackPlt')
            axFrame = plt.axes([0.10, 0.03, 0.40, 0.02], facecolor=axcolor)
            # axFrame = self.divider_im.append_axes('bottom', size='1.5%', pad=0.2)
            self.sFrame = Slider(axFrame, 'frame', 0, len(mapseq_plot) - 1, valinit=0, valfmt='%0.0f')
            axCutwdth = plt.axes([0.65, 0.02, 0.20, 0.01], facecolor=axcolor)
            # axCutwdth = self.divider_im.append_axes('bottom', size='1.5%', pad=0.2)
            self.sCutwdth = Slider(axCutwdth, 'Width[pix]', 1, int(diagpix / 4.0), valinit=5, valfmt='%0.0f')
            axCutang = plt.axes([0.65, 0.04, 0.20, 0.01], facecolor=axcolor)
            self.sCutang = Slider(axCutang, 'Angle[deg]', -45.0, 45.0, valinit=0.0, valfmt='%.1f')
            axCutsmooth = plt.axes([0.65, 0.08, 0.20, 0.01], facecolor=axcolor)
            self.sCutsmooth = Slider(axCutsmooth, 'Smooth', 0.0, 100.0, valinit=10, valfmt='%0.0f')
            self.cutslitbd = CutslitBuilder(ax, cutwidth=self.sCutwdth.val, cutlength=diagpix,
                                            cutang=self.sCutang.val / 180. * np.pi,
                                            scale=self.pixscale)

            # def bStackplt_update(event):
            #     # print(bStackplt.val)
            #     print('button clicked')
            #
            # bStackplt.on_clicked(bStackplt_update)

            def sFrame_update(val):
                frm = self.sFrame.val
                smap = mapseq_plot[int(frm)]
                im1.set_data(smap.data)
                titletext = self.get_plot_title(smap, title)
                ax.set_title(titletext)
                fig_mapseq.canvas.draw()

            self.sFrame.on_changed(sFrame_update)

            def sCutwdth_update(val):
                wdth = self.sCutwdth.val
                self.cutslitbd.cutwidth = wdth
                self.cutslitbd.update()

            self.sCutwdth.on_changed(sCutwdth_update)

            def sCutang_update(val):
                ang = self.sCutang.val / 180. * np.pi
                self.cutslitbd.cutang = ang
                self.cutslitbd.update()

            self.sCutang.on_changed(sCutang_update)

            def sCutsmooth_update(val):
                cutsmooth = self.sCutsmooth.val
                self.cutslitbd.cutsmooth = cutsmooth
                self.cutslitbd.update()

            self.sCutsmooth.on_changed(sCutsmooth_update)

            nfrms = len(self.tplt)
            ax_pos = ax.get_position().extents

            wbut_index = (ax_pos[2] - ax_pos[0]) * 0.5 / 4
            x0but_index = ax_pos[0] + (ax_pos[2] - ax_pos[0]) * 0.1 + np.arange(4) * wbut_index
            y0but_index = ax_pos[3] + 0.05
            hbut = 0.03
            axfrst = plt.axes([x0but_index[0], y0but_index, wbut_index, hbut])
            axprev = plt.axes([x0but_index[1], y0but_index, wbut_index, hbut])
            axnext = plt.axes([x0but_index[2], y0but_index, wbut_index, hbut])
            axlast = plt.axes([x0but_index[3], y0but_index, wbut_index, hbut])
            self.bfrst_pltmpcub = Button(axfrst, '<<')
            self.bprev_pltmpcub = Button(axprev, '<')
            self.bnext_pltmpcub = Button(axnext, '>')
            self.blast_pltmpcub = Button(axlast, '>>')

            def next_frm(event):
                frm = (int(self.sFrame.val) + 1) % nfrms
                self.sFrame.set_val(frm)

            def prev_frm(event):
                frm = (int(self.sFrame.val) - 1) % nfrms
                self.sFrame.set_val(frm)

            def frst_frm(event):
                self.sFrame.set_val(0)

            def last_frm(event):
                self.sFrame.set_val(nfrms - 1)

            self.bfrst_pltmpcub.on_clicked(frst_frm)
            self.bprev_pltmpcub.on_clicked(prev_frm)
            self.bnext_pltmpcub.on_clicked(next_frm)
            self.blast_pltmpcub.on_clicked(last_frm)

        return ax

    def cutslit_fromfile(self, infile, color=None, mask=None):
        def cutslit_fromfile_(infile, color=None, mask=None):
            if isinstance(infile, str):
                with open('{}'.format(infile), 'rb') as sf:
                    cutslit = pickle.load(sf, encoding='latin1')
            elif isinstance(infile, dict):
                cutslit = infile
            else:
                raise ValueError("infile format error. Must be type of str or dict.")
            if 'cutang' in cutslit.keys():
                self.cutslitbd.cutang = cutslit['cutang']
                self.sCutang.set_val(self.cutslitbd.cutang * 180. / np.pi)
            if 'cutwidth' in cutslit.keys():
                self.sCutwdth.set_val(cutslit['cutwidth'])
            if 'cutsmooth' in cutslit.keys():
                self.sCutsmooth.set_val(cutslit['cutsmooth'])
            if 'scale' in cutslit.keys():
                self.cutslitbd.scale = cutslit['scale']
            else:
                self.cutslitbd.scale = 1.0
            self.cutslitbd.xx = cutslit['x']
            self.cutslitbd.yy = cutslit['y']
            if mask is None:
                self.cutslitbd.clickedpoints.set_data(self.cutslitbd.xx, self.cutslitbd.yy)
            self.cutslitbd.clickedpoints.figure.canvas.draw()
            self.cutslitbd.update(mask=mask)
            if color:
                self.cutslitbd.slitline.set_color(color)
                self.cutslitbd.slitline0.set_color(color)
                self.cutslitbd.slitline1.set_color(color)

        if not self.cutslitbd:
            self.plot_mapseq()
        cutslit_fromfile_(infile, color=color, mask=mask)

    def cutslit_tofile(self, outfile=None, cutslit=None):
        if not cutslit:
            cutslit = self.cutslit
        with open('{}'.format(outfile), 'wb') as sf:
            pickle.dump(cutslit, sf)

    def make_stackplot(self, mapseq, frm_range=[], threshold=None, gamma=1.0, get_peak=False, trackslit_diffrot=False,
                       negval=False, movingcut=[]):
        '''
        movingcut: [x,y]. x and y are an array of offset in X and Y direction, respectively. the length of x/y is nframes
        '''
        stackplt = []
        print('making the stack plot...')
        if type(frm_range) is list:
            if len(frm_range) == 2:
                if not (0 <= frm_range[0] < len(mapseq)):
                    frm_range[0] = 0
                if not (0 <= frm_range[-1] < len(mapseq)):
                    frm_range[-1] = len(mapseq)
            else:
                frm_range = [0, len(mapseq)]
        maplist = []
        nframe = frm_range[-1] - frm_range[0]
        if movingcut == []:
            movingcut = [np.zeros(nframe), np.zeros(nframe)]
        else:
            pass
        for idx, smap in enumerate(tqdm(mapseq)):
            if frm_range[0] <= idx <= frm_range[-1]:
                data = smap.data.copy()
                if threshold is not None:
                    if isinstance(threshold, dict):
                        if 'outside' in threshold.keys():
                            thrhd = threshold['outside']
                            data = ma.masked_outside(data, thrhd[0], thrhd[1])
                        elif 'inside' in threshold.keys():
                            thrhd = threshold['inside']
                            data = ma.masked_inside(data, thrhd[0], thrhd[1])
                        else:
                            data = ma.masked_array(data)
                    else:
                        data = ma.masked_less(data, threshold)
                else:
                    data = ma.masked_array(data)
                data = data ** gamma
                maplist.append(sunpy.map.Map(data.data, mapseq[idx].meta))

                if trackslit_diffrot:
                    if idx == frm_range[0]:
                        fov = stpu.get_map_corner_coord(smap)
                    else:
                        pass
                else:
                    fov = stpu.get_map_corner_coord(smap)
                intens = getimprofile(data, self.cutslitbd.cutslitplt, xrange=fov[:2].value + movingcut[0][idx],
                                      yrange=fov[2:].value + movingcut[1][idx], get_peak=get_peak)
                if negval:
                    stackplt.append(-intens['y'])
                else:
                    stackplt.append(intens['y'])
            else:
                stackplt.append(np.zeros_like(self.cutslitbd.cutslitplt['dist']) * np.nan)
                maplist.append(mapseq[idx])
        mapseq = sunpy.map.Map(maplist, sequence=True)
        if len(stackplt) > 1:
            stackplt = np.vstack(stackplt)
            self.stackplt = stackplt.transpose()
        else:
            print('Too few timestamps. Failed to make a stack plot map.')
        return mapseq

    def stackplt_wrap(self):
        cutslitplt = self.cutslitbd.cutslitplt
        dspec = {'dspec': self.stackplt, 'x': np.hstack(
            (self.tplt.plot_date, self.tplt.plot_date[-1] + np.nanmean(np.diff(self.tplt.plot_date)))),
                 'y': np.hstack((cutslitplt['dist'], cutslitplt['dist'][-1] + np.nanmean(np.diff(cutslitplt['dist'])))),
                 'ytitle': 'Distance [arcsec]',
                 'ctitle': 'DN counts per second', 'cutslit': self.cutslit}
        return dspec

    def stackplt_tofile(self, outfile=None, stackplt=None):
        if not stackplt:
            dspec = self.stackplt_wrap()
        with open('{}'.format(outfile), 'wb') as sf:
            pickle.dump(dspec, sf)

    def stackplt_fromfile(self, infile, doplot=False, **kwargs):
        with open('{}'.format(infile), 'rb') as sf:
            stackplt = pickle.load(sf, encoding='latin1')
        self.cutslit_fromfile(stackplt['cutslit'])
        self.stackplt = stackplt['dspec']
        if doplot:
            if 'refresh' in kwargs.keys():
                kwargs.pop('refresh')
            if 'doplot' in kwargs.keys():
                kwargs.pop('doplot')
            self.plot_stackplot(refresh=False, **kwargs)

    def plot_stackplot(self, mapseq=None, fov=None, hdr=False, norm=None, vmax=None, vmin=None, cmap=None,
                       layout_vert=False,
                       diff=False, uni_cm=True, sav_img=False, out_dir=None, dpi=100, anim=False, frm_range=[],
                       cutslitplt=None, silent=False, refresh=True, threshold=None, gamma=1.0, get_peak=False,
                       trackslit_diffrot=False, negval=False, movingcut=[]):
        if mapseq:
            try:
                mapseq_plot = deepcopy(mapseq)
            except:
                mapseq_plot_list = [deepcopy(m) for m in mapseq]
                mapseq_plot = sunpy.map.Map(mapseq_plot_list, sequence=True)
        else:
            try:
                mapseq_plot = deepcopy(self.mapseq_plot)
            except:
                mapseq_plot_list = [deepcopy(m) for m in self.mapseq_plot]
                mapseq_plot = sunpy.map.Map(mapseq_plot_list, sequence=True)
        if mapseq_plot is None:
            print('No mapseq found. Load a mapseq first!')
            return
        if not isinstance(mapseq_plot, MapSequence):
            print('mapseq must be a instance of MapSequence')
            return
        # if hdr:
        #     maplist = []
        #     for idx, smap in enumerate(tqdm(mapseq_plot)):
        #         # smap = DButil.sdo_aia_scale_hdr(smap)
        #         maplist.append(sunpy.map.Map(smap.data, mapseq_plot[idx].meta))
        #     mapseq_plot = sunpy.map.Map(maplist,sequence=True)
        if type(frm_range) is list:
            if len(frm_range) == 2:
                if not (0 <= frm_range[0] < len(mapseq_plot)):
                    frm_range[0] = 0
                if not (0 <= frm_range[-1] < len(mapseq_plot)):
                    frm_range[-1] = len(mapseq_plot)
            else:
                frm_range = [0, len(mapseq_plot)]

        if refresh:
            mapseq_plot = self.make_stackplot(mapseq_plot, frm_range=frm_range, threshold=threshold, gamma=gamma,
                                              get_peak=get_peak, trackslit_diffrot=trackslit_diffrot, negval=negval,
                                              movingcut=movingcut)

        if layout_vert:
            fig_mapseq = plt.figure(figsize=(7, 7))
        else:
            fig_mapseq = plt.figure(figsize=(14, 7))
        self.fig_mapseq = fig_mapseq
        try:
            clrange = DButil.sdo_aia_scale_dict(mapseq_plot[0].meta['wavelnth'])
        except:
            clrange = {'high': None, 'log': False, 'low': None}
        if not vmax:
            vmax = clrange['high']
        if not vmin:
            vmin = clrange['low']
        if (norm is None) and (not diff):
            if clrange['log']:
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)

        cutslitplt = self.cutslitbd.cutslitplt
        if not cmap:
            try:
                cmap = cm.get_cmap('sdoaia{}'.format(mapseq_plot[0].meta['wavelnth']))
            except:
                pass

        dspec = self.stackplt_wrap()
        dspec['args'] = {'norm': norm, 'cmap': cmap}

        dtplot = np.mean(np.diff(self.tplt.plot_date))
        dspec['axvspan'] = [self.tplt[0].plot_date, self.tplt[0].plot_date + dtplot]
        if sav_img:
            if out_dir is None:
                out_dir = '../'

            ax, im1, ax2, im2, vspan = self.plot_map(mapseq_plot[frm_range[0]], dspec, norm=None, vmax=vmax, vmin=vmin,
                                                     cmap=cmap, diff=diff, returnImAx=True, uni_cm=uni_cm,
                                                     layout_vert=layout_vert)
            plt.subplots_adjust(bottom=0.10)
            cuttraj, = ax.plot(cutslitplt['xcen'], cutslitplt['ycen'], color='white', ls='solid')
            cuttrajs1, = ax.plot(cutslitplt['xs0'], cutslitplt['ys0'], color='white', ls='dotted')
            cuttrajs2, = ax.plot(cutslitplt['xs1'], cutslitplt['ys1'], color='white', ls='dotted')
            dists = cutslitplt['dist']
            dist_ticks = ax2.axes.get_yticks()
            dist_ticks_idx = []
            for m, dt in enumerate(dist_ticks):
                ddist_med = np.median(np.abs(np.diff(dists)))
                if np.min(np.abs(dists - dt)) < (1.5 * ddist_med):
                    dist_ticks_idx.append(np.argmin(np.abs(dists - dt)))

            maj_t = LineTicks(cuttraj, dist_ticks_idx, 10, lw=2, label=['{:.0f}"'.format(dt) for dt in dist_ticks])
            ax2.set_xlim(dspec['x'][frm_range[0]], dspec['x'][frm_range[-1]])

            def update_slit(frm):
                if movingcut != []:
                    cuttraj.set_xdata(cutslitplt['xcen'] - movingcut[0][frm])
                    cuttraj.set_ydata(cutslitplt['ycen'] - movingcut[1][frm])
                    cuttrajs1.set_xdata(cutslitplt['xs0'] - movingcut[0][frm])
                    cuttrajs1.set_ydata(cutslitplt['ys0'] - movingcut[1][frm])
                    cuttrajs2.set_xdata(cutslitplt['xs1'] - movingcut[0][frm])
                    cuttrajs2.set_ydata(cutslitplt['ys1'] - movingcut[1][frm])

            if anim:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.animation as animation
                # nframe = len(mapseq_plot)
                nframe = frm_range[-1] - frm_range[0]

                def update_frame2(num):
                    frm = int(num) + frm_range[0]
                    smap = mapseq_plot[frm]
                    # smap.data[smap.data<1]=1
                    im1.set_data(smap.data)
                    vspan_xy = vspan.get_xy()
                    vspan_xy[np.array([0, 1, 4]), 0] = self.tplt[frm].plot_date
                    if frm < len(self.tplt) - 1:
                        vspan_xy[np.array([2, 3]), 0] = self.tplt[frm + 1].plot_date
                    else:
                        vspan_xy[np.array([2, 3]), 0] = self.tplt[frm].plot_date
                    vspan.set_xy(vspan_xy)
                    update_slit(frm)
                    ax.set_title(
                        '{} {} {} {}'.format(smap.observatory, smap.detector, smap.wavelength, smap.meta['t_obs']))
                    fig_mapseq.canvas.draw()
                    return

                ani = animation.FuncAnimation(fig_mapseq, update_frame2, nframe, interval=50, blit=False)

            if silent:
                if fov:
                    ax.set_xlim(fov[:2])
                    ax.set_ylim(fov[2:])
            else:
                prompt = ''
                while not (prompt.lower() in ['y', 'n']):
                    # try:
                    #     input = raw_input
                    # except NameError:
                    #     pass
                    prompt = input('Satisfied with current FOV? [y/n]')
                if prompt.lower() == 'n':
                    return
            if anim:
                print('Saving movie to {}'.format(out_dir))
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
                ani.save('{0}/Stackplot-{2}{1}.mp4'.format(out_dir, mapseq_plot[0].meta['wavelnth'],
                                                           mapseq_plot[0].detector), writer=writer)
            else:
                plt.ioff()
                print('Saving images to {}'.format(out_dir))
                for frm, smap in enumerate(tqdm(mapseq_plot)):
                    if frm < frm_range[0] or frm > frm_range[1] - 1: continue
                    im1.set_data(smap.data)
                    if smap.meta.has_key('t_obs'):
                        tstr = smap.meta['t_obs']
                    else:
                        tstr = smap.meta['date-obs']
                    ax.set_title('{} {} {} {}'.format(smap.observatory, smap.detector, smap.wavelength, tstr))
                    vspan_xy = vspan.get_xy()
                    vspan_xy[np.array([0, 1, 4]), 0] = self.tplt[frm].plot_date
                    if frm < len(self.tplt) - 1:
                        vspan_xy[np.array([2, 3]), 0] = self.tplt[frm + 1].plot_date
                    else:
                        vspan_xy[np.array([2, 3]), 0] = self.tplt[frm].plot_date
                    vspan.set_xy(vspan_xy)
                    update_slit(frm)
                    t_map = Time(tstr)
                    fig_mapseq.canvas.draw()
                    try:
                        outname = '{0}/Stackplot-{3}{1}-{2}.png'.format(out_dir, smap.meta['wavelnth'],
                                                                        t_map.iso.replace(' ', 'T').replace(':',
                                                                                                            '').replace(
                                                                            '-', '')[:-4],
                                                                        smap.detector)
                    except:
                        outname = '{0}/Stackplot-{2}-{1}.png'.format(out_dir, t_map.iso.replace(' ', 'T').replace(':',
                                                                                                                  '').replace(
                            '-', '')[:-4], smap.detector)
                    fig_mapseq.savefig(outname, format='png', dpi=dpi)
                plt.ion()
        else:
            ax, im1, ax2, im2, vspan = self.plot_map(mapseq_plot[frm_range[0]], dspec, norm=None, vmax=vmax, vmin=vmin,
                                                     diff=diff, returnImAx=True, uni_cm=uni_cm, cmap=cmap,
                                                     layout_vert=layout_vert)
            plt.subplots_adjust(bottom=0.10)
            cuttraj, = ax.plot(cutslitplt['xcen'], cutslitplt['ycen'], color='white', ls='solid')
            cuttrajs1, = ax.plot(cutslitplt['xs0'], cutslitplt['ys0'], color='white', ls='dotted')
            cuttrajs2, = ax.plot(cutslitplt['xs1'], cutslitplt['ys1'], color='white', ls='dotted')
            dists = cutslitplt['dist']
            dist_ticks = ax2.axes.get_yticks()
            dist_ticks_idx = []
            for m, dt in enumerate(dist_ticks):
                ddist_med = np.median(np.abs(np.diff(dists)))
                if np.min(np.abs(dists - dt)) < (1.5 * ddist_med):
                    dist_ticks_idx.append(np.argmin(np.abs(dists - dt)))

            maj_t = LineTicks(cuttraj, dist_ticks_idx, 10, lw=2, label=['{:.0f}"'.format(dt) for dt in dist_ticks])
            axcolor = 'lightgoldenrodyellow'
            ax2.set_xlim(dspec['x'][frm_range[0]], dspec['x'][frm_range[-1]])
            ax2_pos = ax2.get_position().extents
            axframe2 = plt.axes([ax2_pos[0], 0.03, ax2_pos[2]-ax2_pos[0], 0.02], facecolor=axcolor)
            # axframe2 = plt.axes([ax2_pos[0], ax2_pos[1], ax2_pos[2] - ax2_pos[0], ax2_pos[3] - ax2_pos[1]],
            #                    facecolor=axcolor, frame_on=False)
            self.sframe2 = Slider(axframe2, '', frm_range[0], frm_range[-1] - 1, valinit=frm_range[0],
                                  valfmt='frm %0.0f',
                                  alpha=0.0)
            nfrms = len(self.tplt)

            def update_slit(frm):
                if movingcut!=[]:
                    cuttraj.set_xdata(cutslitplt['xcen'] - movingcut[0][frm])
                    cuttraj.set_ydata(cutslitplt['ycen'] - movingcut[1][frm])
                    cuttrajs1.set_xdata(cutslitplt['xs0'] - movingcut[0][frm])
                    cuttrajs1.set_ydata(cutslitplt['ys0'] - movingcut[1][frm])
                    cuttrajs2.set_xdata(cutslitplt['xs1'] - movingcut[0][frm])
                    cuttrajs2.set_ydata(cutslitplt['ys1'] - movingcut[1][frm])

            def update2(val):
                tmode = '{}'.format(self.fig_mapseq.canvas.toolbar.mode)
                if tmode == '':
                    frm = int(self.sframe2.val)
                    smap = mapseq_plot[frm]
                    im1.set_data(smap.data)
                    if smap.meta.has_key('t_obs'):
                        tstr = smap.meta['t_obs']
                    else:
                        tstr = smap.meta['date-obs']
                    ax.set_title('{} {} {} {}'.format(smap.observatory, smap.detector, smap.wavelength, tstr))
                    vspan_xy = vspan.get_xy()
                    vspan_xy[np.array([0, 1, 4]), 0] = self.tplt[frm].plot_date
                    if frm < nfrms - 1:
                        vspan_xy[np.array([2, 3]), 0] = self.tplt[frm + 1].plot_date
                    else:
                        vspan_xy[np.array([2, 3]), 0] = self.tplt[frm].plot_date
                    vspan.set_xy(vspan_xy)
                    update_slit(frm)
                    fig_mapseq.canvas.draw()
                else:
                    self.fig_mapseq.canvas.toolbar.set_message(
                        'Uncheck toolbar button {} first!'.format(tmode))

            self.sframe2.on_changed(update2)

            wbut_index = (ax2_pos[2] - ax2_pos[0]) * 0.5 / 4
            x0but_index = ax2_pos[0] + np.arange(4) * wbut_index
            y0but_index = ax2_pos[3] + 0.01
            hbut = 0.03
            axfrst = plt.axes([x0but_index[0], y0but_index, wbut_index, hbut])
            axprev = plt.axes([x0but_index[1], y0but_index, wbut_index, hbut])
            axnext = plt.axes([x0but_index[2], y0but_index, wbut_index, hbut])
            axlast = plt.axes([x0but_index[3], y0but_index, wbut_index, hbut])
            self.bfrst = Button(axfrst, '<<')
            self.bprev = Button(axprev, '<')
            self.bnext = Button(axnext, '>')
            self.blast = Button(axlast, '>>')

            def next_frm(event):
                frm = (int(self.sframe2.val) + 1) % nfrms
                self.sframe2.set_val(frm)

            def prev_frm(event):
                frm = (int(self.sframe2.val) - 1) % nfrms
                self.sframe2.set_val(frm)

            def frst_frm(event):
                self.sframe2.set_val(0)

            def last_frm(event):
                self.sframe2.set_val(nfrms - 1)

            self.bfrst.on_clicked(frst_frm)
            self.bprev.on_clicked(prev_frm)
            self.bnext.on_clicked(next_frm)
            self.blast.on_clicked(last_frm)

            wbut_anal = 1.5 * wbut_index
            axtraject = plt.axes([ax2_pos[2] - wbut_anal, y0but_index, wbut_anal, hbut])
            self.btraject = Button(axtraject, 'trajectory')

            def stackplt_traject(event):
                self.stackplt_traject_fromfile(self.stackplt_wrap(), frm_range=frm_range, cmap=cmap,
                                               norm=norm, gamma=gamma)

            self.btraject.on_clicked(stackplt_traject)

            axlghtcurv = plt.axes([ax2_pos[2] - 2.1 * wbut_anal, y0but_index, wbut_anal, hbut])
            self.blghtcurv = Button(axlghtcurv, 'lightcurve')

            def stackplt_lghtcurv(event):
                self.stackplt_lghtcurv_fromfile(self.stackplt_wrap(), frm_range=frm_range, cmap=cmap, norm=norm, gamma=gamma)

            self.blghtcurv.on_clicked(stackplt_lghtcurv)

        return

    def stackplt_traject_fromfile(self, infile, frm_range=[], cmap='inferno', norm=None, gamma=1.0):
        if isinstance(infile, str):
            with open('{}'.format(infile), 'rb') as sf:
                stackplt = pickle.load(sf, encoding='latin1')
        elif isinstance(infile, dict):
            stackplt = infile
        else:
            raise ValueError("infile format error. Must be type of str or dict.")
        # fig_stpanal = plt.figure()
        # axs_stpanal = []
        # axs_stpanal.append(fig_stpanal.add_subplot(2, 1, 1))
        # axs_stpanal.append(fig_stpanal.add_subplot(4, 1, 3, sharex=axs_stpanal[-1]))
        # axs_stpanal.append(fig_stpanal.add_subplot(4, 1, 4, sharex=axs_stpanal[-1]))
        # axs_stpanal.append(fig_stpanal.add_subplot(3, 1, 1))
        # axs_stpanal.append(fig_stpanal.add_subplot(4, 1, 3, sharex=axs_stpanal[-1]))
        fig_stpanal, axs_stpanal = plt.subplots(nrows=2, sharex=True)
        axs_stpanal = list(axs_stpanal.ravel())
        axs_stpanal[1].set_autoscalex_on(False)
        axs_stpanal[1].set_autoscaley_on(True)
        axs_stpanal[1].set_ylabel('Speed [arcsec s$^{-1}$]')
        # axs_stpanal[2].set_autoscalex_on(False)
        # axs_stpanal[2].set_autoscaley_on(True)
        # axs_stpanal[2].set_ylabel('Acceleration [arcsec s$^{-2}$]')
        x, y, dspec = stackplt['x'], stackplt['y'], stackplt['dspec']
        x = (x - x[0]) * 24 * 3600
        ax = axs_stpanal[0]
        im = ax.pcolormesh(x, y, dspec ** gamma, cmap=cmap,
                           norm=norm, rasterized=True)
        ax.set_xlim(x[frm_range[0]], x[frm_range[-1]])
        ax = axs_stpanal[-1]
        ax.set_xlabel('seconds since {}'.format(Time(stackplt['x'][0], format='plot_date').iso[:-4]))
        self.spacetimeslitbd = SpaceTimeSlitBuilder(axs_stpanal, self.stackplt_wrap(), cutlength=50, color='red')

        axCutsmooth = plt.axes([0.35, 0.01, 0.40, 0.015])
        self.stCutsmooth = Slider(axCutsmooth, 'Smooth', 0.0, 100.0, valinit=10, valfmt='%0.0f')

        def sCutsmooth_update(val):
            self.spacetimeslitbd.cutsmooth = self.stCutsmooth.val
            self.spacetimeslitbd.update()

        self.stCutsmooth.on_changed(sCutsmooth_update)
        return

    def stackplt_lghtcurv_fromfile(self, infile, frm_range=[], cmap='inferno', norm=None, gamma=1.0,
                                   log=False, axes=None):
        if isinstance(infile, str):
            with open('{}'.format(infile), 'rb') as sf:
                stackplt = pickle.load(sf, encoding='latin1')
        elif isinstance(infile, dict):
            stackplt = infile
        else:
            raise ValueError("infile format error. Must be type of str or dict.")
        if axes is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        else:
            ax = axes
        ax.set_autoscalex_on(False)
        x, y, dspec = stackplt['x'], stackplt['y'], stackplt['dspec']
        if log:
            im = ax.pcolormesh(x, y, dspec ** gamma, cmap=cmap,
                               norm=norm, rasterized=True)
        else:
            im = ax.pcolormesh(x, y, dspec ** gamma, cmap=cmap,
                               norm=norm, rasterized=True)

        ax.set_xlim(x[frm_range[0]], x[frm_range[-1]])
        date_format = mdates.DateFormatter('%H:%M:%S')
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(date_format)
        for xlabel in ax.get_xmajorticklabels():
            xlabel.set_rotation(30)
            xlabel.set_horizontalalignment("right")
        ax.yaxis.set_label_text(stackplt['ytitle'])
        ax.set_autoscale_on(False)
        ax.set_xlabel('Time')
        self.lightcurvebd = LightCurveBuilder(stackplt, ax, color='red')
        return

    def mapseq_info(self, mapseq=None):
        if mapseq:
            trange = Time([mapseq[0].date, mapseq[-1].date])
            fov = stpu.get_map_corner_coord(mapseq[0])
            pixscale = np.nanmean([ll.value for ll in mapseq[0].scale])
            binpix = int(np.round(pixscale / self.instrum_meta['SDO/AIA']['scale'].value))
            return {'trange': trange, 'fov': fov, 'pixscale': pixscale, 'binpix': binpix}
        else:
            self.trange = Time([self.mapseq[0].date, self.mapseq[-1].date])
            self.fov = stpu.get_map_corner_coord(self.mapseq[0])
            self.pixscale = np.nanmean([ll.value for ll in self.mapseq[0].scale])
            self.binpix = int(np.round(
                np.mean([ll.value for ll in self.mapseq[0].scale]) / self.instrum_meta['SDO/AIA']['scale'].value))
            return {'trange': self.trange, 'fov': self.fov, 'pixscale': self.pixscale, 'binpix': self.binpix}

    # def mapseq2image(self,mapseq=None,figsize=(7,5)):
    #     if mapseq:
    #         pass
    #     else:n
    #         mapseq = self.mapseq_plot

    @property
    def cutslit(self):
        return {'x': self.cutslitbd.clickedpoints.get_xdata(), 'y': self.cutslitbd.clickedpoints.get_ydata(),
                'cutslit': self.cutslitbd.cutslitplt, 'cutlength': self.cutslitbd.cutlength,
                'cutwidth': self.cutslitbd.cutwidth,
                'cutang': self.cutslitbd.cutang, 'cutsmooth': self.cutslitbd.cutsmooth,
                'scale': self.cutslitbd.scale}

    @property
    def tplt(self, mapseq=None):
        if not mapseq:
            mapseq = self.mapseq
        t = []

        smap = mapseq[0]
        if smap.meta.has_key('date-obs'):
            key = 'date-obs'
        else:
            if smap.meta.has_key('date_obs'):
                key = 'date_obs'
            else:
                if smap.meta.has_key('t_obs'):
                    key = 't_obs'
                else:
                    print('Check you fits header. No time keywords found in the header!')
                    return None

        for idx, smap in enumerate(mapseq):
            tstr = smap.meta[key]
            t.append(tstr)
        t = Time(t)
        if key == 't_obs':
            t = Time(t.mjd - self.exptime_orig / 2.0 / 24. / 3600., format='mjd')
        return t

    @classmethod
    def set_fits_dir(cls, fitsdir):
        cls.fitsdir = fitsdir  # def __repr__(self):  #     if self.mpcube:  #         print('')
