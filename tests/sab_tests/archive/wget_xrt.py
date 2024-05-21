import subprocess
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm


def run_cmd(cmd, verbose=False, *args, **kwargs):
    ''' Runs commands to the system terminal '''

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )

    std_out, std_err = process.communicate()

    if verbose:
        print(std_out.strip(), std_err)
    pass


def con_url(year, month, day, **kwargs):
    ''' Constructs url for xrt_lvl_0 image using date time info '''

    year = str(year)
    month = ('0' if int(month) < 10 else '') + str(month)
    day = ('0' if int(day) < 10 else '') + str(day)

    time = ['00', '00', '00', '0']
    time_vars = ['hour', 'minute', 'sec', 'enum']
    for i, d in enumerate(time_vars):
        if d in kwargs:
            time[i] = (('0' if (int(kwargs.get(d, '0')) < 10 and i < 3) else '')
                       + str(kwargs.get(d, '0')))
    hour, minute, sec, enum = time[0], time[1], time[2], time[3]
    folder = "H" + hour + '00'

    xrt_lvl0 = 'https://data.darts.isas.jaxa.jp/pub//hinode/xrt/level0/'

    url = xrt_lvl0 + '{}/{}/{}/{}/XRT{}{}{}_{}{}{}.{}.fits'.format(
        year, month, day, folder, year, month, day, hour, minute, sec, enum
    )

    return url


def timerange(t1, t2):
    ''' Returns an array of datetimes between t1 and t2 '''

    start = t1
    end = t2

    delta = end-start

    dts = []
    for i in range(delta.seconds + 1):
        dt = start + timedelta(seconds=i)
        dts.append(dt)

    return dts


# Set range of date-times to search for data
t1 = datetime(2012, 7, 19, 10, 30, 00)
t2 = datetime(2012, 7, 19, 15, 00, 00)

dts = timerange(t1, t2)

# Construct array of urls for all possible filenames between t1 and t2
# test_url = 'https://data.darts.isas.jaxa.jp/pub//hinode/xrt/level0/2013/05/15/H0400/XRT20130515_040058.0.fits'
# urls = [test_url]

urls = []
for dt in dts:
    for i in range(0,9):
        url = con_url(str(dt.year), str(dt.month), str(dt.day),
                      hour=str(dt.hour), minute=str(dt.minute), sec=str(dt.second), enum=str(i))
        urls.append(url)

# print(urls)

# Make event directory environmental variable in bash
directory = '/media/saber/My_Passport/Sabastian/xrt/2012_07_19/xrt_lvl_0'
options = '-nv -m --force-directories -nH --cut-dirs=2 -P {}'.format(directory)

for url in tqdm(urls):
    command = 'wget {} {}'.format(options, url)
    run_cmd(command, verbose=False)
