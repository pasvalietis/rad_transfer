import subprocess

def run_cmd(cmd, verbose=False, *args, **kwargs):
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


def con_url(year, month, day, folder='H0400', **kwargs):

    year = str(year)
    month = str(month)
    day = str(day)
    folder = str(folder)

    time = ['00', '00', '00', '0']
    time_vars = ['hour', 'minute', 'sec', 'enum']
    for i, d in enumerate(time_vars):
        if d in kwargs:
            time[i] = str(kwargs.get(d, '00' if i < 3 else '0'))
    hour, minute, sec, enum = time[0], time[1], time[2], time[3]

    xrt_lvl0 = 'https://data.darts.isas.jaxa.jp/pub//hinode/xrt/level0/'

    url = xrt_lvl0 + '{}/{}/{}/{}/XRT{}{}{}_{}{}{}.{}.fits'.format(
        year, month, day, folder, year, month, day, hour, minute, sec, enum
    )

    return url


#url = 'https://data.darts.isas.jaxa.jp/pub//hinode/xrt/level0/2013/05/15/H0400/XRT20130515_040058.0.fits'
url = con_url('2013', '05', '15',
              hour='04', minute='00', sec='58', enum='0')
print(url)

# make event directory environmental variable in bash
directory = 'events'
options = '-nv -m --force-directories -nH --cut-dirs=2 -P {}'.format(directory)
command = 'wget {} {}'.format(options, url)

run_cmd(command, verbose=True)
