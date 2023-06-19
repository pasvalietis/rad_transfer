from scipy.io.idl import readsav
import matplotlib.pyplot as plt
import numpy as np
import re

#resp_file = '../hinode_xrt/xrt_temp_response.npy'
#xrt_trm = np.load(resp_file, allow_pickle=True)

temp_resp_file = '../hinode_xrt/xrt_resp_full.sav'
out = readsav(temp_resp_file, python_dict=True)#, uncompressed_file_name='./unc_map.sav')
#%%
temp_resp = {}

for i in range(15):
    temp_resp[re.findall('.*;', out['tresp_xrt'][i][3].decode('utf-8'))[0][:-1]] = (out['tresp_xrt'][i][6])[:26]

temp_resp['temps'] = (out['tresp_xrt'][1][4])[:26]

# for channel in temp_resp.keys():
#     if channel != 'temps':
#         plt.loglog(temp_resp['temps'], temp_resp[channel], label = channel)
#     plt.xlim(3e5, 1e8)
#     plt.ylim(1e-30, 1e-24)
#     plt.legend()
#
# plt.show()
#%%
#plt.savefig('XRT_responses.png')
#
#
outfile = ('xrt_temp_response.npy')
np.save(outfile, temp_resp)