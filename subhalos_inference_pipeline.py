import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import tensorflow_probability as tfp
import h5py
from samana.forward_model import forward_model
import os
import sys
import random
from scipy import stats
from emulator_pipeline import norm_transform_inv, Coupling, RealNVP

### SETTING UP INPUT DATA FOR FORWARD MODELING ANALYSIS

# Reading in the data and converting it from a string to a list of lists and floats.
necessary_data = open("necessaryDataPipeline.txt", "r")
lines = necessary_data.readlines()
necessary_data.close()

for i in range(len(lines)):
    lines[i] = lines[i].strip('\n')
    if i < 2:
        lines[i] = lines[i][1:-2]
        lines[i].split()
        lines[i] = [float(i) for i in lines[i].split()]
    else:
        lines[i] = float(lines[i])

# Reading in emulator weights
emulator = RealNVP(num_coupling_layers=12)
emulator.load_weights('../data/emulatorModelPipeline.weights.h5')

# Creating function that takes emulator weights (and necessary data file) to produce a population of un-normalized emulator subhalos
def emulator_data(emulator = emulator, lines = lines, num_iterations = 1):

    data_min = np.array(lines[0])
    data_max = np.array(lines[1])
    massTree = lines[2]
    massResolution = lines[3]
    massHost = lines[4]
    radiusVirialHost = lines[5]
    countSubhalosMean = lines[6]

    s_I = 0.18
    p = 1/(1 + countSubhalosMean*s_I**2)
    r = 1/s_I**2
    sigma = np.sqrt(r*(1 - p)/p**2)

    # initializing arrays to go from latent space to data
    i = 0 # used to count number of iterations
    n = 0
    reg_massInfall = []
    reg_massBound = []
    reg_concentration = []
    reg_redshift = []
    reg_projectedRadius = []
    reg_x = []
    reg_y = []
    reg_truncationRadius = []
    sample_amount = 1
    z = np.arange(stats.nbinom.ppf(0, r, p),stats.nbinom.ppf(0.9999999999999999, r, p))
    prob = stats.nbinom.pmf(z,r,p)
    prob = np.nan_to_num(prob)
    N = np.random.choice(z, p = prob)
    data = np.array([])

    min_concentration =  3.4845380492242852 # minimum concentration value from Galacticus data
    while n < num_iterations:
        samples = emulator.distribution.sample(N)
        x, _ = emulator.predict(samples, batch_size=65336)
        xt = norm_transform_inv(x, data_min, data_max, -1, 1)
        clip = (xt[:,0] > np.log10(2.0*massResolution/massTree)) & (xt[:,2] <= 0.0) & (xt[:,2] > -xt[:,0]+np.log10(massResolution/massTree)) & (xt[:,3] >= 0.5) & (xt[:,2] < np.log10(1e9/(massHost * 10**xt[:,0])))

        if len(data) > N:
            data = data[:int(N)]
        elif len(data) < N:
            sample_amount += 1
            i += 1

            for j in range(len(xt[clip])):
                if len(data) == 0:
                    data = np.array([xt[clip][0]])
                else:
                    data = np.vstack((data, xt[clip][j]))
            continue
        else:
            pass
        if isinstance(data[0], float):
            reg_massInfall.append(massHost * (10**data[0]))
            reg_concentration.append(data[1])
            reg_massBound = [np.array(reg_massInfall[0] * 10**data[2])]
            reg_redshift.append(data[3])
            reg_truncationRadius.append(radiusVirialHost * (10**data[4]))
            reg_projectedRadius.append(radiusVirialHost * (10**data[-1]))
            r2d_Mpc = radiusVirialHost * (10**data[-1])
            x = [0]*len(data)
            y = [0]*len(data)
        else:
            reg_massInfall.append(massHost * (10**data[:,0]))

            for i in range(len(data)): # We're doing this because the ith bound mass depends on the ith infall mass, whereas every other quantity depends on a  single scalar
                reg_massBound.append(reg_massInfall[0][i] * (10**data[i][2]))
            reg_massBound = [np.array(reg_massBound)]

            reg_concentration.append(data[:,1])
            reg_redshift.append(data[:,3])
            reg_truncationRadius.append(radiusVirialHost * (10**data[:,4]))
            reg_projectedRadius.append(radiusVirialHost * (10**data[:,-1]))
            r2d_Mpc = radiusVirialHost * (10**data[:,-1])
            x = [0]*len(data)
            y = [0]*len(data)

        for i in range(len(data)):
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)

            theta = np.arccos(1 - 2*r1) # [0,pi] variable
            phi = 2 * np.pi * r2 # [0,2pi] variable

            x[i] = r2d_Mpc[i] * np.cos(phi)
            y[i] = r2d_Mpc[i] * np.sin(phi)
        reg_x.append(x)
        reg_y.append(y)
        n += 1

    return reg_massInfall, reg_concentration, reg_massBound, reg_redshift, reg_truncationRadius, np.array(reg_x), np.array(reg_y)

# Constructing input parameters for the forward_model() function
output_path = os.getcwd() + '/test/'
job_index = sys.argv[1]
n_keep = 2
summary_statistic_tolerance = 1e5

from samana.Data.Mocks.baseline_smooth_mock import BaselineSmoothMockModel
from samana.Data.Mocks.baseline_smooth_mock import BaselineSmoothMock
data_class = BaselineSmoothMock()
model = BaselineSmoothMockModel
preset_model_name = 'WDM'

kwargs_sample_realization = {}
kwargs_sample_realization['LOS_normalization'] = ['FIXED', 0.]
kwargs_sample_realization['log_m_host'] = ['FIXED', 13.3]
kwargs_sample_realization['cone_opening_angle_arcsec'] = ['FIXED', 8.0]
kwargs_sample_realization['log_mlow'] = ['FIXED', 6.0]
kwargs_sample_realization['log_mhigh'] = ['FIXED', 9.0]
kwargs_sample_realization['sigma_sub'] = ['FIXED', 0.0]
kwargs_sample_realization['log_mc'] = ['FIXED', 4.0]

# parameter for emulator data (keep commented out when not working with emulator)
#kwargs_sample_realization['emulator_input'] = ['FIXED', emulator_data]

kwargs_sample_source = {'source_size_pc': ['FIXED', 5]}
kwargs_sample_macro_fixed = {
    'a4_a': ['FIXED', 0.0], 
    'a3_a': ['FIXED', 0.0],
    'delta_phi_m3': ['GAUSSIAN', -np.pi/6, np.pi/6]
}
kwargs_model_class = {'shapelets_order': 10} # source complexity

### RUNNING THE FORWARD MODEL FUNCTION ###
forward_model(output_path, job_index, n_keep, data_class, model, preset_model_name, 
              kwargs_sample_realization, kwargs_sample_source, kwargs_sample_macro_fixed,
              tolerance=summary_statistic_tolerance, log_mlow_mass_sheets = 6.0, kwargs_model_class = kwargs_model_class, verbose=False, test_mode=False)

f = open(output_path + 'job_'+str(job_index)+'/parameters.txt', 'r')
param_names = f.readlines()[0]
print('PARAMETER NAMES:')
print(param_names)
f.close()

print('code executed!')
