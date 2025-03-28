import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import h5py
from scipy import stats
from emulator_pipeline import norm_transform, norm_transform_inv, Coupling, RealNVP

num_files = 1 # refers to number of Galacticus .hdf5 files

### SETTING UP EMULATOR DATA INPUT ###

# Initializing arrays that get fed into the emulator
tot_countSubhalosMean = np.array([])
tot_weights = np.array([])
tot_massInfallNormalized = np.array([])
tot_concentrationNormalized = np.array([])
tot_massBoundNormalized = np.array([])
tot_redshiftLastIsolatedNormalized = np.array([])
tot_truncationRadiusNormalized = np.array([])
tot_projectedRadiusNormalized = np.array([])

# Reading in Galacticus data/filling in the arrays initialized above
for i in range(num_files):
    f = h5py.File('darkMatterOnlySubHalosPipeline.hdf5', 'r')
    mergerTreeBuildMassesGroup = f['Parameters/mergerTreeBuildMasses']
    massResolutionGroup = f['Parameters/mergerTreeMassResolution']

    massTree = mergerTreeBuildMassesGroup.attrs['massTree'][0]
    countTree = mergerTreeBuildMassesGroup.attrs['treeCount'][0]

    massResolution = massResolutionGroup.attrs['massResolution']
    weight = f['Outputs/Output1/nodeData/nodeSubsamplingWeight'][:]
    treeIndex = f['Outputs/Output1/nodeData/mergerTreeIndex'][:]
    isCentral = f['Outputs/Output1/nodeData/nodeIsIsolated'][:]
    massInfall = f['Outputs/Output1/nodeData/basicMass'][:]
    massBound = f['Outputs/Output1/nodeData/satelliteBoundMass'][:]
    concentration = f['Outputs/Output1/nodeData/concentration'][:]
    truncationRadius = f['Outputs/Output1/nodeData/radiusTidalTruncationNFW'][:]
    scaleRadius = f['Outputs/Output1/nodeData/darkMatterProfileScale'][:]
    redshiftLastIsolated = f['Outputs/Output1/nodeData/redshiftLastIsolated'][:]
    positionOrbitalX = f['Outputs/Output1/nodeData/positionOrbitalX'][:]
    positionOrbitalY = f['Outputs/Output1/nodeData/positionOrbitalY'][:]
    positionOrbitalZ = f['Outputs/Output1/nodeData/positionOrbitalZ'][:]
    satelliteTidalHeating = f['Outputs/Output1/nodeData/satelliteTidalHeatingNormalized'][:]
    radiusVirial = f['Outputs/Output1/nodeData/darkMatterOnlyRadiusVirial'][:]
    velocityVirial = f['Outputs/Output1/nodeData/darkMatterOnlyVelocityVirial'][:]
    radiusProjected = np.sqrt(positionOrbitalX**2+positionOrbitalY**2)
    subhalos = (isCentral == 0) & (massInfall > 2.0*massResolution)
    radiusOrbital = np.sqrt(positionOrbitalX**2+positionOrbitalY**2+positionOrbitalZ**2)
    centrals = (isCentral == 1)
    countSubhalos = np.zeros(countTree)
    for j in range(countTree):
        selectTree = (isCentral == 0) & (massInfall > 2.0*massResolution) & (treeIndex == j+1)
        countSubhalos[j] = np.sum(weight[selectTree])
    countSubhalosMean = np.mean(countSubhalos)

    tot_countSubhalosMean = np.append(tot_countSubhalosMean, countSubhalosMean)

    overMassive = (massBound > massInfall)
    massBound[overMassive] = massInfall[overMassive]

    massHost = massInfall[centrals][0]
    radiusVirialHost = radiusVirial[centrals][0]
    velocityVirialHost = velocityVirial[centrals][0]
    massInfallNormalized = np.log10(massInfall[subhalos]/massHost)
    massBoundNormalized = np.log10(massBound[subhalos]/massInfall[subhalos])
    concentrationNormalized = concentration[subhalos]
    redshiftLastIsolatedNormalized = redshiftLastIsolated[subhalos]
    radiusOrbitalNormalized = np.log10(np.sqrt(positionOrbitalX[subhalos]**2+positionOrbitalY[subhalos]**2+positionOrbitalZ[subhalos]**2)/radiusVirialHost)
    satelliteTidalHeatingNormalized = np.log10(1.0e-6+satelliteTidalHeating[subhalos]/velocityVirial[subhalos]**2*radiusVirial[subhalos]**2)
    truncationRadiusNormalized = np.log10(truncationRadius[subhalos]/radiusVirialHost)
    projectedRadiusNormalized = np.log10(np.sqrt(positionOrbitalX[subhalos]**2 + positionOrbitalY[subhalos]**2)/radiusVirialHost)

    tot_weights = np.append(tot_weights, weight[subhalos])
    tot_massInfallNormalized = np.append(tot_massInfallNormalized, massInfallNormalized)
    tot_concentrationNormalized = np.append(tot_concentrationNormalized, concentrationNormalized)
    tot_massBoundNormalized = np.append(tot_massBoundNormalized, massBoundNormalized)
    tot_redshiftLastIsolatedNormalized = np.append(tot_redshiftLastIsolatedNormalized, redshiftLastIsolatedNormalized)
    tot_truncationRadiusNormalized = np.append(tot_truncationRadiusNormalized, truncationRadiusNormalized)
    tot_projectedRadiusNormalized = np.append(tot_projectedRadiusNormalized, projectedRadiusNormalized)

# Creating a 6D object (before shifted to hypercube coordinates)
data=np.array(
    list(
        zip(
            tot_massInfallNormalized,
            tot_concentrationNormalized,
            tot_massBoundNormalized,
            tot_redshiftLastIsolatedNormalized,
            tot_truncationRadiusNormalized,
            tot_projectedRadiusNormalized
        )
    )
)

# Creating the augmented_normalized_data object which is what actually gets input into the emulator
data_min, data_max, normalized_data = norm_transform(data,-1,1)
augmented_normalized_data = np.hstack((normalized_data, np.expand_dims(tot_weights,1)))
np.random.shuffle(augmented_normalized_data)

# Creating a .txt file that gets used to un-normalize the emulator output in the inference part of the pipeline
min_array = np.nanmin(data, axis = 0)
max_array = np.nanmax(data, axis = 0)
with open('necessaryDataPipeline.txt', 'w', newline='') as file:
    file.write(str(min_array).replace('\n','').replace(' ',' ') + '\n')
    file.write(str(max_array).replace('\n','').replace(' ',' ') + '\n')
    file.write(str(massTree) + '\n')
    file.write(str(massResolution) + '\n')
    file.write(str(massHost) + '\n')
    file.write(str(radiusVirialHost) + '\n')
    file.write(str(countSubhalosMean) + '\n')
    file.close()

### TRAINING THE EMULATOR ###
model = RealNVP(num_coupling_layers=12)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

history = model.fit(
    augmented_normalized_data, batch_size=256, epochs=50, verbose=2, validation_split=0.2
)

model.save_weights('../data/emulatorModelPipeline.weights.h5')

### CREATING EMULATOR DATA ###
# Loading weights from emulator
emulator = RealNVP(num_coupling_layers=12)
emulator.load_weights('../data/emulatorModelPipeline.weights.h5')

# Generating emulator subhalos
num_subhalos = 500

samples = emulator.distribution.sample(num_subhalos)
x, _ = emulator.predict(samples)
xt = norm_transform_inv(x, np.nanmin(data, axis = 0), np.nanmax(data, axis = 0), -1, 1)

# Creating clip to impose physical constraints on emulator halos
clip = (xt[:,0] > np.log10(2.0*massResolution/massTree)) & (xt[:,2] <= 0.0) & (xt[:,2] > -xt[:,0]+np.log10(massResolution/massTree))# & (xt[:,3] >= 0.5) & (xt[:,2] < np.log10(1e9/(massHost * 10**xt[:,0])))

### MAKING PLOTS ###
# Generate a weighted subsample of the original data with same number of subhalos as emulator subhalo population after clip is applied
w = weight[subhalos]
i = np.arange(0, w.size, 1, dtype=int)
subsample = np.random.choice(i, size= num_subhalos, replace=True, p=w/np.sum(w))

# loss vs epoch plot
plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["train", "validation"], loc="upper right")
plt.ylabel("loss", fontsize = 'large')
plt.xlabel("epoch", fontsize = 'large')
plt.savefig('plots/loss_pipeline.pdf')

# Concentration CDFs
cs = np.linspace(3, 15, 100)
x_gal = []
x_em = []
for c in cs:
    n = np.sum(data[subsample, 1] < c)
    x_gal.append(n)

    n = np.sum(xt[clip, 1] < c)
    x_em.append(n)

x_gal = np.array(x_gal)/len(data[subsample, 1])
x_em = np.array(x_em)/len(xt[clip, 1])

max_diff = np.max(np.abs(x_gal - x_em))

for i in range(len(x_gal)):
    if np.abs(x_gal[i] - x_em[i]) == max_diff:
        print('concentration at max_diff: ', cs[i])
        c0 = cs[i]

plt.figure()
plt.plot(cs, x_gal, 'k-', label = 'Galacticus')
plt.plot(cs, x_em, 'r-', label = 'Emulator')
plt.axvline(x = c0, linestyle = '--', color = 'grey')
plt.xlim(3.5, 4)
plt.ylim(0, 1)
plt.xlabel('Concentration', fontsize = 'large')
plt.ylabel('CDF', fontsize = 'large')
plt.legend()
plt.savefig('plots/con_cdf_pipeline.pdf')

# 2Sample KS tests
print('KS concentration: ', stats.ks_2samp(data[subsample, 1], xt[clip, 1]))
print('KS bound mass: ', stats.ks_2samp(data[subsample, 2], xt[clip, 2]))
print('KS infall redshift: ', stats.ks_2samp(data[subsample, 3], xt[clip, 3]))
print('KS truncation radius: ', stats.ks_2samp(data[subsample, 4], xt[clip, 4]))
print('KS projected radius: ', stats.ks_2samp(data[subsample, 5], xt[clip, 5]))

# 2D density plots
from scipy.stats import gaussian_kde

concentration_density_galacticus = np.vstack([data[:, 0][subsample], data[:,1][subsample]])
z1_galacticus = gaussian_kde(concentration_density_galacticus)(concentration_density_galacticus)
concentration_density_generated = np.vstack([xt[clip, 0], xt[clip, 1]])
z1_generated = gaussian_kde(concentration_density_generated)(concentration_density_generated)

mass_bound_density_galacticus = np.vstack([data[:, 0][subsample], data[:, 2][subsample]])
z2_galacticus = gaussian_kde(mass_bound_density_galacticus)(mass_bound_density_galacticus)
mass_bound_density_generated = np.vstack([xt[clip, 0], xt[clip, 2]])
z2_generated = gaussian_kde(mass_bound_density_generated)(mass_bound_density_generated)

redshift_infall_density_galacticus = np.vstack([data[:, 0][subsample], data[:, 3][subsample]])
z3_galacticus = gaussian_kde(redshift_infall_density_galacticus)(redshift_infall_density_galacticus)
redshift_infall_density_generated = np.vstack([xt[clip, 0], xt[clip, 3]])
z3_generated = gaussian_kde(redshift_infall_density_generated)(redshift_infall_density_generated)

orbital_radius_density_galacticus = np.vstack([data[:, 0][subsample], data[:, 4][subsample]])
z4_galacticus = gaussian_kde(orbital_radius_density_galacticus)(orbital_radius_density_galacticus)
orbital_radius_density_generated = np.vstack([xt[clip, 0], xt[clip, 4]])
z4_generated = gaussian_kde(orbital_radius_density_generated)(orbital_radius_density_generated)

truncation_radius_density_galacticus = np.vstack([data[:, 0][subsample], data[:, 5][subsample]])
z5_galacticus = gaussian_kde(truncation_radius_density_galacticus)(truncation_radius_density_galacticus)
truncation_radius_density_generated = np.vstack([xt[clip, 0], xt[clip, 5]])
z5_generated = gaussian_kde(truncation_radius_density_generated)(truncation_radius_density_generated)

f, axes = plt.subplots(5, 2)
f.set_size_inches(15, 18)

axes[0, 0].scatter(data[:, 0][subsample], data[:, 1][subsample], c = z1_galacticus, s=9)
axes[0, 0].set(title="Galacticus", xlabel="mass infall", ylabel="concentration")
axes[0, 1].scatter(xt[clip, 0], xt[clip, 1], c = z1_generated, s=9)
axes[0, 1].set(title="Generated", xlabel="mass infall", ylabel="concentration")
axes[1, 0].scatter(data[:, 0][subsample], data[:, 2][subsample], c = z2_galacticus, s=9)
axes[1, 0].set(title="Galacticus", xlabel="mass infall", ylabel="mass bound")
axes[1, 1].scatter(xt[clip, 0], xt[clip, 2], c = z2_generated, s=9)
axes[1, 1].set(title="Generated", xlabel="mass infall", ylabel="mass bound")
axes[2, 0].scatter(data[:, 0][subsample], data[:, 3][subsample], c = z3_galacticus, s=9)
axes[2, 0].set(title="Galacticus", xlabel="mass infall", ylabel="redshift infall")
axes[2, 1].scatter(xt[clip, 0], xt[clip, 3], c = z3_generated, s=9)
axes[2, 1].set(title="Generated", xlabel="mass infall", ylabel="redshift infall")
axes[3, 0].scatter(data[:, 0][subsample], data[:, 4][subsample], c = z4_galacticus, s=9)
axes[3, 0].set(title="Galacticus", xlabel="mass infall", ylabel="orbital radius")
axes[3, 1].scatter(xt[clip, 0], xt[clip, 4], c = z4_generated, s=9)
axes[3, 1].set(title="Generated", xlabel= "mass infall", ylabel="orbital radius")
axes[4, 0].scatter(data[:, 0][subsample], data[:, 5][subsample], c = z5_galacticus, s=9)
axes[4, 0].set(title="Galacticus", xlabel="mass infall", ylabel="truncation radius")
axes[4, 0].set_ylim([-4.0, 0])
axes[4, 1].scatter(xt[clip, 0], xt[clip, 5], c = z5_generated, s=9)
axes[4, 1].set(title="Generated", xlabel="mass infall", ylabel="truncation radius")
axes[4, 1].set_ylim([-4.0, 0])
plt.savefig('plots/density_pipeline.png')

# Histograms of subhalo parameters 
f, axes = plt.subplots(6)
f.set_size_inches(15, 20)

axes[0].hist(data[:, 0][subsample], bins = 70, range = (-5, 0), label = 'Galacticus', fill = True, edgecolor = 'blue')
axes[0].hist(xt[clip, 0], bins = 70, range = (-5, 0), label = 'Generated', fill = False, edgecolor = 'orange')
axes[0].set(title = 'Infall Mass')
axes[0].legend()
axes[1].hist(data[:, 1][subsample], bins = 70, range = (0, 30), label = 'Galacticus', fill = True, edgecolor = 'blue')
axes[1].hist(xt[clip, 1], bins = 70, range = (0, 30), label = 'Generated', fill = False, edgecolor = 'orange')
axes[1].set(title = 'Concentration')
axes[1].legend()
axes[2].hist(data[:, 2][subsample], bins = 70, range = (-4, 0), label = 'Galacticus', fill = True, edgecolor = 'blue')
axes[2].hist(xt[clip, 2], bins = 70, range = (-4, 0), label = 'Generated', fill = False, edgecolor = 'orange')
axes[2].set(title = 'Bound Mass')
axes[2].legend()
axes[3].hist(data[:, 3][subsample], bins = 70, range = (0, 10), label = 'Galacticus', fill = True, edgecolor = 'blue')
axes[3].hist(xt[clip, 3], bins = 70, range = (0, 10), label = 'Generated', fill = False, edgecolor = 'orange')
axes[3].set(title = 'Infall Redshift')
axes[3].legend()
axes[4].hist(data[:, 4][subsample], bins = 70, range = (-2, 2), label = 'Galacticus', fill = True, edgecolor = 'blue')
axes[4].hist(xt[clip, 4], bins = 70, range = (-2, 2), label = 'Generated', fill = False, edgecolor = 'orange')
axes[4].set(title = 'Orbital radius')
axes[4].legend()
axes[5].hist(data[:, 5][subsample], bins = 70, range = (-4, 0), label = 'Galacticus', fill = True, edgecolor = 'blue')
axes[5].hist(xt[clip, 5], bins = 70, range = (-4, 0), label = 'Generated', fill = False, edgecolor = 'orange')
axes[5].set(title = 'Truncation radius')
axes[5].legend()
plt.savefig('plots/histograms_pipeline.png')

# N.B. distribution plot
N = countSubhalosMean
s_I = 0.18
p = 1/(1 + N*s_I**2)
r = 1/s_I**2
x = np.arange(stats.nbinom.ppf(0.01, r, p),stats.nbinom.ppf(0.99, r, p))
plt.plot(x, stats.nbinom.pmf(x, r, p), 'ko', ms=1, label='nbinom pmf')
plt.savefig('plots/negative_binomial.png')

print('code executed!')
