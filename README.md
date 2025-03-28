# normalizing-flows
This repo contains the code used to run a forward modeling lensing analysis on a population of subhalos generated from a normalizing flows algorithm. The code is split into 3 main steps:

1. Generate a set of Galacticus merger trees. The file "darkMatterOnlySubHalosPipeline.xml" is the input parameter file that gets read into Galacticus, "dmosh_pipeline.sh" is the job script used to run Galacticus on this parameter file, and "darkMatterOnlySubHalosPipeline.hdf5" is the output file after Galacticus has generated the merger trees.
2. Train the emulator on Galacticus data. The script "normalizing_flows_pipeline.py" reads in the Galacticus data from step 1, and feeds the data into the emulator. "flows_pipeline.sh" is the job file used to run normalizing_flows_pipeline.py.
3. Perform the forward modeling analysis to generate S_lens values. The script "subhalos_inference_pipeline.py" uses the emulator to generate a population of emulated subhalos, and runs it through a forward modeling analysis from the analysis code samana. The .sh file used to run this final step is "inference_pipeline.sh".

NOTE: In steps 2 and 3, both files "normalizing_flows_pipeline.py" and "subhalos_inference_pipeline.py" import functions/classes from the script "emulator_pipeline.py". This script contains functions for transforming Galacticus to and from it's normalized coordinates and hypercube coordinates, as well as the code which builds the normalizing flows architecture.
