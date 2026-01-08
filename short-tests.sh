#!/bin/bash

set -e

if [ ! -n "$CUDA_INSTALL_PATH" ]; then
	echo "ERROR ** Install CUDA Toolkit and set CUDA_INSTALL_PATH.";
	exit;
fi

#Make the simulator
export PATH=$CUDA_INSTALL_PATH/bin:$PATH;
source ./gpu-simulator/setup_environment.sh
./setup_hotspot.sh
cd HotSpot && make
cd ..
make -j -C ./gpu-simulator

#Get the pre-run trace files
rm -rf ./hw_run/rodinia_2.0-ft
wget https://engineering.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/latest/rodinia_2.0-ft.tgz
mkdir -p ./hw_run
tar -xzvf rodinia_2.0-ft.tgz -C ./hw_run
trash rodinia_2.0-ft.tgz

#Run the tests on the trace
./util/job_launching/run_simulations.py -C QV100-SASS -B rodinia_2.0-ft -T ./hw_run/rodinia_2.0-ft/9.1 -N Accelwattch_Test

# Wait for them to finish , monitor this simulation
./util/job_launching/monitor_func_test.py -v -N Accelwattch_Test

# Collect the stats
./util/job_launching/get_stats.py -N myTest | tee per-app-fromlaunch.csv # divide in app
./util/job_launching/get_stats.py -K -k -N myTest | tee per-kernel-instance.csv # divide in kernel for each app

./util/job_launching/get_stats.py -R -B rodinia_2.0-ft -C QV100-SASS | tee per-app-for-autoplot.csv # get stats for plot
./util/job_launching/get_stats.py -K -k -R -B rodinia_2.0-ft -C QV100-SASS | tee per-app-for-correlation.csv # get stats for correlation analysis


# Accelwattch(A Power Modeling Framework for Modern GPUs) tests
./util/job_launching/run_simulations.py -B rodinia_2.0-ft -C QV100-Lyhong_SIM -T ./hw_run/rodinia_2.0-ft/9.1 -N Accelwattch_Test

# Collect the stats
./util/job_launching/get_stats.py -N Accelwattch_Test | tee accelwattch-per-app-fromlaunch.csv
./util/job_launching/get_stats.py -K -k -N Accelwattch_Test | tee accelwattch-per-kernel-instance.csv

# Accelwattch HW tests
./util/job_launching/run_simulations.py -B rodinia_2.0-ft -a -C QV100-Accelwattch_SASS_HW -T ./hw_run/rodinia_2.0-ft/9.1 -N AccelwattchHW_Test # require corresponding hw_perf.csv

# run your own traces on your device
./util/tracer_nvbit/install_nvbit.sh
make -C ./util/tracer_nvbit/
git clone https://github.com/accel-sim/gpu-app-collection
source ./gpu-app-collection/src/setup_environment
make -j -C ./gpu-app-collection/src rodinia_2.0-ft
make -C ./gpu-app-collection/src data

./util/tracer_nvbit/run_hw_trace.py -B rodinia_2.0-ft -D 0 # traces result in hw_run/traces/device-0/<cuda-version>
