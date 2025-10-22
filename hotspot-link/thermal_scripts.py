#!/usr/bin/env python
import os
import sys
import subprocess

# num_shaders = 80
# interval_s = 1e-9       # time interval in seconds
num_shaders = int(sys.argv[1])
interval_s = float(sys.argv[2])
power_values = [float(x) for x in sys.argv[3:]]
assert len(power_values) == num_shaders, "Error: number of power values must match num_shaders."

base_dir = os.path.dirname(__file__)
hotspot_dir = base_dir[:base_dir.rfind("/")]
hotspot_dir = os.path.join(hotspot_dir, "HotSpot")
hotspot_binary = os.path.join(hotspot_dir, "hotspot")
floorplan = os.path.join(base_dir, "4090.flp")

# Init delete primary file
powerLogFile = os.path.join(base_dir, "PeriodicPower.log")
powerInstantaneousFile = os.path.join(base_dir, "InstantaneousPower.log")
thermalLogFile = os.path.join(base_dir, "PeriodicThermal.log")
temperatureInitFile = os.path.join(base_dir, "Temperature.init")
instantaneousTemperatureFile = os.path.join(base_dir, "InstantaneousTemperature.log")
first_run = not os.path.exists(powerLogFile)
if first_run:
    for file in [powerLogFile, powerInstantaneousFile, thermalLogFile, instantaneousTemperatureFile, temperatureInitFile]:
        if os.path.exists(file):
            os.remove(file)

# establish log files
powerLogFileName = open(os.path.join(base_dir,"PeriodicPower.log"), 'a')
powerInstantaneousFileName = open(os.path.join(base_dir,"InstantaneousPower.log"), 'w')
thermalLogFileName = open(os.path.join(base_dir,"PeriodicThermal.log"), 'a')

# write the headings
Headings = ""
for i_cycle in range(num_shaders):
    Headings += "SM_"+str(i_cycle)+"\t"
Initial_flag = os.stat(os.path.join(base_dir,"PeriodicPower.log")).st_size == 0
if Initial_flag:
    powerLogFileName.write(Headings + "\n")
    thermalLogFileName.write(Headings + "\n")
powerInstantaneousFileName.write(Headings + "\n")

# write the power
Readings = ""
for i_cycle in range(num_shaders):
    # Readings += str(12) +"\t"
    Readings += str(power_values[i_cycle]) + "\t"
powerInstantaneousFileName.write (Readings+"\n")
powerInstantaneousFileName.close ()
powerLogFileName.write (Readings+"\n")
powerLogFileName.close()

# hotspot command
hotspot_args = ['-c', os.path.join(base_dir, "hotspot.config"),
                '-f', floorplan,
                '-sampling_intvl', str(interval_s),
                '-p', os.path.join(base_dir, "InstantaneousPower.log"),
                '-o', os.path.join(base_dir, "InstantaneousTemperature.log")]
if not Initial_flag:
    hotspot_args += ['-init_file', os.path.join(base_dir, "Temperature.init")]

# call hotspot
temperatures = subprocess.check_output([hotspot_binary] + hotspot_args)

# write the temperature
with open(os.path.join(base_dir, "Temperature.init"), 'w') as f:
    f.write(temperatures.decode('utf-8'))
with open(os.path.join(base_dir, "InstantaneousTemperature.log"), 'r') as instTemperatureFile:
    instTemperatureFile.readline()
    thermalLogFileName.write(instTemperatureFile.readline())
thermalLogFileName.close()