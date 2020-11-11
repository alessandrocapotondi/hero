#!/bin/bash

#################################################
######       PARAMETER DEFINITIONS        #######
#################################################

progs=(2mm 3mm atax axpy bicg conv2d gemm seq)
#progs=(axpy bicg conv2d seq)
#progs=(axpy)
#back=(axpy bicg conv2d seq)
back=(${progs[*]})
samples=64
out="measurements"

#################################################
######        FUNCTION DEFINITIONS        #######
#################################################

# Function to launch CMUX in the background
launch_cmux() {
  taskset -c 1 ./cmux &
  cmux_pid=$!
  echo "CMUX launched with pid ${cmux_pid}"
  sleep 1
}

# Function to kill CMUX
kill_cmux() {
   echo "Killing CMUX with pid ${cmux_pid}"
   kill ${cmux_pid}
}

# Function to launch programs in the background
launch_background() {
  taskset -c 0 ./$1 &
  bg1_pid=$!
  taskset -c 2 ./$1 &
  bg2_pid=$!
  echo "${bprog} launched with pids ${bg1_pid} and ${bg2_pid}"
  sleep 1
}

# Function to kill the two background programs
kill_background() {
  echo "Killing ${bprog} with pid ${bg1_pid} and ${bg2_pid}"
  kill ${bg1_pid} ${bg2_pid}
}

# Function to perform measurements
run_benchmark() {
  echo "Running ${prog} with background ${bprog}"
  for (( i=1; i<=${samples}; i++ )); 
  do 
    taskset -c 3 ./${prog}_PREM.elf \
	    >> ${out}/${prog}_${bprog}$1
  done
}


#################################################
######             MAIN SCRIPT            #######
#################################################

# Prepare the platform
source ./sourceme.sh
mkdir ${out}
./prepare_board.sh &> /dev/null
renice -n -20 $$

# First, PREM without interference
launch_cmux
for prog in ${progs[*]}
do
  run_benchmark "noint.csv"
done

# Second, PREM with non-PREM interference
for bprog in ${back[*]}
do
  launch_background ${bprog}_noise.elf
  for prog in ${progs[*]}
  do
    run_benchmark "_noPREM.csv"
  done
  kill_background
done
kill_cmux

# Third, PREM with PREM interference
for bprog in ${back[*]}
do
  launch_cmux
  launch_background ${bprog}_noise_PREM.elf
  for prog in ${progs[*]}
  do
    run_benchmark "_PREM.csv"
  done
  kill_background
  kill_cmux
done
