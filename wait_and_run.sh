#!/bin/bash

# Set up logging
log_file="gpu_monitor.log"
exec 1> >(tee -a "$log_file") 2>&1

echo "Starting GPU process monitor at $(date)"

# Function to check if a process exists
check_process() {
    if ps -p $1 > /dev/null; then
        return 0    # Process exists
    else
        return 1    # Process doesn't exist
    fi
}

# Function to check if both processes are done
check_both_processes() {
    local pid1=$1
    local pid2=$2
    
    if ! check_process $pid1 && ! check_process $pid2; then
        return 0    # Both processes are done
    else
        return 1    # At least one process is still running
    fi
}

# Main monitoring loop
echo "Monitoring processes 628013 and 628014..."

while true; do
    if check_both_processes 628013 628014; then
        echo "Both processes have completed at $(date)"
        echo "Starting sequence_run.sh..."
        
        # Execute the sequence run script
        if bash sequency_run.sh; then
            echo "sequence_run.sh completed successfully"
            break
        else
            echo "Error: sequence_run.sh failed with exit code $?"
            exit 1
        fi
    fi
    
    # Wait for 30 seconds before checking again
    sleep 30
done

echo "Monitor script completed at $(date)"
exit 0