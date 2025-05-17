#!/bin/bash
#SBATCH -J pipeline
#SBATCH -o %x.%J.out
##SBATCH -e %x.%J.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 04:00:00

export OPENBLAS_NUM_THREADS=2

source /cosma/apps/dp004/dc-mcgi1/python_env/pipeline/bin/activate

### L025m5

# swift-pipeline -C ./colibre \
#     -c SOAP/halo_properties_0076.hdf5 \
#     -s snapshots/colibre_0076/colibre_0076.hdf5 \
#     -i "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0025N0752/Thermal_approx_grav" \
#     -o "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0025N0752/Thermal_approx_grav/pipeline_z2"
#
# swift-pipeline -C ./colibre \
#     -c SOAP_uncompressed/halo_properties_0015.hdf5 \
#     -s swift_snapshots/swift_015/snap_015.0.hdf5 \
#     -i "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0025N0752" \
#     -o "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0025N0752/pipeline_z2"

swift-pipeline -C ./colibre \
    -c SOAP_uncompressed/halo_properties_0015.hdf5 SOAP/halo_properties_0076.hdf5 \
    -n EAGLE COLIBRE_THERMAL \
    -s swift_snapshots/swift_015/snap_015.0.hdf5 snapshots/colibre_0076/colibre_0076.hdf5 \
    -i "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0025N0752" "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0025N0752/Thermal_approx_grav" \
    -o "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0025N0752/L25_comparison_z2" \
    -j 10

#### L100m6

# swift-pipeline -C ./colibre \
#     -c SOAP/halo_properties_0076.hdf5 \
#     -s snapshots/colibre_0076/colibre_0076.hdf5 \
#     -i "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0100N1504/Thermal" \
#     -o "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0100N1504/Thermal/pipeline_z2"
#
# swift-pipeline -C ./colibre \
#     -c SOAP_uncompressed/halo_properties_0015.hdf5 \
#     -s swift_snapshots/swift_015/snap_015.0.hdf5 \
#     -i "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0100N1504" \
#     -o "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0100N1504/pipeline_z2"

swift-pipeline -C ./colibre \
    -c SOAP_uncompressed/halo_properties_0015.hdf5 SOAP/halo_properties_0076.hdf5 \
    -n EAGLE COLIBRE_THERMAL \
    -s swift_snapshots/swift_015/snap_015.0.hdf5 snapshots/colibre_0076/colibre_0076.hdf5 \
    -i "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0100N1504" "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0100N1504/Thermal" \
    -o "/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/L0100N1504/L100_comparison_z2" \
    -j 10

echo "Job complete!"
