# kinect-imu-fusion
Classification of Kinect and IMU sensor data

program used in bachelor's thesis of Johanna Peham, June 2019
under supervision of Andreas Kimmig (KIT-IMI) and Kinemic GmbH

##Use-Case:
action data with model car from LEGO parts as example of assembly with tracking of depth sensor and IMU, for later analysis and prediction

Kinect skeleton data consists of 25 joints, here finger tips and wrist are used
Kinect works with 30 Hz
for every frame the coordinates of the skeleton and points-of-interest are drawn onto the live-feed and saved into CSV-files after the recording ended
points-of-interest (POI) mark specific points, necessary for use-case: the position of the tires of the lego car and of fields, that contain extra parts used in the assembly
the position of those POI can be set with the keys 1 and 2 for tires, and 3 and 4 for the fields and the position of the finger tips of the right hand.

the IMU wristband from kinemic GmbH collects accelerometer and gyroscope data with 50 Hz for all three axes from the right hand of the subject

the subject would go through the different motions of assembly while marking the time-slices of those actions with a key logger
in the thesis a foot pedal was used, that acts as a key from the keyboard

the collected data of those two sensors was formatted, fused together and segmented.
for predicition different classificators and features were used
and for illustiation different plots
