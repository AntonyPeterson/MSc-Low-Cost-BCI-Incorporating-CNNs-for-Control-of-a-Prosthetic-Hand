import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque
import os
import random
import tensorflow as tf
import pandas as pd
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from scipy import signal
import time
from math import inf
from pyfirmata import Arduino, SERVO
from time import sleep

parser = argparse.ArgumentParser ()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
parser.add_argument ('--timeout', type = int, help  = 'timeout for device discovery or connection', required = False, default = 0)
parser.add_argument ('--ip-port', type = int, help  = 'ip port', required = False, default = 0)
parser.add_argument ('--ip-protocol', type = int, help  = 'ip protocol, check IpProtocolType enum', required = False, default = 0)
parser.add_argument ('--ip-address', type = str, help  = 'ip address', required = False, default = '')
parser.add_argument ('--serial-port', type = str, help  = 'serial port', required = False, default = 'COM4')
parser.add_argument ('--mac-address', type = str, help  = 'mac address', required = False, default = '')
parser.add_argument ('--other-info', type = str, help  = 'other info', required = False, default = '')
parser.add_argument ('--streamer-params', type = str, help  = 'streamer params', required = False, default = '')
parser.add_argument ('--serial-number', type = str, help  = 'serial number', required = False, default = '')
parser.add_argument ('--board-id', type = int, help  = 'board id, check docs to get a list of supported boards', required = False, default=0)
parser.add_argument ('--log', action = 'store_true')
args = parser.parse_args ()

######Setup Arduino#####

port = 'COM6'
hand_board = Arduino(port)
#Function to write servo motor angles to pin on board to move fingers
def rotateServo(pin, angle):
    hand_board.digital[pin].write(angle)
    sleep(0.01)
    
def MoveFingers(pins,angles):
    for index in range(5):
        rotateServo(pins[index],angles[index])
def EnsurePositive(angles):
    for index in range(5):
        angles[index] = max(angles[index],0)

folder = '5class tuned rhtng 8chan fpz bias hand control trials'   #Create folder to save trial results

if os.path.isdir(folder) == False:
    os.mkdir(folder)



###Ending angles for each gesture

gesture_ends = {'Tripod':[160-30,70,160-70,0,0],'Grasp':[160-140,140,160-140,140,140],'Pinch':[160-20,70,160-0,0,0],'Point':[160-0,0,160-140,140,140], 'Hook':[160-0,70,160-70,70,70]}

#####Assign pin numbers to finger names######

thumb = 3
index = 4
middle = 7
ring = 8
pinky = 9

pins = [thumb,index,middle,ring,pinky]

#####Rotation directions#####

cw_pins = [thumb, middle]
ccw_pins = [index, ring, pinky]


#####Initialise Servos#####

for pin in ccw_pins:
    hand_board.digital[pin].mode = SERVO
    hand_board.digital[pin].write(0)
for pin in cw_pins:
    hand_board.digital[pin].mode = SERVO
    hand_board.digital[pin].write(180)



#######Load Model######

window_len = 500   #2sec

#5 class MI and EMG tuned model
model = tf.keras.models.load_model('Best Models\\tuned5classRHTNGMIEMG-Active8chanfpzbias 05to20-08 train 23&24-08 test-96,8%acc.h5')

#####Setup Signal Filters#####

b,a = signal.butter(2, Wn = [48,52], btype = 'bandstop', fs = 250)
d,c = signal.butter(2, Wn = [4,38], btype = 'bandpass', fs = 250)


#####Setup BCI#####

BoardShim.enable_dev_board_logger ()
params = BrainFlowInputParams()
params.serial_port = args.serial_port
board = BoardShim(args.board_id, params)
board.prepare_session ()
board.config_board('x1040110Xx2040110Xx3040110Xx4040110Xx5040110Xx6040110Xx7040110Xx8040110X') #configures board appropriately for active electrodes
board.start_stream (45000)   #EEG starts streaming
print('stream started')

while True:
    class_predictions = []
    class_probabilities = []
    prediction_count = 0
    #Set hand to starting/open position
    thumb_angle = 0
    index_angle = 0
    middle_angle = 0
    ring_angle = 0
    pinky_angle = 0
    angles = [160-thumb_angle, index_angle, 160-middle_angle, ring_angle, pinky_angle]
    MoveFingers(pins,angles)

    #Text input for user to select which gesture to perform
    gesture = input('Enter Gesture to perform:')
    trial_number = input('Enter trial number:')
    file_name = gesture + ' trial ' + trial_number
    end_angles = gesture_ends[gesture]

    #3 second countdown
    print('Get Ready...')
    sleep(1)
    print(3)
    sleep(1)
    print(2)
    sleep(1)
    print(1)
    sleep(1)
    print('GO!')



    #####Main Loop#####

    while prediction_count < 100:

        main_start = time.time()
        
        
        data = board.get_current_board_data(window_len)  #get latest 2 sec of EEG data
        eeg_data = data[1:9]   #get only data for the 8 electrodes

        
        if eeg_data.shape[1] == window_len:   #ensures full 2sec window before starting classification

            start = time.time()
            #Filter data
            filtered_data=np.ndarray((eeg_data.shape[0],window_len))
            for chan in range(len(eeg_data)):
                data_to_filter = eeg_data[chan]
                notch_filt = signal.filtfilt(b,a,data_to_filter, padtype='even')  #notch filter
                filtered_data[chan] = signal.filtfilt(d,c,notch_filt, padtype='even')  #bandpass filter


            CNN_input = filtered_data.reshape(1,filtered_data.shape[0],filtered_data.shape[1],1)

            end = time.time()
            duration = end - start
            print('Filter time: ' + str(duration))

            start = time.time()
            

            prediction = model.predict(CNN_input)   #Prediction = array of probabilities for each class
            prediction_count+=1
            print('Prediction:')   #prediction printed to screen
            print(prediction)
            class_probabilities.append(np.array(prediction))

            output = np.argmax(prediction)  #class that was predicted
            print(output)
            class_predictions.append(output)
            
            if prediction.max() > 0:   #can change this value to a suitable probability threshold to minimise low confidence predictions

                if output == 0:  #RH - pinch
                    if thumb_angle<20:
                        thumb_angle+=1
                    if index_angle<70:
                        index_angle+=1
                    middle_angle = ring_angle = pinky_angle = 0

                    if thumb_angle == 20 and index_angle == 70: # play sound when gesture complete
                        print('\a')


                elif output == 1:  #TNG - point
                    index_angle = 0
                    thumb_angle = 0
                    if middle_angle<140:
                        middle_angle+=2
                    if middle_angle>140:
                        middle_angle=140
                    ring_angle = pinky_angle = middle_angle

                    if middle_angle == 140:    #play sound when gesture complete
                        print('\a')

                elif output == 2:   #Blinking - hook
                    thumb_angle=0
                    if index_angle < 70:
                        index_angle+=1
                    if index_angle > 70:
                        index_angle = 70
                    middle_angle=ring_angle=pinky_angle=index_angle

                    if index_angle == 70:    #play sound when gesture complete
                        print('\a')

                elif output == 3: #Eyebrows - tripod pinch
                    if thumb_angle<30:
                        thumb_angle+=1
                    if index_angle<70 and middle_angle <70:
                        index_angle+=1
                        middle_angle+=1
                    ring_angle = pinky_angle = 0

                    if thumb_angle == 30 and index_angle==middle_angle==70:  #play sound when gesture complete
                        print('\a')
                
                elif output == 4: #Teeth Clench - grasp
                    if thumb_angle<140:
                        thumb_angle+=2
                    if thumb_angle>140:
                        thumb_angle = 140
                    index_angle=middle_angle=ring_angle=pinky_angle=thumb_angle
                    if thumb_angle == 140: #play sound when gesture complete
                        print('\a')

                
            else:
                print('Probability too small')


            angles = [160-thumb_angle, index_angle, 160-middle_angle, ring_angle, pinky_angle]  #latest servo angles
            EnsurePositive(angles)  #Ensures angles are positive, meant for thumb and middle angle to prevent them from being negative
            MoveFingers(pins,angles)  #write servo angles to board to move fingers
            end = time.time() 
            duration = end-start
            print('Prediction and movement time: ' + str(duration))
            

        main_end = time.time()
        total_duration = main_end - main_start
        print('Time per sample update: ' + str(total_duration))


    np.savetxt(folder + '\\' + file_name + ' class pred.csv', class_predictions) #save predictions
 
    np.save(folder + '\\' + file_name + ' class probabilities.npy', class_probabilities)  #save probability arrays
    print('Files Saved...')

#end communication with headset and hand boards    
hand_board.exit()
board.release_session ()

