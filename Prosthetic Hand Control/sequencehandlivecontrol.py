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
import msvcrt

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
def rotateServo(pin, angle):
    hand_board.digital[pin].write(angle)
    sleep(0.01)
def MoveFingers(pins,angles):
    for index in range(5):
        rotateServo(pins[index],angles[index])
def EnsurePositive(angles):
    for index in range(5):
        angles[index] = max(angles[index],0)

folder = '5class tuned rhtng 8chan fpz bias hand control sequence TEST'

if os.path.isdir(folder) == False:
    os.mkdir(folder)


###Ending angles for each gesture, hook not performed

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

#5 class MI and EMG tuned model
model = tf.keras.models.load_model('Best Models\\tuned5classRHTNGMIEMG-Active8chanfpzbias 05to20-08 train 23&24-08 test-96,8%acc.h5')

#####Setup Signal Filters#####

b,a = signal.butter(2, Wn = [48,52], btype = 'bandstop', fs = 250)  #notch filter
d,c = signal.butter(2, Wn = [4,38], btype = 'bandpass', fs = 250)  #bandpass filter


#####Setup BCI#####

BoardShim.enable_dev_board_logger ()
params = BrainFlowInputParams()
params.serial_port = args.serial_port
board = BoardShim(args.board_id, params)
board.prepare_session ()
board.config_board('x1040110Xx2040110Xx3040110Xx4040110Xx5040110Xx6040110Xx7040110Xx8040110X') #configures board appropriately for active electrodes
board.start_stream (45000)  #EEG starts streaming
print('stream started')


class_predictions = []
class_probabilities = []
times = []
pred_filt_times = []
prediction_count = 0
#set hand to open position:
thumb_angle = 0
index_angle = 0
middle_angle = 0
ring_angle = 0
pinky_angle = 0
angles = [160-thumb_angle, index_angle, 160-middle_angle, ring_angle, pinky_angle]
MoveFingers(pins,angles)


trial_number = input('Enter trial number:')  #user enters trial number to manually start the trial when they are ready
file_name = 'Sequence trial ' + trial_number

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

while True:

    if msvcrt.kbhit():   #ends loop if user presses any key on keyboard
        break
    main_start = time.time() 
    
    

    data = board.get_current_board_data(window_len)  #get latest 2 sec of EEG data
    eeg_data = data[1:9]  #get data for the 8 electrodes
    
    if eeg_data.shape[1] == window_len:   #ensures full 2sec window before starting classification

        #Filter data
        filtered_data=np.ndarray((eeg_data.shape[0],window_len))
        for chan in range(len(eeg_data)):
            data_to_filter = eeg_data[chan]
            notch_filt = signal.filtfilt(b,a,data_to_filter, padtype='even')
            filtered_data[chan] = signal.filtfilt(d,c,notch_filt, padtype='even')


        CNN_input = filtered_data.reshape(1,filtered_data.shape[0],filtered_data.shape[1],1)        

        prediction = model.predict(CNN_input)  ##Prediction = array of probabilities for each class 
        end = time.time()
        duration = end-main_start
        pred_filt_times.append(duration)  #Record time to process live EEG and make a prediction

        prediction_count+=1
        print('Prediction:')
        print(prediction)
        class_probabilities.append(np.array(prediction))

        output = np.argmax(prediction) #Predicted class
        print(output)
        class_predictions.append(output)
        
        if prediction.max() > 0:  #can change this value to a suitable probability threshold to minimise low confidence predictions 

            if output == 0:  #RH - pinch
                if thumb_angle<20:
                    thumb_angle+=1
                if index_angle<70:
                    index_angle+=1
                middle_angle = ring_angle = pinky_angle = 0

                if thumb_angle == 20 and index_angle == 70: #play sound when gesture complete
                    print('\a')


            elif output == 1:  #TNG - point
                index_angle = 0
                thumb_angle = 0
                if middle_angle<140:
                    middle_angle+=2
                if middle_angle>140:
                    middle_angle=140
                ring_angle = pinky_angle = middle_angle

                if middle_angle == 140:  #play sound when gesture complete
                    print('\a')

            elif output == 2:   #Blinking - reset hand to open position
                
                thumb_angle=0
                middle_angle=0
                ring_angle=0
                pinky_angle=0
                index_angle=0
                time.sleep(1)  #1 sec delay after so user can begin has sufficient time to begin performing next task


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
        EnsurePositive(angles)
        MoveFingers(pins,angles)  #write latest angles to board to control fingers to moves
        

        main_end = time.time()
        total_duration = main_end - main_start
        times.append(total_duration)  #total time to process EEG, make a prediction and send a command to the hand

    

#save predictions, probabilities and times after key pressed to end trial
np.savetxt(folder + '\\' + file_name + ' class pred.csv', class_predictions)
np.save(folder + '\\' + file_name + ' class probabilities.npy', class_probabilities)
np.savetxt(folder + '\\' + file_name + ' times.csv', times)
np.savetxt(folder + '\\' + file_name + ' predandfilt times.csv', pred_filt_times)
print('Files Saved...')

#end communication with headset and hand boards
hand_board.exit()
board.release_session ()

