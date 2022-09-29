from kivy.base import runTouchApp
from kivy.lang import Builder

from kivy.app import App 
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.properties import ListProperty, ObjectProperty, StringProperty, NumericProperty, BooleanProperty
from kivy.graphics.vertex_instructions import (Rectangle, Ellipse, Line)

from kivy.graphics.context_instructions import Color
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.core.window import Window
import random

from kivy.uix.screenmanager import Screen, ScreenManager, FadeTransition

import argparse
import time
import numpy as np
import pandas as pd
import shutil
import os
from datetime import date
import threading
import matplotlib.pyplot as plt

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


class FirstScreen(Screen):
	taskname = StringProperty("")
	random_status = BooleanProperty(False)
	equal_choice_status = BooleanProperty(False)
	mi_status = BooleanProperty()
	

class SecondScreen(Screen):
	taskname = StringProperty("")
	participant_name = StringProperty("")
	trial_duration = NumericProperty(5.)
	no_trials = NumericProperty(5)
	trial_interval = NumericProperty(10)
	dynamic_label = StringProperty('')
	image_source = StringProperty('')
	choices = ListProperty([])
	equal_choice_status = BooleanProperty()
	mi_status = BooleanProperty()

	def create_choices(self):
		if self.mi_status:
			classes = ['Foot', 'Left Hand', 'Right Hand', 'Tongue'] #,'Nothing']  #uncomment to add nothing class to MI sequence without adding new button
		else:
			classes = ['Teeth Clench', 'Blinking', 'Eyebrow Raise', 'Close Eyes', 'Nothing']
		if self.equal_choice_status:
			self.choices=[]
			choice_count = 0
			while choice_count < self.no_trials:
				choice = random.choice(classes)
				if self.choices.count(choice) < self.no_trials/len(classes):
					self.choices.append(choice)
					choice_count+=1

	
	

class ThirdScreen(Screen):
	
	taskname = StringProperty("")
	choices = ListProperty([])
	participant_name = StringProperty("")
	trial_duration = NumericProperty()
	no_trials = NumericProperty()
	trial_interval = NumericProperty()
	dynamic_label = StringProperty('')
	trial_start_number = NumericProperty()	
	count = NumericProperty()
	interval_duration = NumericProperty()
	trial_number = NumericProperty()
	random_status = BooleanProperty()
	equal_choice_status = BooleanProperty()
	background_colour = ObjectProperty([1,1,1,1])
	text_colour = ObjectProperty([1,1,1,1])
	image_source = StringProperty('')
	button_disabled = BooleanProperty(False)
	mi_status = BooleanProperty()

	data_collected = threading.Event()

	BoardShim.enable_dev_board_logger ()
	params = BrainFlowInputParams()
	params.serial_port = 'COM4'
	board = BoardShim(BoardIds.CYTON_BOARD.value, params)
	#board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params) #use for testing GUI without BCI hardware


	def create_folders(self):

		if os.path.isdir(self.participant_name) == False:
			os.mkdir(self.participant_name)
		if os.path.isdir(self.participant_name + '\\' + self.taskname) == False:
			os.mkdir(self.participant_name + '\\'+ self.taskname)
		if os.path.isdir(self.participant_name + '\\' + self.taskname + '\\' + str(date.today())) == False:
			os.mkdir(self.participant_name + '\\' + self.taskname + '\\' + str(date.today()))

	
	
	def Label_Change(self, dt):
		
		self.board.prepare_session()
		self.board.config_board('x1040110Xx2040110Xx3040110Xx4040110Xx5040110Xx6040110Xx7040110Xx8040110X') #comment out when using synthetic board
		print('Session Ready...')
		Clock.schedule_interval(self.Callback_Clock , 1)

	def Callback_Clock(self,dt):

		self.count = self.count - 1

		if self.count >= 1:
			self.dynamic_label = str(self.count) + '...'
		else:
			self.text_colour = [0,1,0,1]
			self.background_colour = [0,1,0,1]
			self.dynamic_label = 'Begin!'   
			self.Stop()
			

	def Stop(self):
		
		Clock.unschedule(self.Callback_Clock)
		Clock.schedule_once(self.init_thread, 0.05)
	
	def init_thread(self, dt):
	
		thread = threading.Thread(target = self.start_stop_eeg)

		thread.start()
		thread.join()
	
		

	def start_stop_eeg(self):
		
		
		print('Beginning Trial' + str(self.trial_start_number+1))
		self.board.start_stream(45000)
		BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the thread')
			
		time.sleep(self.trial_duration)

		self.board.stop_stream()
		print('stream stopped...')
		data = self.board.get_board_data()
		self.board.release_session()

		sampling_rate = BoardShim.get_sampling_rate(2)
		eeg_channels = BoardShim.get_eeg_channels(2)
		df = pd.DataFrame(np.transpose(data))
		print('Data from the board')
		print(df.head(10))

		DataFilter.write_file (data, 'test' + str(self.trial_start_number+1) +'.csv', 'w')
		restored_data = DataFilter.read_file ('test' + str(self.trial_start_number+1) +'.csv')
		restored_df = pd.DataFrame (np.transpose (restored_data))
		shutil.move('test' + str(self.trial_start_number+1) + '.csv', self.participant_name + '\\' + self.taskname + '\\' + str(date.today()))
		print ('Data From the File')
		print (restored_df.head (10))
		print('Samples: ' + str(len(data[1])))
		print('Sampling Rate:' + str(sampling_rate))
		print(eeg_channels)

		
		print('Interval')

		self.stop_label()		
	

	def stop_label(self):

		self.text_colour = [1,0,0,1]
		self.background_colour = [1,0,0,1]
		self.dynamic_label = 'Stop!'
		Clock.schedule_once(self.init_interval_countdown, 2)

	def init_interval_countdown(self, dt):

		Clock.schedule_interval(self.interval_countdown, 1)

	def interval_countdown(self, dt):

		self.image_source = 'Interval.png'
		self.text_colour = [1,1,1,1]
		self.background_colour = [1,1,1,1]
		self.interval_duration = self.interval_duration - 1
		if self.interval_duration >= 0:
			self.dynamic_label = 'Interval: ' + str(self.interval_duration + 1) +'...'
		else:
			self.Stop2()
			self.trial_number += 1
			self.loop()
			

	def Stop2(self):

		Clock.unschedule(self.interval_countdown)
		Clock.unschedule(self.init_interval_countdown)
		Clock.unschedule(self.stop_label)
		

	def startEEGstream(self, dt): 

		if self.mi_status:
			classes = ['Foot', 'Left Hand', 'Right Hand', 'Tongue']
		else:
			classes = ['Teeth Clench', 'Blinking', 'Eyebrow Raise', 'Close Eyes']
		if self.random_status:
			self.taskname = random.choice(classes)

		elif self.equal_choice_status:
			self.taskname = self.choices[self.trial_number-1]


		self.create_folders()

		self.text_colour = [1,1,1,1]
		self.background_colour = [1,1,1,1]
		self.dynamic_label = 'Prepare to Perform ' + self.taskname + ' MI for ' + str(self.trial_duration) + ' sec. (Trial: ' + str(self.trial_number) + '/' + str(self.no_trials) +')'
		self.image_source = self.taskname +'.png'
		self.trial_start_number = len(os.listdir(self.participant_name + '\\' + self.taskname + '\\' + str(date.today())))
		self.interval_duration = self.trial_interval
		self.count=4


		Clock.schedule_once(self.Label_Change, 2)
		


	def loop(self):
		

		if self.trial_number <= self.no_trials:
			Clock.unschedule(self.startEEGstream)
			Clock.schedule_once(self.startEEGstream, 0)
		else:
			self.text_colour = [0,0,1,1]
			self.dynamic_label = 'Session Complete.'
			self.image_source = 'Complete.png'
			print('COMPLETE')
			self.button_disabled = False


		
		

class MyScreenManager(ScreenManager):
	pass
	

	
root_widget = Builder.load_string('''
	

<FirstScreen>:
	name: 'first'
	BoxLayout:
		orientation: 'vertical'
		spacing: 5
		padding: 10
		Label:
			text: 'Select Task to Perform:'
			font_size: 50
			text_size: self.width, None
			size: self.texture_size
			size_hint: 1, 0.1
			halign: 'center'
			valign: 'middle'
			canvas.after:
				Color:
					rgba: 0,0,0,0
				Rectangle:
					pos: self.pos
					size: self.size
		BoxLayout:
			orientation: 'horizontal'
			spacing: 5
			padding: 10
			size_hint_x: 1
			BoxLayout:
				orientation: 'vertical'
				spacing: 5
				padding: 5
				Label:
					text: 'Motor Imagery:'
					font_size: 40
					text_size: self.width, None
					size: self.texture_size
					size_hint_y: 0.5
					halign: 'center'
					valign: 'middle'
				BoxLayout:
					orientation: 'horizontal'
					spacing: 5
					Button:
						id: RHbutton
						text: 'Right Hand'
						font_size: 40
						text_size: self.width, None
						size: self.texture_size
						halign: 'center'
						valign: 'middle'
						on_release: root.taskname = RHbutton.text
						on_release: root.random_status = False
						on_release: root.equal_choice_status = False
						on_release: root.mi_status = True
						on_release: app.root.current = 'second'
						 	
					Button:
						id: LHbutton
						text: 'Left Hand'
						font_size: 40
						text_size: self.width, None
						size: self.texture_size
						halign: 'center'
						valign: 'middle'
						on_release: root.taskname = LHbutton.text
						on_release: root.random_status = False
						on_release: root.equal_choice_status = False
						on_release: root.mi_status = True
						on_release: app.root.current = 'second'
						
						
				BoxLayout:
					orientation: 'horizontal'
					spacing: 5
					Button:
						id: footbutton
						text: 'Foot'
						font_size: 40
						text_size: self.width, None
						size: self.texture_size
						halign: 'center'
						valign: 'middle'
						on_release: root.taskname = footbutton.text
						on_release: root.random_status = False
						on_release: root.equal_choice_status = False
						on_release: root.mi_status = True
						on_release: app.root.current = 'second'
						
						
					Button:
						id: tonguebutton
						text: 'Tongue'
						font_size: 40
						text_size: self.width, None
						size: self.texture_size
						halign: 'center'
						valign: 'middle'
						on_release: root.taskname = tonguebutton.text
						on_release: root.random_status = False
						on_release: root.equal_choice_status = False
						on_release: root.mi_status = True
						on_release: app.root.current = 'second'
				BoxLayout:
					orientation: 'horizontal'
					spacing: 5
					Button:
						id: randombutton
						text: 'Random Sequence'
						font_size: 40
						size_hint: 1, 1
						text_size: self.width, None
						halign: 'center'
						valign: 'middle'
						on_release: root.taskname = randombutton.text
						on_release: root.random_status = True
						on_release: root.equal_choice_status = False
						on_release: root.mi_status = True
						on_release: app.root.current = 'second'

					Button:
						id: equalrandombutton
						text: 'Random Sequence (Equal Trials)'
						font_size: 30
						size_hint: 1, 1
						text_size: self.width, None
						halign: 'center'
						valign: 'middle'
						on_release: root.taskname = equalrandombutton.text
						on_release: root.random_status = False
						on_release: root.equal_choice_status = True
						on_release: root.mi_status = True
						on_release: app.root.current = 'second'

			BoxLayout:
				orientation: 'vertical'
				spacing: 5
				padding: 5
				Label:
					text: 'Muscle:'
					font_size: 40
					text_size: self.width, None
					size: self.texture_size
					size_hint_y: 0.5
					halign: 'center'
					valign: 'middle'
				BoxLayout:
					orientation: 'horizontal'
					spacing: 5
					Button:
						id: Teethbutton
						text: 'Teeth Clench'
						font_size: 40
						text_size: self.width, None
						size: self.texture_size
						halign: 'center'
						valign: 'middle'
						on_release: root.taskname = Teethbutton.text
						on_release: root.random_status = False
						on_release: root.equal_choice_status = False
						on_release: root.mi_status = False
						on_release: app.root.current = 'second'
						 	
					Button:
						id: Blinkbutton
						text: 'Blinking'
						font_size: 40
						text_size: self.width, None
						size: self.texture_size
						halign: 'center'
						valign: 'middle'
						on_release: root.taskname = Blinkbutton.text
						on_release: root.random_status = False
						on_release: root.equal_choice_status = False
						on_release: root.mi_status = False
						on_release: app.root.current = 'second'
						
						
				BoxLayout:
					orientation: 'horizontal'
					spacing: 5
					Button:
						id: Eyebrowsbutton
						text: 'Eyebrow Raise'
						font_size: 40
						text_size: self.width, None
						size: self.texture_size
						halign: 'center'
						valign: 'middle'
						on_release: root.taskname = Eyebrowsbutton.text
						on_release: root.random_status = False
						on_release: root.equal_choice_status = False
						on_release: root.mi_status = False
						on_release: app.root.current = 'second'
						
						
					Button:
						id: Eyesclosedbutton
						text: 'Close Eyes'
						font_size: 40
						text_size: self.width, None
						size: self.texture_size
						halign: 'center'
						valign: 'middle'
						on_release: root.taskname = Eyesclosedbutton.text
						on_release: root.random_status = False
						on_release: root.equal_choice_status = False
						on_release: root.mi_status = False
						on_release: app.root.current = 'second'
				BoxLayout:
					orientation: 'horizontal'
					spacing: 5
					Button:
						id: randombutton2
						text: 'Random Sequence'
						font_size: 40
						size_hint: 1, 1
						text_size: self.width, None
						halign: 'center'
						valign: 'middle'
						on_release: root.taskname = randombutton2.text
						on_release: root.random_status = True
						on_release: root.equal_choice_status = False
						on_release: root.mi_status = False
						on_release: app.root.current = 'second'

					Button:
						id: equalrandombutton2
						text: 'Random Sequence (Equal Trials)'
						font_size: 30
						size_hint: 1, 1
						text_size: self.width, None
						halign: 'center'
						valign: 'middle'
						on_release: root.taskname = equalrandombutton2.text
						on_release: root.random_status = False
						on_release: root.equal_choice_status = True
						on_release: root.mi_status = False
						on_release: app.root.current = 'second'
				

<SecondScreen>:
	
	name: 'second'
	BoxLayout:
		orientation: 'vertical'	
		spacing: 5
		padding: 30
		Label:
			text: 'Session Information:'
			font_size: 50
			text_size: self.width, None
			size: self.texture_size
			halign: 'center'
			valign: 'middle'
		BoxLayout:
			orientation: 'horizontal'
			Label:
				id: task_label
				text: 'Task:'
				font_size: self.height - 20
				height: 50
				size_hint_y: None
				halign: 'center'
				valign: 'middle'
				pos: self.pos

			Label:
				id: task_text
				text: root.taskname
				font_size: self.height - 20
				height: 50
				size_hint_y: None
				valign: 'middle'
				halign: 'center'
				multiline: False
		BoxLayout:
			orientation: 'horizontal'
			Label:
				id: participant_label
				text: 'Participant:'
				font_size: self.height - 20
				height: 50
				size_hint_y: None
				halign: 'center'
				valign: 'middle'
				pos: self.pos

			TextInput:
				id: participant_text
				font_size: self.height - 20
				height: 50
				size_hint_y: None
				valign: 'middle'
				halign: 'center'
				multiline: False
				on_text: root.participant_name = participant_text.text
		BoxLayout:
			orientation: 'horizontal'
			Label:
				id: trials_label
				text: 'Number of Trials:'
				font_size: self.height - 20
				height: 50
				size_hint_y: None
				halign: 'center'
				valign: 'middle'

			TextInput:
				id: trials_text
				text: str(root.no_trials)
				font_size: self.height - 20
				height: 50
				size_hint_y: None
				valign: 'middle'
				halign: 'center'
				multiline: False
				input_filter: 'int'
				on_text: root.no_trials = int(trials_text.text)
		BoxLayout:
			orientation: 'horizontal'
			Label:
				id: duration_label
				text: 'Trial Duration (sec):'
				font_size: self.height - 20
				height: 50
				size_hint_y: None
				halign: 'center'
				valign: 'middle'
				pos: self.pos

			TextInput:
				id: duration_text
				font_size: self.height - 20
				height: 50
				size_hint_y: None
				valign: 'middle'
				halign: 'center'
				input_filter: 'float'
				text: str(root.trial_duration)
				multiline: False
				on_text: root.trial_duration = float(duration_text.text)
		BoxLayout:
			orientation: 'horizontal'
			Label:
				id: interval_label
				text: 'Interval (sec):'
				font_size: self.height - 20
				height: 50
				size_hint_y: None
				halign: 'center'
				valign: 'middle'
				pos: self.pos

			TextInput:
				id: interval_text
				text: str(root.trial_interval)
				font_size: self.height - 20
				height: 50
				size_hint_y: None
				valign: 'middle'
				halign: 'center'
				multiline: False
				input_filter: 'int'
				on_text: root.trial_interval = int(interval_text.text)
		
		BoxLayout:
			orientation: 'horizontal'
			spacing: 5
			padding: 10
			Button:
				text: 'Back'
				font_size: 50
				on_release: app.root.current = 'first'
			Button:
				text: 'Next'
				font_size: 50
				on_press: root.create_choices()
				on_press: root.dynamic_label = 'Prepare to Perform ' + root.taskname + ' MI for ' + str(root.trial_duration) + ' sec.'; root.image_source = root.taskname + '.png'; root.text_colour = [1,1,1,1]
				on_release: app.root.current = 'third'
				
				

<ThirdScreen>:
	
	name: 'third' 
	BoxLayout:
		orientation: 'vertical'
		spacing: 15
		padding: 30

		Label:
			id: changing_label
			text: root.dynamic_label
			font_size: 50
			text_size: self.width, None
			size: self.texture_size
			halign: 'center'
			valign: 'middle'
			size_hint: 1, 0.3 
			color: root.text_colour

		Image:
			source: root.image_source
			allow_stretch: True
			keep_ratio: True
			canvas.before:
				Color:
					rgba: root.background_colour
				Rectangle:
					pos: self.pos
					size: self.size
			
		BoxLayout:
			orientation: 'horizontal'
			spacing: 5
			size_hint: 1, 0.3
			Button:
				text: 'Back'
				font_size: 50
				disabled: root.button_disabled
				on_press: root.text_colour = [1,1,1,1]
				on_release: app.root.current = 'second'
				
				
			Button:
				id: start_button
				text: 'Start'
				font_size: 50
				disabled: root.button_disabled
				on_press: root.trial_number = 1
				on_release: root.loop(); root.button_disabled = True


	

			
MyScreenManager:
	
	FirstScreen:
		id: firstscreen
	SecondScreen:
		id: secondscreen
		taskname: firstscreen.taskname
		dynamic_label: thirdscreen.dynamic_label
		image_source: thirdscreen.image_source
		equal_choice_status: firstscreen.equal_choice_status
		mi_status: firstscreen.mi_status
	ThirdScreen:
		id: thirdscreen
		taskname: firstscreen.taskname
		participant_name: secondscreen.participant_name
		trial_duration: secondscreen.trial_duration
		no_trials: secondscreen.no_trials
		trial_interval: secondscreen.trial_interval
		dynamic_label: secondscreen.dynamic_label
		random_status: firstscreen.random_status
		image_source: secondscreen.image_source
		choices: secondscreen.choices
		equal_choice_status: firstscreen.equal_choice_status
		mi_status: firstscreen.mi_status



''')

class ScreenManagerApp(App):
	def build(self):
		return root_widget

ScreenManagerApp().run()

	