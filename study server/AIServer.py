import socket
import tflearn
import tensorflow as tf
import numpy as np
import sys
import os
import random
from collections import deque
from zipfile import ZipFile

def preprocess(songs = [], charts = [], ti=1024):
    '''
    this function preprocesses the dataset for use in the model

    1. load appropriate files
    2. slice input into needed chunks then flatten
    3. add these to train/test lists as specified
    '''
    fail_count = 0

    trainX = []
    trainY = []
    testX = []
    testY = []

    test_data = []
    num = 0
    for chart in charts:
        # split the testing / training data
        training = True
        song = songs[0]

        # load the song data from memory map and reshape appropriately        
        song_mm = np.load(song, mmap_mode="r")
        song_data = np.frombuffer(buffer=song_mm, dtype=np.float32, count=-1)
        song_data = song_data[0:song_mm.shape[0]*song_mm.shape[1]]
        song_data = np.reshape(song_data, song_mm.shape)

        
        # NEW: set up note data for retraining
        nd = []
        for i in range(len(chart)):
            if i != 0 and i < len(chart) - 2:
                n = chart[i].split(",")
                n[0] = int(n[0]) + 1
                n[1] = int(float(n[1]))
                nd.append(n)
        
        nd.sort(key=sort_notes)
        timestamp = 0
        note_data = []
        while len(nd) > 0:

            note_type = nd[0][0]
            note_time = nd[0][1]
            while abs(timestamp * 23 - note_time) > 23:
                note_data.append(np.zeros(7))
                timestamp += 1

            if note_type == 1:
                note_data.append(np.array([0,1,0,0,0,0,0]))
            elif note_type == 2:
                note_data.append(np.array([0,0,1,0,0,0,0]))
            elif note_type == 3:
                note_data.append(np.array([0,0,0,1,0,0,0]))
            elif note_type == 4:
                note_data.append(np.array([0,0,0,0,1,0,0]))
            else:
                note_data.append(np.zeros(7))
            timestamp += 1

            del nd[0]
        
        note_data = np.array(note_data, dtype=np.int32)

        #note_mm = np.load(chart, mmap_mode="r")
        #note_data = np.frombuffer(note_mm, dtype=np.int32, count=-1)
        #note_data = np.reshape(note_data, [len(note_mm), 7])

        # pad the note data with 0's so it matches the end of the song, adding exactly enough to make life easier for myself below
        diff = len(song_data) - len(note_data)
        padding = []
        for i in range(diff + 16):
            padding.append(np.zeros(7))

        note_data = np.append(note_data, padding, axis=0)

        # package up the last 16 blocks of data, which require padding in the song_input because the song ends
        for h in range(16):
            song_input = []
            note_input = []
            output_chunk = []
            for k in range(16):
                if h - k < 0:
                    song_input.append(np.zeros([80]))
                    if k < 12:
                        note_input.append(np.zeros([7]))
                    elif k != 15:
                        note_input.append(np.ones([7]))
                    if k > 11:
                        output_chunk.append(np.zeros([7]))
                else:
                    song_input.append(song_data[h-k])
                    if k < 12:
                        note_input.append(note_data[h-k])
                    elif k != 15:
                        note_input.append(np.ones([7]))
                    if k > 11: 
                        output_chunk.append(note_data[h-k])
            song_input = np.array(song_input).flatten()
            note_input = np.array(note_input).flatten()
            input_chunk = np.concatenate([song_input, note_input])
            output_chunk = np.concatenate(output_chunk)
            output_chunk = np.reshape(output_chunk, [4, 7])

            if training:
                trainX.append(input_chunk)
                trainY.append(output_chunk)
            else:
                testX.append(input_chunk)
                testY.append(output_chunk)
        
        # package up the data in 1445-size 1D tensors as needed for input, then append to appropriate Train / Test list
        for j in range(16, len(song_data)):
            song_input = []
            note_input = []
            output_chunk = []
            for k in range(16):
                song_input.append(song_data[j-k])
                if k < 12:
                    note_input.append(note_data[j-k])
                elif k != 15:
                    note_input.append(np.ones([7]))
                if k > 11:
                    output_chunk.append(note_data[j-k])
    
            song_input = np.array(song_input).flatten()
            note_input = np.array(note_input).flatten()
            input_chunk = np.concatenate([song_input, note_input])
            output_chunk = np.concatenate(output_chunk)
            output_chunk = np.reshape(output_chunk, [4, 7])
            if training and j < ti: # CHANGE RETRAINING AMOUNT HERE 
                trainX.append(input_chunk)
                trainY.append(output_chunk)

        num += 1

    # ensure things are working
    #print(len(trainX), "train X", len(trainY), "train Y")
    #print(len(trainX[0]), "trainX 0", len(testX[0]), "testX 0")
    #print(len(testX), "test X", len(testY), "test Y")
    #print(len(trainY[0]), "trainY 0", len(testY[0]), "testY 0")

    return trainX,trainY,testX,testY

def sort_notes(n):
    return n[1]


# NOTE: fill these in with your appropriate details
TCP_IP = 'VVVVVV'
TCP_PORT = 'VVVVVV'

BUFFER_SIZE = 524288
### importing model for predictions ###

# unpack the input data
net = tflearn.input_data([None, 1385])
song = tf.slice(net, [0,0], [-1, 1280])
song = tf.reshape(song, [-1, 16, 80])
prev_notes = tf.slice(net, [0,1280], [-1, 105])
prev_notes = tf.reshape(prev_notes, [-1, 7, 15])

# two conv layers with a 20% dropout layer after the first and max_pooling after each
song_encoder = tflearn.conv_1d(song, nb_filter=16, filter_size=3, activation="relu")
song_encoder = tflearn.dropout(song_encoder, keep_prob=0.8)
song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=2)

song_encoder = tflearn.conv_1d(song, nb_filter=32, filter_size=3, activation="relu")
song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=2)

song_encoder = tflearn.fully_connected(song_encoder, n_units=128, activation="relu")
song_encoder = tf.reshape(song_encoder, [-1,8,16])

# split song data into past chunks and current chunk
past_chunks = tf.slice(song_encoder, [0,0,0], [-1, 8, 15])
curr_chunk = tf.slice(song_encoder, [0,0,15], [-1, 8, 1])

# combine note data with processed song data
lstm_input = tf.unstack(past_chunks, axis=1)
lstm_input = tf.math.multiply(lstm_input, prev_notes)
lstm_input = tf.reshape(lstm_input, [-1]) # flatten this to add on the current chunk

# add on the final segment which does not have data yet
curr_chunk = tf.math.multiply(curr_chunk, tf.ones([8, 15]))
curr_chunk = tf.reshape(curr_chunk, [-1])
lstm_input = tf.concat([lstm_input, curr_chunk], 0)

lstm_input = tf.reshape(lstm_input, [-1, 16, 88]) # reshape to desired shape

# 2 lstm layers, then a final fully connected softmax layer
lstm_input = tflearn.lstm(lstm_input, 64, dropout=0.8, activation="relu")
lstm_input = tf.reshape(lstm_input, [-1, 8, 8])

lstm_input = tflearn.lstm(song_encoder, 64, dropout=0.8, activation="relu")

lstm_input = tflearn.fully_connected(lstm_input, n_units=28, activation="softmax")
lstm_input = tflearn.reshape(lstm_input, [-1,4,7])

# setting up final parameters
network = tflearn.regression(lstm_input, optimizer = "adam", loss="categorical_crossentropy", learning_rate=0.001, batch_size=4)
model = tflearn.DNN(network)

cwd = os.getcwd()
model_dir = cwd + "/model"
os.chdir(model_dir)
model.load("model.tfl")
os.chdir(cwd)
print("moded loaded, ready to connect!")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

conn, addr = s.accept()
print("connected!")

while 1:
    try:
        data = conn.recv(BUFFER_SIZE)

        if len(data)>0:
            data = data.decode(errors="ignore") # avoiding weirdness on godot's end adding some non-ascii garbo to the start of sent info
            data = data.split(sep="|")
            print(data)
            if data[0] == "":
                print("fuck!")
                del data[0]
            data[0] = data[0][len(data[0]) - 1]
            send_byt = ""

            # mode 1: make predictions across a given series of time (polygon)
            if int(data[0]) == 1:
                # load the song data from memory map and reshape appropriately 
                npy_file = "polygon Input.npy"     
                song_mm = np.load(npy_file, mmap_mode="r")
                song_data = np.frombuffer(buffer=song_mm, dtype=np.float32, count=-1)
                song_data = song_data[0:song_mm.shape[0]*song_mm.shape[1]]
                song_data = np.reshape(song_data, song_mm.shape)
                
                # RETRAINING STEP commented out here
                #trainX, trainY, testX, testY = preprocess(songs = [npy_file], charts = [data], ti=1024)
                #model.fit(trainX, trainY, show_metric=True, batch_size=4, n_epoch=5)

                # create the given song chunk
                # predict for the current song chunk
                # feed this prediction information back into the model

                note_queue = deque([])

                for i in range(16):
                    note_queue.append(np.zeros(7))
                
                predictions = []

                #while len(note_queue) < 16:
                    #note_queue.append(np.zeros(7))

                for j in range(len(song_data)):
                    input_chunk = []
                    song = []
                    note_data = []

                    for i in range(16):
                        if j - i < 0:
                            song.append(np.zeros(80))
                        else:
                            song.append(song_data[j-i])
                        if i < 12:
                            note_data.append(note_queue[i])
                        elif i != 15:
                            note_data.append(np.ones(7))

                    song = np.array(song).flatten()
                    note_data = np.array(note_data).flatten()
                    input_chunk = np.concatenate([song, note_data])
                    input_chunk = np.expand_dims(input_chunk, axis=0)
                    p = model.predict(input_chunk)
                    note_queue.popleft()
                    predictions.append(p[0])
                    note_queue.append(p[0][0])

                note_selections = []

                for k in range(len(predictions)):
                    guess = np.zeros([7])
                    for n in range(4):
                        try:
                            selection = np.array(predictions[k+n][3-n])
                            guess = np.add(guess, selection)
                        except IndexError:
                            # we have reached the end
                            break
                    prob = guess / np.sum(guess)
                    try:
                        choice = np.random.choice([0,1,2,3,4,5,6], p=prob)
                        note_selections.append(choice)
                    except ValueError:
                        print('uh oh')
                
                indices = data[len(data) - 2].split(sep=",")
                ind1 = int(int(float(indices[0][1:])) / 23)
                ind2 = int(int(float(indices[1][:len(indices[1]) - 1])) / 23)
                send_list = [str(1)]
                for i in range(ind1, ind2):
                    if note_selections[i] != 0:
                        send_list.append(str(note_selections[i]) + "," + str(i * 23))
                send_byt = "|".join(send_list)

            # mode 2: export to osz
            elif int(data[0]) == 2:
                '''
                Create the .osu file based on the note selections
                '''
                
                # template for beginning of file
                osu_file = """osu file format v14

[General]
AudioFilename: audio1.mp3
AudioLeadIn: 0
PreviewTime: 0
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 1
LetterboxInBreaks: 0
WidescreenStoryboard: 0

[Editor]
DistanceSpacing: 0.8
BeatDivisor: 4
GridSize: 32
TimelineZoom: 3.14

[Metadata]
Title:polygon
TitleUnicode:polygon
Artist:Sota Fujimori
ArtistUnicode:Sota Fujimori
Creator:You and KiaiTime
Version:TaikoNation v1
Source:
Tags:
BeatmapID:-1
BeatmapSetID:-1

[Difficulty]
HPDrainRate:6
CircleSize:2
OverallDifficulty:6
ApproachRate:10
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
324,344.827586206897,4,1,0,70,1,0


[HitObjects]
"""
                file_name = "Collab Editor.osu"
                outfile = open(file_name, "w+")
                del data[0]
                del data[len(data) - 1]

                for note in data:
                    # add each note to the string with its corresponding time
                    note_data = note.split(",")
                    note_type = int(note_data[0]) + 1
                    timestamp = int(float(note_data[1]))
                    if note_type == 1:
                        osu_file += ("256,192," + str(timestamp) + ",1,0,0:0:0:0:\n")
                    elif note_type == 2:
                        osu_file += ("256,192," + str(timestamp) + ",1,2,0:0:0:0:\n")
                    elif note_type == 3:
                        osu_file += ("256,192," + str(timestamp) + ",1,4,0:0:0:0:\n")
                    elif note_type == 4:
                        osu_file += ("256,192," + str(timestamp) + ",1,6,0:0:0:0:\n")
                outfile.write(osu_file)
                outfile.close()

                
                package = "Collab Editor.osz"
                with ZipFile(package, mode="w") as oszf:
                    oszf.write("audio1.mp3")
                    oszf.write(file_name)

                send_byt = "file exported!"
                # mode 3: make predictions across a given series of time (citrus)
            elif int(data[0]) == 3:
                # load the song data from memory map and reshape appropriately 
                npy_file = "citrus Input.npy"     
                song_mm = np.load(npy_file, mmap_mode="r")
                song_data = np.frombuffer(buffer=song_mm, dtype=np.float32, count=-1)
                song_data = song_data[0:song_mm.shape[0]*song_mm.shape[1]]
                song_data = np.reshape(song_data, song_mm.shape)
                
                # RETRAINING STEP
                #trainX, trainY, testX, testY = preprocess(songs = [npy_file], charts = [data], ti=1024)
                #model.fit(trainX, trainY, show_metric=True, batch_size=4, n_epoch=5)

                # create the given song chunk
                # predict for the current song chunk
                # feed this prediction information back into the model

                note_queue = deque([])

                for i in range(16):
                    note_queue.append(np.zeros(7))
                
                predictions = []

                #while len(note_queue) < 16:
                    #note_queue.append(np.zeros(7))

                for j in range(len(song_data)):
                    input_chunk = []
                    song = []
                    note_data = []

                    for i in range(16):
                        if j - i < 0:
                            song.append(np.zeros(80))
                        else:
                            song.append(song_data[j-i])
                        if i < 12:
                            note_data.append(note_queue[i])
                        elif i != 15:
                            note_data.append(np.ones(7))

                    song = np.array(song).flatten()
                    note_data = np.array(note_data).flatten()
                    input_chunk = np.concatenate([song, note_data])
                    input_chunk = np.expand_dims(input_chunk, axis=0)
                    p = model.predict(input_chunk)
                    note_queue.popleft()
                    predictions.append(p[0])
                    note_queue.append(p[0][0])

                note_selections = []

                for k in range(len(predictions)):
                    guess = np.zeros([7])
                    for n in range(4):
                        try:
                            selection = np.array(predictions[k+n][3-n])
                            guess = np.add(guess, selection)
                        except IndexError:
                            # we have reached the end
                            break
                    prob = guess / np.sum(guess)
                    try:
                        choice = np.random.choice([0,1,2,3,4,5,6], p=prob)
                        note_selections.append(choice)
                    except ValueError:
                        print('uh oh')
                
                indices = data[len(data) - 2].split(sep=",")
                ind1 = int(int(float(indices[0][1:])) / 23)
                ind2 = int(int(float(indices[1][:len(indices[1]) - 1])) / 23)
                send_list = [str(1)]
                for i in range(ind1, ind2):
                    if note_selections[i] != 0:
                        send_list.append(str(note_selections[i]) + "," + str(i * 23))
                send_byt = "|".join(send_list)

            # mode 4: export to osz (citrus)
            elif int(data[0]) == 4:
                '''
                Create the .osu file based on the note selections
                '''
                
                # template for beginning of file
                osu_file = """osu file format v14

[General]
AudioFilename: audio2.mp3
AudioLeadIn: 0
PreviewTime: 0
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 1
LetterboxInBreaks: 0
WidescreenStoryboard: 0

[Editor]
DistanceSpacing: 0.8
BeatDivisor: 4
GridSize: 32
TimelineZoom: 3.14

[Metadata]
Title:citrus
TitleUnicode:citrus
Artist:Kamome Sano
ArtistUnicode:Kamome Sano
Creator:You and KiaiTime
Version:TaikoNation v1
Source:
Tags:
BeatmapID:-1
BeatmapSetID:-1

[Difficulty]
HPDrainRate:6
CircleSize:2
OverallDifficulty:6
ApproachRate:10
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
1550,344.827586206897,4,1,0,70,1,0


[HitObjects]
"""
                file_name = "Collab Editor.osu"
                outfile = open(file_name, "w+")
                del data[0]
                del data[len(data) - 1]

                for note in data:
                    # add each note to the string with its corresponding time
                    note_data = note.split(",")
                    note_type = int(note_data[0]) + 1
                    timestamp = int(float(note_data[1]))
                    if note_type == 1:
                        osu_file += ("256,192," + str(timestamp) + ",1,0,0:0:0:0:\n")
                    elif note_type == 2:
                        osu_file += ("256,192," + str(timestamp) + ",1,2,0:0:0:0:\n")
                    elif note_type == 3:
                        osu_file += ("256,192," + str(timestamp) + ",1,4,0:0:0:0:\n")
                    elif note_type == 4:
                        osu_file += ("256,192," + str(timestamp) + ",1,6,0:0:0:0:\n")
                outfile.write(osu_file)
                outfile.close()

                
                package = "Collab Editor.osz"
                with ZipFile(package, mode="w") as oszf:
                    oszf.write("audio2.mp3")
                    oszf.write(file_name)

                send_byt = "file exported!"
            elif int(data[0]) == 5:
                npy_file = "polygon Input.npy" 
                # RETRAINING STEP
                trainX, trainY, testX, testY = preprocess(songs = [npy_file], charts = [data], ti=1024)
                model.fit(trainX, trainY, show_metric=True, batch_size=4, n_epoch=5)
                send_byt = "training complete!"
            elif int(data[0]) == 6:
                npy_file = "citrus Input.npy" 
                # RETRAINING STEP
                trainX, trainY, testX, testY = preprocess(songs = [npy_file], charts = [data], ti=1024)
                model.fit(trainX, trainY, show_metric=True, batch_size=4, n_epoch=5)
                send_byt = "training complete!"
            else:
                send_byt = "invalid mode"
            send_byt = send_byt.encode()
            conn.send(send_byt)
            data = ''
        conn.close()
        conn, addr = s.accept()#Reset
            
    except KeyboardInterrupt:
        conn.close()
        del conn
        break
