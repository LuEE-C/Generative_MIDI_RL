import numpy as np
import os
import mido


def convert_midi_to_nptensor(directory='../Data/', cutoff=10, max_lines=10000000, name='midi_piano', overlap=True):
    if overlap is True:
        name = name + '_overlap'
        jump = 1
    else:
        jump = cutoff

    if os.path.isfile('../TransformedData/' + name + '_' + str(cutoff) + '.npy'):
        x = np.load('../TransformedData/' + name + '_' + str(cutoff) + '.npy')

    else:
        list_files = [directory + f for f in os.listdir(directory)]
        x = np.zeros((max_lines, cutoff, 3), dtype=np.int32)
        current_line = 0

        for file in list_files:
            mid = mido.MidiFile(file)

            for track in mid.tracks:
                if track.name == 'Piano right':
                    added_lines = 0
                    while added_lines + cutoff < len(track):
                        try:
                            sample_to_add = []
                            for i in range(cutoff):
                                tick = [track[i + added_lines].note,
                                        track[i + added_lines].velocity,
                                        track[i + added_lines].time]
                                sample_to_add.append(tick)
                            x[current_line] = np.array(sample_to_add)
                            current_line += 1
                        #Skip meta messages
                        except AttributeError:
                            pass
                        added_lines += jump
        x = x[:current_line]
        np.save('../TransformedData/' + name + '_' + str(cutoff) + '.npy', x)
    return x


def make_midi_file(midi, file_name):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for i in range(midi.shape[1]):
        track.append(mido.Message('note_on', note=midi[0, i, 0], velocity=midi[0, i, 1], time=midi[0, i, 2]))
    mid.save('../' + file_name)

if __name__ == '__main__':
    convert_midi_to_nptensor()