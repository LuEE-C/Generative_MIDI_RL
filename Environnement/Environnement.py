import numpy as np
import numba as nb
from Environnement import data_util

class Environnement:
    def __init__(self, cutoff=10):
        self.datas = data_util.convert_midi_to_nptensor(cutoff=cutoff)
        self.datas = self.datas.astype(np.float64)

        # Clipping outliers
        for i in range(3):
            self.datas[:, :, i] = np.clip(self.datas[:, :, i], np.percentile(self.datas[:, :, i], 5), np.percentile(self.datas[:, :, i], 95))

        self.datas_range = [self.datas[:, :, i].max() - self.datas[:, :, i].min() for i in range(3)]
        self.datas_max = [self.datas[:, :, i].max() for i in range(3)]

        # Normalizing data set to go from -1 to 1 to match the tanh output
        for i in range(3):
            self.datas[:, :, i] -= self.datas_max[i]
            self.datas[:, :, i] += self.datas_range[i]/2
            self.datas[:, :, i] /= self.datas_range[i]/2
        self.index = 0
        np.random.shuffle(self.datas)

    @nb.jit
    def query_state(self, batch_size):

        state = self.datas[self.index: self.index + batch_size]
        self.index += batch_size
        # End of epoch, shuffle dataset for next epoch
        if self.index + batch_size >= self.datas.shape[0]:
            self.index = 0
            np.random.shuffle(self.datas)
            return state, True
        else:
            return state, False

    def make_midi(self, midi, file_name):
        for i in range(3):
            midi[:,:,i] *= self.datas_range[i]/2
            midi[:,:,i] -= self.datas_range[i]/2
            midi[:,:,i] += self.datas_max[i]
        midi = midi.astype(np.int64)
        midi[:,:,0] = np.clip(midi[:,:,0], 0, 127)
        midi[:,:,1] = np.clip(midi[:,:,1], 0, 127)
        midi = np.clip(midi, 0, 10000)
        for i in range(3):
            print(int(midi[:,:,i].mean()), int(midi[:,:,i].std()), end="\t")
        print()
        data_util.make_midi_file(midi, file_name)


if __name__ == '__main__':
    env = Environnement()
    print(env.query_state(2))