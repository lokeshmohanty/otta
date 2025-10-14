import atexit
import pickle
from pathlib import Path

class ExperimentTracker():
    def __init__(self, path: Path, cfg: dict, hparams: dict):
        ''' Setup the tracker '''
        self.cfg = cfg
        self.hparams = hparams
        self.tracks = {}
        self.metrics = {}
        self.path = path / "data.pkl"
        atexit.register(self.exit)

    def log(self, ob, **kwargs):
        '''Logs metrics'''
        for k, v in ob.items():
            self.metrics[k] = v

        for k, v in kwargs.items():
            self.metrics[k] = v
        
    def track(self, **kwargs):
        '''Tracks running metrics'''
        for k, v in kwargs.items():
            if k in self.tracks:
                self.tracks[k].append(v)
            else:
                self.tracks[k] = [v]

    def exit(self):
        with open(self.path, 'wb') as f:
            pickle.dump({
                "cfg": self.cfg,
                "hparams": self.hparams,
                "tracks": self.tracks,
                "metrics": self.metrics,
                }, f)

