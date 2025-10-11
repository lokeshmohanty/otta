import atexit
import pickle
from pathlib import Path

class ExperimentTracker():
    def __init__(self, name: str, cfg: dict, hparams: dict):
        ''' Setup the tracker '''
        self.cfg = cfg
        self.hparams = hparams
        self.tracks = {}
        self.metrics = {}
        self.path = Path("logs") / name
        atexit.register(self.exit)

    def log(self, **kwargs):
        '''Logs metrics'''
        for key in kwargs.keys():
            if key not in self.metrics:
                self.metrics[key] = kwargs[key]
        
    def track(self, **kwargs):
        '''Tracks running metrics'''
        for key in kwargs.keys():
            if key in self.tracks:
                self.tracks[key].append(kwargs[key])
            else:
                self.tracks[key] = [kwargs[key]]

    def exit(self):
        with open(self.path, 'wb') as f:
            pickle.dump({
                "tracks": self.tracks,
                "cfg": self.cfg,
                "hparams": self.hparams,
                "metrics": self.metrics,
                }, f)



