class Callback(object):
    def __init__(self, *args, **kwargs):
        self.state_dict = {}

    def on_epoch_begin(self, *args, **kwargs):
        raise NotImplementedError

    def on_epoch_end(self, *args, **kwargs):
        raise NotImplementedError

    def on_batch_begin(self, *args, **kwargs):
        raise NotImplementedError

    def on_batch_end(self, *args, **kwargs):
        raise NotImplementedError
