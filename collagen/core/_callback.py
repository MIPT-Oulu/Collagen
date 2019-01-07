class Callback(object):
    def __init__(self, *args, **kwargs):
        self.state_dict = {}

    def on_epoch_begin(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass

    def on_batch_begin(self, *args, **kwargs):
        pass

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_backward_begin(self, *args, **kwargs):
        pass

    def on_backward_end(self, *args, **kwargs):
        pass

    def on_optimizer_step_begin(self, *args, **kwargs):
        pass

    def on_optimizer_step_end(self, *args, **kwargs):
        pass

    def on_split_begin(self, *args, **kwargs):
        pass

    def on_split_end(self, *args, **kwargs):
        pass

    def on_sample_begin(self, *args, **kwargs):
        pass

    def on_sample_end(self, *args, **kwargs):
        pass

    def on_itemloader_begin(self, *args, **kwargs):
        pass

    def on_itemloader_end(self, *args, **kwargs):
        pass

    def on_parse_item(self, *args, **kwargs):
        pass

    def on_forward_begin(self, *args, **kwargs):
        pass

    def on_forward_end(self, *args, **kwargs):
        pass