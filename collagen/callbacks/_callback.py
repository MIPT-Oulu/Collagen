class Callback(object):
    def __init__(self, type = None, *args, **kwargs):
        self.__type = type
        self.state_dict = {}

    def get_type(self):
        return self.__type

    def get_name(self):
        pass

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

    def on_loss_begin(self, *args, **kwargs):
        pass

    def on_loss_end(self, *args, **kwargs):
        pass

    def on_train_begin(self, *args, **kwargs):
        pass

    def on_train_end(self, *args, **kwargs):
        pass

    def on_test_begin(self, *args, **kwargs):
        pass

    def on_test_end(self, *args, **kwargs):
        pass

    def on_minibatch_begin(self, *args, **kwargs):
        pass

    def on_minibatch_end(self, *args, **kwargs):
        pass

    # GAN
    def on_gan_g_batch_begin(self, *args, **kwargs):
        pass

    def on_gan_g_batch_end(self, *args, **kwargs):
        pass

    def on_gan_d_batch_begin(self, *args, **kwargs):
        pass

    def on_gan_d_batch_end(self, *args, **kwargs):
        pass

    def __str__(self):
        return ""


class BackwardCallback(Callback):
    def __init__(self, retain_graph=True, create_graph=False):
        super().__init__()
        self.__retain_graph = retain_graph
        self.__create_graph = create_graph

    def on_backward_begin(self, session, **kwargs):
        session.set_backward_param(retain_graph=self.__retain_graph, create_graph=self.__create_graph)
