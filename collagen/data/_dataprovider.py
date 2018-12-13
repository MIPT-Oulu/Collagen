import gc


class DataProvider(object):
    def __init__(self, item_loaders: dict):
        self.__loaders = item_loaders
        self.__state_dict = {}

    def sample(self, **kwargs):
        raise NotImplementedError

    def state_dict(self):
        return self.__state_dict

    def empty_state(self):
        del self.__state_dict
        gc.collect()
        self.__state_dict = {}


