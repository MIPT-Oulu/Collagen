from collagen.data.utils._utils import Singleton

class DataProvider(Singleton):
    def __init__(self, map_itemloader: dict):
        self.__map_itemloader = map_itemloader

    def sample(self, dict_batch_size: dict):
        """Sample from dictionary of multiple itemloaders.
        
        Parameters
        ----------
        dict_batch_size : dict
            Map from name to batch size of corresponding ``itemloader``
        
        """
        return NotImplementedError

    def state(self):
        """TBD
        
        """
        return NotImplementedError

    def empty(self):
        """TBD
        
        """
        return NotImplementedError

            
        