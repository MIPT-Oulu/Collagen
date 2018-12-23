import gc
from ._itemloader import ItemLoader
from multiprocessing.pool import Pool


class DataProvider(object):
    """ Provides data from single or multiple ``ItemLoader``s
    Parameters
    ----------
    item_loaders : ``collagen.data._itemloader.ItemLoader``
        Dictionary that maps names to ``ItemLoader`` objects
    """
    def __init__(self, item_loaders: dict):
        self.__loaders = item_loaders
        self.__state_dict = {}

        for il_name in self.__loaders:
            len_il = self.__loaders[il_name].__len__()
            self.__state_dict[il_name] = {"total": len_il,
                                          "samples": None,
                                          "num_sampled_data": 0,
                                          "num_available_data": len_il,
                                          "loop": 0}

    def sample(self, **kwargs):
        """ Sample :attr:__loaders with specified number of data

        Parameters
        ----------
        kwargs : dict
            Dictionary of the names, corresponding to the itemloaders stored in ``DataProvider``, and the number of batches,
            which needs to be drawn from each of them.
        Returns
        -------
        list_samples : list
            List of samples

        """
        list_samples = []

        sampling_args = []
        for itemloader_name, k in kwargs.items():
            if itemloader_name in self.__loaders:
                sampling_args.append((itemloader_name, k))
            else:
                raise ValueError("Not found argument {} in itemloader list".format(itemloader_name))

        for il_name, k in sampling_args:
            list_samples.append(self.__sample(il_name, k))

        return list_samples

    def __sample(self, itemloader_name : str, k : int):
        """ Internal function to sample a single ``ItemLoader`` by name with k samples

        Parameters
        ----------
        itemloader_name : str
            ``ItemLoader`` name
        k : int
            The number of samples

        Returns
        -------
        samplers : list
            List of sampled data
        """

        samples = []

        samples = self.__loaders[itemloader_name].sample(k)
        num_sample = len(samples)
        self.__state_dict[itemloader_name]["samples"] = samples

        # Update state_dict
        if self.__state_dict[itemloader_name]["num_sampled_data"] + num_sample > self.__state_dict[itemloader_name]["total"]:
            self.__state_dict[itemloader_name]["loop"] += 1
            self.__state_dict[itemloader_name]["num_sampled_data"] = num_sample
            self.__state_dict[itemloader_name]["num_available_data"] = self.__state_dict[itemloader_name]["total"]
        else:
            self.__state_dict[itemloader_name]["num_sampled_data"] += num_sample
            self.__state_dict[itemloader_name]["num_available_data"] -= num_sample

        return samples

    def state_dict(self):
        """ Return :attr:__state_dict
        """
        return self.__state_dict

    def empty_state(self):
        """ Clean :attr:__state_dict
        """
        del self.__state_dict
        gc.collect()
        self.__state_dict = {}