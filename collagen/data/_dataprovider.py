import gc
from ._itemloader import ItemLoader
from multiprocessing.pool import Pool


class DataProvider(object):
    """ Class provider data from single or multiple ``ItemLoader``s
    """
    def __init__(self, item_loaders: dict, num_workers: int = 0):
        """
        Parameters
        ----------
        item_loaders : ``collagen.data._itemloader.Itemloader``
            Dictionary that maps names to ``ItemLoader`` objects
        num_workers : int, optional
            The number of workers for sampling data asynchronously
        """
        self.__loaders = item_loaders
        self.__state_dict = {}
        self.__num_workers = num_workers
        if self.__num_workers < 0:
            raise ValueError("num_workers must non-negative, but found {}".format(num_workers))

        for il_name in self.__loaders:
            len_il = self.__loaders[il_name].__len__()
            self.__state_dict[il_name] = {"total": len_il,
                                          "num_sampled_data": 0,
                                          "num_available_data": len_il,
                                          "loop": 0}

    def sample(self, **kwargs):
        """ Sample :attr:__loaders with specified number of data

        Parameters
        ----------
        kwargs : dict
            Dictionary of itemloader names and the number sample k
        Returns
        -------
        list_samples : list
            List of samples

        """
        list_samples = []

        # self.__requested_itemloader_names = [il_name for il_name,k in kwargs.items() if il_name in self.__loaders]
        sampling_args = []
        for itemloader_name, k in kwargs.items():
            if itemloader_name in self.__loaders:
                sampling_args.append((itemloader_name, k))
            else:
                # TODO: Warning here
                pass

        if self.__num_workers == 0:
            for il_name, k in sampling_args:
                list_samples.append(self.__sample(il_name, k))
        else:
            with Pool(processes=self.__num_workers) as pool:
                list_samples = pool.starmap_async(self.__sample, sampling_args)

        return list_samples

    def __sample(self, itemloader_name : str, k : int):
        """ Internal function to sample a single ``ItemLoader`` by name with k samples

        Parameters
        ----------
        itemloader_name : str
            ``ItemLoader`` name
        k : int
            The number of sample

        Returns
        -------
        samplers : list
            List of sampled data
        """

        samples = []

        try:
            samples = self.__loaders[itemloader_name].sample(k)
            num_sample = len(samples)
            # Update state_dict
            self.__state_dict[itemloader_name]["num_sampled_data"] += num_sample
            self.__state_dict[itemloader_name]["num_available_data"] -= num_sample
            self.__state_dict[itemloader_name]["samples"] = samples
        except StopIteration:
            self.__state_dict[itemloader_name]["loop"] += 1
            self.__state_dict[itemloader_name]["num_sampled_data"] = 0
            self.__state_dict[itemloader_name]["num_available_data"] = self.__state_dict[itemloader_name]["total"]
        # TODO: handle drop_last==False

        return samples

    def state_dict(self):
        """ Get :attr:__state_dict
        """
        return self.__state_dict

    def empty_state(self):
        """ Clean :attr:__state_dict
        """
        del self.__state_dict
        gc.collect()
        self.__state_dict = {}