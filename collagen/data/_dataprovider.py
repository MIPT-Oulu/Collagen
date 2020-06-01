import gc


class DataProvider(object):
    """ Provides data from single or multiple ``ItemLoader``s

    Parameters
    ----------
    item_loaders : ``collagen.data.ItemLoader``
        Dictionary that maps names to ``ItemLoader`` objects
    """

    def __init__(self, item_loaders: dict):
        self.__loaders = item_loaders
        self.__state_dict = {}

        for itemloader_name in self.__loaders:
            # Auto set name for itemloader
            if not self.__loaders[itemloader_name].name or self.__loaders[itemloader_name].name is None:
                self.__loaders[itemloader_name].name = itemloader_name
            itemloader_len = len(self.__loaders[itemloader_name])
            self.__state_dict[itemloader_name] = {"total": itemloader_len,
                                                  "samples": None,
                                                  "num_sampled": 0,
                                                  "num_left": itemloader_len,
                                                  "num_loops": 0}

    def sample(self, **kwargs):
        """ Samples :attr:__loaders with specified number of data

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
                raise ValueError("Not found argument `{}` in itemloader list".format(itemloader_name))

        for il_name, k in sampling_args:
            list_samples.append(self.__sample(il_name, k))

        return list_samples

    def __sample(self, itemloader_name: str, k: int):
        """Gets `k` samples from the itemloader specified by `itemloader_name`.

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

        samples = self.__loaders[itemloader_name].sample(k)
        num_samples = len(samples)
        self.__state_dict[itemloader_name]["samples"] = samples

        # Update state_dict
        if self.__state_dict[itemloader_name]["num_sampled"] + num_samples > self.__state_dict[itemloader_name][
            "total"]:
            self.__state_dict[itemloader_name]["num_loops"] += 1
            self.__state_dict[itemloader_name]["num_sampled"] = num_samples
            self.__state_dict[itemloader_name]["num_left"] = self.__state_dict[itemloader_name]["total"] - num_samples
        else:
            self.__state_dict[itemloader_name]["num_sampled"] += num_samples
            self.__state_dict[itemloader_name]["num_left"] -= num_samples

        return samples

    def state_dict(self):
        """ Returns :attr:__state_dict
        """
        return self.__state_dict

    def empty_state(self):
        """ Cleans :attr:__state_dict
        """
        del self.__state_dict
        gc.collect()
        self.__state_dict = {}

    def get_loader_names(self):
        return [name for name in self.__loaders]

    def get_loader_by_name(self, name):
        if name in self.__loaders:
            return self.__loaders[name]
        elif isinstance(name, tuple) or isinstance(name, list):
            return tuple([self.__loaders[s] for s in name])
        else:
            raise ValueError("`{}` not found in list of loader names".format(name))

    def set_epoch(self, epoch):
        for loader in self.__loaders:
            self.__loaders[loader].set_epoch(epoch)
