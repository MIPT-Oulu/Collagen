from collagen.data import ItemLoader


class VisualizationSampler(ItemLoader):
    """VisualizationSampler samples images from given test set, forward them through the encoder to get reconstructions
    """
    def __init__(self, viz_loader, device, bs, ae):
        """
        constructor constructs the sampler object
        Parameters
        ----------
        viz_loader: ItemLoader
            ItemLoader object to sample images for visualization
        device: torch.device
            torch device indicating cpu or gpu computation
        bs: int
            Batch Size
        ae: AutoEncoder
            The AutoEncoder object to be used to reconstruct test images
        """
        self.batch_size = bs
        self.viz_loader = viz_loader
        self.device = device
        self.ae = ae
        self.__name = 'viz'

    def sample(self, k=1):
        """
        samples data from underlaying itemloader
        Parameters
        ----------
        k: int
            number of times to be sampled

        Returns
        -------
        samples: dict
            sampled data as a dictionary
        """
        samples = []
        for _ in range(k):
            data = self.viz_loader.sample()
            # hard coded zero index, we will only visualize the first set of images even if there are more
            reconstruction = self.ae(data[0]['data'].to(self.device))
            samples.append(
                {'name': self.__name, 'data': reconstruction.detach(), 'target':data[0]['data'].detach()})

        return samples

    def __len__(self):
        return 1