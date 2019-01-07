class GANStrategy(object):
    def __init__(self, data_provider: DataProvider,
                 real_loader_name: str, fake_loader_name: str,
                 g_session: Session,
                 d_session: Session,
                 g_train_callbacks: Tuple[Callback] or Callback = None,
                 d_train_callbacks: Tuple[Callback] or Callback = None):

        self.__data_provider = data_provider

        # Trains with real and then with fake.
        self.__d_strategy = Trainer(self.__data_provider, session=d_session,
                                    train_loader_names=(real_loader_name, fake_loader_name),
                                    val_loader_names=None,
                                    train_callbacks=d_train_callbacks)

        self.__g_strategy = Trainer(self.__data_provider, session=g_session,
                                    train_loader_names=fake_loader_name,
                                    val_loader_names=fake_loader_name,
                                    train_callbacks=g_train_callbacks)

    def train(self, data_key: Tuple[str] or str = 'img',
              target_key: Tuple[str] or str = 'target',
              latent_key: Tuple[str] or str = 'latent',
              accumulate_grad=False, cast_target=None):

        # TODO: Verify accumulated grads
        self.__d_strategy.train(data_key=data_key, target_key=target_key,
                                accumulate_grad=accumulate_grad, cast_target=cast_target)
        self.__g_strategy.train(data_key=latent_key, target_key=target_key,
                                accumulate_grad=accumulate_grad, cast_target=cast_target)
