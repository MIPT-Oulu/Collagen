from collagen.core import Callback


class UpdateSWA(Callback):
    def __init__(self, swa_model, student_model, start_cycle_epoch, cycle_interval):
        super().__init__(ctype='update_weight')
        self.__swa_model = swa_model
        self.__student_model = student_model
        self.__num_params = 1
        self.__start_cycle_epoch = start_cycle_epoch
        self.__cycle_interval = cycle_interval
        self.swa_model.load_state_dict(self.student_model.state_dict())

    @property
    def swa_model(self):
        return self.__swa_model

    @property
    def student_model(self):
        return self.__student_model

    def on_epoch_begin(self, epoch, *args, **kwargs):
        if ((epoch >= self.__start_cycle_epoch)) and ((epoch - self.__start_cycle_epoch) % self.__cycle_interval) == 0:
            self.__num_params += 1
            inv = 1. / float(self.__num_params)
            for swa_p, src_p in zip(self.swa_model.parameters(), self.student_model.parameters()):
                swa_p.data.add_(-inv * swa_p.data)
                swa_p.data.add_(inv * src_p.data)

    def reset(self):
        self.__num_params = 0