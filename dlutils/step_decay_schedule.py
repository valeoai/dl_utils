import numpy as np
from keras.callbacks import LearningRateScheduler


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    """

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)


lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)

model.fit(X_train, Y_train, callbacks=[lr_sched])
