from diffusers import DPMSolverMultistepScheduler

# TODO
class Scheduler():
    def __init__(self):
        pass

    def get_scheduler(self, type:str, args):
        assert isinstance(type, str)
        type = type.lower()
        if type == 'dpm':
            return DPMSolverMultistepScheduler( # TODO: Dig deeper here!
                num_train_timesteps=args.num_train_timesteps,
                beta_schedule=args.beta_schedule,
                solver_order=3,
                prediction_type='epsilon',
                algorithm_type='dpmsolver++'
            )
        else:
            raise ValueError('Error: Illegal type for noise scheduler.')