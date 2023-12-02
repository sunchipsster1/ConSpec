class ConSpec:
    def __init__(
            self,
            state_shape, # Not sure if this is necessary?
            # Probably also add other parameters related to when you freeze the prototypes etc.
            intrinsic_reward_type="smooth", # I don't remember exactly but there were 2 different intrinsic reward types in the paper
            num_prototypes=16,
            buffer_size=64
    ):
        self.state_shape = state_shape
        self.num_prototypes = num_prototypes
        self.buffer_size = buffer_size

    
    def get_prototypes(self):
        """
        Returns the current prototypes
        """
        pass

    def train_prototypes_step(self):
        """
        Trains the prototypes for one step
        """
        pass
    
    def add_trajectory(self, trajectory):
        """
        args:
            trajectory: list of transitions (s, a, s', r)
        Adds a trajectory to the buffer
        """
        pass

    def get_intrinsic_reward_for_trajectory(self, trajectory):
        """
        Returns the intrinsic reward for an entire trajectory
        args:
            trajectory: list of transitions (s, a, s', r)
        returns:
            instrinsic_reward: np.array of shape (len(trajectory),) containing the intrinsic reward for each transition
        """
        pass

    def get_intrinsic_reward_for_transition(self, transition):
        """
        Returns the intrinsic reward for a single transition (s, a, s', r)
        args:
            transition: tuple (s, a, s', r)
        returns:
            instrinsic_reward: float containing the intrinsic reward for the transition
        """
        pass

    def freeze_prototypes(self):
        """
        Freezes the prototypes
        """
        # Could be useful for experiments where you want to freeze prototypes early
        pass

    def unfreeze_prototypes(self):
        """
        Unfreezes the prototypes
        """
        # Same as above
        pass