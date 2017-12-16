import numpy as np

class HParams:
    def __init__(self):
        # Parameters for Model Free Path
        self.model_output_dim = 200
        self.model_conv_output_dim = 512

        # Parameters for encoding simulated trajectories
        self.conv_output_dim = 256
        self.encoder_output_dim = 100
        # How long of a trajectory to simulate
        self.traj_length = 2
        # How many trajectories to simulate
        self.traj_num = 2

        # Parameters for I3A core architecture
        # Input dimension after concatenating model free 
        # and model based paths
        self.joint_input_dim = self.traj_num * self.encoder_output_dim + self.model_output_dim
        self.lstm_output_dim = 50


hp = HParams()
