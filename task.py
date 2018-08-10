import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0.
        
        ## Add the vertical velocity value to a reward when the quadcopter flew up and down in z-direction.
        ## If the velocity goes up, then the reward gets increased. Otherwise, it gets decreased.
        reward += self.sim.v[2]

        ## Check x and y position movement
        pos_xy = 1 - 2*(abs(self.sim.pose[:2] - self.target_pos[:2])).sum()
        if pos_xy < -1: ## Give a penalty if x and y position has been changed too much
            reward -= 1
        elif pos_xy == 1: ## reward if maintains the vertical in x0 and y0
            reward += 2
        else: ## If x and y moved very little, it is considered as a well-maintenance in x and y position
            reward += 1

        ## Check angular velocity change
        av = 1 - 2*(abs(self.sim.angular_v[:3])).sum()
        ## If the angular velocities are irregular enough to disrupt the quadcopter's vertical ascending movement.
        ## then give a negative reward. Otherwise, a positive reward
        if av > 1:
            reward -= 1
        elif av < -1:
            reward += 1
        
        ## Checked the vertical distance between the current position and the target position.
        z_distance = 1 - 2*(abs(self.sim.pose[2] - self.target_pos[2]))
        if z_distance < -1: ## if the z-distance is getting bigger, then give a negative reward.
            reward -= 1
        elif z_distance > 0: ## if the z-distance is 0, then give a big positive reward.
            reward += 5
        else:                ## if the z-distance is getting closer, then give a positive reward.
            reward += 2
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)

            if(self.sim.pose[2] > self.target_pos[2]):
                reward += 200
                done = True
                
        next_state = np.concatenate(pose_all)

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state