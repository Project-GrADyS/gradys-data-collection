from arguments import EnvironmentArgs

class ProgressiveScaler:
    def __init__(self, env_args: EnvironmentArgs, scaling_limits: [int, int, int, int]):
        self.env_args = env_args
        self.scaling_limits = scaling_limits

        self._scaling_steps = []
        for a in range(self.env_args.min_drone_count, self.env_args.max_drone_count + 1):
            for s in range(self.env_args.min_sensor_count, self.env_args.max_sensor_count + 1):
                self._scaling_steps.append((self.env_args.min_drone_count, a, self.env_args.min_sensor_count, s))

        self._current_scaling_step = 0 if self.env_args.progressive_scaling else len(self._scaling_steps) - 1
        self._current_confidence = 0

    def scale(self, all_collected_rate: float):
        if not self.env_args.progressive_scaling:
            return

        if all_collected_rate >= self.env_args.progressive_scaling_cutoff:
            self._current_confidence += 1
        else:
            self._current_confidence = 0

        if (self._current_confidence >= self.env_args.progressive_scaling_confidence
                and self._current_scaling_step < len(self._scaling_steps) - 1):
            self._current_scaling_step += 1
            self._current_confidence = 0

        self.scaling_limits[0] = self._scaling_steps[self._current_scaling_step][0]
        self.scaling_limits[1] = self._scaling_steps[self._current_scaling_step][1]
        self.scaling_limits[2] = self._scaling_steps[self._current_scaling_step][2]
        self.scaling_limits[3] = self._scaling_steps[self._current_scaling_step][3]