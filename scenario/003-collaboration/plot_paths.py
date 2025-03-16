from main import Actor
from environment import GrADySEnvironment
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class data_linewidth_plot():
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        self.fig.canvas.draw()

        self.ppd = 72./self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([],[],**kwargs)
        if "label" in kwargs: kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw =  ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.fig.canvas.draw_idle())
        self.timer.start()


# Loading the model
models = [
    (2, 12, "runs/results/results__a_2-s_12__1__1723034080/a_2-s_12-checkpoint4999.cleanrl_model"),
    (2, 24, "runs/results/results__a_2-s_24__1__1723034081/a_2-s_24-checkpoint4999.cleanrl_model"),
    (2, 36, "runs/results/results__a_2-s_36__1__1723034081/a_2-s_36-checkpoint4999.cleanrl_model"),
    (4, 12, "runs/results/results__a_4-s_12__1__1723034081/a_4-s_12-checkpoint4999.cleanrl_model"),
    (4, 24, "runs/results/results__a_4-s_24__1__1723034081/a_4-s_24-checkpoint4999.cleanrl_model"),
    (4, 36, "runs/results/results__a_4-s_36__1__1723034081/a_4-s_36-checkpoint4999.cleanrl_model"),
    (8, 12, "runs/results/results__a_8-s_12__1__1726583488/a_8-s_12-checkpoint4999.cleanrl_model"),
    (8, 24, "runs/results/results__a_8-s_24__1__1723034081/a_8-s_24-checkpoint4999.cleanrl_model"),
    (8, 36, "runs/results/results__a_8-s_36__1__1723034081/a_8-s_36-checkpoint4999.cleanrl_model"),
]

completion_times = []

for num_agents, num_sensors, model_path in models:
    print(f"Running with {num_agents} agents and {num_sensors} sensors")
    print(f"Loading model from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_model = torch.load(model_path, map_location=device, weights_only=True)[0]

    def get_agent_positions(environment: GrADySEnvironment):
        agent_nodes = [environment.simulator.get_node(agent_id) for agent_id in environment.agent_node_ids]
        return {f"Agent {agent_id}": agent_node.position for agent_id, agent_node in enumerate(agent_nodes)}

    def get_sensor_positions(environment: GrADySEnvironment):
        sensor_nodes = [environment.simulator.get_node(sensor_id) for sensor_id in environment.sensor_node_ids]
        return {f"Sensor {sensor_id}": sensor_node.position for sensor_id, sensor_node in enumerate(sensor_nodes)}

    if __name__ == "__main__":
        env = GrADySEnvironment(
            algorithm_iteration_interval=0.5,
            num_drones=num_agents,
            num_sensors=num_sensors,
            max_seconds_stalled=30,
            scenario_size=100,
            render_mode=None,
            state_num_closest_sensors=12,
            state_num_closest_drones=num_agents-1,
            min_sensor_priority=1,
            full_random_drone_position=False,
            speed_action=True
        )
        actor = Actor(env.action_space(0), env.observation_space(0)).to(device)
        actor.load_state_dict(actor_model)

        success_count = 0
        collection_times_sum = 0

        # Running the model
        obs, _ = env.reset()

        position_list = []

        for agent, position in get_agent_positions(env).items():
            position_list.append({
                "agent": agent,
                "x": position[0],
                "y": position[1],
                "timestamp": env.simulator._current_timestamp
            })

        sensor_positions = []
        for sensor, position in get_sensor_positions(env).items():
            sensor_positions.append({
                "sensor": sensor,
                "x": position[0],
                "y": position[1]
            })

        while True:
            with torch.no_grad():
                actions = {}
                for agent in env.agents:
                    actions[agent] = actor(torch.Tensor(obs[agent]).to(device)).cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            for agent, position in get_agent_positions(env).items():
                position_list.append({
                    "agent": agent,
                    "x": position[0],
                    "y": position[1],
                    "timestamp": env.simulator._current_timestamp
                })

            obs = next_obs
            if len(infos) > 0 and "avg_reward" in infos[env.agents[0]]:
                break
        env.close()


        position_df = pd.DataFrame.from_records(position_list)
        position_df = position_df.set_index("timestamp")

        sensor_df = pd.DataFrame.from_records(sensor_positions)

        # Plot agents paths over time
        # Overlay sensor positions
        sns.set_theme()
        sns.set_context("talk")
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

        sns.scatterplot(data=sensor_df, x="x", y="y", ax=ax, marker='x', color='black',
                        label='Sensors Positions', s=100, linewidth=2)

        grouped = position_df.groupby("agent")
        # Plot a line for each agent
        for name, group in grouped:
            line = plt.plot(group['x'], group['y'], marker='o', linestyle='-', ms=5, label=name)
            data_linewidth_plot(group['x'], group['y'], marker=None, linestyle='-', linewidth=40, color=line[0].get_color(), label=None, alpha=0.1)

        plt.legend()
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()

        plt.savefig(f"statistics/paths/path-a{num_agents}s{num_sensors}.png")
