import os
from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from torch_robotics.torch_utils.torch_utils import to_numpy
import matplotlib.collections as mcoll


def create_fig_and_axes(dim=2):
    fig = plt.figure(layout='tight')
    if dim == 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()

    return fig, ax


class PlanningVisualizer:

    def __init__(self, task=None, planner=None):
        self.task = task
        self.env = self.task.env
        self.robot = self.task.robot
        self.planner = planner

        self.colors = {'collision': 'black', 'free': 'orange'}
        self.colors_robot = {'collision': 'black', 'free': 'darkorange'}
        self.cmaps = {'collision': 'Greys', 'free': 'Oranges'}
        self.cmaps_robot = {'collision': 'Greys', 'free': 'YlOrRd'}

    def render_robot_trajectories(self, fig=None, ax=None, render_planner=False, trajs=None, traj_best=None, **kwargs):
        if fig is None or ax is None:
            fig, ax = create_fig_and_axes(dim=self.env.dim)

        if render_planner:
            self.planner.render(ax)
        self.env.render(ax)
        if trajs is not None:
            _, trajs_coll_idxs, _, trajs_free_idxs, _ = self.task.get_trajs_collision_and_free(trajs, return_indices=True)
            kwargs['colors'] = []
            for i in range(len(trajs_coll_idxs) + len(trajs_free_idxs)):
                kwargs['colors'].append(self.colors['collision'] if i in trajs_coll_idxs else self.colors['free'])
        self.robot.render_trajectories(ax, trajs=trajs, **kwargs)
        if traj_best is not None:
            kwargs['colors'] = ['blue']
            self.robot.render_trajectories(ax, trajs=traj_best.unsqueeze(0), **kwargs)

        return fig, ax

    def render_multi_robot_trajectories_steps(self, fig=None, axs=None, render_planner=False, 
                                        start_goal_pairs=None, trajs=None, t=-1, **kwargs):
        """
            Render denoising steps for one planning instance
            Input: 
                Trajs: (B, T, N, H, D)
        """
        if fig is None or axs is None:
            fig, axs = plt.subplots(2, 3, figsize=(9, 6), layout="tight")
            axs = axs.flatten()
        if render_planner:
            self.planner.render(ax)
        _, T, N, H, D = trajs.shape
        ts = [24, 19, 14, 9, 4, 0] # decides the timesteps to render.
        fig.suptitle(f"Step {t} / {T-1}")
        if trajs is not None:
            trajs = trajs[0] # (T, N, H, D) render the first solution 
            for i, t in enumerate(ts):
                ax = axs[i]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xlabel("")
                ax.set_ylabel("")
                self.env.render(ax)
                # Plot each agent
                for agent_idx in range(N):
                    start_state, goal_state = start_goal_pairs[agent_idx]
                    self.robot.render(ax, start_state, color='green', cmap='Greens')
                    self.robot.render(ax, goal_state, color='red', cmap='Greens')
                    agent_color = plt.cm.get_cmap("tab20")(agent_idx % 20)
                    traj = trajs[t, agent_idx] # always render the first result in batch
                    kwargs['colors'] = [agent_color]
                    kwargs['linewidth'] = [5]
                    self.robot.render_trajectories(ax, trajs=traj.unsqueeze(0), **kwargs)
        return fig, ax

   
    def render_multi_robot_trajectories(self, fig=None, axs=None, render_planner=False, 
                                        start_goal_pairs=None, trajs=None, t=-1, **kwargs):
        """
            Plot the first 10 trajectories in the results
            Trajs: (B, T, N, H, D)
        """
        if fig is None or axs is None:
            fig, axs = plt.subplots(2, 5, figsize=(15, 6), layout="tight")
            axs = axs.flatten()
        if render_planner:
            self.planner.render(ax)
        B, T, N, H, D = trajs.shape
        if t < 0:
            t = T-1
        fig.suptitle(f"Step {t} / {T-1}")
        num_to_plot = min(B, 10)
        if trajs is not None:
            for b in range(num_to_plot):
                ax = axs[b]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xlabel("")
                ax.set_ylabel("")

                self.env.render(ax)
                # Plot each agent
                for agent_idx in range(N):
                    start_state, goal_state = start_goal_pairs[agent_idx]
                    self.robot.render(ax, start_state, color='green', cmap='Greens')
                    self.robot.render(ax, goal_state, color='red', cmap='Greens')
                    agent_color = plt.cm.get_cmap("tab20")(agent_idx % 20)
                    traj = trajs[b, t, agent_idx, :, :]
                    kwargs['colors'] = [agent_color]
                    kwargs['linewidth'] = [5]
                    self.robot.render_trajectories(ax, trajs=traj.unsqueeze(0), **kwargs)
        return fig, ax

    def animate_robot_trajectories(
            self, trajs=None, start_state=None, goal_state=None,
            plot_trajs=False,
            n_frames=10,
            **kwargs
    ):
        if trajs is None:
            return

        assert trajs.ndim == 3
        B, H, D = trajs.shape

        idxs = np.round(np.linspace(0, H - 1, n_frames)).astype(int)
        trajs_selection = trajs[:, idxs, :]

        fig, ax = create_fig_and_axes(dim=self.env.dim)
        def animate_fn(i):
            ax.clear()
            ax.set_title(f"step: {idxs[i]}/{H-1}")
            if plot_trajs:
                self.render_robot_trajectories(
                    fig=fig, ax=ax, trajs=trajs, start_state=start_state, goal_state=goal_state, **kwargs
                )
            else:
                self.env.render(ax)

            # TODO - implement batched version
            qs = trajs_selection[:, i, :]  # batch, q_dim
            if qs.ndim == 1:
                qs = qs.unsqueeze(0)  # interface (batch, q_dim)
            for q in qs:
                self.robot.render(
                    ax, q=q,
                    color=self.colors_robot['collision'] if self.task.compute_collision(q, margin=0.0) else self.colors_robot['free'],
                    arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.,
                    cmap=self.cmaps['collision'] if self.task.compute_collision(q, margin=0.0) else self.cmaps['free'],
                    **kwargs
                )

            if start_state is not None:
                self.robot.render(ax, start_state, color='green', cmap='Greens')
            if goal_state is not None:
                self.robot.render(ax, goal_state, color='purple', cmap='Purples')

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)

    def animate_multi_robot_trajectories(
                self,
                start_goal_pairs=None,
                trajs=None,
                n_interpolation=10,
                n_frames=5,
                **kwargs
        ):
        """
            Given solution trajectory start_goal_pairs and solution trajectories, generate
            video of agents following the waypoints and insert interpolation points between waypoints.

            Input:
                start_goal_pairs: start and goal location per-agent
                trajs: (B, T, N, H, D) denoised trajectories
        """
        if trajs is None:
            return

        assert trajs.ndim == 5, "Expected trajectories with shape (B, T, N, H, D)"

        fig, ax = create_fig_and_axes(dim=self.env.dim)
        final_trajs = trajs[0, -1]

        def interpolate_trajectory(path, inserts):
            if inserts <= 0 or path.shape[1] < 2:
                return path
            if isinstance(path, torch.Tensor):
                alphas = torch.linspace(
                    0.0, 1.0, inserts + 2, device=path.device, dtype=path.dtype
                )[1:-1]
                start = path[:, :-1]
                delta = path[:, 1:] - start
                interpolated = start.unsqueeze(2) + delta.unsqueeze(2) * alphas.view(1, 1, -1, 1)
                stacked = torch.cat([start.unsqueeze(2), interpolated], dim=2)
                stacked = stacked.reshape(path.shape[0], -1, path.shape[-1])
                return torch.cat([stacked, path[:, -1:, :]], dim=1)

            path_np = np.asarray(path)
            alphas_np = np.linspace(0.0, 1.0, inserts + 2)[1:-1]
            start_np = path_np[:, :-1]
            delta_np = path_np[:, 1:] - start_np
            interpolated_np = start_np[:, :, None, :] + delta_np[:, :, None, :] * alphas_np[None, None, :, None]
            stacked_np = np.concatenate([start_np[:, :, None, :], interpolated_np], axis=2)
            stacked_np = stacked_np.reshape(path_np.shape[0], -1, path_np.shape[-1])
            return np.concatenate([stacked_np, path_np[:, -1:, :]], axis=1)

        interp_final_trajs = interpolate_trajectory(final_trajs, n_interpolation)
        interp_final_trajs = to_numpy(interp_final_trajs)
        num_agents, num_steps, _ = interp_final_trajs.shape

        frame_indices = np.arange(num_steps, dtype=int)
        trail_window = max(int(n_frames), 1)

        cmap = plt.cm.get_cmap("tab20")
        agent_colors = [cmap(i % 20) for i in range(num_agents)]

        def animate_fn(frame_idx):
            ax.clear()
            self.env.render(ax)

            if start_goal_pairs is not None:
                for start_state, goal_state in start_goal_pairs:
                    if start_state is not None:
                        self.robot.render(ax, start_state, color='green', cmap='Greens')
                    if goal_state is not None:
                        self.robot.render(ax, goal_state, color='purple', cmap='Purples')

            step = frame_indices[frame_idx]
            trail_start = max(0, step - trail_window + 1)

            for agent_idx in range(num_agents):
                trail_steps = range(trail_start, step + 1)
                trail_length = len(trail_steps)
                base_color = agent_colors[agent_idx]
                rgb = base_color[:3]
                for trail_pos, t_step in enumerate(trail_steps, start=1):
                    alpha = trail_pos / trail_length
                    color = (rgb[0], rgb[1], rgb[2], float(alpha))
                    self.robot.render(ax, q=interp_final_trajs[agent_idx, t_step], color=color, cmap='Blues')

            ax.set_title(f"step: {step}/{num_steps - 1}")

        create_animation_video(fig, animate_fn, n_frames=len(frame_indices), **kwargs)

    def animate_opt_iters_robots(
            self, trajs=None, traj_best=None, start_state=None, goal_state=None,
            n_frames=10,
            **kwargs
    ):
        # trajs: steps, batch, horizon, q_dim
        if trajs is None:
            return

        assert trajs.ndim == 4
        S, B, H, D = trajs.shape

        idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        trajs_selection = trajs[idxs]

        fig, ax = create_fig_and_axes(dim=self.env.dim)

        def animate_fn(i):
            ax.clear()
            ax.set_title(f"iter: {idxs[i]}/{S-1}")
            self.render_robot_trajectories(
                fig=fig, ax=ax, trajs=trajs_selection[i],
                traj_best=traj_best if i == n_frames - 1 else None,
                start_state=start_state, goal_state=goal_state, **kwargs
            )
            if start_state is not None:
                self.robot.render(ax, start_state, color='green', cmap='Greens')
            if goal_state is not None:
                self.robot.render(ax, goal_state, color='purple', cmap='Purples')

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)

    def animate_opt_iters_multi_robots(
                self, start_goal_pairs=None, trajs=None,
                **kwargs
        ):
        """
            Animate optimization iterations. 
            
            Input:
                start_goal_pairs: start goal pairs for each agent.
                trajs: (B, T, N, H, D) solution trajectories.
        """
        # trajs: steps, batch, horizon, q_dim
        if trajs is None:
            return

        B, T, N, H, D = trajs.shape

        fig, axs = plt.subplots(2, 5, figsize=(15, 6), layout="tight")
        axs = axs.flatten()

        def animate_fn(i, axs):
            # clear axs
            for ax in axs:
                ax.clear()
            self.render_multi_robot_trajectories(
                fig=fig, axs=axs, trajs=trajs,
                start_goal_pairs=start_goal_pairs, t=i, **kwargs
            )
        create_animation_video(fig, partial(animate_fn, axs=axs), n_frames=T, **kwargs)
        
    def plot_joint_space_state_trajectories(
            self,
            fig=None, axs=None,
            trajs=None,
            traj_best=None,
            pos_start_state=None, pos_goal_state=None,
            vel_start_state=None, vel_goal_state=None,
            set_joint_limits=True,
            **kwargs
    ):
        if trajs is None:
            return
        trajs_np = to_numpy(trajs)

        assert trajs_np.ndim == 3
        B, H, D = trajs_np.shape

        # Separate trajectories in collision and free (not in collision)
        trajs_coll, trajs_free = self.task.get_trajs_collision_and_free(trajs)

        trajs_coll_pos_np = to_numpy([])
        trajs_coll_vel_np = to_numpy([])
        if trajs_coll is not None:
            trajs_coll_pos_np = to_numpy(self.robot.get_position(trajs_coll))
            trajs_coll_vel_np = to_numpy(self.robot.get_velocity(trajs_coll))

        trajs_free_pos_np = to_numpy([])
        trajs_free_vel_np = to_numpy([])
        if trajs_free is not None:
            trajs_free_pos_np = to_numpy(self.robot.get_position(trajs_free))
            trajs_free_vel_np = to_numpy(self.robot.get_velocity(trajs_free))

        if pos_start_state is not None:
            pos_start_state = to_numpy(pos_start_state)
        if vel_start_state is not None:
            vel_start_state = to_numpy(vel_start_state)
        if pos_goal_state is not None:
            pos_goal_state = to_numpy(pos_goal_state)
        if vel_goal_state is not None:
            vel_goal_state = to_numpy(vel_goal_state)

        if fig is None or axs is None:
            fig, axs = plt.subplots(self.robot.q_dim, 2, squeeze=False)
        axs[0, 0].set_title('Position')
        axs[0, 1].set_title('Velocity')
        axs[-1, 0].set_xlabel('Timesteps')
        axs[-1, 1].set_xlabel('Timesteps')
        timesteps = np.arange(H).reshape(1, -1)
        for i, ax in enumerate(axs):
            for trajs_filtered, color in zip([(trajs_coll_pos_np, trajs_coll_vel_np), (trajs_free_pos_np, trajs_free_vel_np)],
                                             ['black', 'orange']):
                # Positions and velocities
                for j, trajs_filtered_ in enumerate(trajs_filtered):
                    if trajs_filtered_.size > 0:
                        timesteps_ = np.repeat(timesteps, trajs_filtered_.shape[0], axis=0)
                        plot_multiline(ax[j], timesteps_, trajs_filtered_[..., i], color=color, **kwargs)

            if traj_best is not None:
                traj_best_pos = self.robot.get_position(traj_best)
                traj_best_vel = self.robot.get_velocity(traj_best)
                traj_best_pos_np = to_numpy(traj_best_pos)
                traj_best_vel_np = to_numpy(traj_best_vel)
                plot_multiline(ax[0], timesteps, traj_best_pos_np[..., i].reshape(1, -1), color='blue', **kwargs)
                plot_multiline(ax[1], timesteps, traj_best_vel_np[..., i].reshape(1, -1), color='blue', **kwargs)

            # Start and goal
            if pos_start_state is not None:
                ax[0].scatter(0, pos_start_state[i], color='green')
            if vel_start_state is not None:
                ax[1].scatter(0, vel_start_state[i], color='green')
            if pos_goal_state is not None:
                ax[0].scatter(H-1, pos_goal_state[i], color='purple')
            if vel_goal_state is not None:
                ax[1].scatter(H-1, vel_goal_state[i], color='purple')
            # Y label
            ax[0].set_ylabel(f'q_{i}')
            # Set limits
            if set_joint_limits:
                ax[0].set_ylim(self.robot.q_min_np[i], self.robot.q_max_np[i])
                # ax[1].set_ylim(self.robot.q_vel_min_np[i], self.robot.q_vel_max_np[i])

        return fig, axs

    def animate_opt_iters_joint_space_state(
            self, trajs=None, traj_best=None, n_frames=10, **kwargs
    ):
        # trajs: steps, batch, horizon, q_dim
        if trajs is None:
            return

        assert trajs.ndim == 4
        S, B, H, D = trajs.shape

        idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        trajs_selection = trajs[idxs]

        fig, axs = self.plot_joint_space_state_trajectories(trajs=trajs_selection[0], **kwargs)

        def animate_fn(i):
            [ax.clear() for ax in axs.ravel()]
            fig.suptitle(f"iter: {idxs[i]}/{S-1}")
            self.plot_joint_space_state_trajectories(
                fig=fig, axs=axs,
                trajs=trajs_selection[i], **kwargs
            )
            if i == n_frames -1 and traj_best is not None:
                self.plot_joint_space_state_trajectories(
                    fig=fig, axs=axs,
                    trajs=trajs_selection[i],
                    traj_best=traj_best, **kwargs
                )

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)


def create_animation_video(fig, animate_fn, anim_time=5, n_frames=100, video_filepath='video.mp4', **kwargs):
    str_start = "Creating animation"
    print(f'{str_start}...')
    ani = FuncAnimation(
        fig,
        animate_fn,
        frames=n_frames,
        interval=anim_time * 1000 / n_frames,
        repeat=False
    )
    print(f'...finished {str_start}')

    str_start = "Saving video..."
    print(f'{str_start}...')
    ani.save(os.path.join(video_filepath), fps=max(1, int(n_frames / anim_time)), dpi=90)
    print(f'...finished {str_start}')


def plot_multiline(ax, X, Y, color='blue', linestyle='solid', **kwargs):
    segments = np.stack((X, Y), axis=-1)
    line_segments = mcoll.LineCollection(segments, colors=[color] * len(segments), linestyle=linestyle)
    ax.add_collection(line_segments)
    points = np.reshape(segments, (-1, 2))
    ax.scatter(points[:, 0], points[:, 1], color=color, s=2 ** 2)
