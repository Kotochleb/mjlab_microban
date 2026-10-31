"""Play a tracking policy in viser and show which body frames it tracks.

Takes the same arguments as `uv run play` (the viser viewer is always used), e.g.:
    uv run python src/mjlab_microban/scripts/view_tracking.py Mjlab-Tracking-Microban \
        --wandb-run-path kotochleb-cnrs-laas/mjlab_microban_tracking/efuotrwp \
        --motion-file src/mjlab_microban/motion/S0_normal_walk_0_seg00_first320.npz

On top of the regular play viewer, every body of the motion command gets:
- a frame at the target pose the terms compare against: the reference anchor pose for
  the anchor, the reference pose re-anchored on the robot (body_*_relative_w) for the
  other bodies, so it differs from the ghost, which shows the raw reference;
- a smaller frame at the current robot body pose and a line from it to the target;
- a label with its current position/orientation error and the reward and termination
  terms acting on it, read from the env config.
"""

import dataclasses
import sys
from dataclasses import dataclass

import mjlab
import mjlab.scripts.play as play
import numpy as np
import torch
import tyro
import viser
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.math import quat_error_magnitude
from mjlab.viewer import ViserPlayViewer

ANCHOR_RGB = (230, 60, 60)
TRACKED_RGB = (236, 236, 0)
ERROR_RGB = (255, 140, 0)


@dataclass
class TrackingTerm:
    name: str
    kind: str  # "reward" or "termination"
    func: str
    bodies: tuple[str, ...]
    weight: float | None
    params: dict


def tracking_terms(cfg: ManagerBasedRlEnvCfg, command_name: str = "motion") -> list[TrackingTerm]:
    """Reward and termination terms fed by the motion command, with the bodies they act on."""
    motion = cfg.commands[command_name]
    terms = []
    for kind, group in (("reward", cfg.rewards), ("termination", cfg.terminations)):
        for name, term in group.items():
            if term is None or term.params.get("command_name") != command_name:
                continue
            func = term.func.__name__
            if "anchor" in func:
                bodies = (motion.anchor_body_name,)
            else:
                bodies = tuple(term.params.get("body_names") or motion.body_names)
            # Keep the scalar parameters (std, threshold, ...) for display.
            params = {k: v for k, v in term.params.items() if isinstance(v, (int, float))}
            terms.append(TrackingTerm(name, kind, func, bodies, getattr(term, "weight", None), params))
    return terms


def term_summary(term: TrackingTerm) -> str:
    values = [f"w={term.weight:g}"] if term.weight is not None else []
    values += [f"{k}={v:g}" for k, v in term.params.items()]
    return f"{term.name} ({', '.join(values)})"


class TrackingViserPlayViewer(ViserPlayViewer):
    """ViserPlayViewer drawing the target and current pose of every tracked body."""

    def setup(self) -> None:
        super().setup()
        env = self.env.unwrapped
        self._command: MotionCommand = env.command_manager.get_term("motion")
        cfg = self._command.cfg
        self._bodies = list(cfg.body_names)
        self._anchor_idx = self._bodies.index(cfg.anchor_body_name)
        terms = tracking_terms(env.cfg)
        self._label_terms = {
            body: "\n".join(f"  {term_summary(t)}" for t in terms if body in t.bodies) for body in self._bodies
        }
        # Only report the errors that some term on the body actually uses.
        self._shows_pos_err = [any(b in t.bodies and "pos" in t.func for t in terms) for b in self._bodies]
        self._shows_ori_err = [any(b in t.bodies and "ori" in t.func for t in terms) for b in self._bodies]

        scene = self._server.scene
        self._target_frames = []
        self._robot_frames = []
        self._labels = []
        for i, body in enumerate(self._bodies):
            is_anchor = i == self._anchor_idx
            self._target_frames.append(
                scene.add_frame(
                    f"/tracking/target/{body}",
                    axes_length=0.08 if is_anchor else 0.06,
                    axes_radius=0.004,
                    origin_radius=0.012,
                    origin_color=ANCHOR_RGB if is_anchor else TRACKED_RGB,
                )
            )
            self._robot_frames.append(
                scene.add_frame(f"/tracking/robot/{body}", axes_length=0.04, axes_radius=0.003, origin_radius=0.006)
            )
            self._labels.append(scene.add_label(f"/tracking/label/{body}", body, font_screen_scale=0.7))
        self._error_lines = scene.add_line_segments(
            "/tracking/error", np.zeros((len(self._bodies), 2, 3)), ERROR_RGB, thickness=0.004
        )

        with self._server.gui.add_folder("Tracked frames"):
            show_targets = self._server.gui.add_checkbox("Target frames", initial_value=True)
            show_robot = self._server.gui.add_checkbox("Robot frames", initial_value=True)
            show_errors = self._server.gui.add_checkbox("Error lines", initial_value=True)
            show_labels = self._server.gui.add_checkbox("Labels", initial_value=True)
            rows = [f"| `{t.name}` | {t.kind} | {', '.join(t.bodies)} |" for t in terms]
            self._server.gui.add_markdown("| term | kind | bodies |\n|---|---|---|\n" + "\n".join(rows))

        def bind(checkbox: viser.GuiCheckboxHandle, handles: list) -> None:
            @checkbox.on_update
            def _(_) -> None:
                for handle in handles:
                    handle.visible = checkbox.value

        bind(show_targets, self._target_frames)
        bind(show_robot, self._robot_frames)
        bind(show_errors, [self._error_lines])
        bind(show_labels, self._labels)

        # Refresh the overlay with every scene update, so it shares the scene offset
        # and runs under the same lock as the robot meshes.
        scene_update = self._scene.update

        def update(wp_data, env_idx: int | None = None) -> None:
            scene_update(wp_data, env_idx)
            self._update_tracking_overlay()

        self._scene.update = update

    def _update_tracking_overlay(self) -> None:
        c = self._command
        i = self._scene.env_idx
        a = self._anchor_idx
        target_pos = c.body_pos_relative_w[i].clone()
        target_quat = c.body_quat_relative_w[i].clone()
        target_pos[a] = c.anchor_pos_w[i]
        target_quat[a] = c.anchor_quat_w[i]
        robot_pos = c.robot_body_pos_w[i]
        robot_quat = c.robot_body_quat_w[i]
        pos_err = torch.linalg.norm(target_pos - robot_pos, dim=-1).cpu().numpy()
        ori_err = quat_error_magnitude(target_quat, robot_quat).cpu().numpy()

        offset = self._scene._scene_offset
        target_pos = target_pos.cpu().numpy() + offset
        target_quat = target_quat.cpu().numpy()
        robot_pos = robot_pos.cpu().numpy() + offset
        robot_quat = robot_quat.cpu().numpy()
        for j, body in enumerate(self._bodies):
            self._target_frames[j].position = target_pos[j]
            self._target_frames[j].wxyz = target_quat[j]
            self._robot_frames[j].position = robot_pos[j]
            self._robot_frames[j].wxyz = robot_quat[j]
            self._labels[j].position = robot_pos[j]
            errors = []
            if self._shows_pos_err[j]:
                errors.append(f"{100 * pos_err[j]:.1f} cm")
            if self._shows_ori_err[j]:
                errors.append(f"{np.rad2deg(ori_err[j]):.0f} deg")
            self._labels[j].text = (
                f"{body}{' (anchor)' if j == a else ''}: {', '.join(errors)}\n{self._label_terms[body]}"
            )
        self._error_lines.points = np.stack([robot_pos, target_pos], axis=1)


def main() -> None:
    # Same argument parsing as mjlab's play script.
    import mjlab.tasks  # noqa: F401  (populates the task registry)

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(
        play.PlayConfig,
        args=remaining_args,
        default=play.PlayConfig(viewer="viser"),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    args = dataclasses.replace(args, viewer="viser")

    # run_play builds the env and the policy, then hands them to this viewer.
    play.ViserPlayViewer = TrackingViserPlayViewer
    play.run_play(chosen_task, args)


if __name__ == "__main__":
    main()
