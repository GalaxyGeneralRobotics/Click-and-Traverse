import functools

# --- Set environment variables ---
import os
from collections.abc import Mapping
from dataclasses import dataclass

import tyro
from absl import logging

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- TensorFlow GPU setup ---
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.keras.mixed_precision.set_global_policy("float32")

import jax
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import tf2onnx
from mujoco_playground import wrapper
import cat_ppo

# --- MLP model definition ---
class MLP(tf.keras.Model):
    def __init__(
        self,
        layer_sizes,
        activation=tf.nn.relu,
        kernel_init="lecun_uniform",
        activate_final=False,
        bias=True,
        layer_norm=False,
    ):
        super().__init__()
        self.activation = activation
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        self.model = tf.keras.Sequential(name="MLP_0")

        for i, size in enumerate(layer_sizes):
            self.model.add(
                tf.keras.layers.Dense(
                    size,
                    activation=None,
                    use_bias=bias,
                    kernel_initializer=kernel_init,
                    name=f"hidden_{i}",
                )
            )
            if i != len(layer_sizes) - 1 or activate_final:
                if layer_norm:
                    self.model.add(tf.keras.layers.LayerNormalization(name=f"ln_{i}"))

    def call(self, inputs):
        x = inputs
        for layer in self.model.layers:
            x = layer(x)
            if isinstance(layer, tf.keras.layers.Dense):
                if self.activate_final or not layer.name.endswith(
                    f"{len(self.model.layers) // (2 if self.layer_norm else 1) - 1}"
                ):
                    x = self.activation(x)
        loc, _ = tf.split(x, 2, axis=-1)
        return tf.tanh(loc)


# --- Utility functions ---
def build_tf_policy_network(
    action_size,
    hidden_layer_sizes,
    activation="swish",
    kernel_init="lecun_uniform",
    layer_norm=False,
):
    if activation == "swish":
        activation = tf.nn.swish
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

    return MLP(
        layer_sizes=list(hidden_layer_sizes) + [action_size * 2],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
    )


def transfer_weights(jax_params, tf_model):
    for name, params in jax_params.items():
        try:
            tf_layer = tf_model.get_layer("MLP_0").get_layer(name=name)
        except ValueError:
            logging.error(f"Layer {name} not found in TF model.")
            continue
        if isinstance(tf_layer, tf.keras.layers.Dense):
            tf_layer.set_weights([np.array(params["kernel"]), np.array(params["bias"])])
        else:
            logging.error(f"Unhandled layer type: {type(tf_layer)}")
    logging.info("Weights transferred successfully.")


def get_latest_ckpt(path):
    from pathlib import Path

    ckpts = [ckpt for ckpt in Path(path).glob("*") if not ckpt.name.endswith(".json")]
    ckpts.sort(key=lambda x: int(x.name))
    return ckpts[-1] if ckpts else None


def convert_jax2onnx(
    ckpt_dir,
    output_path,
    inference_fn,
    hidden_layer_sizes,
    obs_size: int | Mapping[str, tuple[int, ...] | int],
    action_size: int,
    policy_obs_key,
    jax_params,
    activation="swish",
):
    rand_obs = {
        "state": np.random.randn(1, obs_size["state"][0]).astype(np.float32),
        "privileged_state": np.random.randn(1, obs_size["privileged_state"][0]).astype(
            np.float32
        ),
    }

    jax_pred, _ = inference_fn(rand_obs, jax.random.PRNGKey(0))
    jax_pred = np.array(jax_pred[0])

    tf_model = build_tf_policy_network(
        action_size=action_size,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )

    example_input = tf.ones((1, obs_size[policy_obs_key][0]))
    tf_model(example_input)  # build model

    transfer_weights(jax_params[1]["params"], tf_model)

    test_input = [rand_obs[policy_obs_key].reshape(1, -1)]
    tf_pred = tf_model(test_input)[0][0].numpy()

    tf_model.output_names = ["continuous_actions"]

    # Single input signature for ONNX conversion
    # spec = [
    #     tf.TensorSpec(
    #         shape=(1, obs_size[policy_obs_key][0]), dtype=tf.float32, name="obs"
    #     )
    # ]

    # Dynamic shape for ONNX conversion
    spec = (tf.TensorSpec([None, obs_size[policy_obs_key][0]], tf.float32, name="obs"),)
    tf2onnx.convert.from_keras(
        tf_model, input_signature=spec, opset=11, output_path=output_path
    )

# --- CLI args ---
@dataclass
class Args:
    task: str
    exp_name: str


# --- Main entry point ---
def main(args: Args):
    import brax.training.agents.ppo.train as ppo
    from brax.training.agents.ppo.networks import make_ppo_networks

    import cat_ppo

    ckpt_path = cat_ppo.get_path_log(args.exp_name) / "checkpoints"
    latest_ckpt = get_latest_ckpt(ckpt_path)

    if latest_ckpt is None:
        raise FileNotFoundError("No checkpoint found.")

    logging.info(f"Using checkpoint: {latest_ckpt}")
    output_path = f"{latest_ckpt}/policy.onnx"

    env_class = cat_ppo.registry.get(args.task, "train_env_class")
    task_cfg = cat_ppo.registry.get(args.task, "config")
    env_cfg = task_cfg.env_config
    policy_config = task_cfg.policy_config
    env = env_class(task_type=env_cfg.task_type, config=env_cfg)

    policy_obs_key = policy_config.network_factory.policy_obs_key

    network_factory = functools.partial(
        make_ppo_networks, **policy_config.network_factory
    )
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=0,
        episode_length=policy_config.episode_length,
        normalize_observations=False,
        restore_checkpoint_path=latest_ckpt,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_envs=1,
    )

    make_inference_fn, params, _ = train_fn(environment=env)
    inference_fn = make_inference_fn(params, deterministic=True)

    obs_size = env.observation_size
    act_size = env.action_size

    convert_jax2onnx(
        ckpt_dir=latest_ckpt,
        output_path=output_path,
        inference_fn=inference_fn,
        hidden_layer_sizes=policy_config.network_factory.policy_hidden_layer_sizes,
        obs_size=obs_size,
        action_size=act_size,
        policy_obs_key=policy_obs_key,
        jax_params=params,
        activation="swish",
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
