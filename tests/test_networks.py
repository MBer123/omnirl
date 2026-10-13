import io
import unittest

import torch

from omnirl.networks import (
    CategoricalQNetwork,
    DeepQNetwork,
    DeterministicActorNetwork,
    DuelingQNetwork,
    QCriticNetwork,
    QuantileCriticNetwork,
    QuantileQNetwork,
    StochasticActorNetwork,
    ValueCriticNetwork,
)
from omnirl.networks.base import CNN, MLP, ResNet


class NetworkTest(unittest.TestCase):
    def test_base_network_outputs(self):
        mlp = MLP(input_dim=6, mlp_hidden_dims=(16, 8))
        self.assertEqual(mlp(torch.randn(4, 6)).shape, (4, 8))
        self.assertEqual(mlp.output_dim, 8)

        cnn_kwargs = {
            "state_dim": (8, 8, 3),
            "conv_layers": ((4, 3, 1, 0),),
        }
        cnn = CNN(**cnn_kwargs)
        self.assertEqual(cnn(torch.randn(2, 8, 8, 3)).shape, (2, 256))
        self.assertEqual(cnn.output_dim, 256)

        projected_cnn = CNN(**cnn_kwargs, conv_hidden_dims=(16, 8))
        self.assertEqual(projected_cnn(torch.randn(2, 8, 8, 3)).shape, (2, 8))
        self.assertEqual(projected_cnn.output_dim, 8)

        resnet = ResNet(
            state_dim=(32, 32, 3),
            stem_channels=8,
            stage_channels=(8, 16),
            blocks_per_stage=(1, 1),
            resnet_hidden_dims=(32,),
        )
        self.assertEqual(resnet(torch.randn(2, 32, 32, 3)).shape, (2, 32))
        self.assertEqual(resnet.output_dim, 32)

    def test_explicit_layer_type_and_network_kwargs(self):
        states = torch.randn(4, 6)
        network_kwargs = {"mlp_hidden_dims": (16, 8)}

        actor = StochasticActorNetwork(
            state_dim=6,
            action_dim=3,
            layer_type="MLP",
            network_kwargs=network_kwargs,
        )
        critic = ValueCriticNetwork(
            state_dim=6,
            layer_type="MLP",
            network_kwargs=network_kwargs,
        )

        self.assertEqual(actor(states).shape, (4, 3))
        self.assertEqual(critic(states).shape, (4, 1))

    def test_mlp_actor_outputs(self):
        states = torch.randn(4, 6)

        discrete_actor_kwargs = {
            "state_dim": 6,
            "action_dim": 3,
            "layer_type": "MLP",
            "network_kwargs": {"mlp_hidden_dims": (16, 8)},
        }
        discrete_actor = StochasticActorNetwork(**discrete_actor_kwargs)
        self.assertEqual(discrete_actor(states).shape, (4, 3))

        continuous_actor_kwargs = {
            "state_dim": 6,
            "action_dim": 2,
            "is_continuous": True,
            "layer_type": "MLP",
            "log_std_min": -1.0,
            "log_std_max": 0.5,
            "network_kwargs": {"mlp_hidden_dims": (16, 8)},
        }
        continuous_actor = StochasticActorNetwork(**continuous_actor_kwargs)
        means, log_stds = continuous_actor(states)
        self.assertEqual(means.shape, (4, 2))
        self.assertEqual(log_stds.shape, (4, 2))
        self.assertTrue(torch.all(log_stds >= -1.0))
        self.assertTrue(torch.all(log_stds <= 0.5))

        deterministic_actor_kwargs = {
            "state_dim": 6,
            "action_dim": 2,
            "layer_type": "MLP",
            "network_kwargs": {"mlp_hidden_dims": (16, 8)},
        }
        deterministic_actor = DeterministicActorNetwork(**deterministic_actor_kwargs)
        actions = deterministic_actor(states)
        self.assertEqual(actions.shape, (4, 2))
        self.assertTrue(torch.all(actions >= -1.0))
        self.assertTrue(torch.all(actions <= 1.0))

    def test_mlp_critic_outputs(self):
        states = torch.randn(4, 6)
        actions = torch.randn(4, 2)

        value_critic_kwargs = {
            "state_dim": 6,
            "layer_type": "MLP",
            "network_kwargs": {"mlp_hidden_dims": (16, 8)},
        }
        value_critic = ValueCriticNetwork(**value_critic_kwargs)
        self.assertEqual(value_critic(states).shape, (4, 1))

        continuous_q_kwargs = {
            "state_dim": 6,
            "action_dim": 2,
            "layer_type": "MLP",
            "network_kwargs": {"mlp_hidden_dims": (16, 8)},
        }
        continuous_q_critic = QCriticNetwork(**continuous_q_kwargs)
        self.assertEqual(continuous_q_critic(states, actions).shape, (4, 1))

        discrete_q_kwargs = {
            "state_dim": 6,
            "action_dim": 3,
            "layer_type": "MLP",
            "network_kwargs": {"mlp_hidden_dims": (16, 8)},
        }
        discrete_q_critic = DeepQNetwork(**discrete_q_kwargs)
        self.assertEqual(discrete_q_critic(states).shape, (4, 3))

        quantile_critic_kwargs = {
            "state_dim": 6,
            "action_dim": 2,
            "layer_type": "MLP",
            "num_quantiles": 7,
            "network_kwargs": {"mlp_hidden_dims": (16, 8)},
        }
        quantile_critic = QuantileCriticNetwork(**quantile_critic_kwargs)
        self.assertEqual(quantile_critic(states, actions).shape, (4, 7))

    def test_cnn_network_outputs(self):
        state_dim = (8, 8, 3)
        states = torch.randn(2, *state_dim)
        actions = torch.randn(2, 2)
        network_kwargs = {
            "conv_layers": ((4, 3, 1, 0),),
            "conv_hidden_dims": (8,),
        }

        actor_kwargs = {
            "state_dim": state_dim,
            "action_dim": 2,
            "is_continuous": True,
            "layer_type": "CNN",
            "network_kwargs": network_kwargs,
        }
        actor = StochasticActorNetwork(**actor_kwargs)
        means, log_stds = actor(states)
        self.assertEqual(means.shape, (2, 2))
        self.assertEqual(log_stds.shape, (2, 2))

        value_critic = ValueCriticNetwork(
            state_dim=state_dim,
            layer_type="CNN",
            network_kwargs=network_kwargs,
        )
        self.assertEqual(value_critic(states).shape, (2, 1))

        q_critic_kwargs = {
            "state_dim": state_dim,
            "action_dim": 2,
            "layer_type": "CNN",
            "network_kwargs": network_kwargs,
        }
        q_critic = QCriticNetwork(**q_critic_kwargs)
        self.assertEqual(q_critic(states, actions).shape, (2, 1))

        quantile_critic_kwargs = {
            "state_dim": state_dim,
            "action_dim": 2,
            "layer_type": "CNN",
            "num_quantiles": 5,
            "network_kwargs": network_kwargs,
        }
        quantile_critic = QuantileCriticNetwork(**quantile_critic_kwargs)
        self.assertEqual(quantile_critic(states, actions).shape, (2, 5))

    def test_q_network_outputs(self):
        mlp_states = torch.randn(4, 6)
        cnn_states = torch.randn(2, 8, 8, 3)
        network_kwargs = {
            "conv_layers": ((4, 3, 1, 0),),
            "conv_hidden_dims": (8,),
        }

        network_cases = (
            (DeepQNetwork, {}, (4, 3), (2, 3)),
            (DuelingQNetwork, {}, (4, 3), (2, 3)),
            (CategoricalQNetwork, {"num_atoms": 5}, (4, 3, 5), (2, 3, 5)),
            (QuantileQNetwork, {"num_quantiles": 7}, (4, 3, 7), (2, 3, 7)),
        )

        for network_cls, kwargs, mlp_shape, cnn_shape in network_cases:
            with self.subTest(network=network_cls.__name__, layer_type="MLP"):
                mlp_kwargs = {
                    "state_dim": 6,
                    "action_dim": 3,
                    "layer_type": "MLP",
                    "network_kwargs": {"mlp_hidden_dims": (16, 8)},
                    **kwargs,
                }
                network = network_cls(**mlp_kwargs)
                self.assertEqual(network(mlp_states).shape, mlp_shape)

            with self.subTest(network=network_cls.__name__, layer_type="CNN"):
                cnn_kwargs = {
                    "state_dim": (8, 8, 3),
                    "action_dim": 3,
                    "layer_type": "CNN",
                    "network_kwargs": network_kwargs,
                    **kwargs,
                }
                network = network_cls(**cnn_kwargs)
                self.assertEqual(network(cnn_states).shape, cnn_shape)

        dueling_kwargs = {
            "state_dim": 6,
            "action_dim": 3,
            "layer_type": "MLP",
            "network_kwargs": {"mlp_hidden_dims": (16, 8)},
        }
        dueling_network = DuelingQNetwork(**dueling_kwargs)
        features = dueling_network.layers(mlp_states)
        state_values = dueling_network.output["V"](features)
        advantages = dueling_network.output["A"](features)
        q_values = dueling_network(mlp_states)

        mean_advantages = advantages.mean(dim=-1, keepdim=True)
        expected_q_values = state_values + advantages - mean_advantages
        mean_q_values = q_values.mean(dim=-1, keepdim=True)

        self.assertTrue(torch.allclose(q_values, expected_q_values))
        self.assertTrue(torch.allclose(mean_q_values, state_values))

    def test_resnet_network_outputs(self):
        state_dim = (32, 32, 3)
        states = torch.randn(2, *state_dim)
        actions = torch.randn(2, 2)
        network_kwargs = {
            "stem_channels": 8,
            "stage_channels": (8, 16),
            "blocks_per_stage": (1, 1),
            "resnet_hidden_dims": (32,),
        }

        actor = StochasticActorNetwork(
            state_dim=state_dim,
            action_dim=3,
            layer_type="RESNET",
            network_kwargs=network_kwargs,
        )
        self.assertEqual(actor(states).shape, (2, 3))

        deterministic_actor = DeterministicActorNetwork(
            state_dim=state_dim,
            action_dim=2,
            layer_type="RESNET",
            network_kwargs=network_kwargs,
        )
        self.assertEqual(deterministic_actor(states).shape, (2, 2))

        value_critic = ValueCriticNetwork(
            state_dim=state_dim,
            layer_type="RESNET",
            network_kwargs=network_kwargs,
        )
        self.assertEqual(value_critic(states).shape, (2, 1))

        q_critic = QCriticNetwork(
            state_dim=state_dim,
            action_dim=2,
            layer_type="RESNET",
            network_kwargs=network_kwargs,
        )
        self.assertEqual(q_critic(states, actions).shape, (2, 1))

        discrete_q_critic = DeepQNetwork(
            state_dim=state_dim,
            action_dim=3,
            layer_type="RESNET",
            network_kwargs=network_kwargs,
        )
        self.assertEqual(discrete_q_critic(states).shape, (2, 3))

        quantile_critic = QuantileCriticNetwork(
            state_dim=state_dim,
            action_dim=2,
            layer_type="RESNET",
            num_quantiles=5,
            network_kwargs=network_kwargs,
        )
        self.assertEqual(quantile_critic(states, actions).shape, (2, 5))

        network_cases = (
            (DeepQNetwork, {}, (2, 3)),
            (DuelingQNetwork, {}, (2, 3)),
            (CategoricalQNetwork, {"num_atoms": 5}, (2, 3, 5)),
            (QuantileQNetwork, {"num_quantiles": 7}, (2, 3, 7)),
        )
        for network_cls, kwargs, expected_shape in network_cases:
            with self.subTest(network=network_cls.__name__):
                network = network_cls(
                    state_dim=state_dim,
                    action_dim=3,
                    layer_type="RESNET",
                    network_kwargs=network_kwargs,
                    **kwargs,
                )
                self.assertEqual(network(states).shape, expected_shape)

    def test_network_checkpoint_roundtrip(self):
        actor_kwargs = {
            "state_dim": 4,
            "action_dim": 2,
            "is_continuous": True,
            "layer_type": "MLP",
            "network_kwargs": {"mlp_hidden_dims": (8, 8)},
        }
        critic_kwargs = {
            "state_dim": 4,
            "action_dim": 2,
            "layer_type": "MLP",
            "network_kwargs": {"mlp_hidden_dims": (8, 8)},
        }
        cnn_actor_kwargs = {
            "state_dim": (8, 8, 3),
            "action_dim": 2,
            "layer_type": "CNN",
            "network_kwargs": {
                "conv_layers": ((4, 3, 1, 0),),
                "conv_hidden_dims": (8,),
            },
        }
        states = torch.randn(4, 4)
        actions = torch.randn(4, 2)
        images = torch.randn(2, 8, 8, 3)
        image_actions = torch.randn(2, 2)
        network_cases = (
            (StochasticActorNetwork, actor_kwargs, (states,)),
            (QCriticNetwork, critic_kwargs, (states, actions)),
            (DeepQNetwork, critic_kwargs, (states,)),
            (StochasticActorNetwork, cnn_actor_kwargs, (images,)),
            (DeterministicActorNetwork, cnn_actor_kwargs, (images,)),
            (QCriticNetwork, cnn_actor_kwargs, (images, image_actions)),
        )
        for network_cls, kwargs, inputs in network_cases:
            with self.subTest(
                network=network_cls.__name__, layer_type=kwargs["layer_type"]
            ):
                network = network_cls(**kwargs)
                with io.BytesIO() as checkpoint:
                    torch.save(network.state_dict(), checkpoint)
                    checkpoint.seek(0)
                    weights = torch.load(checkpoint, weights_only=True)

                restored = network_cls(**kwargs)
                restored.load_state_dict(weights)
                torch.testing.assert_close(restored(*inputs), network(*inputs))


if __name__ == "__main__":
    unittest.main()
