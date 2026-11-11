import importlib
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from omnirl.runtime.runner import Runner

ALGORITHMS = {
    "a2c": "omnirl.algorithms.on_policy.a2c",
    "ddpg": "omnirl.algorithms.off_policy.ddpg",
    "impala": "omnirl.algorithms.off_policy.impala",
    "ppo": "omnirl.algorithms.on_policy.ppo",
    "sac": "omnirl.algorithms.off_policy.sac",
    "td3": "omnirl.algorithms.off_policy.td3",
    "tqc": "omnirl.algorithms.off_policy.tqc",
    "c51_dqn": "omnirl.algorithms.off_policy.c51_dqn",
    "ddqn": "omnirl.algorithms.off_policy.ddqn",
    "dqn": "omnirl.algorithms.off_policy.dqn",
    "dueling_dqn": "omnirl.algorithms.off_policy.dueling_dqn",
    "per_dqn": "omnirl.algorithms.off_policy.per_dqn",
    "qr_dqn": "omnirl.algorithms.off_policy.qr_dqn",
}

CONFIG_ROOT = Path("examples")


class StructureTest(unittest.TestCase):
    def test_algorithm_exports(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                module = importlib.import_module(ALGORITHMS[name])
                for component in ["Learner", "Memory", "Policy"]:
                    self.assertTrue(hasattr(module, component))
                self.assertTrue(callable(module.Policy.set_weights))
                self.assertTrue(callable(module.Policy.choose_action))
                self.assertTrue(callable(module.Policy.sample_action))

    def test_algorithm_example_coverage(self):
        template_paths = list((CONFIG_ROOT / "algorithms").glob("*.yaml"))
        self.assertEqual({path.stem for path in template_paths}, set(ALGORITHMS))
        for path in template_paths:
            with self.subTest(algorithm=path.stem):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    config["agent"]["module"], ALGORITHMS[path.stem]
                )

    def test_ppo_environment_examples(self):
        cases = {
            "cartpole": ("CartPole-v1", "MLP"),
            "bipedal_walker": ("BipedalWalker-v3", "MLP"),
            "car_racing": ("CarRacing-v3", "CNN"),
        }
        for name, (env_id, layer_type) in cases.items():
            with self.subTest(example=name):
                example_dir = CONFIG_ROOT / name
                agent = yaml.safe_load(
                    (example_dir / "ppo.yaml").read_text(encoding="utf-8")
                )
                train = yaml.safe_load(
                    (example_dir / "train.yaml").read_text(encoding="utf-8")
                )
                evaluation = yaml.safe_load(
                    (example_dir / "test.yaml").read_text(encoding="utf-8")
                )
                self.assertEqual(agent["agent"]["module"], ALGORITHMS["ppo"])
                self.assertGreaterEqual(
                    agent["modules"]["memory"]["batch_size"],
                    train["runner"]["collection_fragment_length"],
                )
                for component in ("actor", "critic"):
                    self.assertEqual(
                        agent["modules"]["network"][component]["layer_type"], layer_type
                    )
                self.assertEqual(train["env"]["id"], env_id)
                self.assertEqual(evaluation["env"], train["env"])
                self.assertEqual(
                    evaluation["runner"]["ckpt_dir"], train["runner"]["ckpt_dir"]
                )
                self.assertEqual(
                    evaluation["runner"]["eval_model"],
                    f"agent_{train['runner']['max_updates']}.pth",
                )

    def test_config_module_paths(self):
        paths = list(CONFIG_ROOT.rglob("*.yaml"))
        self.assertTrue(paths)

        for path in paths:
            with self.subTest(config=str(path)):
                with path.open("r", encoding="utf-8") as file:
                    config = yaml.safe_load(file)
                self.assertNotIn("pipeline", config)
                if "agent" in config:
                    importlib.import_module(config["agent"]["module"])
                else:
                    importlib.import_module(config["env"]["module"])

    def test_config_fields(self):
        runner_params = set(inspect.signature(Runner.__init__).parameters)

        for path in CONFIG_ROOT.rglob("*.yaml"):
            with self.subTest(config=str(path)):
                with path.open("r", encoding="utf-8") as file:
                    config = yaml.safe_load(file)

                if "agent" in config:
                    self.assertEqual(set(config), {"agent", "modules"})
                    self.assertEqual(set(config["agent"]), {"module"})
                    module_names = set(config["modules"])
                    allowed_modules = {"learner", "memory", "policy", "network"}
                    self.assertLessEqual(module_names, allowed_modules)

                    network = config["modules"].get("network")
                    if network:
                        network_parts = (
                            [network["actor"], network["critic"]]
                            if "actor" in network
                            else [network]
                        )
                        for network_part in network_parts:
                            self.assertIn("layer_type", network_part)
                            self.assertIn(
                                network_part["layer_type"],
                                {"MLP", "CNN", "RESNET"},
                            )

                    module = importlib.import_module(config["agent"]["module"])
                    components = {
                        "learner": module.Learner.__ray_metadata__.modified_class,
                        "memory": module.Memory.__ray_metadata__.modified_class,
                        "policy": module.Policy,
                    }
                    for name, component in components.items():
                        kwargs = set((config["modules"].get(name) or {}).keys())
                        params = set(inspect.signature(component.__init__).parameters)
                        self.assertLessEqual(kwargs, params)
                else:
                    allowed_fields = {"env", "runner", "device", "seed"}
                    self.assertLessEqual(set(config), allowed_fields)
                    env_fields = set(config["env"])
                    self.assertIn("module", env_fields)
                    self.assertNotIn("role", env_fields)
                    self.assertLessEqual(set(config["runner"]), runner_params)
                    self.assertLessEqual(set(config.get("device", {})), {"learner"})
                    seed_fields = set(config.get("seed", {}))
                    allowed_seed_fields = {"learner", "sampler", "evaluator"}
                    self.assertLessEqual(seed_fields, allowed_seed_fields)

    def test_runner_config_loading(self):
        example_dir = CONFIG_ROOT / "cartpole"

        with (example_dir / "train.yaml").open("r", encoding="utf-8") as file:
            runner_config = yaml.safe_load(file)
        with (example_dir / "ppo.yaml").open("r", encoding="utf-8") as file:
            agent_config = yaml.safe_load(file)
        runner_config["runner"]["collection_fragment_length"] = 8

        agent = object()
        environment = object()
        runner_args = (agent, environment, runner_config, agent_config)
        with patch.object(Runner, "__init__", return_value=None) as init:
            Runner.from_config(*runner_args)
        captured_kwargs = init.call_args.kwargs

        self.assertIs(captured_kwargs["agent"], agent)
        self.assertIs(captured_kwargs["environment"], environment)
        expected_env_config = {"id": "CartPole-v1", "render_mode": None}
        self.assertEqual(captured_kwargs["env_config"], expected_env_config)
        self.assertEqual(captured_kwargs["agent_config"], agent_config["modules"])
        self.assertEqual(captured_kwargs["collection_fragment_length"], 8)
        self.assertEqual(captured_kwargs["learner_seed"], 0)
        self.assertEqual(captured_kwargs["sampler_seed"], 0)
        self.assertEqual(captured_kwargs["evaluator_seed"], 0)

    def test_environment_arguments_are_forwarded_opaquely(self):
        runner_config = {
            "env": {
                "module": "omnirl.envs.example",
                "game": "example",
                "custom_option": 42,
            }
        }
        agent_config = {"modules": {}}
        agent = object()
        environment = object()

        runner_args = (agent, environment, runner_config, agent_config)
        with patch.object(Runner, "__init__", return_value=None) as init:
            Runner.from_config(*runner_args)
        captured_kwargs = init.call_args.kwargs

        expected_env_config = {"game": "example", "custom_option": 42}
        self.assertEqual(captured_kwargs["env_config"], expected_env_config)

    def test_network_location(self):
        network_files = list(Path("omnirl/algorithms").rglob("network.py"))
        self.assertEqual(network_files, [])

    def test_evaluator_location(self):
        evaluator_files = list(Path("omnirl/algorithms").rglob("evaluator.py"))
        self.assertEqual(evaluator_files, [])

    def test_sampler_location(self):
        sampler_files = list(Path("omnirl/algorithms").rglob("sampler.py"))
        self.assertEqual(sampler_files, [])

    def test_layer_boundaries(self):
        algorithm_paths = Path("omnirl/algorithms").rglob("*.py")
        algorithm_files = [path.read_text(encoding="utf-8") for path in algorithm_paths]
        algorithm_sources = "\n".join(algorithm_files)
        for forbidden in [
            "omnirl.envs",
            "omnirl.runtime",
            "env.reset(",
            "env.step(",
            "action_space",
        ]:
            self.assertNotIn(forbidden, algorithm_sources)

        runtime_paths = Path("omnirl/runtime").rglob("*.py")
        runtime_files = [path.read_text(encoding="utf-8") for path in runtime_paths]
        runtime_sources = "\n".join(runtime_files)
        for forbidden in [
            "import gymnasium",
            "from gymnasium",
            "omnirl.envs",
            "omnirl.algorithms",
            ".reset(",
            ".step(",
            ".render(",
            "action_space",
            "render_mode",
            "env_id",
        ]:
            self.assertNotIn(forbidden, runtime_sources)

        environment_paths = Path("omnirl/envs").rglob("*.py")
        environment_files = [
            path.read_text(encoding="utf-8") for path in environment_paths
        ]
        environment_sources = "\n".join(environment_files)
        for forbidden in ["omnirl.algorithms", "omnirl.networks", "omnirl.runtime"]:
            self.assertNotIn(forbidden, environment_sources)


if __name__ == "__main__":
    unittest.main()
