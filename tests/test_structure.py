import importlib
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import DEFAULT, Mock, patch

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
                self.assertEqual(config["agent"]["module"], ALGORITHMS[path.stem])

    def test_ppo_environment_examples(self):
        cases = {
            "cartpole": ("CartPole-v1", "MLP"),
            "bipedal_walker": ("BipedalWalker-v3", "MLP"),
            "car_racing": ("CarRacing-v3", "CNN"),
        }
        for name, (env_id, layer_type) in cases.items():
            with self.subTest(example=name):
                example_dir = CONFIG_ROOT / name
                agent_path = example_dir / "ppo.yaml"
                train_path = example_dir / "train.yaml"
                evaluation_path = example_dir / "test.yaml"
                agent = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
                train = yaml.safe_load(train_path.read_text(encoding="utf-8"))
                evaluation = yaml.safe_load(evaluation_path.read_text(encoding="utf-8"))
                modules = agent["modules"]
                train_runner = train["runner"]
                evaluation_runner = evaluation["runner"]
                self.assertEqual(agent["agent"]["module"], ALGORITHMS["ppo"])
                batch_size = modules["memory"]["batch_size"]
                fragment_length = train["sampler"]["collection_fragment_length"]
                self.assertGreaterEqual(batch_size, fragment_length)
                for component in ("actor", "critic"):
                    network = modules["network"][component]
                    self.assertEqual(network["layer_type"], layer_type)
                self.assertEqual(train["env"]["id"], env_id)
                self.assertEqual(evaluation["env"], train["env"])
                expected_ckpt_dir = train_runner["ckpt_dir"]
                self.assertEqual(evaluation_runner["ckpt_dir"], expected_ckpt_dir)
                expected_model = f"agent_{train['learner']['max_updates']}.pth"
                self.assertEqual(evaluation_runner["eval_model"], expected_model)
                self.assertEqual(train["evaluator"]["num_envs_per_evaluator"], 1)
                self.assertEqual(evaluation["evaluator"]["num_envs_per_evaluator"], 1)

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
                        network_parts = [network]
                        if "actor" in network:
                            network_parts = [network["actor"], network["critic"]]
                        for network_part in network_parts:
                            self.assertIn("layer_type", network_part)
                            layer_type = network_part["layer_type"]
                            self.assertIn(layer_type, {"MLP", "CNN", "RESNET"})

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
                    allowed_fields = {
                        "env",
                        "runner",
                        "learner",
                        "sampler",
                        "evaluator",
                        "memory",
                        "seed",
                    }
                    self.assertLessEqual(set(config), allowed_fields)
                    env_fields = set(config["env"])
                    self.assertIn("module", env_fields)
                    self.assertNotIn("role", env_fields)
                    self.assertLessEqual(set(config["runner"]), runner_params)
                    for role in ("learner", "sampler", "evaluator", "memory"):
                        fields = set(config.get(role) or {})
                        self.assertLessEqual(fields, runner_params)
                    seed_fields = set(config.get("seed", {}))
                    allowed_seed_fields = {"learner", "sampler", "evaluator"}
                    self.assertLessEqual(seed_fields, allowed_seed_fields)

    def test_runner_config_loading(self):
        example_dir = CONFIG_ROOT / "cartpole"

        with (example_dir / "train.yaml").open("r", encoding="utf-8") as file:
            runner_config = yaml.safe_load(file)
        with (example_dir / "ppo.yaml").open("r", encoding="utf-8") as file:
            agent_config = yaml.safe_load(file)
        runner_config["learner"]["learner_device"] = "gpu"
        runner_config["memory"]["num_buffers"] = 2
        runner_config["sampler"].update(
            collection_fragment_length=8,
            num_sampler_gpus=2,
            num_samplers_per_gpu=4,
        )
        evaluator_config = runner_config["evaluator"]
        evaluator_config.update(
            num_envs_per_evaluator=4,
            num_evaluator_gpus=1,
            num_evaluators_per_gpu=2,
        )

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
        self.assertEqual(captured_kwargs["num_learners"], 1)
        self.assertEqual(captured_kwargs["num_samplers"], 1)
        self.assertEqual(captured_kwargs["num_envs_per_sampler"], 4)
        self.assertEqual(captured_kwargs["collection_fragment_length"], 8)
        self.assertEqual(captured_kwargs["num_evaluators"], 1)
        self.assertEqual(captured_kwargs["num_envs_per_evaluator"], 4)
        self.assertEqual(captured_kwargs["num_episodes_per_env"], 5)
        self.assertEqual(captured_kwargs["num_buffers"], 2)
        self.assertEqual(captured_kwargs["learner_device"], "gpu")
        self.assertEqual(captured_kwargs["num_sampler_gpus"], 2)
        self.assertEqual(captured_kwargs["num_samplers_per_gpu"], 4)
        self.assertEqual(captured_kwargs["num_evaluator_gpus"], 1)
        self.assertEqual(captured_kwargs["num_evaluators_per_gpu"], 2)
        self.assertEqual(captured_kwargs["learner_seed"], 0)
        self.assertEqual(captured_kwargs["sampler_seed"], 0)
        self.assertEqual(captured_kwargs["evaluator_seed"], 0)
        expected_learner_config = {
            "num_learners": 1,
            "max_updates": 100,
            "prefetch_depth": 1,
            "weight_publish_interval": 1,
            "learner_device": "gpu",
        }
        self.assertEqual(runner_config["learner"], expected_learner_config)

    def test_training_entry_saves_run_config(self):
        with patch.dict("os.environ"):
            entry = importlib.import_module("run")
        example_dir = CONFIG_ROOT / "cartpole"
        args = Mock(
            mode="train",
            runner_config=example_dir / "train.yaml",
            agent_config=example_dir / "ppo.yaml",
        )
        runner_config = entry.load_yaml(args.runner_config)
        agent_config = entry.load_yaml(args.agent_config)
        expected_config = {**agent_config, **runner_config}

        with tempfile.TemporaryDirectory() as ckpt_dir:
            runner = Mock(spec=["ckpt_path", "fit"])
            runner.ckpt_path = str(Path(ckpt_dir) / "checkpoint.ckpt")
            with patch.object(entry, "parse_args", return_value=args):
                with patch.object(Runner, "from_config", return_value=runner):
                    entry.main()
            config_path = Path(ckpt_dir) / "run_config.yaml"
            saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_config, expected_config)
            runner.fit.assert_called_once_with()

    def test_evaluation_config_loading(self):
        config_path = CONFIG_ROOT / "cartpole" / "test.yaml"
        runner_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        runner_args = (object(), object(), runner_config, {"modules": {}})

        with patch.object(Runner, "__init__", return_value=None) as init:
            Runner.from_config(*runner_args)
        captured_kwargs = init.call_args.kwargs

        self.assertEqual(captured_kwargs["eval_model"], "agent_100.pth")
        self.assertNotIn("num_samplers", captured_kwargs)
        self.assertEqual(captured_kwargs["num_evaluators"], 1)
        self.assertEqual(captured_kwargs["num_envs_per_evaluator"], 1)
        self.assertEqual(captured_kwargs["num_episodes_per_env"], 10)
        self.assertEqual(captured_kwargs["num_evaluator_gpus"], 0)
        self.assertEqual(captured_kwargs["num_evaluators_per_gpu"], 1)
        self.assertEqual(captured_kwargs["evaluator_seed"], 0)
        self.assertNotIn("num_learners", captured_kwargs)
        self.assertNotIn("learner_device", captured_kwargs)

    def test_runner_device_allocation(self):
        gpu_bundles = [(1, 1)] * 2 + [(2, 1)] * 2 + [(1, 0), (2, 1)] + [(1, 0)] * 3
        cpu_bundles = [(1, 0)] * 2 + [(2, 1)] * 2 + [(1, 0), (2, 1)] + [(1, 0)] * 3
        cpu_sampler_bundles = [(1, 1)] * 2 + [(1, 0)] * 5 + [(2, 1)] + [(1, 0)] * 3
        shared_gpu_bundles = [(1, 1)] * 2 + [(3, 1)] + [(1, 0)] * 2
        shared_gpu_bundles += [(3, 1)] + [(1, 0)] * 2
        spare_gpu_bundles = [(1, 0)] * 2 + [(4, 1), (1, 1), (0, 1), (2, 1)]
        spare_gpu_bundles += [(1, 0)] * 3

        gpu_expected = (gpu_bundles, [2, 2, 3, 3, 4], [5, 5, 6])
        cpu_expected = (cpu_bundles, [2, 2, 3, 3, 4], [5, 5, 6])
        cpu_sampler_expected = (cpu_sampler_bundles, [2, 3, 4, 5, 6], [7, 7, 8])
        shared_gpu_expected = (shared_gpu_bundles, [2, 2, 2, 3, 4], [5, 5, 5])
        spare_gpu_expected = (spare_gpu_bundles, [2, 2, 2, 2, 3], [5, 5, 6])
        all_cpu_expected = ([(1, 0)] * 12, [2, 3, 4, 5, 6], [7, 8, 9])
        learner_gpu_bundles = [(1, 1)] * 2 + [(1, 0)] * 10
        learner_gpu_expected = (learner_gpu_bundles, [2, 3, 4, 5, 6], [7, 8, 9])
        cases = [
            (5, "gpu", (2, 2), (1, 2), gpu_expected),
            (8, "gpu", (2, 2), (1, 2), gpu_expected),
            (3, "cpu", (2, 2), (1, 2), cpu_expected),
            (3, "gpu", (0, 1), (1, 2), cpu_sampler_expected),
            (4, "gpu", (1, 3), (1, 3), shared_gpu_expected),
            (4, "cpu", (3, 4), (1, 2), spare_gpu_expected),
            (0, "cpu", (0, 1), (0, 1), all_cpu_expected),
            (2, "cpu", (0, 1), (0, 1), all_cpu_expected),
            (0, "cpu", (0, 0), (0, 0), all_cpu_expected),
            (0, None, (0, 1), (0, 1), all_cpu_expected),
            (2, None, (0, 1), (0, 1), all_cpu_expected),
            (3, "gpu", (0, 1), (0, 1), learner_gpu_expected),
        ]
        runner_patches = {
            "ray": DEFAULT,
            "Sampler": DEFAULT,
            "Evaluator": DEFAULT,
            "WeightStore": DEFAULT,
            "FileWriter": DEFAULT,
            "setup_logger": DEFAULT,
        }
        resource_patches = {"ray": DEFAULT, "placement_group": DEFAULT}
        for available_gpus, device, sampler, evaluator, expected in cases:
            with self.subTest(available_gpus=available_gpus, device=device):
                runner_patch = patch.multiple("omnirl.runtime.runner", **runner_patches)
                resource_target = "omnirl.runtime.resources"
                resource_patch = patch.multiple(resource_target, **resource_patches)
                with runner_patch as mocks, resource_patch as resource_mocks:
                    mocks["ray"].is_initialized.return_value = True
                    resources = {"CPU": 12, "GPU": available_gpus}
                    resource_mocks["ray"].available_resources.return_value = resources
                    group = resource_mocks["placement_group"].return_value
                    group.wait.return_value = True
                    mocks["ray"].get.return_value = 0
                    agent = Mock()
                    environment = Mock()
                    environment.EnvProvider.get_spec.return_value = {
                        "env_spec": {},
                        "state_spec": {},
                        "action_spec": {},
                    }
                    environment.EvaluationLoop.default_metrics = {}
                    runner_kwargs = {
                        "agent": agent,
                        "environment": environment,
                        "agent_config": {},
                        "env_config": {},
                        "num_learners": 2,
                        "num_samplers": 5,
                        "num_evaluators": 3,
                        "num_envs_per_evaluator": 4,
                        "evaluator_seed": 7,
                        "learner_device": device,
                        "num_sampler_gpus": sampler[0],
                        "num_samplers_per_gpu": sampler[1],
                        "num_evaluator_gpus": evaluator[0],
                        "num_evaluators_per_gpu": evaluator[1],
                    }
                    if device is None:
                        runner_kwargs.pop("learner_device")
                    runner = Runner(**runner_kwargs)
                    create_evaluator = mocks["Evaluator"].options.return_value.remote
                    for i, call in enumerate(create_evaluator.call_args_list):
                        self.assertEqual(call.kwargs["num_envs_per_evaluator"], 4)
                        self.assertEqual(call.kwargs["seed"], 2007 + i * 4)
                    bundle_specs, sampler_indices, evaluator_indices = expected
                    bundles = []
                    for cpus, gpus in bundle_specs:
                        bundle = {"CPU": cpus, "GPU": gpus} if gpus else {"CPU": cpus}
                        bundles.append(bundle)
                    resource_mocks["placement_group"].assert_called_once_with(bundles)
                    group.wait.assert_called_once_with(timeout_seconds=30)
                    self.assertIs(runner.resources.group, group)
                    learner_gpu = device == "gpu"
                    indices = (
                        [0, 1],
                        sampler_indices,
                        evaluator_indices,
                        [len(bundles) - 2],
                        [len(bundles) - 1],
                    )
                    actors = (
                        agent.Learner,
                        mocks["Sampler"],
                        mocks["Evaluator"],
                        agent.Memory,
                        mocks["WeightStore"],
                    )
                    fractions = {0: 0, 1: 1, 2: 0.5, 3: 1 / 3, 4: 0.25}
                    sampler_share = fractions[sampler[1]]
                    evaluator_share = fractions[evaluator[1]]
                    shares = (int(learner_gpu), sampler_share, evaluator_share, 0, 0)
                    for actor, bundle_indices, share in zip(actors, indices, shares):
                        options = actor.options.call_args_list
                        calls = actor.options.return_value.remote.call_args_list
                        self.assertEqual(len(options), len(bundle_indices))
                        self.assertEqual(len(calls), len(bundle_indices))
                        for option, args, index in zip(options, calls, bundle_indices):
                            gpus = share if bundle_specs[index][1] else 0
                            self.assertEqual(option.kwargs["num_cpus"], 1)
                            self.assertEqual(option.kwargs["num_gpus"], gpus)
                            if actor in actors[:3]:
                                expected_device = "gpu" if gpus else "cpu"
                                self.assertEqual(args.kwargs["device"], expected_device)
                            strategy = option.kwargs["scheduling_strategy"]
                            self.assertIs(strategy.placement_group, group)
                            bundle_index = strategy.placement_group_bundle_index
                            self.assertEqual(bundle_index, index)

    def test_runner_invalid_device_allocation(self):
        gpu_config = {
            "learner_device": "gpu",
            "num_sampler_gpus": 1,
            "num_evaluator_gpus": 1,
        }
        cases = [
            ({"CPU": 6, "GPU": 0}, {"learner_device": "gpu"}),
            ({"CPU": 6, "GPU": 1}, {"learner_device": "gpu"}),
            ({"CPU": 6, "GPU": 1}, {"num_sampler_gpus": 2}),
            ({"CPU": 6, "GPU": 1}, {"num_evaluator_gpus": 2}),
            ({"CPU": 6, "GPU": 2}, gpu_config),
            ({}, {}),
            ({"CPU": 5}, {}),
            ({"CPU": 6}, {}),
        ]
        runner_patches = {
            "ray": DEFAULT,
            "Sampler": DEFAULT,
            "Evaluator": DEFAULT,
            "WeightStore": DEFAULT,
            "FileWriter": DEFAULT,
        }
        resource_patches = {
            "ray": DEFAULT,
            "placement_group": DEFAULT,
            "remove_placement_group": DEFAULT,
        }
        for resources, kwargs in cases:
            with self.subTest(resources=resources, kwargs=kwargs):
                runner_patch = patch.multiple("omnirl.runtime.runner", **runner_patches)
                resource_target = "omnirl.runtime.resources"
                resource_patch = patch.multiple(resource_target, **resource_patches)
                with runner_patch as mocks, resource_patch as resource_mocks:
                    mocks["ray"].is_initialized.return_value = True
                    resource_mocks["ray"].available_resources.return_value = resources
                    group = resource_mocks["placement_group"].return_value
                    group.wait.return_value = False
                    agent = Mock()
                    runner_kwargs = {
                        "learner_device": "cpu",
                        "num_learners": 2,
                        **kwargs,
                    }
                    resource_name = "GPUs" if kwargs else "CPUs"
                    message = f"Not enough {resource_name} available"
                    message += " for the current configuration."
                    timeout = not kwargs and resources.get("CPU", 0) == 6
                    if timeout:
                        message = "Could not reserve resources within 30 seconds."
                    runner = Runner.__new__(Runner)
                    error_type = RuntimeError if timeout else ValueError
                    with self.assertRaisesRegex(error_type, message):
                        runner.__init__(agent, Mock(), {}, {}, **runner_kwargs)
                    agent.Memory.options.assert_not_called()
                    agent.Memory.remote.assert_not_called()
                    agent.Learner.options.assert_not_called()
                    mocks["Sampler"].options.assert_not_called()
                    mocks["Evaluator"].options.assert_not_called()
                    mocks["WeightStore"].options.assert_not_called()
                    mocks["WeightStore"].remote.assert_not_called()
                    mocks["FileWriter"].assert_not_called()
                    if timeout:
                        group.wait.assert_called_once_with(timeout_seconds=30)
                        remove_group = resource_mocks["remove_placement_group"]
                        remove_group.assert_called_once_with(group)
                        self.assertIsNone(runner.resources.group)
                    else:
                        resource_mocks["placement_group"].assert_not_called()
                        resource_mocks["remove_placement_group"].assert_not_called()

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
        environment_files = []
        for path in environment_paths:
            environment_files.append(path.read_text(encoding="utf-8"))
        environment_sources = "\n".join(environment_files)
        for forbidden in ["omnirl.algorithms", "omnirl.networks", "omnirl.runtime"]:
            self.assertNotIn(forbidden, environment_sources)


if __name__ == "__main__":
    unittest.main()
