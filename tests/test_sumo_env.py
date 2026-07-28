from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import numpy as np
from gymnasium import spaces

from omnirl.envs.sumo import (
    CollectionLoop,
    EnvAdapter,
    EvaluationLoop,
    aggregate_evaluations,
    build_env,
    get_env_spec,
)
from omnirl.envs.sumo.tripinfo import parse_tripinfo


class FakeSumoLoopEnv:
    """Deterministic two-step SUMO-like episodes for backend protocol tests."""

    render_mode = None
    traffic_signal_ids = ("traffic_light",)

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        observation_space_kwargs = {
            "low": 0.0,
            "high": 1.0,
            "shape": (3,),
            "dtype": np.float32,
        }
        self.observation_space = spaces.Box(**observation_space_kwargs)
        self.reset_seeds = []
        self.episode = -1
        self.steps = 0

    def reset(self, seed=None, options=None):
        self.reset_seeds.append(seed)
        self.episode += 1
        self.steps = 0
        return np.asarray([self.episode, 0, 1], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        info = {
            "traffic_light_stopped": float(self.steps + 2),
        }
        state = [self.episode, self.steps, action]
        observation = np.asarray(state, dtype=np.float32)
        return observation, 1.0, False, self.steps == 2, info

    def render(self):
        return None

    def close(self):
        return None

    def finish_episode_metrics(self):
        if self.episode == 0:
            sums = {"delay": 10.0, "trip_time": 20.0, "waiting_time": 4.0}
            num_trips = 1
        else:
            sums = {"delay": 30.0, "trip_time": 50.0, "waiting_time": 8.0}
            num_trips = 2
        return {"trip_metric_sums": sums, "num_trips": num_trips}


class SumoEnvironmentTest(unittest.TestCase):
    def test_benchmark_scenario_has_four_phases_and_nonstationary_turns(self):
        repository_root = Path(__file__).resolve().parents[1]
        experiment_root = repository_root / "experiments" / "environment_integration"
        scenario = experiment_root / "sumo" / "single_intersection" / "scenario"
        network = ET.parse(scenario / "single-intersection.net.xml").getroot()
        routes = ET.parse(scenario / "single-intersection.rou.xml").getroot()

        traffic_lights = network.findall("tlLogic")
        green_phases = [
            phase
            for phase in traffic_lights[0].findall("phase")
            if "y" not in phase.attrib["state"] and "G" in phase.attrib["state"]
        ]
        controlled_lanes = {
            connection.attrib["from"] + "_" + connection.attrib["fromLane"]
            for connection in network.findall("connection")
            if connection.attrib.get("tl") == "t"
        }
        traffic_periods = {
            (flow.attrib["begin"], flow.attrib["end"])
            for flow in routes.findall("flow")
        }
        expected_periods = {
            ("0", "1200"),
            ("1200", "2400"),
            ("2400", "3600"),
        }
        vehicle_types = routes.find("vTypeDistribution").findall("vType")

        self.assertEqual(len(traffic_lights), 1)
        self.assertEqual(len(green_phases), 4)
        self.assertEqual(len(controlled_lanes), 12)
        self.assertEqual(len(routes.findall("route")), 12)
        self.assertEqual(len(routes.findall("flow")), 36)
        self.assertEqual(traffic_periods, expected_periods)
        self.assertEqual(len(vehicle_types), 3)

    def test_builder_forces_single_agent_without_collection_metrics(self):
        raw_env = Mock()
        raw_env.ts_ids = ["traffic_light"]
        environment_cls = Mock(return_value=raw_env)
        config = {
            "net_file": "network.net.xml",
            "route_file": "traffic.rou.xml",
            "num_seconds": 3600,
            "render_mode": "human",
        }
        loader_target = "omnirl.envs.sumo.builder._load_sumo_environment"

        with patch(loader_target, return_value=environment_cls):
            env = build_env(config, role="collection")

        self.assertIsInstance(env, EnvAdapter)
        kwargs = environment_cls.call_args.kwargs
        self.assertTrue(kwargs["single_agent"])
        self.assertFalse(kwargs["use_gui"])
        self.assertIsNone(kwargs["render_mode"])
        self.assertFalse(kwargs["add_system_info"])
        self.assertFalse(kwargs["add_per_agent_info"])
        self.assertIsNone(kwargs["out_csv_name"])
        self.assertEqual(kwargs["max_depart_delay"], -1)
        self.assertEqual(kwargs["time_to_teleport"], -1)
        self.assertEqual(kwargs["additional_sumo_cmd"], None)
        self.assertEqual(kwargs["num_seconds"], 3600)

    def test_evaluation_role_preserves_human_rendering(self):
        raw_env = Mock()
        raw_env.ts_ids = ["traffic_light"]
        environment_cls = Mock(return_value=raw_env)
        config = {
            "net_file": "network.net.xml",
            "route_file": "traffic.rou.xml",
            "render_mode": "human",
            "additional_sumo_cmd": "--no-step-log",
        }
        loader_target = "omnirl.envs.sumo.builder._load_sumo_environment"

        with patch(loader_target, return_value=environment_cls):
            env = build_env(config, role="evaluation")

        kwargs = environment_cls.call_args.kwargs
        write_unfinished = "--tripinfo-output.write-unfinished true"
        write_undeparted = "--tripinfo-output.write-undeparted true"

        self.assertEqual(env.render_mode, "human")
        self.assertTrue(kwargs["use_gui"])
        self.assertEqual(kwargs["render_mode"], "human")
        self.assertTrue(kwargs["add_per_agent_info"])
        self.assertIn("--no-step-log", kwargs["additional_sumo_cmd"])
        self.assertIn("--tripinfo-output ", kwargs["additional_sumo_cmd"])
        self.assertIn(write_unfinished, kwargs["additional_sumo_cmd"])
        self.assertIn(write_undeparted, kwargs["additional_sumo_cmd"])
        self.assertIn("--eager-insert", kwargs["additional_sumo_cmd"])
        env.close()

    def test_builder_rejects_multi_signal_networks(self):
        raw_env = Mock()
        raw_env.ts_ids = ["north", "south"]
        environment_cls = Mock(return_value=raw_env)
        config = {
            "net_file": "network.net.xml",
            "route_file": "traffic.rou.xml",
        }
        loader_target = "omnirl.envs.sumo.builder._load_sumo_environment"

        with patch(loader_target, return_value=environment_cls):
            with self.assertRaisesRegex(ValueError, "exactly one traffic signal"):
                build_env(config)

        raw_env.close.assert_called_once_with()

    def test_builder_rejects_missing_and_backend_owned_fields(self):
        missing_route_config = {"net_file": "network.net.xml"}
        backend_owned_config = {
            "net_file": "network.net.xml",
            "route_file": "traffic.rou.xml",
            "single_agent": False,
        }

        with self.assertRaisesRegex(ValueError, "route_file"):
            build_env(missing_route_config)
        with self.assertRaisesRegex(ValueError, "single_agent"):
            build_env(backend_owned_config)

    def test_spec_is_discrete_and_has_no_return_metric(self):
        env = Mock()
        observation_space_kwargs = {
            "low": 0.0,
            "high": 1.0,
            "shape": (7,),
            "dtype": np.float32,
        }
        env.observation_space = spaces.Box(**observation_space_kwargs)
        env.action_space = spaces.Discrete(4)
        config = {
            "net_file": "network.net.xml",
            "route_file": "traffic.rou.xml",
        }
        expected_metrics = {
            "delay_mean": 0.0,
            "trip_time_mean": 0.0,
            "waiting_time_mean": 0.0,
            "queue_length_mean": 0.0,
        }

        with patch("omnirl.envs.sumo.spec.build_env", return_value=env):
            spec = get_env_spec(config)

        self.assertEqual(spec.state_dim, 7)
        self.assertEqual(spec.state_dtype, np.dtype(np.float32))
        self.assertEqual(spec.action_dim, 4)
        self.assertFalse(spec.is_continuous)
        self.assertIsNone(spec.action_low)
        self.assertIsNone(spec.action_high)
        self.assertEqual(spec.default_eval_metrics, expected_metrics)
        self.assertNotIn("return_mean", spec.default_eval_metrics)
        env.close.assert_called_once_with()

    def test_adapter_rejects_out_of_range_actions(self):
        raw_env = Mock()
        raw_env.ts_ids = ["traffic_light"]
        raw_env.action_space = spaces.Discrete(2)
        env = EnvAdapter(raw_env)

        with self.assertRaisesRegex(ValueError, "outside"):
            env.step(2)
        raw_env.step.assert_not_called()

    def test_collection_preserves_time_limit_semantics_and_advances_seed(self):
        env = FakeSumoLoopEnv()
        loop = CollectionLoop(lambda: env, seed=17)
        action = {"env_action": 1, "data": {"action": 1}}

        rollout = loop.collect(lambda observation: action, num_steps=3)
        expected_terminals = [False, False, False]
        expected_dones = [False, True, False]

        self.assertEqual(env.reset_seeds, [17, 18])
        np.testing.assert_array_equal(rollout["rewards"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(rollout["terminals"], expected_terminals)
        np.testing.assert_array_equal(rollout["dones"], expected_dones)
        np.testing.assert_array_equal(rollout["policy_data"]["action"], [1, 1, 1])
        self.assertEqual(rollout["states"].shape, (3, 3))
        self.assertEqual(rollout["next_states"].shape, (3, 3))

    def test_evaluation_reports_only_traffic_metrics(self):
        env = FakeSumoLoopEnv()
        loop = EvaluationLoop(lambda: env, seed=23)

        result = loop.run(lambda observation: 0, num_episodes=2)
        expected_trip_time = 70.0 / 3.0

        self.assertEqual(env.reset_seeds, [23, 24])
        self.assertEqual(result["num_episodes"], 2)
        self.assertEqual(result["num_trips"], 3)
        self.assertEqual(result["num_queue_samples"], 4)
        self.assertAlmostEqual(result["metrics"]["delay_mean"], 40.0 / 3.0)
        actual_trip_time = result["metrics"]["trip_time_mean"]
        self.assertAlmostEqual(actual_trip_time, expected_trip_time)
        self.assertAlmostEqual(result["metrics"]["waiting_time_mean"], 4.0)
        self.assertAlmostEqual(result["metrics"]["queue_length_mean"], 3.5)
        self.assertNotIn("return_mean", result["metrics"])

    def test_distributed_aggregation_uses_exact_sufficient_statistics(self):
        eval_infos = [
            {
                "num_episodes": 2,
                "num_trips": 2,
                "num_queue_samples": 2,
                "queue_length_sum": 10.0,
                "trip_metric_sums": {
                    "delay": 30.0,
                    "trip_time": 50.0,
                    "waiting_time": 10.0,
                },
                "metrics": {},
            },
            {
                "num_episodes": 1,
                "num_trips": 1,
                "num_queue_samples": 1,
                "queue_length_sum": 2.0,
                "trip_metric_sums": {
                    "delay": 30.0,
                    "trip_time": 40.0,
                    "waiting_time": 8.0,
                },
                "metrics": {},
            },
        ]

        result = aggregate_evaluations(eval_infos, version=9)

        self.assertEqual(result["version"], 9)
        self.assertEqual(result["num_episodes"], 3)
        self.assertAlmostEqual(result["delay_mean"], 20.0)
        self.assertAlmostEqual(result["trip_time_mean"], 30.0)
        self.assertAlmostEqual(result["waiting_time_mean"], 6.0)
        self.assertAlmostEqual(result["queue_length_mean"], 4.0)
        self.assertNotIn("return_mean", result)

    def test_evaluation_rejects_missing_sumo_metrics(self):
        env = FakeSumoLoopEnv()
        step_result = (
            np.zeros(3, dtype=np.float32),
            0.0,
            False,
            True,
            {},
        )
        env.step = Mock(return_value=step_result)
        loop = EvaluationLoop(lambda: env, seed=0)

        with self.assertRaisesRegex(ValueError, "missing queue metrics"):
            loop.run(lambda observation: 0, num_episodes=1)

    def test_tripinfo_parser_matches_resco_vehicle_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            tripinfo_path = Path(directory) / "tripinfo.xml"
            tripinfo_xml = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
    <tripinfo id="ghost_blocker" duration="100" waitingTime="100"
              timeLoss="100" departDelay="100" arrival="-1"/>
    <tripinfo id="a" duration="20" waitingTime="4"
              timeLoss="6" departDelay="1" arrival="20"/>
    <tripinfo id="b" duration="40" waitingTime="8"
              timeLoss="10" departDelay="3" arrival="-1"/>
</tripinfos>
"""
            tripinfo_path.write_text(tripinfo_xml, encoding="utf-8")

            result = parse_tripinfo(tripinfo_path)

        expected_sums = {
            "delay": 20.0,
            "trip_time": 60.0,
            "waiting_time": 12.0,
        }
        self.assertEqual(result["num_trips"], 2)
        self.assertEqual(result["trip_metric_sums"], expected_sums)


if __name__ == "__main__":
    unittest.main()
