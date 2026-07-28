from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


TRIP_METRICS = ("delay", "trip_time", "waiting_time")


def _finite_attribute(element, name):
    """Read one required finite floating-point tripinfo attribute."""
    if name not in element.attrib:
        raise ValueError(f"SUMO tripinfo is missing attribute {name!r}.")
    value = float(element.attrib[name])
    if not np.isfinite(value):
        raise ValueError(f"SUMO tripinfo attribute {name!r} is not finite.")
    return value


def parse_tripinfo(path):
    """Parse the vehicle-level statistics used by the RESCO benchmark."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"SUMO did not create tripinfo output at {path}.")

    metric_sums = {name: 0.0 for name in TRIP_METRICS}
    num_trips = 0

    try:
        elements = ET.iterparse(path, events=("end",))
        for _, element in elements:
            if element.tag != "tripinfo":
                continue
            if element.attrib.get("id", "").startswith("ghost"):
                element.clear()
                continue

            time_loss = _finite_attribute(element, "timeLoss")
            depart_delay = _finite_attribute(element, "departDelay")
            metric_sums["delay"] += time_loss + depart_delay
            metric_sums["trip_time"] += _finite_attribute(element, "duration")
            metric_sums["waiting_time"] += _finite_attribute(
                element, "waitingTime"
            )
            num_trips += 1
            element.clear()
    except (ET.ParseError, OSError, ValueError) as error:
        raise ValueError(f"Cannot parse SUMO tripinfo output {path}: {error}") from error

    return {"trip_metric_sums": metric_sums, "num_trips": num_trips}
