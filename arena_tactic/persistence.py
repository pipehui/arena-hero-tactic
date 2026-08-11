from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from uuid import UUID

from arena_hero import CoreState, Position, UnitType

from .config import DEFAULT_CONFIG, TacticConfig
from .geometry import DIRECTION_ORDER, add_direction, cardinal_neighbors, diamond, vision_is_clear
from .projection import attack_cells
from .rules import CORE_VISION_RADIUS, UNIT_VISION_RADIUS
from .schema import EXPLORATION_MEMORY_SCHEMA_VERSION
from .state import TacticMemory
from .models import (
    CrisisForceBaseline,
    EnemyCoreIntel,
    EnemyTrack,
    ThreatHeatCell,
    WorkerScoutPhase,
    WorkerScoutState,
)
from .world import _decayed_threat_heat, _update_threat_heat


EXPLORATION_MEMORY_FILENAME = "balanced_tactic_memory.json"


def _position(value: object) -> Position | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value[:2])
    ):
        return value[0], value[1]
    return None


def _uuid(value: object) -> UUID | None:
    if not isinstance(value, str):
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _triples(value: object):
    if not isinstance(value, list):
        return
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        position = _position(item)
        amount = item[2]
        if position is not None and isinstance(amount, int) and not isinstance(amount, bool):
            yield position, amount


class ExplorationMemoryStore:
    """Atomic durable-world checkpoint with best-effort legacy migration.

    Service admission and SDK-facing action state are intentionally not
    durable.  Value-only Worker scout leases are durable so a watchdog restart
    does not turn every explorer into an unassigned waiter.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        config: TacticConfig = DEFAULT_CONFIG,
        save_interval_ticks: int = 16,
    ) -> None:
        if save_interval_ticks <= 0:
            raise ValueError("save_interval_ticks must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / EXPLORATION_MEMORY_FILENAME
        self.config = config
        self.save_interval_ticks = save_interval_ticks
        self._last_saved_tick: int | None = None
        self.restored_through_tick: int | None = None
        self.restored_visit_count = 0

    def load(self) -> TacticMemory:
        memory = TacticMemory(home_force_high_water=self.config.home_force_floor)
        saved_tick: int | None = None
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                saved_tick = self._restore(memory, payload)
            except (OSError, ValueError, TypeError, KeyError) as error:
                warnings.warn(
                    f"Ignoring invalid exploration checkpoint: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                memory = TacticMemory(home_force_high_water=self.config.home_force_floor)
                saved_tick = None
        replay_tick = self._catch_up(memory, after_tick=saved_tick)
        latest = replay_tick if replay_tick is not None else saved_tick
        if latest is not None:
            for cell, seen_tick in tuple(memory.resource_memory.items()):
                if latest - seen_tick > self.config.resource_memory_ttl:
                    memory.resource_memory.pop(cell, None)
            for cell, record in tuple(memory.threat_heat.items()):
                if record.score(latest) <= 0:
                    memory.threat_heat.pop(cell, None)
            for enemy_id, track in tuple(memory.enemy_tracks.items()):
                if latest - track.last_seen_tick > self.config.enemy_track_ttl:
                    memory.enemy_tracks.pop(enemy_id, None)
            for core_id, intel in tuple(memory.enemy_core_intel.items()):
                if latest - intel.last_seen_tick > self.config.raid_intel_ttl:
                    memory.enemy_core_intel.pop(core_id, None)
        self.restored_through_tick = latest
        self.restored_visit_count = sum(memory.visit_counts.values())
        self._last_saved_tick = saved_tick
        return memory

    def save(self, memory: TacticMemory, *, tick: int, force: bool = False) -> bool:
        if (
            not force
            and self._last_saved_tick is not None
            and tick - self._last_saved_tick < self.save_interval_ticks
        ):
            return True
        payload = {
            "schema_version": EXPLORATION_MEMORY_SCHEMA_VERSION,
            "saved_tick": tick,
            "core_id": None if memory.core_id is None else str(memory.core_id),
            "core_position": None if memory.core_position is None else list(memory.core_position),
            "core_position_history": [list(cell) for cell in memory.core_position_history],
            "known_obstacles": [list(cell) for cell in sorted(memory.known_obstacles)],
            "known_passable": [list(cell) for cell in sorted(memory.known_passable)],
            "cell_last_visible": self._counter_rows(memory.cell_last_visible),
            "worker_cell_last_visible": self._counter_rows(memory.worker_cell_last_visible),
            "visit_counts": self._counter_rows(memory.visit_counts),
            "congestion_counts": self._counter_rows(memory.congestion_counts),
            "resource_memory": self._counter_rows(memory.resource_memory),
            "resource_seen_count": self._counter_rows(memory.resource_seen_count),
            "resource_harvest_count": self._counter_rows(memory.resource_harvest_count),
            "threat_heat": [
                [
                    cell[0],
                    cell[1],
                    record.risk,
                    record.updated_tick,
                    record.expires_tick,
                    record.source,
                ]
                for cell, record in sorted(memory.threat_heat.items())
                if record.score(tick) > 0
            ],
            "enemy_tracks": [
                {
                    "enemy_id": str(enemy_id),
                    "unit_type": track.unit_type.value,
                    "samples": [
                        [sample_tick, position[0], position[1]]
                        for sample_tick, position in track.samples
                    ],
                    "last_seen_tick": track.last_seen_tick,
                }
                for enemy_id, track in sorted(
                    memory.enemy_tracks.items(),
                    key=lambda item: item[0].bytes,
                )
                if tick - track.last_seen_tick <= self.config.enemy_track_ttl
            ],
            "enemy_core_intel": [
                {
                    "enemy_id": str(core_id),
                    "position": list(intel.position),
                    "hp": intel.hp,
                    "shield": intel.shield,
                    "state": intel.state.value,
                    "destination": (
                        None if intel.destination is None else list(intel.destination)
                    ),
                    "last_seen_tick": intel.last_seen_tick,
                    "sighting_count": intel.sighting_count,
                }
                for core_id, intel in sorted(
                    memory.enemy_core_intel.items(),
                    key=lambda item: item[0].bytes,
                )
                if tick - intel.last_seen_tick <= self.config.raid_intel_ttl
            ],
            "worker_scout_states": [
                {
                    "worker_id": str(worker_id),
                    "slot": state.slot,
                    "sector_index": state.sector_index,
                    "stage": state.stage,
                    "phase": state.phase.value,
                    "target": None if state.target is None else list(state.target),
                    "assigned_tick": state.assigned_tick,
                    "best_route_cost": state.best_route_cost,
                    "stalled_ticks": state.stalled_ticks,
                    "backoff_until": state.backoff_until,
                    "last_scan_tick": state.last_scan_tick,
                    "reachable_candidates": state.reachable_candidates,
                }
                for worker_id, state in sorted(
                    memory.worker_scout_states.items(),
                    key=lambda item: item[0].bytes,
                )
            ],
            "last_congestion_decay_tick": memory.last_congestion_decay_tick,
            "opening_complete": memory.opening_complete,
            "home_force_high_water": memory.home_force_high_water,
            "crisis_force_baseline": (
                None
                if memory.crisis_force_baseline is None
                else {
                    "vanguards": memory.crisis_force_baseline.vanguards,
                    "rangers": memory.crisis_force_baseline.rangers,
                    "started_tick": memory.crisis_force_baseline.started_tick,
                    "phase": memory.crisis_force_baseline.phase,
                    "safe_ticks": memory.crisis_force_baseline.safe_ticks,
                }
            ),
            "strategic_relocation_pending": memory.strategic_relocation_pending,
            "strategic_relocation_safe_ticks": memory.strategic_relocation_safe_ticks,
            "strategic_relocation_goal": (
                None
                if memory.strategic_relocation_goal is None
                else list(memory.strategic_relocation_goal)
            ),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except OSError as error:
            warnings.warn(
                f"Could not save exploration checkpoint: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        self._last_saved_tick = tick
        return True

    @staticmethod
    def _counter_rows(values) -> list[list[int]]:
        return [[cell[0], cell[1], int(value)] for cell, value in sorted(values.items())]

    def _restore(self, memory: TacticMemory, payload: object) -> int | None:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint root must be an object")
        schema = payload.get("schema_version")
        if not isinstance(schema, int) or not 1 <= schema <= EXPLORATION_MEMORY_SCHEMA_VERSION:
            raise ValueError("unsupported exploration checkpoint schema")
        saved_tick = payload.get("saved_tick")
        if not isinstance(saved_tick, int) or isinstance(saved_tick, bool):
            saved_tick = None

        memory.core_id = _uuid(payload.get("core_id"))
        memory.core_position = _position(payload.get("core_position"))
        if schema >= 8:
            memory.core_position_history = tuple(
                cell
                for raw in payload.get("core_position_history", [])
                if (cell := _position(raw)) is not None
            )[-4:]
            if not memory.core_position_history and memory.core_position is not None:
                memory.core_position_history = (memory.core_position,)
        obstacle_key = "known_obstacles" if schema >= 6 else "obstacles"
        for value in payload.get(obstacle_key, []):
            if (cell := _position(value)) is not None:
                memory.known_obstacles.add(cell)
        for value in payload.get("known_passable", []):
            if (cell := _position(value)) is not None:
                memory.known_passable.add(cell)

        fields = (
            ("cell_last_visible", memory.cell_last_visible),
            ("worker_cell_last_visible", memory.worker_cell_last_visible),
            ("visit_counts", memory.visit_counts),
            ("congestion_counts", memory.congestion_counts),
            ("resource_memory", memory.resource_memory),
            ("resource_seen_count", memory.resource_seen_count),
            ("resource_harvest_count", memory.resource_harvest_count),
        )
        for key, target in fields:
            old_key = key
            if schema < 6 and key == "visit_counts":
                old_key = "patrol_visits"
            for cell, value in _triples(payload.get(old_key, [])):
                target[cell] = value

        if schema >= 9:
            for raw in payload.get("threat_heat", []):
                if not isinstance(raw, (list, tuple)) or len(raw) < 6:
                    continue
                cell = _position(raw)
                risk, updated_tick, expires_tick, source = raw[2:6]
                if (
                    cell is None
                    or not isinstance(risk, int)
                    or isinstance(risk, bool)
                    or risk <= 0
                    or not isinstance(updated_tick, int)
                    or isinstance(updated_tick, bool)
                    or not isinstance(expires_tick, int)
                    or isinstance(expires_tick, bool)
                    or expires_tick <= updated_tick
                    or not isinstance(source, str)
                ):
                    continue
                record = ThreatHeatCell(
                    position=cell,
                    risk=risk,
                    updated_tick=updated_tick,
                    expires_tick=expires_tick,
                    source=source,
                )
                if saved_tick is None or record.score(saved_tick) > 0:
                    memory.threat_heat[cell] = record

        if schema >= 11:
            for raw in payload.get("enemy_tracks", []):
                if not isinstance(raw, dict):
                    continue
                enemy_id = _uuid(raw.get("enemy_id"))
                try:
                    unit_type = UnitType(raw.get("unit_type"))
                except (TypeError, ValueError):
                    continue
                samples: list[tuple[int, Position]] = []
                for sample in raw.get("samples", []):
                    if (
                        not isinstance(sample, (list, tuple))
                        or len(sample) < 3
                        or not isinstance(sample[0], int)
                        or isinstance(sample[0], bool)
                    ):
                        continue
                    position = _position(sample[1:3])
                    if position is not None:
                        samples.append((sample[0], position))
                last_seen_tick = raw.get("last_seen_tick")
                if (
                    enemy_id is None
                    or not samples
                    or not isinstance(last_seen_tick, int)
                    or isinstance(last_seen_tick, bool)
                    or samples[-1][0] != last_seen_tick
                ):
                    continue
                memory.enemy_tracks[enemy_id] = EnemyTrack(
                    id=enemy_id,
                    unit_type=unit_type,
                    samples=tuple(samples[-4:]),
                    last_seen_tick=last_seen_tick,
                )
            for raw in payload.get("enemy_core_intel", []):
                if not isinstance(raw, dict):
                    continue
                core_id = _uuid(raw.get("enemy_id"))
                position = _position(raw.get("position"))
                destination = _position(raw.get("destination"))
                try:
                    state = CoreState(raw.get("state"))
                except (TypeError, ValueError):
                    continue
                hp = raw.get("hp")
                shield = raw.get("shield")
                last_seen_tick = raw.get("last_seen_tick")
                sighting_count = raw.get("sighting_count")
                values = (hp, shield, last_seen_tick, sighting_count)
                if (
                    core_id is None
                    or position is None
                    or any(
                        not isinstance(value, int) or isinstance(value, bool)
                        for value in values
                    )
                    or hp < 0
                    or shield < 0
                    or sighting_count <= 0
                ):
                    continue
                memory.enemy_core_intel[core_id] = EnemyCoreIntel(
                    id=core_id,
                    position=position,
                    hp=hp,
                    shield=shield,
                    state=state,
                    destination=destination,
                    last_seen_tick=last_seen_tick,
                    sighting_count=sighting_count,
                )

        if schema >= 10:
            for raw in payload.get("worker_scout_states", []):
                if not isinstance(raw, dict):
                    continue
                worker_id = _uuid(raw.get("worker_id"))
                target = _position(raw.get("target"))
                try:
                    phase = WorkerScoutPhase(raw.get("phase"))
                except (TypeError, ValueError):
                    continue
                integer_fields = {
                    key: raw.get(key)
                    for key in (
                        "slot",
                        "sector_index",
                        "stage",
                        "assigned_tick",
                        "stalled_ticks",
                        "backoff_until",
                        "reachable_candidates",
                    )
                }
                if (
                    worker_id is None
                    or any(
                        not isinstance(value, int) or isinstance(value, bool)
                        for value in integer_fields.values()
                    )
                    or integer_fields["slot"] < 0
                    or not 0 <= integer_fields["sector_index"] < 8
                    or integer_fields["stage"] < 0
                    or integer_fields["stalled_ticks"] < 0
                    or integer_fields["reachable_candidates"] < 0
                ):
                    continue
                best_route_cost = raw.get("best_route_cost")
                if not isinstance(best_route_cost, int) or isinstance(best_route_cost, bool):
                    best_route_cost = None
                last_scan_tick = raw.get("last_scan_tick")
                if not isinstance(last_scan_tick, int) or isinstance(last_scan_tick, bool):
                    last_scan_tick = None
                memory.worker_scout_states[worker_id] = WorkerScoutState(
                    worker_id=worker_id,
                    slot=integer_fields["slot"],
                    sector_index=integer_fields["sector_index"],
                    stage=integer_fields["stage"],
                    phase=phase,
                    target=target,
                    assigned_tick=integer_fields["assigned_tick"],
                    best_route_cost=best_route_cost,
                    stalled_ticks=integer_fields["stalled_ticks"],
                    backoff_until=integer_fields["backoff_until"],
                    last_scan_tick=last_scan_tick,
                    reachable_candidates=integer_fields["reachable_candidates"],
                )

        decay_tick = payload.get("last_congestion_decay_tick")
        if schema >= 8 and isinstance(decay_tick, int) and not isinstance(decay_tick, bool):
            memory.last_congestion_decay_tick = decay_tick

        high_water = payload.get("home_force_high_water")
        if not isinstance(high_water, int):
            # Accept the short-lived nested form emitted by one schema-5 build.
            defense = payload.get("defense_strength")
            if isinstance(defense, dict):
                high_water = defense.get(
                    "home_force_high_water",
                    defense.get("hostile_force_high_water", defense.get("high_water")),
                )
        if isinstance(high_water, int) and high_water > 0:
            memory.home_force_high_water = max(self.config.home_force_floor, high_water)
        if schema >= 12:
            raw_baseline = payload.get("crisis_force_baseline")
            if isinstance(raw_baseline, dict):
                vanguards = raw_baseline.get("vanguards")
                rangers = raw_baseline.get("rangers")
                started_tick = raw_baseline.get("started_tick")
                phase = raw_baseline.get("phase")
                safe_ticks = raw_baseline.get("safe_ticks", 0)
                if (
                    isinstance(vanguards, int)
                    and not isinstance(vanguards, bool)
                    and vanguards >= 0
                    and isinstance(rangers, int)
                    and not isinstance(rangers, bool)
                    and rangers >= 0
                    and (started_tick is None or isinstance(started_tick, int))
                    and phase in {"SAFE", "ACTIVE", "REBUILD"}
                    and isinstance(safe_ticks, int)
                    and not isinstance(safe_ticks, bool)
                    and safe_ticks >= 0
                ):
                    memory.crisis_force_baseline = CrisisForceBaseline(
                        vanguards=vanguards,
                        rangers=rangers,
                        started_tick=started_tick,
                        phase=phase,
                        safe_ticks=safe_ticks,
                    )
        memory.opening_complete = bool(payload.get("opening_complete", False))
        memory.strategic_relocation_pending = bool(
            payload.get("strategic_relocation_pending", False)
        )
        relocation_safe_ticks = payload.get("strategic_relocation_safe_ticks", 0)
        if isinstance(relocation_safe_ticks, int) and relocation_safe_ticks >= 0:
            memory.strategic_relocation_safe_ticks = relocation_safe_ticks
        memory.strategic_relocation_goal = _position(
            payload.get("strategic_relocation_goal")
        )
        # Legacy schemas persisted controller-relative missions and Core
        # service choreography.  They cannot be trusted after reconnecting to
        # a newer complete Turn, so current schemas deliberately migrate only
        # the durable map/economy fields above.
        return saved_tick

    def _catch_up(self, memory: TacticMemory, *, after_tick: int | None) -> int | None:
        # Logs can overlap after a reconnect or a watchdog restart.  Replaying
        # them in filename order used to count a duplicated Turn twice and,
        # worse, could leave ``memory.last_tick`` pointing at an older Turn
        # when files were copied or renamed.  Keep the last complete state for
        # each Tick and apply the authoritative states in game order.
        states_by_tick: dict[int, dict] = {}
        for path in sorted(self.directory.glob("arena_hero_*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            record = json.loads(line)
                        except (ValueError, TypeError):
                            continue
                        if record.get("record_type") != "turn":
                            continue
                        tick = record.get("tick")
                        if not isinstance(tick, int) or (after_tick is not None and tick <= after_tick):
                            continue
                        state = record.get("state")
                        if not isinstance(state, dict):
                            continue
                        states_by_tick[tick] = state
            except OSError:
                continue
        for tick in sorted(states_by_tick):
            self._learn_logged_state(memory, tick, states_by_tick[tick])
        return max(states_by_tick, default=None)

    def _learn_logged_state(self, memory: TacticMemory, tick: int, state: dict) -> None:
        visible_resources: set[Position] = set()
        sources: list[tuple[Position, int, bool]] = []
        visible_hostiles: list[tuple[UUID, Position, UnitType]] = []
        for obj in state.get("objects", []):
            if not isinstance(obj, dict):
                continue
            kind = obj.get("kind")
            if kind == "OBSTACLE":
                for raw in obj.get("positions", []):
                    if (cell := _position(raw)) is not None:
                        memory.known_obstacles.add(cell)
            elif kind == "RESOURCE":
                for raw in obj.get("positions", []):
                    if (cell := _position(raw)) is not None:
                        visible_resources.add(cell)
                        memory.resource_memory[cell] = tick
            elif kind == "CORE" and obj.get("controlled") is True:
                core_id = _uuid(obj.get("id"))
                core_position = _position(obj.get("position"))
                if core_id is not None and core_position is not None:
                    if memory.core_id != core_id:
                        memory.reset_for_core(core_id, core_position)
                        memory.home_force_high_water = self.config.home_force_floor
                    else:
                        memory.core_id = core_id
                elif core_id is not None:
                    memory.core_id = core_id
                if core_position is not None:
                    if memory.core_position != core_position:
                        memory.core_position_history = (
                            *memory.core_position_history,
                            core_position,
                        )[-4:]
                    memory.core_position = core_position
                    memory.visit_counts[core_position] += 1
                    sources.append((core_position, CORE_VISION_RADIUS, False))
            elif kind == "CORE" and obj.get("controlled") is False:
                core_id = _uuid(obj.get("id"))
                position = _position(obj.get("position"))
                destination = _position(obj.get("destination"))
                try:
                    core_state = CoreState(obj.get("state"))
                except (TypeError, ValueError):
                    continue
                hp = obj.get("hp")
                shield = obj.get("shield")
                if (
                    core_id is None
                    or position is None
                    or not isinstance(hp, int)
                    or isinstance(hp, bool)
                    or not isinstance(shield, int)
                    or isinstance(shield, bool)
                ):
                    continue
                previous = memory.enemy_core_intel.get(core_id)
                sightings = 1
                if (
                    previous is not None
                    and tick - previous.last_seen_tick <= self.config.raid_intel_ttl
                ):
                    sightings = previous.sighting_count + int(
                        tick > previous.last_seen_tick
                    )
                memory.enemy_core_intel[core_id] = EnemyCoreIntel(
                    id=core_id,
                    position=position,
                    hp=hp,
                    shield=shield,
                    state=core_state,
                    destination=destination,
                    last_seen_tick=tick,
                    sighting_count=sightings,
                )
            elif kind == "UNIT" and obj.get("controlled") is True:
                if (cell := _position(obj.get("position"))) is not None:
                    memory.visit_counts[cell] += 1
                    memory.known_passable.add(cell)
                    try:
                        unit_type = UnitType(obj.get("unit_type"))
                    except (TypeError, ValueError):
                        continue
                    sources.append(
                        (
                            cell,
                            UNIT_VISION_RADIUS[unit_type],
                            unit_type is UnitType.WORKER,
                        )
                    )
            elif kind == "UNIT" and obj.get("controlled") is False:
                enemy_id = _uuid(obj.get("id"))
                position = _position(obj.get("position"))
                try:
                    unit_type = UnitType(obj.get("unit_type"))
                except (TypeError, ValueError):
                    continue
                if enemy_id is not None and position is not None:
                    previous = memory.enemy_tracks.get(enemy_id)
                    sample = tick, position
                    if previous is None or tick > previous.last_seen_tick + 1:
                        samples = (sample,)
                    elif tick == previous.last_seen_tick:
                        samples = (*previous.samples[:-1], sample)
                    else:
                        samples = (*previous.samples, sample)[-4:]
                    memory.enemy_tracks[enemy_id] = EnemyTrack(
                        id=enemy_id,
                        unit_type=unit_type,
                        samples=samples,
                        last_seen_tick=tick,
                    )
                    if unit_type in {UnitType.VANGUARD, UnitType.RANGER}:
                        visible_hostiles.append((enemy_id, position, unit_type))
        obstacles = frozenset(memory.known_obstacles)
        visible_cells: set[Position] = set()
        for origin, radius, from_worker in sources:
            for cell in diamond(origin, radius):
                if not vision_is_clear(origin, cell, obstacles):
                    continue
                visible_cells.add(cell)
                memory.cell_last_visible[cell] = tick
                if from_worker:
                    memory.worker_cell_last_visible[cell] = tick
                if cell not in obstacles:
                    memory.known_passable.add(cell)
        for cell in tuple(memory.resource_memory):
            if cell in visible_cells and cell not in visible_resources:
                memory.resource_memory.pop(cell, None)
        self._learn_logged_threat_heat(
            memory,
            tick,
            visible_hostiles,
            state.get("events", []),
        )
        for enemy_id, track in tuple(memory.enemy_tracks.items()):
            if tick - track.last_seen_tick > self.config.enemy_track_ttl:
                memory.enemy_tracks.pop(enemy_id, None)
        for core_id, intel in tuple(memory.enemy_core_intel.items()):
            if tick - intel.last_seen_tick > self.config.raid_intel_ttl:
                memory.enemy_core_intel.pop(core_id, None)
        memory.last_tick = tick

    def _learn_logged_threat_heat(
        self,
        memory: TacticMemory,
        tick: int,
        visible_hostiles: list[tuple[UUID, Position, UnitType]],
        events: object,
    ) -> None:
        obstacles = frozenset(memory.known_obstacles)
        for _, position, unit_type in visible_hostiles:
            current_zone = {position, *attack_cells(position, unit_type, obstacles)}
            for cell in current_zone:
                _update_threat_heat(
                    memory,
                    position=cell,
                    risk=self.config.threat_heat_visible_risk,
                    tick=tick,
                    ttl=self.config.threat_heat_visible_ttl,
                    source="VISIBLE_ATTACK_ZONE",
                )
            possible = {position}
            possible.update(
                add_direction(position, direction)
                for direction in DIRECTION_ORDER
                if add_direction(position, direction) not in obstacles
            )
            projected: set[Position] = set()
            for candidate in possible:
                projected.add(candidate)
                projected.update(attack_cells(candidate, unit_type, obstacles))
            for cell in projected - current_zone:
                _update_threat_heat(
                    memory,
                    position=cell,
                    risk=self.config.threat_heat_projected_risk,
                    tick=tick,
                    ttl=self.config.threat_heat_projected_ttl,
                    source="PROJECTED_ATTACK_ZONE",
                )
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                position = _position(event.get("position"))
                event_type = event.get("event_type")
                if position is None or event_type not in {
                    "UNIT_DAMAGED",
                    "UNIT_DESTROYED",
                }:
                    continue
                destroyed = event_type == "UNIT_DESTROYED"
                center_risk = (
                    self.config.threat_heat_destroyed_risk
                    if destroyed
                    else self.config.threat_heat_damage_risk
                )
                neighbor_risk = (
                    self.config.threat_heat_destroyed_neighbor_risk
                    if destroyed
                    else self.config.threat_heat_damage_neighbor_risk
                )
                ttl = (
                    self.config.threat_heat_destroyed_ttl
                    if destroyed
                    else self.config.threat_heat_damage_ttl
                )
                _update_threat_heat(
                    memory,
                    position=position,
                    risk=center_risk,
                    tick=tick,
                    ttl=ttl,
                    source=str(event_type),
                )
                for _, neighbor in cardinal_neighbors(position):
                    _update_threat_heat(
                        memory,
                        position=neighbor,
                        risk=neighbor_risk,
                        tick=tick,
                        ttl=ttl,
                        source=f"{event_type}_NEARBY",
                    )
        _decayed_threat_heat(
            memory,
            tick=tick,
            cell_limit=self.config.threat_heat_cell_limit,
        )
