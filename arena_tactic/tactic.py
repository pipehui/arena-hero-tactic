from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from arena_hero import CommandSource, Direction, Received, Turn, UnitType

from .config import DEFAULT_CONFIG, TacticConfig
from .kernel import DecisionKernel
from .geometry import add_direction
from .models import IntentAction, IntentResolution, ManualMoveLease, MoveAttempt
from .schema import STRATEGY_LOG_SCHEMA_VERSION
from .state import TacticMemory
from .projection import TacticalMap


class BalancedTactic:
    """Compatibility facade over the modular survival/economy kernel."""

    def __init__(
        self,
        config: TacticConfig = DEFAULT_CONFIG,
        *,
        memory: TacticMemory | None = None,
    ) -> None:
        self.config = config
        self.memory = memory or TacticMemory()
        self._kernel = DecisionKernel(config, self.memory)
        self._last_world = None
        self._last_decision_trace: dict[str, object] = {
            "schema_version": STRATEGY_LOG_SCHEMA_VERSION,
            "mode": "INITIALIZING",
            "tasks": [],
            "resolution": {"selected_count": 0, "rejected": []},
            "world": {},
            "economy": {},
            "combat": {},
            "core_safety": None,
        }

    @property
    def last_decision_trace(self) -> dict[str, object]:
        """Return detached JSON-compatible values for replay logging."""

        return deepcopy(self._last_decision_trace)

    @property
    def last_tactical_map(self) -> TacticalMap | None:
        """Return the immutable team map used for the latest decision."""

        return self._kernel.last_tactical_map

    def choose_actions(self, turn: Turn) -> None:
        # The runtime logs ``last_decision_trace`` even when planning raises.
        # Reset it before touching the Turn so an exceptional Tick can never
        # be mislabeled with the previous Tick's successful decisions.
        self._last_decision_trace = {
            "schema_version": STRATEGY_LOG_SCHEMA_VERSION,
            "mode": "DECIDING",
            "tick": turn.tick,
            "tasks": [],
            "resolution": {"selected_count": 0, "rejected": []},
            "world": {},
            "economy": {},
            "combat": {},
            "core_safety": None,
        }
        turn.clear()
        world, resolution, trace = self._kernel.decide(turn)
        self._last_world = world
        self._emit(turn, resolution)
        self._last_decision_trace = trace

    def observe_receipt(self, receipt: Received) -> None:
        """Protect successful Manual movement from immediate task reversal."""

        if receipt.source is not CommandSource.MANUAL:
            return
        for unit_id, action in receipt.plan.unit_actions.items():
            # A Manual attack replaces the Agent action for that actor.  Do
            # not attribute its result to an earlier Agent prediction.
            self.memory.last_ranger_shots.pop(unit_id, None)
            self.memory.last_vanguard_sweeps.pop(unit_id, None)
            if getattr(action, "type", None) != "MOVE":
                # Pydantic action types expose an enum-like string through
                # model_dump even when a future SDK changes the attribute.
                data = action.model_dump(mode="python")
                if data.get("type") != "MOVE":
                    continue
                direction = data.get("direction")
            else:
                direction = getattr(action, "direction", None)
            if direction is None:
                continue
            try:
                direction = Direction(direction)
            except (TypeError, ValueError):
                continue
            snapshot = (
                None
                if self._last_world is None
                else self._last_world.friendly(unit_id)
            )
            if snapshot is not None and snapshot.unit_type in {
                UnitType.VANGUARD,
                UnitType.RANGER,
            }:
                # Manual combat commands never become online policy.  Clear
                # stale attribution above, but let the next Turn's global
                # assignment respond to the current battlefield immediately.
                self.memory.manual_move_leases.pop(unit_id, None)
                continue
            self.memory.unit_missions.pop(unit_id, None)
            scout = self.memory.worker_scout_states.get(unit_id)
            if scout is not None:
                self.memory.worker_scout_states[unit_id] = replace(
                    scout,
                    target=None,
                    assigned_tick=receipt.tick,
                    best_route_cost=None,
                    stalled_ticks=0,
                )
            if self.memory.beacon_mission_actor_id == unit_id:
                self.memory.beacon_mission_actor_id = None
                self.memory.beacon_mission_target = None
            self.memory.manual_move_leases[unit_id] = ManualMoveLease(
                direction=direction,
                expires_tick=receipt.tick + self.config.manual_move_protection_ticks,
            )
            origin = self.memory.last_positions.get(unit_id)
            if origin is not None:
                self.memory.last_move_attempts[unit_id] = MoveAttempt(
                    actor_id=unit_id,
                    tick=receipt.tick,
                    origin=origin,
                    destination=add_direction(origin, direction),
                    direction=direction,
                )

    @staticmethod
    def _emit(turn: Turn, resolution: IntentResolution) -> None:
        units = {unit.id: unit for unit in turn.units}
        for intent in resolution.selected:
            if intent.actor_id is None:
                if turn.core is not None:
                    BalancedTactic._emit_core(turn.core, intent)
                continue
            controller = units.get(intent.actor_id)
            if controller is None:
                continue
            action = intent.action
            if action is IntentAction.WAIT:
                controller.wait()
            elif action is IntentAction.MOVE:
                assert intent.direction is not None
                controller.move(intent.direction)
            elif action is IntentAction.HARVEST:
                controller.harvest()
            elif action is IntentAction.DEPOSIT:
                controller.deposit()
            elif action is IntentAction.SWEEP:
                assert intent.direction is not None
                controller.sweep(intent.direction)
            elif action is IntentAction.SHOOT_CELL:
                assert intent.expected_cell is not None
                controller.shoot_cell(intent.expected_cell)
            elif action is IntentAction.SHOOT:
                assert intent.target_id is not None and intent.expected_cell is not None
                controller.shoot(intent.target_id, expected_cell=intent.expected_cell)
            elif action is IntentAction.HEAL:
                controller.heal()
            elif action is IntentAction.PICKUP_BEACON:
                controller.pickup_beacon()
            elif action is IntentAction.DROP_BEACON:
                controller.drop_beacon()
            elif action is IntentAction.SELF_DESTRUCT:
                controller.self_destruct()
            else:
                raise ValueError(f"Unsupported Unit intent action: {action.value}")

    @staticmethod
    def _emit_core(core, intent) -> None:
        action = intent.action
        if action is IntentAction.WAIT:
            core.wait()
        elif action is IntentAction.SPAWN:
            assert intent.unit_type is not None
            core.spawn(intent.unit_type)
        elif action is IntentAction.HEAL:
            core.heal()
        elif action is IntentAction.REPAIR_SHIELD:
            core.repair_shield()
        elif action is IntentAction.START_MOVE:
            assert intent.direction is not None
            core.start_move(intent.direction)
        elif action is IntentAction.CANCEL_MOVE:
            core.cancel_move()
        elif action is IntentAction.PICKUP_BEACON:
            core.pickup_beacon()
        elif action is IntentAction.DROP_BEACON:
            core.drop_beacon()
        elif action is IntentAction.SELF_DESTRUCT:
            core.self_destruct()
        else:
            raise ValueError(f"Unsupported Core intent action: {action.value}")
