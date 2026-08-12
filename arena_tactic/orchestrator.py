from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from arena_hero import Position, Turn, UnitType

from .beacon import BeaconPlanner
from .combat import CombatPlanner
from .config import DEFAULT_CONFIG, TacticConfig
from .context import DecisionContext
from .core_safety import CoreSafetyPlanner
from .defense import DefensePlanner
from .geometry import manhattan
from .models import (
    ActionIntent,
    CoreServiceQueue,
    FireMission,
    HomeCounterSiegeDecision,
    HomeCombatAssignment,
    IntentAction,
    IntentResolution,
    MoveAttempt,
    UnitMission,
    WorldModel,
)
from .production import ProductionPlanner
from .projection import TacticalMap, build_tactical_map
from .raid import RaidPlanner
from .recovery import RecoveryPlanner
from .resolver import IntentResolver
from .resource_allocator import ResourceAllocator
from .service import CoreServiceChoreography, CoreServicePlanner, service_protected_positions
from .state import TacticMemory
from .trace import DecisionTraceBuilder
from .worker import WorkerPlanner
from .world import build_world_model


@dataclass(frozen=True, slots=True)
class _PlanningProducts:
    intents: tuple[ActionIntent, ...]
    choreography_intents: tuple[ActionIntent, ...]
    fire_missions: tuple[FireMission, ...]
    legal_opportunities: tuple[tuple[UUID, UUID, Position], ...]
    production_candidates: tuple[dict[str, object], ...]
    home_combat_assignment: HomeCombatAssignment
    counter_siege: HomeCounterSiegeDecision


class DecisionKernel:
    """Small orchestration layer over pure-value tactical planners."""

    def __init__(
        self,
        config: TacticConfig = DEFAULT_CONFIG,
        memory: TacticMemory | None = None,
    ) -> None:
        self.config = config
        self.memory = memory or TacticMemory()
        self.resources = ResourceAllocator(config, self.memory)
        self.combat = CombatPlanner(config, self.memory)
        self.service = CoreServicePlanner(config, self.memory)
        self.recovery = RecoveryPlanner(config, self.memory)
        self.workers = WorkerPlanner(config, self.memory, self.resources)
        self.defense = DefensePlanner(config, self.memory, self.combat)
        self.raids = RaidPlanner(config, self.memory)
        self.beacon = BeaconPlanner(config, self.memory)
        self.core_safety = CoreSafetyPlanner(config, self.memory)
        self.production = ProductionPlanner(config, self.memory, self.combat)
        self.trace = DecisionTraceBuilder(config, self.memory)
        self.last_tactical_map: TacticalMap | None = None

    def decide(self, turn: Turn) -> tuple[WorldModel, IntentResolution, dict[str, object]]:
        world = build_world_model(turn, self.memory, self.config)
        projection = build_tactical_map(world, self.config)
        if world.core is None:
            result = self._without_core(world, projection)
            self.last_tactical_map = self._resolved_tactical_map(
                projection,
                result[1],
            )
            return result

        self.combat.sync_engagements(world, projection)
        evacuation, planned_core_intents = self.core_safety.intents(world, projection)
        core_intents = tuple(planned_core_intents)
        context = self._context(world, projection, core_intents)
        products = self._plan(context, core_intents)
        resolution = self._resolve(context, products)
        self.defense.observe_resolution(world, resolution)
        self._remember_move_attempts(world, resolution)
        assigned_map = self._map_with_resource_assignments(context.tactical_map)
        resolved_map = self._resolved_tactical_map(assigned_map, resolution)
        self.last_tactical_map = resolved_map
        trace = self.trace.build(
            world,
            resolved_map,
            resolution,
            context.service,
            products.fire_missions,
            products.legal_opportunities,
            evacuation,
            products.production_candidates,
            products.home_combat_assignment,
            products.counter_siege,
        )
        return world, resolution, trace

    def _without_core(
        self,
        world: WorldModel,
        projection: TacticalMap,
    ) -> tuple[WorldModel, IntentResolution, dict[str, object]]:
        empty_service = CoreServiceQueue(service="NONE", admission_id=None)
        waits = tuple(
            ActionIntent.simple(
                unit.id,
                IntentAction.WAIT,
                UnitMission.WAIT,
                99,
                reason="CORE_UNAVAILABLE",
            )
            for unit in world.friendlies
        )
        resolution = IntentResolver(
            decision_node_limit=self.config.decision_node_limit
        ).resolve(world, waits)
        trace = self.trace.build(
            world,
            projection,
            resolution,
            empty_service,
            (),
            (),
            None,
            (),
        )
        return world, resolution, trace

    def _context(
        self,
        world: WorldModel,
        projection: TacticalMap,
        core_intents: tuple[ActionIntent, ...],
    ) -> DecisionContext:
        core_starting_move = any(
            intent.action is IntentAction.START_MOVE for intent in core_intents
        )
        core_move_target = next(
            (
                intent.target_position
                for intent in core_intents
                if intent.action is IntentAction.START_MOVE
            ),
            None,
        )
        service = self.service.plan(
            world,
            projection,
            core_starting_move=core_starting_move,
            projected_core_destination=core_move_target,
        )
        protected = service_protected_positions(world, service)
        combat_active = any(
            enemy.unit_type in {UnitType.VANGUARD, UnitType.RANGER}
            and manhattan(enemy.position, world.core.position)
            <= self.config.combat_exclusive_radius
            for enemy in world.enemies
        )
        tactical_map = projection.with_operations(
            service_positions=protected,
            projected_core_position=core_move_target,
        )
        return DecisionContext(
            world=world,
            tactical_map=tactical_map,
            service=service,
            protected_positions=protected,
            core_starting_move=core_starting_move,
            combat_active=combat_active,
        )

    def _plan(
        self,
        context: DecisionContext,
        core_intents: tuple[ActionIntent, ...],
    ) -> _PlanningProducts:
        world = context.world
        projection = context.tactical_map
        service = context.service
        fire_missions, fire_intents, legal_opportunities = self.combat.fire_intents(
            world,
            projection,
        )
        worker_intents = self.workers.intents(world, projection, service)
        egress_intents = self.service.combat_egress_intents(world, projection, service)
        recovery_intents = self.recovery.intents(world, projection, service)
        home_combat_assignment = self.combat.home_combat_assignment(
            world,
            projection,
        )
        counter_siege, counter_siege_intents = self.raids.counter_siege_intents(
            world,
            projection,
            context.protected_positions,
        )

        intents: list[ActionIntent] = []
        intents.extend(core_intents)
        intents.extend(worker_intents)
        intents.extend(egress_intents)
        intents.extend(self.recovery.survival_intents(world, projection))
        intents.extend(recovery_intents)
        intents.extend(fire_intents)
        intents.extend(counter_siege_intents)
        intents.extend(
            self.beacon.intents(
                world,
                projection,
                context.protected_positions,
            )
        )
        intents.extend(
            self.combat.vanguard_intents(
                world,
                projection,
                home_combat_assignment,
            )
        )
        intents.extend(
            self.defense.intents(
                world,
                projection,
                context.protected_positions,
                home_combat_assignment.assigned_vanguard_ids
                | frozenset(counter_siege.member_ids),
            )
        )
        intents.extend(
            self.raids.intents(
                world,
                projection,
                context.protected_positions,
            )
        )
        production_intents, production_candidates = self.production.intents(
            world,
            projection,
            reserved_resources=service.reserved_resources,
            timeline=service.timeline,
        )
        intents.extend(production_intents)
        intents.extend(self._fallback_waits(world))
        return _PlanningProducts(
            intents=tuple(intents),
            choreography_intents=(
                *worker_intents,
                *egress_intents,
                *recovery_intents,
            ),
            fire_missions=fire_missions,
            legal_opportunities=legal_opportunities,
            production_candidates=production_candidates,
            home_combat_assignment=home_combat_assignment,
            counter_siege=counter_siege,
        )

    @staticmethod
    def _fallback_waits(world: WorldModel) -> tuple[ActionIntent, ...]:
        unit_waits = tuple(
            ActionIntent.simple(
                unit.id,
                IntentAction.WAIT,
                UnitMission.WAIT,
                99,
                reason="NO_LEGAL_TASK",
            )
            for unit in world.friendlies
        )
        core_wait = ActionIntent.simple(
            None,
            IntentAction.WAIT,
            UnitMission.WAIT,
            99,
            reason="NO_CORE_ACTION",
        )
        return (*unit_waits, core_wait)

    def _resolve(
        self,
        context: DecisionContext,
        products: _PlanningProducts,
    ) -> IntentResolution:
        choreography = CoreServiceChoreography.build(
            context.world,
            context.service,
            products.choreography_intents,
        )
        resolver = IntentResolver(
            decision_node_limit=self.config.decision_node_limit,
            combat_exclusive=context.combat_active,
            combat_exclusive_center=context.world.core.position,
            combat_exclusive_radius=self.config.combat_exclusive_radius,
            protected_positions=context.protected_positions,
            actor_priority_ceilings=choreography.priority_ceiling_map(),
            actor_move_blocks={
                actor_id: frozenset((failure.destination,))
                for actor_id, failure in self.memory.failed_unit_moves.items()
                if failure.expires_tick >= context.world.tick
            },
        )
        return resolver.resolve(context.world, products.intents)

    @staticmethod
    def _resolved_tactical_map(
        tactical_map: TacticalMap,
        resolution: IntentResolution,
    ) -> TacticalMap:
        planned_positions = {
            intent.actor_id: intent.target_position
            for intent in resolution.selected
            if intent.actor_id is not None
            and intent.action is IntentAction.MOVE
            and intent.target_position is not None
        }
        projected_core = next(
            (
                intent.target_position
                for intent in resolution.selected
                if intent.actor_id is None
                and intent.action is IntentAction.START_MOVE
                and intent.target_position is not None
            ),
            tactical_map.projected_core_position,
        )
        return tactical_map.with_operations(
            planned_positions=planned_positions,
            reserved_positions=frozenset(resolution.reserved_positions),
            projected_core_position=projected_core,
        )

    def _map_with_resource_assignments(
        self,
        tactical_map: TacticalMap,
    ) -> TacticalMap:
        assignments: dict[Position, list[UUID]] = {}
        resource_positions = {resource.position for resource in tactical_map.resources}
        for worker_id, mission in self.memory.unit_missions.items():
            if (
                mission.mission is UnitMission.HARVEST
                and mission.target in resource_positions
            ):
                assignments.setdefault(mission.target, []).append(worker_id)
        return tactical_map.with_resource_assignments(
            {
                position: tuple(worker_ids)
                for position, worker_ids in assignments.items()
            }
        )

    def _remember_move_attempts(
        self,
        world: WorldModel,
        resolution: IntentResolution,
    ) -> None:
        self.memory.last_move_attempts = {
            intent.actor_id: MoveAttempt(
                actor_id=intent.actor_id,
                tick=world.tick,
                origin=world.actor_position(intent.actor_id),
                destination=intent.target_position,
                direction=intent.direction,
            )
            for intent in resolution.selected
            if intent.actor_id is not None
            and intent.action is IntentAction.MOVE
            and intent.target_position is not None
            and intent.direction is not None
            and world.actor_position(intent.actor_id) is not None
        }
