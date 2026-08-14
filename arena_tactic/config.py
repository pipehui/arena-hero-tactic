from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TacticConfig:
    """Tuning values for the modular intent kernel.

    Official HP, prices, storage and action ranges deliberately do not live
    here; those values come from the SDK and the v0.14 rules.
    """

    decision_node_limit: int = 4_096
    # Information-gain targets are capped to a 24-cell search radius.  A 2,048
    # node A* budget covers that diamond (including substantial detours) while
    # keeping large multi-Worker Turns deterministically bounded.
    path_node_limit: int = 2_048
    # Resource assignment uses complete route-distance fields, not geometric
    # distance.  Keep this separate from point-to-point A* so operators can
    # reduce matching cost without silently changing ordinary navigation.
    # Route-aware matching stops after 1,536 visited cells.  Actual movement
    # still uses the larger A* budget below; this bound prevents a handful of
    # fog-separated remembered nodes from multiplying full-map BFS scans for
    # every Worker.
    distance_field_node_limit: int = 1_536
    exploration_candidate_limit: int = 24
    exploration_search_radius: int = 24
    # Selecting a brand-new information target requires a full weighted field.
    # Bound those expensive scans per Tick; sticky missions still navigate on
    # every Tick and the UUID order is rotated for deterministic fairness.
    exploration_new_goal_budget: int = 1
    exploration_refresh_ticks: int = 64
    exploration_target_failure_limit: int = 3
    exploration_target_backoff_ticks: int = 16
    exploration_scout_hold_ticks: int = 64
    exploration_stall_ticks: int = 3
    exploration_sector_radii: tuple[int, ...] = (10, 20, 30, 40)
    exploration_sector_step: int = 10
    resource_memory_ttl: int = 256
    resource_assignment_persistence_bonus: int = 6
    congestion_decay_ticks: int = 64
    enemy_track_ttl: int = 6
    enemy_core_occupancy_memory_ttl: int = 8
    enemy_core_control_ttl: int = 512
    danger_envelope_ttl: int = 6
    threat_heat_visible_risk: int = 8
    threat_heat_visible_ttl: int = 24
    threat_heat_projected_risk: int = 4
    threat_heat_projected_ttl: int = 16
    threat_heat_damage_risk: int = 16
    threat_heat_damage_neighbor_risk: int = 8
    threat_heat_damage_ttl: int = 40
    threat_heat_destroyed_risk: int = 24
    threat_heat_destroyed_neighbor_risk: int = 12
    threat_heat_destroyed_ttl: int = 64
    threat_heat_cell_limit: int = 2_048
    # Team-level tactical awareness.  These values describe uncertain areas
    # around combat enemies seen by *any* friendly observer.  They influence
    # every Worker's route without pretending that a fogged enemy occupies a
    # concrete cell.
    # Two Worker vision radii (Gameplay v0.14: 3 + 3).  Shared sightings
    # inside this near field trigger retreat; farther sightings only bias the
    # global risk map and route costs.
    global_worker_threat_awareness_radius: int = 6
    global_worker_corridor_projection_ticks: int = 4
    global_worker_corridor_width: int = 2
    global_worker_corridor_risk: int = 12
    loop_history_length: int = 8
    loop_repeat_limit: int = 2
    unit_move_failure_ttl: int = 2

    opening_worker_target: int = 4
    worker_ratio_percent: int = 50
    worker_only_population_threshold: int = 25
    # Mature-force stockpiling means 35 Workers plus 35 combat Units, not 35
    # total population.  Keep the aggregate threshold as a compatibility guard
    # and expose the two composition targets explicitly.
    population_stockpile_threshold: int = 70
    stockpile_worker_target: int = 35
    stockpile_combat_target: int = 35
    minimum_vanguards: int = 6
    minimum_rangers: int = 6
    home_force_floor: int = 12

    worker_escape_trigger_radius: int = 4
    worker_escape_clearance_radius: int = 5
    worker_escape_safe_ticks: int = 2
    worker_escape_nonfatal_hit_budget: int = 1
    worker_escape_lookahead_nodes: int = 32
    worker_escape_plan_depth: int = 4
    worker_escape_plan_node_limit: int = 64
    worker_escape_replan_ticks: int = 2
    worker_escape_max_loop_period: int = 4
    worker_escape_waypoint_lease_ticks: int = 4
    enemy_core_worker_exclusion_radius: int = 6
    enemy_core_worker_clear_radius: int = 8
    # Once Core storage is saturated, Workers still return home, but spread
    # across dedicated non-combat rings instead of joining the deposit lane.
    # A small hysteresis prevents one point of healing/repair from summoning
    # the entire workforce back to the Core entrance.
    worker_full_storage_guard_radii: tuple[int, ...] = (8, 10)
    worker_full_storage_near_reserve_count: int = 4
    worker_full_storage_parking_min_radius: int = 12
    worker_full_storage_parking_max_radius: int = 20
    # During a home-defense alert, yield the single-occupancy combat area but
    # remain close enough for service.  Avoid the 5/10/15 combat patrol rings.
    worker_full_storage_combat_guard_radii: tuple[int, ...] = (14, 16)
    worker_full_storage_release_space: int = 5
    worker_full_storage_replenishers: int = 1

    home_warning_radius: int = 30
    home_defense_hold_ticks: int = 4
    home_engage_radius: int = 13
    home_pursuit_radius: int = 18
    outer_screen_min_radius: int = 13
    outer_screen_max_radius: int = 30
    outer_screen_continue_radius: int = 32
    outer_screen_acquire_distance: int = 5
    outer_screen_vanguards: int = 2
    outer_screen_rangers: int = 2
    outer_screen_home_vanguard_reserve: int = 4
    outer_screen_home_ranger_reserve: int = 4
    outer_screen_hold_ticks: int = 4
    outer_screen_fog_ttl: int = 2
    outer_screen_stance_lease_ticks: int = 4
    outer_screen_no_progress_ticks: int = 2
    outer_screen_reverse_suppress_ticks: int = 2
    vanguard_engage_distance: int = 4
    vanguard_intercept_lease_ticks: int = 4
    combat_exclusive_radius: int = 13
    peaceful_squad_radii: tuple[int, ...] = (5, 10, 15)
    squad_max_separation: int = 4
    squad_reassembly_no_progress_ticks: int = 2
    squad_reassembly_break_ticks: int = 4
    formation_candidate_limit: int = 8
    formation_target_stall_ticks: int = 2
    formation_target_backoff_ticks: int = 8
    formation_pair_cooldown_ticks: int = 8
    formation_partner_hold_ticks: int = 2
    formation_max_route_distance: int = 12
    formation_yield_ticks: int = 2
    tactical_position_lease_ticks: int = 4
    ranger_repeat_miss_limit: int = 2
    ranger_miss_suppress_ticks: int = 2

    core_retreat_radius: int = 18
    core_retreat_projected_attackers: int = 2
    core_retreat_safe_ticks: int = 4
    core_move_failure_ttl: int = 8
    strategic_site_min_distance: int = 12
    strategic_site_max_distance: int = 20

    service_lane_depth: int = 2
    # An injured Unit may reserve treatment resources from anywhere, but it
    # only owns the single Core admission slot once it reaches the local
    # service area.  This prevents a distant casualty from freezing a ready
    # cargo pipeline for dozens of Ticks.
    service_patient_ready_radius: int = 3
    recovery_urgent_percent: int = 50
    manual_move_protection_ticks: int = 2

    raid_start_radius: int = 24
    raid_confirmed_start_radius: int = 30
    raid_confirmed_sightings: int = 2
    raid_continue_radius: int = 30
    raid_intel_ttl: int = 64
    raid_force_margin: int = 2
    raid_containment_radius: int = 45
    raid_containment_continue_radius: int = 50
    raid_containment_core_count: int = 2
    raid_peace_home_reserve: int = 8
    raid_min_siege_members: int = 4
    raid_long_range_start_radius: int = 60
    raid_long_range_continue_radius: int = 70
    raid_long_range_min_members: int = 4
    raid_long_range_max_route: int = 80
    raid_long_range_search_reserve_ticks: int = 16
    raid_long_range_max_campaign_ticks: int = 96
    raid_initial_pair_count: int = 2
    raid_escalation_pair_step: int = 1
    home_return_radius: int = 18
    home_return_handoff_radius: int = 12
    beacon_acquire_radius: int = 12
    beacon_min_workers: int = 4
    beacon_guard_radius: int = 2

    def __post_init__(self) -> None:
        positive = (
            self.decision_node_limit,
            self.path_node_limit,
            self.distance_field_node_limit,
            self.exploration_candidate_limit,
            self.exploration_search_radius,
            self.exploration_new_goal_budget,
            self.exploration_refresh_ticks,
            self.exploration_target_failure_limit,
            self.exploration_target_backoff_ticks,
            self.exploration_scout_hold_ticks,
            self.exploration_stall_ticks,
            self.exploration_sector_step,
            self.resource_memory_ttl,
            self.resource_assignment_persistence_bonus,
            self.congestion_decay_ticks,
            self.enemy_track_ttl,
            self.enemy_core_occupancy_memory_ttl,
            self.enemy_core_control_ttl,
            self.danger_envelope_ttl,
            self.threat_heat_visible_risk,
            self.threat_heat_visible_ttl,
            self.threat_heat_projected_risk,
            self.threat_heat_projected_ttl,
            self.threat_heat_damage_risk,
            self.threat_heat_damage_neighbor_risk,
            self.threat_heat_damage_ttl,
            self.threat_heat_destroyed_risk,
            self.threat_heat_destroyed_neighbor_risk,
            self.threat_heat_destroyed_ttl,
            self.threat_heat_cell_limit,
            self.global_worker_threat_awareness_radius,
            self.global_worker_corridor_projection_ticks,
            self.global_worker_corridor_width,
            self.global_worker_corridor_risk,
            self.loop_history_length,
            self.loop_repeat_limit,
            self.unit_move_failure_ttl,
            self.opening_worker_target,
            self.worker_ratio_percent,
            self.worker_only_population_threshold,
            self.population_stockpile_threshold,
            self.stockpile_worker_target,
            self.stockpile_combat_target,
            self.minimum_vanguards,
            self.minimum_rangers,
            self.home_force_floor,
            self.worker_escape_trigger_radius,
            self.worker_escape_clearance_radius,
            self.worker_escape_safe_ticks,
            self.worker_escape_lookahead_nodes,
            self.worker_escape_plan_depth,
            self.worker_escape_plan_node_limit,
            self.worker_escape_replan_ticks,
            self.worker_escape_max_loop_period,
            self.worker_escape_waypoint_lease_ticks,
            self.enemy_core_worker_exclusion_radius,
            self.enemy_core_worker_clear_radius,
            self.worker_full_storage_release_space,
            self.worker_full_storage_replenishers,
            self.worker_full_storage_near_reserve_count,
            self.worker_full_storage_parking_min_radius,
            self.worker_full_storage_parking_max_radius,
            self.home_warning_radius,
            self.home_defense_hold_ticks,
            self.home_engage_radius,
            self.home_pursuit_radius,
            self.outer_screen_min_radius,
            self.outer_screen_max_radius,
            self.outer_screen_continue_radius,
            self.outer_screen_acquire_distance,
            self.outer_screen_vanguards,
            self.outer_screen_rangers,
            self.outer_screen_home_vanguard_reserve,
            self.outer_screen_home_ranger_reserve,
            self.outer_screen_hold_ticks,
            self.outer_screen_fog_ttl,
            self.outer_screen_stance_lease_ticks,
            self.outer_screen_no_progress_ticks,
            self.outer_screen_reverse_suppress_ticks,
            self.vanguard_engage_distance,
            self.vanguard_intercept_lease_ticks,
            self.combat_exclusive_radius,
            self.squad_max_separation,
            self.squad_reassembly_no_progress_ticks,
            self.squad_reassembly_break_ticks,
            self.formation_candidate_limit,
            self.formation_target_stall_ticks,
            self.formation_target_backoff_ticks,
            self.formation_pair_cooldown_ticks,
            self.formation_partner_hold_ticks,
            self.formation_max_route_distance,
            self.formation_yield_ticks,
            self.tactical_position_lease_ticks,
            self.ranger_repeat_miss_limit,
            self.ranger_miss_suppress_ticks,
            self.core_retreat_radius,
            self.core_retreat_projected_attackers,
            self.core_retreat_safe_ticks,
            self.core_move_failure_ttl,
            self.strategic_site_min_distance,
            self.strategic_site_max_distance,
            self.service_lane_depth,
            self.service_patient_ready_radius,
            self.recovery_urgent_percent,
            self.manual_move_protection_ticks,
            self.raid_start_radius,
            self.raid_confirmed_start_radius,
            self.raid_confirmed_sightings,
            self.raid_continue_radius,
            self.raid_intel_ttl,
            self.raid_force_margin,
            self.raid_containment_radius,
            self.raid_containment_continue_radius,
            self.raid_containment_core_count,
            self.raid_peace_home_reserve,
            self.raid_min_siege_members,
            self.raid_long_range_start_radius,
            self.raid_long_range_continue_radius,
            self.raid_long_range_min_members,
            self.raid_long_range_max_route,
            self.raid_long_range_search_reserve_ticks,
            self.raid_long_range_max_campaign_ticks,
            self.raid_initial_pair_count,
            self.raid_escalation_pair_step,
            self.home_return_radius,
            self.home_return_handoff_radius,
            self.beacon_acquire_radius,
            self.beacon_min_workers,
            self.beacon_guard_radius,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("tactic limits must be positive")
        if not 1 <= self.worker_ratio_percent <= 100:
            raise ValueError("worker ratio must be within 1..100")
        if self.population_stockpile_threshold <= self.worker_only_population_threshold:
            raise ValueError("stockpile population must exceed worker-only threshold")
        if self.population_stockpile_threshold < (
            self.stockpile_worker_target + self.stockpile_combat_target
        ):
            raise ValueError("stockpile population must cover both force targets")
        if self.outer_screen_min_radius < self.home_engage_radius:
            raise ValueError("outer screen must start at or beyond home engagement")
        if self.outer_screen_max_radius < self.outer_screen_min_radius:
            raise ValueError("outer screen radius must be ordered")
        if self.outer_screen_continue_radius < self.outer_screen_max_radius:
            raise ValueError("outer screen continuation must cover acquisition radius")
        if self.squad_reassembly_break_ticks < self.squad_reassembly_no_progress_ticks:
            raise ValueError("reassembly break must not precede its no-progress limit")
        if self.formation_pair_cooldown_ticks < self.squad_reassembly_break_ticks:
            raise ValueError("pair cooldown must cover the reassembly break window")
        if (
            not self.exploration_sector_radii
            or tuple(sorted(set(self.exploration_sector_radii)))
            != self.exploration_sector_radii
        ):
            raise ValueError("exploration sector radii must be unique and ordered")
        if not 1 <= self.recovery_urgent_percent <= 100:
            raise ValueError("recovery threshold must be within 1..100")
        if self.worker_escape_clearance_radius <= self.worker_escape_trigger_radius:
            raise ValueError("escape clearance must exceed its trigger radius")
        if self.enemy_core_worker_clear_radius <= self.enemy_core_worker_exclusion_radius:
            raise ValueError("enemy Core clearance must exceed its exclusion radius")
        if self.enemy_core_control_ttl <= self.raid_intel_ttl:
            raise ValueError("enemy Core control memory must outlive raid intel")
        if self.worker_escape_plan_node_limit < self.worker_escape_plan_depth:
            raise ValueError("escape planning nodes must cover its depth")
        if self.worker_escape_max_loop_period * 2 > self.loop_history_length:
            raise ValueError("escape loop period must fit twice in position history")
        if (
            not self.worker_full_storage_guard_radii
            or tuple(sorted(set(self.worker_full_storage_guard_radii)))
            != self.worker_full_storage_guard_radii
            or any(radius <= 0 for radius in self.worker_full_storage_guard_radii)
        ):
            raise ValueError("Worker home-guard radii must be unique and ordered")
        if self.worker_full_storage_guard_radii[0] <= self.service_lane_depth + 2:
            raise ValueError("Worker home guard must stay outside Core service")
        if (
            self.worker_full_storage_parking_min_radius
            <= self.worker_full_storage_guard_radii[-1]
            or self.worker_full_storage_parking_max_radius
            < self.worker_full_storage_parking_min_radius
        ):
            raise ValueError("Worker parking must be ordered outside the near reserve")
        if (
            not self.worker_full_storage_combat_guard_radii
            or tuple(sorted(set(self.worker_full_storage_combat_guard_radii)))
            != self.worker_full_storage_combat_guard_radii
            or any(
                radius <= self.combat_exclusive_radius
                for radius in self.worker_full_storage_combat_guard_radii
            )
        ):
            raise ValueError("Combat Worker guard must be ordered outside combat exclusivity")
        if set(self.worker_full_storage_combat_guard_radii) & set(
            self.peaceful_squad_radii
        ):
            raise ValueError("Combat Worker guard must not overlap combat patrol rings")
        if self.global_worker_corridor_width >= self.global_worker_threat_awareness_radius:
            raise ValueError("global threat awareness must exceed corridor width")
        if self.home_engage_radius > self.home_pursuit_radius:
            raise ValueError("pursuit radius must include the engagement radius")
        if self.home_pursuit_radius > self.home_warning_radius:
            raise ValueError("warning radius must include the pursuit radius")
        if self.combat_exclusive_radius < self.home_engage_radius:
            raise ValueError("combat exclusivity must include the home engagement area")
        if self.core_retreat_radius < self.home_engage_radius:
            raise ValueError("Core retreat radius must include the engagement area")
        if self.strategic_site_max_distance < self.strategic_site_min_distance:
            raise ValueError("strategic site range is inverted")
        if self.service_patient_ready_radius < self.service_lane_depth:
            raise ValueError("patient readiness must include the service lane")
        if self.raid_continue_radius < self.raid_start_radius:
            raise ValueError("raid continuation radius must include its start radius")
        if not (
            self.raid_start_radius
            <= self.raid_confirmed_start_radius
            <= self.raid_continue_radius
        ):
            raise ValueError("confirmed raid radius must lie within raid limits")
        if self.raid_containment_radius < self.raid_confirmed_start_radius:
            raise ValueError("containment radius must include confirmed raid range")
        if self.raid_containment_continue_radius < self.raid_containment_radius:
            raise ValueError("containment continuation must include its start range")
        if self.raid_peace_home_reserve < 2:
            raise ValueError("containment raids must leave at least two defenders")
        if self.raid_min_siege_members < 2:
            raise ValueError("a siege requires at least two members")
        if self.raid_long_range_start_radius <= self.raid_containment_radius:
            raise ValueError("long-range raids must extend beyond containment range")
        if self.raid_long_range_continue_radius < self.raid_long_range_start_radius:
            raise ValueError("long-range continuation must cover its start range")
        if self.raid_long_range_min_members < 4:
            raise ValueError("long-range raids require at least a 2V+2R group")
        if self.raid_long_range_max_route < self.raid_long_range_start_radius:
            raise ValueError("long-range route budget must cover its start radius")
        if self.raid_long_range_max_campaign_ticks < 64:
            raise ValueError("long-range campaign must preserve a useful search window")
        if self.home_return_handoff_radius >= self.home_return_radius:
            raise ValueError("home handoff must be inside the return trigger")
        if (
            not self.peaceful_squad_radii
            or tuple(sorted(set(self.peaceful_squad_radii)))
            != self.peaceful_squad_radii
        ):
            raise ValueError("peaceful squad radii must be unique and ordered")


DEFAULT_CONFIG = TacticConfig()
