//! Derived game state the 1.21.11 protocol doesn't hand a client as one
//! ready-made field, reconstructed here from packets and from the bot's own
//! inputs so `build_observation` can fill what the sim trained against
//! instead of a hardcoded neutral.
//!
//! Two sources:
//!   * **packets** (`on_packet`, fed from `Event::Packet`): scoreboard team
//!     membership (`ClientboundSetPlayerTeam`), other players' main-hand item
//!     (`ClientboundSetEquipment`), our own hurt flash
//!     (`ClientboundHurtAnimation` / `ClientboundDamageEvent`) and shield
//!     lock-out (`ClientboundCooldown` naming the shield item);
//!   * **the bot's own actions** (`note_bow_draw`, `note_hotbar_swap`): the
//!     draw timer and the modern-input-order swap lock-out are things the bot
//!     *does*, so it simply counts the ticks.
//!
//! All timers are in game ticks and advanced once per tick by `advance`.

use std::collections::{HashMap, HashSet};

use azalea::inventory::components::EquipmentSlot;
use azalea::protocol::packets::game::ClientboundGamePacket;
use azalea::protocol::packets::game::c_set_player_team::Method as TeamMethod;
use azalea::registry::builtin::ItemKind;
use azalea::world::MinecraftEntityId;

/// Sim's `hurt_time_left` is the 20-tick (1.0 s) `invulnerableTime` window,
/// not vanilla's shorter 10-tick `hurtTime` render timer - match the sim.
const HURT_WINDOW_TICKS: u32 = 20;

/// Whether another player counts as friend or foe for the observation split
/// and the attack guard.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Relation {
    Teammate,
    Enemy,
}

#[derive(Default)]
pub struct Tracker {
    /// scoreboard team name -> member entry names (player names for players).
    teams: HashMap<String, HashSet<String>>,
    /// The team this bot's own name is on, if any.
    my_team: Option<String>,
    /// Other entities' main-hand item, from `ClientboundSetEquipment`.
    mainhand: HashMap<MinecraftEntityId, ItemKind>,
    /// Ticks left on our hurt-invulnerability flash (counts down).
    hurt_ticks: u32,
    /// Ticks left on a shield lock-out from an axe hit (counts down).
    shield_disabled_ticks: u32,
    /// Ticks we've been holding right-click with a bow in hand (counts up).
    bow_draw_ticks: u32,
    /// Ticks left on the post-swap attack/use lock-out (`--input-order
    /// modern`; counts down).
    swap_lockout_ticks: u32,
}

impl Tracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one packet. `my_name` / `my_id` identify this bot so its own
    /// team membership and hurt animation are recognised; `my_id` is `None`
    /// until the bot has spawned.
    pub fn on_packet(
        &mut self,
        packet: &ClientboundGamePacket,
        my_name: &str,
        my_id: Option<MinecraftEntityId>,
    ) {
        match packet {
            ClientboundGamePacket::SetPlayerTeam(p) => {
                self.apply_team(&p.name, &p.method);
                self.recompute_my_team(my_name);
            }
            ClientboundGamePacket::SetEquipment(p) => {
                for (slot, stack) in &p.slots.slots {
                    if *slot == EquipmentSlot::Mainhand {
                        if stack.is_present() {
                            self.mainhand.insert(p.entity_id, stack.kind());
                        } else {
                            self.mainhand.remove(&p.entity_id);
                        }
                    }
                }
            }
            ClientboundGamePacket::HurtAnimation(p) if my_id == Some(p.id) => {
                self.hurt_ticks = HURT_WINDOW_TICKS;
            }
            ClientboundGamePacket::DamageEvent(p) if my_id == Some(p.entity_id) => {
                self.hurt_ticks = HURT_WINDOW_TICKS;
            }
            ClientboundGamePacket::Cooldown(p) if p.item == ItemKind::Shield => {
                self.shield_disabled_ticks = p.duration;
            }
            _ => {}
        }
    }

    fn apply_team(&mut self, name: &str, method: &TeamMethod) {
        match method {
            TeamMethod::Add((_, players)) => {
                self.teams
                    .insert(name.to_owned(), players.iter().cloned().collect());
            }
            TeamMethod::Remove => {
                self.teams.remove(name);
            }
            TeamMethod::Join(players) => {
                self.teams
                    .entry(name.to_owned())
                    .or_default()
                    .extend(players.iter().cloned());
            }
            TeamMethod::Leave(players) => {
                if let Some(members) = self.teams.get_mut(name) {
                    for p in players {
                        members.remove(p);
                    }
                }
            }
            TeamMethod::Change(_) => {}
        }
    }

    fn recompute_my_team(&mut self, my_name: &str) {
        self.my_team = self
            .teams
            .iter()
            .find(|(_, members)| members.contains(my_name))
            .map(|(name, _)| name.clone());
    }

    /// Friend or foe for `player_name`. With no scoreboard teams in play
    /// (the common case), everyone is an enemy - the pre-team behaviour.
    pub fn relation(&self, player_name: &str) -> Relation {
        match &self.my_team {
            Some(team) if self
                .teams
                .get(team)
                .is_some_and(|m| m.contains(player_name)) =>
            {
                Relation::Teammate
            }
            _ => Relation::Enemy,
        }
    }

    /// The main-hand item last broadcast for `id`, if any.
    pub fn mainhand_of(&self, id: MinecraftEntityId) -> Option<ItemKind> {
        self.mainhand.get(&id).copied()
    }

    /// Advance every tick timer by one tick. `drawing_bow` is whether the
    /// bot is holding right-click with a bow in hand *this* tick.
    pub fn advance(&mut self, drawing_bow: bool) {
        self.hurt_ticks = self.hurt_ticks.saturating_sub(1);
        self.shield_disabled_ticks = self.shield_disabled_ticks.saturating_sub(1);
        self.swap_lockout_ticks = self.swap_lockout_ticks.saturating_sub(1);
        self.bow_draw_ticks = if drawing_bow {
            self.bow_draw_ticks.saturating_add(1)
        } else {
            0
        };
    }

    /// Record a hotbar swap the bot just performed, starting the modern
    /// input-order lock-out (`lockout_ticks == 0` for the legacy order).
    pub fn note_hotbar_swap(&mut self, lockout_ticks: u32) {
        self.swap_lockout_ticks = self.swap_lockout_ticks.max(lockout_ticks);
    }

    // --- normalised observation fields (mirroring sim/src/observation.rs) ---

    /// `hurt_time_left / hurt_invulnerability_seconds`.
    pub fn self_hurt(&self, hurt_window_seconds: f64) -> f64 {
        if hurt_window_seconds <= 0.0 {
            return 0.0;
        }
        let window_ticks = (hurt_window_seconds / 0.05).max(1.0);
        (self.hurt_ticks as f64 / window_ticks).clamp(0.0, 1.0)
    }

    /// `shield_disabled_time / axe_shield_disable_seconds`.
    pub fn self_shield_disabled(&self, disable_seconds: f64) -> f64 {
        if disable_seconds <= 0.0 {
            return 0.0;
        }
        let ticks = (disable_seconds / 0.05).max(1.0);
        (self.shield_disabled_ticks as f64 / ticks).clamp(0.0, 1.0)
    }

    /// `bow_draw / bow_max_draw_seconds`.
    pub fn self_bow_draw(&self, max_draw_seconds: f64) -> f64 {
        if max_draw_seconds <= 0.0 {
            return 0.0;
        }
        (self.bow_draw_ticks as f64 * 0.05 / max_draw_seconds).clamp(0.0, 1.0)
    }

    /// `swap_lockout / swap_lockout_seconds`.
    pub fn self_swap_lockout(&self, lockout_seconds: f64) -> f64 {
        let ticks = (lockout_seconds / 0.05).max(1.0);
        (self.swap_lockout_ticks as f64 / ticks).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_teams_means_everyone_is_an_enemy() {
        let t = Tracker::new();
        assert_eq!(t.relation("SomePlayer"), Relation::Enemy);
    }

    #[test]
    fn team_membership_splits_friend_from_foe() {
        let mut t = Tracker::new();
        let mut red = HashSet::new();
        red.insert("Bot".to_string());
        red.insert("Ally".to_string());
        t.teams.insert("red".to_string(), red);
        let mut blue = HashSet::new();
        blue.insert("Foe".to_string());
        t.teams.insert("blue".to_string(), blue);
        t.recompute_my_team("Bot");

        assert_eq!(t.relation("Ally"), Relation::Teammate);
        assert_eq!(t.relation("Foe"), Relation::Enemy);
        assert_eq!(t.relation("Stranger"), Relation::Enemy);
    }

    #[test]
    fn leaving_a_team_drops_the_membership() {
        let mut t = Tracker::new();
        t.apply_team(
            "red",
            &TeamMethod::Join(vec!["Bot".to_string(), "Ally".to_string()]),
        );
        t.recompute_my_team("Bot");
        assert_eq!(t.relation("Ally"), Relation::Teammate);

        t.apply_team("red", &TeamMethod::Leave(vec!["Ally".to_string()]));
        assert_eq!(t.relation("Ally"), Relation::Enemy);
    }

    #[test]
    fn hurt_timer_ramps_then_decays() {
        let mut t = Tracker::new();
        assert_eq!(t.self_hurt(1.0), 0.0);
        t.hurt_ticks = HURT_WINDOW_TICKS;
        assert!((t.self_hurt(1.0) - 1.0).abs() < 1e-9);
        for _ in 0..10 {
            t.advance(false);
        }
        assert!((t.self_hurt(1.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn bow_draw_counts_up_while_drawing_and_resets_when_released() {
        let mut t = Tracker::new();
        for _ in 0..10 {
            t.advance(true);
        }
        // 10 ticks * 0.05 s / 1.0 s max draw
        assert!((t.self_bow_draw(1.0) - 0.5).abs() < 1e-9);
        t.advance(false);
        assert_eq!(t.self_bow_draw(1.0), 0.0);
    }

    #[test]
    fn swap_lockout_is_zero_under_the_legacy_input_order() {
        let mut t = Tracker::new();
        t.note_hotbar_swap(0);
        assert_eq!(t.self_swap_lockout(0.05), 0.0);
        t.note_hotbar_swap(1);
        assert!(t.self_swap_lockout(0.05) > 0.0);
        t.advance(false);
        assert_eq!(t.self_swap_lockout(0.05), 0.0);
    }
}
