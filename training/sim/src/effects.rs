//! Status effects - the vanilla potion effect table, ticked once per game
//! tick on each `Player`.
//!
//! Only splash potions apply effects today (`projectile::SplashPotion`), and
//! they only produce five of the nine effects (Speed, Strength, Poison plus
//! the two instants). Slowness, Weakness, Regeneration and Fire Resistance
//! are fully modelled here so a later enchant / environmental / config
//! source can grant them without touching the effect engine.
//!
//! Numbers follow vanilla: Speed +20% move per level, Slowness -15%,
//! Strength +3 flat melee damage per level, Weakness -4, Regeneration 1 HP
//! every `50 >> amplifier` ticks, Poison 1 HP every `25 >> amplifier` ticks
//! (never lethal), Fire Resistance negates fire/lava damage. Instant Health
//! and Instant Damage carry no duration - the caller applies the HP delta
//! from `instant_magnitude` directly.

use crate::kit::Item;

/// Every status effect. Discriminant order is the wire order of the
/// `self_effects` observation block - keep in sync with `features.py` and
/// bump `protocol::WIRE_VERSION` if it changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Effect {
    Speed = 0,
    Slowness = 1,
    Strength = 2,
    Weakness = 3,
    Regeneration = 4,
    Poison = 5,
    InstantHealth = 6,
    InstantDamage = 7,
    FireResistance = 8,
}

/// Number of `Effect` variants - width of the effect table and the
/// `self_effects` observation block.
pub const EFFECT_COUNT: usize = 9;

impl Effect {
    pub fn index(self) -> usize {
        self as usize
    }

    /// Applied as a one-off HP change, no duration (Instant Health / Damage).
    pub fn is_instant(self) -> bool {
        matches!(self, Effect::InstantHealth | Effect::InstantDamage)
    }

    /// Helpful to the affected player (used to decide friendly-fire gating
    /// of a splash cloud on teammates).
    pub fn is_beneficial(self) -> bool {
        matches!(
            self,
            Effect::Speed
                | Effect::Strength
                | Effect::Regeneration
                | Effect::InstantHealth
                | Effect::FireResistance
        )
    }

    /// The effect (and its amplifier, from config) a splash potion `Item`
    /// applies, or `None` for a non-potion item.
    pub fn from_splash_item(item: Item) -> Option<(Effect, u8)> {
        let c = &crate::config::cfg().combat;
        Some(match item {
            Item::SplashHealing => (Effect::InstantHealth, c.potion_healing_amplifier as u8),
            Item::SplashHarming => (Effect::InstantDamage, c.potion_harming_amplifier as u8),
            Item::SplashPoison => (Effect::Poison, c.potion_poison_amplifier as u8),
            Item::SplashSpeed => (Effect::Speed, c.potion_speed_amplifier as u8),
            Item::SplashStrength => (Effect::Strength, c.potion_strength_amplifier as u8),
            _ => return None,
        })
    }

    /// Seconds a *direct-centre* splash of this effect lasts (scaled down by
    /// the distance falloff at the edge of the cloud). Meaningless for the
    /// instant effects.
    pub fn splash_seconds(self) -> f32 {
        let c = &crate::config::cfg().combat;
        match self {
            Effect::Speed => c.potion_speed_seconds,
            Effect::Strength => c.potion_strength_seconds,
            Effect::Poison => c.potion_poison_seconds,
            _ => 0.0,
        }
    }

    /// HP a single Instant Health / Damage application moves, at `amplifier`
    /// (vanilla: `base << amplifier`).
    pub fn instant_magnitude(self, amplifier: u8) -> f32 {
        let c = &crate::config::cfg().combat;
        let base = match self {
            Effect::InstantHealth => c.instant_health_hp,
            Effect::InstantDamage => c.instant_damage_hp,
            _ => return 0.0,
        };
        base * (1u32 << amplifier.min(6)) as f32
    }
}

#[derive(Debug, Clone, Copy)]
struct Active {
    /// Vanilla amplifier: 0 = level I, 1 = level II, ...
    amplifier: u8,
    ticks_left: u32,
}

/// A player's active timed effects plus the fractional HP accumulators for
/// Regeneration / Poison.
#[derive(Debug, Clone, Default)]
pub struct StatusEffects {
    slots: [Option<Active>; EFFECT_COUNT],
    regen_accum: f32,
    poison_accum: f32,
}

impl StatusEffects {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Grant a *timed* effect. Vanilla stacking: a strictly stronger
    /// amplifier always wins; an equal one only wins if it would last
    /// longer; a weaker one is ignored. Instant effects are a no-op here -
    /// the caller applies their HP delta directly.
    pub fn apply(&mut self, effect: Effect, amplifier: u8, duration_ticks: u32) {
        if effect.is_instant() || duration_ticks == 0 {
            return;
        }
        let slot = &mut self.slots[effect.index()];
        let win = match slot {
            None => true,
            Some(cur) => {
                amplifier > cur.amplifier
                    || (amplifier == cur.amplifier && duration_ticks > cur.ticks_left)
            }
        };
        if win {
            *slot = Some(Active { amplifier, ticks_left: duration_ticks });
        }
    }

    pub fn has(&self, effect: Effect) -> bool {
        self.slots[effect.index()].is_some()
    }

    /// `(amplifier, ticks_left)` for an active effect.
    #[allow(dead_code)] // used in tests; handy for future observation-of-others
    pub fn get(&self, effect: Effect) -> Option<(u8, u32)> {
        self.slots[effect.index()].map(|a| (a.amplifier, a.ticks_left))
    }

    fn amp(&self, effect: Effect) -> Option<f32> {
        self.slots[effect.index()].map(|a| a.amplifier as f32 + 1.0)
    }

    /// Horizontal-movement multiplier from Speed / Slowness (multiplicative,
    /// clamped non-negative).
    pub fn move_multiplier(&self) -> f32 {
        let mut m = 1.0;
        if let Some(l) = self.amp(Effect::Speed) {
            m *= 1.0 + 0.20 * l;
        }
        if let Some(l) = self.amp(Effect::Slowness) {
            m *= (1.0 - 0.15 * l).max(0.0);
        }
        m
    }

    /// Flat melee-damage delta from Strength / Weakness, added to the weapon
    /// base before the attack-charge multiplier (as vanilla applies the
    /// attribute modifier).
    pub fn attack_damage_add(&self) -> f32 {
        let mut d = 0.0;
        if let Some(l) = self.amp(Effect::Strength) {
            d += 3.0 * l;
        }
        if let Some(l) = self.amp(Effect::Weakness) {
            d -= 4.0 * l;
        }
        d
    }

    /// Fire Resistance: no fire / lava damage, and lava never sets a burn.
    pub fn fire_immune(&self) -> bool {
        self.has(Effect::FireResistance)
    }

    /// Advance one tick: accumulate Regeneration / Poison, then count every
    /// timer down. Returns `(heal_hp, poison_hp)` to apply this tick (poison
    /// bypasses armour and i-frames but must never be lethal - the caller
    /// clamps it to leave the player at >= 1 HP).
    pub fn tick(&mut self) -> (f32, f32) {
        let mut heal = 0.0;
        let mut poison = 0.0;

        if let Some(a) = self.slots[Effect::Regeneration.index()] {
            let period = (50u32 >> a.amplifier.min(5)).max(1);
            self.regen_accum += 1.0 / period as f32;
        }
        if let Some(a) = self.slots[Effect::Poison.index()] {
            let period = (25u32 >> a.amplifier.min(4)).max(1);
            self.poison_accum += 1.0 / period as f32;
        }
        // Small epsilon so N accumulations of 1/N land on the exact tick
        // rather than one tick late from float rounding.
        while self.regen_accum >= 1.0 - 1e-6 {
            self.regen_accum -= 1.0;
            heal += 1.0;
        }
        while self.poison_accum >= 1.0 - 1e-6 {
            self.poison_accum -= 1.0;
            poison += 1.0;
        }

        for slot in &mut self.slots {
            if let Some(a) = slot {
                a.ticks_left = a.ticks_left.saturating_sub(1);
                if a.ticks_left == 0 {
                    *slot = None;
                }
            }
        }
        (heal, poison)
    }

    /// `[amplifier+1 if active else 0]` per effect, for the observation.
    pub fn levels(&self) -> [f32; EFFECT_COUNT] {
        let mut out = [0.0; EFFECT_COUNT];
        for (i, slot) in self.slots.iter().enumerate() {
            if let Some(a) = slot {
                out[i] = a.amplifier as f32 + 1.0;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEC: u32 = 20;

    #[test]
    fn stronger_amplifier_replaces_weaker_regardless_of_duration() {
        let mut e = StatusEffects::default();
        e.apply(Effect::Speed, 0, 100 * SEC);
        e.apply(Effect::Speed, 1, 1 * SEC); // shorter but stronger
        assert_eq!(e.get(Effect::Speed).unwrap().0, 1);
    }

    #[test]
    fn equal_amplifier_only_replaces_if_longer() {
        let mut e = StatusEffects::default();
        e.apply(Effect::Strength, 0, 10 * SEC);
        e.apply(Effect::Strength, 0, 5 * SEC);
        assert_eq!(e.get(Effect::Strength).unwrap().1, 10 * SEC);
        e.apply(Effect::Strength, 0, 30 * SEC);
        assert_eq!(e.get(Effect::Strength).unwrap().1, 30 * SEC);
    }

    #[test]
    fn speed_and_slowness_move_multiplier() {
        let mut e = StatusEffects::default();
        e.apply(Effect::Speed, 1, SEC); // Speed II -> +40%
        assert!((e.move_multiplier() - 1.4).abs() < 1e-6);
        e.apply(Effect::Slowness, 0, SEC); // -15%
        assert!((e.move_multiplier() - 1.4 * 0.85).abs() < 1e-6);
    }

    #[test]
    fn strength_and_weakness_damage_add() {
        let mut e = StatusEffects::default();
        e.apply(Effect::Strength, 1, SEC); // +6
        assert_eq!(e.attack_damage_add(), 6.0);
        e.apply(Effect::Weakness, 0, SEC); // -4
        assert_eq!(e.attack_damage_add(), 2.0);
    }

    #[test]
    fn regeneration_and_poison_tick_at_vanilla_periods() {
        let mut e = StatusEffects::default();
        e.apply(Effect::Regeneration, 0, 100 * SEC); // 1 HP / 50 ticks
        e.apply(Effect::Poison, 0, 100 * SEC); // 1 HP / 25 ticks
        let (mut heal, mut poison) = (0.0, 0.0);
        for _ in 0..50 {
            let (h, p) = e.tick();
            heal += h;
            poison += p;
        }
        assert_eq!(heal, 1.0);
        assert_eq!(poison, 2.0);
    }

    #[test]
    fn timed_effect_expires() {
        let mut e = StatusEffects::default();
        e.apply(Effect::Speed, 0, 3);
        for _ in 0..3 {
            e.tick();
        }
        assert!(!e.has(Effect::Speed));
    }

    #[test]
    fn instant_effects_are_never_stored() {
        let mut e = StatusEffects::default();
        e.apply(Effect::InstantHealth, 4, 100);
        e.apply(Effect::InstantDamage, 4, 100);
        assert!(!e.has(Effect::InstantHealth) && !e.has(Effect::InstantDamage));
        assert_eq!(Effect::InstantDamage.instant_magnitude(1), 2.0 * crate::config::cfg().combat.instant_damage_hp);
    }

    #[test]
    fn fire_resistance_flag() {
        let mut e = StatusEffects::default();
        assert!(!e.fire_immune());
        e.apply(Effect::FireResistance, 0, SEC);
        assert!(e.fire_immune());
    }
}
