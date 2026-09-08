//! Per-kit item loadouts and the `Item` enum the policy selects between.
//!
//! The held-item action is a categorical over `Item` *types* (not physical
//! hotbar slots), so a kit can carry as many items as it likes. A player
//! tracks a per-item count in `[u32; ITEM_COUNT]` indexed by `item as
//! usize`; selecting an item the kit doesn't have, or one whose count has
//! run out, is ignored (the held item stays put).

use crate::config::{cfg, Kit};

/// Every item type. The discriminant order is load-bearing: it's the
/// categorical action index and the observation's `held` id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Item {
    Empty = 0,
    Sword = 1,
    Axe = 2,
    Pickaxe = 3,
    Bow = 4,
    Crossbow = 5,
    Planks = 6,
    Cobweb = 7,
    WaterBucket = 8,
    LavaBucket = 9,
    GoldenApple = 10,
    GoldenHead = 11,
    // Splash potions. No kit carries these by default - they're opt-in via
    // `SimConfig::splash_potions`. Thrown with `use_item`; on impact they
    // apply a status effect in a radius (see `projectile`, `effects`).
    SplashHealing = 12,
    SplashHarming = 13,
    SplashPoison = 14,
    SplashSpeed = 15,
    SplashStrength = 16,
}

/// Number of `Item` variants - width of the per-player count array and the
/// `inventory` observation block.
pub const ITEM_COUNT: usize = 17;

/// Physical hotbar slots (vanilla: keys 1-9).
pub const HOTBAR_SLOTS: usize = 9;

/// Width of the held-slot categorical action. `0..HOTBAR_SLOTS` selects a
/// physical slot; `HOTBAR_SLOTS..HOTBAR_ACTION_DIM` is a vanilla number-key
/// "hotkey": drop `Item` id `(a - HOTBAR_SLOTS)` into the currently selected
/// slot (`HOTBAR_SLOTS` itself = `Item::Empty` = clear the slot). One hotbar
/// op per tick - a slot action either selects or hotkeys, never both.
pub const HOTBAR_ACTION_DIM: usize = HOTBAR_SLOTS + ITEM_COUNT;

impl Item {
    #[allow(dead_code)] // kept as the inverse of `index()`; handy in tests / future callers
    pub fn from_index(i: usize) -> Item {
        match i {
            1 => Item::Sword,
            2 => Item::Axe,
            3 => Item::Pickaxe,
            4 => Item::Bow,
            5 => Item::Crossbow,
            6 => Item::Planks,
            7 => Item::Cobweb,
            8 => Item::WaterBucket,
            9 => Item::LavaBucket,
            10 => Item::GoldenApple,
            11 => Item::GoldenHead,
            12 => Item::SplashHealing,
            13 => Item::SplashHarming,
            14 => Item::SplashPoison,
            15 => Item::SplashSpeed,
            16 => Item::SplashStrength,
            _ => Item::Empty,
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn is_ranged(self) -> bool {
        matches!(self, Item::Bow | Item::Crossbow)
    }

    pub fn is_placeable(self) -> bool {
        matches!(
            self,
            Item::Planks | Item::Cobweb | Item::WaterBucket | Item::LavaBucket
        )
    }

    pub fn is_food(self) -> bool {
        matches!(self, Item::GoldenApple | Item::GoldenHead)
    }

    pub fn is_splash_potion(self) -> bool {
        matches!(
            self,
            Item::SplashHealing
                | Item::SplashHarming
                | Item::SplashPoison
                | Item::SplashSpeed
                | Item::SplashStrength
        )
    }
}

/// A resolved kit: which items a player starts with (and how many), the
/// enchant levels, and the armor profile.
#[derive(Debug, Clone)]
pub struct Loadout {
    /// Starting count for every item type (0 = not in this kit).
    pub counts: [u32; ITEM_COUNT],
    /// Which `Item` sits in each physical hotbar slot (index 0 = key 1).
    /// `Item::Empty` is a real, selectable slot (empty hand). This is the
    /// vanilla-authentic "hotbar layout" a PvP player tunes; it's fixed per
    /// kit here rather than a client-side custom scan order.
    pub hotbar: [Item; HOTBAR_SLOTS],
    /// Hotbar slot the player has selected at spawn.
    pub default_slot: usize,
    /// Item the player holds at spawn (`hotbar[default_slot]`).
    pub default_held: Item,
    /// Whether a shield sits in the off-hand.
    pub has_shield: bool,
    /// Loose arrows for the bow (the crossbow is loaded from the same pool).
    pub arrows: u32,
    /// Crossbow starts loaded with one bolt.
    pub crossbow_preloaded: bool,
    pub sharpness: u32,
    pub power: u32,
    pub piercing: u32,
    /// Efficiency level on the pickaxe - speeds up mining placed blocks
    /// (`uhc` = 3). Mining-only, no combat effect.
    pub efficiency: u32,
    /// Full-set armor points / toughness / total Protection EPF.
    pub armor_points: f32,
    pub armor_toughness: f32,
    pub protection_epf: f32,
}

impl Loadout {
    fn empty() -> Self {
        Loadout {
            counts: [0; ITEM_COUNT],
            hotbar: [Item::Empty; HOTBAR_SLOTS],
            default_slot: 0,
            default_held: Item::Sword,
            has_shield: false,
            arrows: 0,
            crossbow_preloaded: false,
            sharpness: 0,
            power: 0,
            piercing: 0,
            efficiency: 0,
            // Full diamond set: 3+8+6+3 points, 2 toughness per piece.
            armor_points: 20.0,
            armor_toughness: 8.0,
            // 2x Protection IV + 2x Protection III.
            protection_epf: 2.0 * 4.0 + 2.0 * 3.0,
        }
    }
}

pub fn loadout(kit: Kit) -> Loadout {
    let mut l = Loadout::empty();
    match kit {
        // Standard 1v1: diamond sword + full diamond armour (2x Protection IV
        // + 2x Protection III, the `empty()` default). No shield.
        Kit::Sword => {
            l.counts[Item::Sword.index()] = 1;
            l.hotbar = [
                Item::Sword,
                Item::Empty,
                Item::Empty,
                Item::Empty,
                Item::Empty,
                Item::Empty,
                Item::Empty,
                Item::Empty,
                Item::Empty,
            ];
        }
        // Sword + shield + full diamond armour + axe + bow + crossbow + 6 arrows.
        Kit::Axe => {
            l.counts[Item::Sword.index()] = 1;
            l.counts[Item::Axe.index()] = 1;
            l.counts[Item::Bow.index()] = 1;
            l.counts[Item::Crossbow.index()] = 1;
            l.hotbar = [
                Item::Sword,
                Item::Axe,
                Item::Bow,
                Item::Crossbow,
                Item::Empty,
                Item::Empty,
                Item::Empty,
                Item::Empty,
                Item::Empty,
            ];
            l.has_shield = true;
            l.arrows = 6;
        }
        // Ultra Hardcore: Sharpness/Power/Piercing gear, placeables,
        // buckets, golden apples and heads. No natural regen (inherent).
        Kit::Uhc => {
            l.counts[Item::Sword.index()] = 1;
            l.counts[Item::Axe.index()] = 1;
            l.counts[Item::Pickaxe.index()] = 1;
            l.counts[Item::Bow.index()] = 1;
            l.counts[Item::Crossbow.index()] = 1;
            l.counts[Item::Planks.index()] = 128;
            l.counts[Item::Cobweb.index()] = 8;
            l.counts[Item::WaterBucket.index()] = 4;
            l.counts[Item::LavaBucket.index()] = 2;
            l.counts[Item::GoldenApple.index()] = 8;
            l.counts[Item::GoldenHead.index()] = 2;
            // Only 9 slots: the golden head and lava bucket don't get one -
            // they stay in `counts` (so eating a golden apple can still fall
            // back correctly) but aren't reachable by the policy. Documented
            // in the READMEs.
            l.hotbar = [
                Item::Sword,
                Item::Axe,
                Item::Pickaxe,
                Item::Bow,
                Item::Crossbow,
                Item::Planks,
                Item::Cobweb,
                Item::WaterBucket,
                Item::GoldenApple,
            ];
            l.has_shield = true;
            l.arrows = 16;
            l.crossbow_preloaded = true;
            l.sharpness = 3;
            l.power = 1;
            l.piercing = 1;
            l.efficiency = 3;
            // Protection III on all four pieces.
            l.protection_epf = 4.0 * 3.0;
        }
    }
    // Splash potions are never part of a kit's default loadout - a config
    // opts into them. They land in `counts` (reachable by the policy's
    // number-key "hotkey" action) but take no fixed hotbar slot.
    let sp = &cfg().splash_potions;
    l.counts[Item::SplashHealing.index()] = sp.healing;
    l.counts[Item::SplashHarming.index()] = sp.harming;
    l.counts[Item::SplashPoison.index()] = sp.poison;
    l.counts[Item::SplashSpeed.index()] = sp.speed;
    l.counts[Item::SplashStrength.index()] = sp.strength;

    l.default_held = l.hotbar[l.default_slot];
    l
}
