package rl.minecraft.ai.client.combat;

import net.minecraft.component.DataComponentTypes;
import net.minecraft.component.type.PotionContentsComponent;
import net.minecraft.entity.effect.StatusEffects;
import net.minecraft.item.ItemStack;
import net.minecraft.item.Items;

/**
 * The item vocabulary the trained policy knows, mirroring
 * {@code training/sim/src/kit.rs}'s {@code Item} enum. The ordinal (0 =
 * EMPTY) is the on-wire id the observation uses for {@code self_held} and the
 * {@code hotbar} layout block, and {@code ordinal() - 1} indexes the
 * {@code inventory} count block (which omits EMPTY).
 */
public enum KitItem {
    EMPTY,
    SWORD,
    AXE,
    PICKAXE,
    BOW,
    CROSSBOW,
    PLANKS,
    COBWEB,
    WATER_BUCKET,
    LAVA_BUCKET,
    GOLDEN_APPLE,
    GOLDEN_HEAD,
    SPLASH_HEALING,
    SPLASH_HARMING,
    SPLASH_POISON,
    SPLASH_SPEED,
    SPLASH_STRENGTH;

    /** Number of real inventory item categories (everything except EMPTY). */
    public static final int COUNT = values().length - 1;

    /** Maps a live Minecraft stack to the kit's item, or EMPTY for anything unmodelled. */
    public static KitItem of(ItemStack stack) {
        if (stack == null || stack.isEmpty()) return EMPTY;
        if (stack.isOf(Items.DIAMOND_SWORD) || stack.isOf(Items.NETHERITE_SWORD)
            || stack.isOf(Items.IRON_SWORD) || stack.isOf(Items.STONE_SWORD)
            || stack.isOf(Items.GOLDEN_SWORD) || stack.isOf(Items.WOODEN_SWORD)) return SWORD;
        if (stack.isOf(Items.DIAMOND_AXE) || stack.isOf(Items.NETHERITE_AXE)
            || stack.isOf(Items.IRON_AXE) || stack.isOf(Items.STONE_AXE)
            || stack.isOf(Items.GOLDEN_AXE) || stack.isOf(Items.WOODEN_AXE)) return AXE;
        if (stack.isOf(Items.DIAMOND_PICKAXE) || stack.isOf(Items.NETHERITE_PICKAXE)
            || stack.isOf(Items.IRON_PICKAXE) || stack.isOf(Items.STONE_PICKAXE)
            || stack.isOf(Items.GOLDEN_PICKAXE) || stack.isOf(Items.WOODEN_PICKAXE)) return PICKAXE;
        if (stack.isOf(Items.BOW)) return BOW;
        if (stack.isOf(Items.CROSSBOW)) return CROSSBOW;
        if (stack.isOf(Items.OAK_PLANKS) || stack.isOf(Items.SPRUCE_PLANKS)
            || stack.isOf(Items.BIRCH_PLANKS) || stack.isOf(Items.JUNGLE_PLANKS)
            || stack.isOf(Items.ACACIA_PLANKS) || stack.isOf(Items.DARK_OAK_PLANKS)
            || stack.isOf(Items.MANGROVE_PLANKS) || stack.isOf(Items.CHERRY_PLANKS)
            || stack.isOf(Items.CRIMSON_PLANKS) || stack.isOf(Items.WARPED_PLANKS)) return PLANKS;
        if (stack.isOf(Items.COBWEB)) return COBWEB;
        if (stack.isOf(Items.WATER_BUCKET)) return WATER_BUCKET;
        if (stack.isOf(Items.LAVA_BUCKET)) return LAVA_BUCKET;
        if (stack.isOf(Items.GOLDEN_APPLE)) return GOLDEN_APPLE;
        if (stack.isOf(Items.ENCHANTED_GOLDEN_APPLE)) return GOLDEN_HEAD;
        if (stack.isOf(Items.SPLASH_POTION)) return splashKind(stack);
        return EMPTY;
    }

    /** The splash potion this stack is, by its primary effect, or EMPTY. */
    private static KitItem splashKind(ItemStack stack) {
        PotionContentsComponent contents = stack.get(DataComponentTypes.POTION_CONTENTS);
        if (contents == null) return EMPTY;
        for (var e : contents.getEffects()) {
            var t = e.getEffectType();
            if (t == StatusEffects.INSTANT_HEALTH) return SPLASH_HEALING;
            if (t == StatusEffects.INSTANT_DAMAGE) return SPLASH_HARMING;
            if (t == StatusEffects.POISON) return SPLASH_POISON;
            if (t == StatusEffects.SPEED) return SPLASH_SPEED;
            if (t == StatusEffects.STRENGTH) return SPLASH_STRENGTH;
        }
        return EMPTY;
    }

    public boolean isFood() {
        return this == GOLDEN_APPLE || this == GOLDEN_HEAD;
    }

    /** A held item whose right-click raises a guard (closest live proxy for a raised shield). */
    public boolean isShieldish() {
        return this == SWORD || this == AXE || this == EMPTY;
    }

    public boolean isRanged() {
        return this == BOW || this == CROSSBOW;
    }

    public static boolean isArrow(ItemStack stack) {
        return stack.isOf(Items.ARROW) || stack.isOf(Items.SPECTRAL_ARROW) || stack.isOf(Items.TIPPED_ARROW);
    }
}
