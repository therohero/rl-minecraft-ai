package rl.minecraft.ai.client.obs;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.entity.Entity;
import net.minecraft.screen.slot.SlotActionType;
import net.minecraft.util.Hand;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.RaycastContext;

import rl.minecraft.ai.client.RlConfig;
import rl.minecraft.ai.client.combat.ClientGuard;
import rl.minecraft.ai.client.combat.KitItem;
import rl.minecraft.ai.client.net.Action;

/**
 * Turns one decoded {@link Action} into real client input for the current
 * tick: movement keys, a humanised look (via {@link ClientGuard}), an attack
 * on the chosen target, use-item, and hotbar selection (including swapping a
 * buried kit item into the hotbar). Mirrors
 * {@code azalea_bot/src/main.rs::apply_action} plus its {@code guard} module
 * - lighter in places a *real* vanilla client already enforces for free
 * (jump only works on the ground, sprint cost is the real hunger bar, click
 * rate is the real attack cooldown), and adds what a real client can still
 * fake: rotation naturalism and hit legality (reach, facing cone, line of
 * sight, aim-settle). Use only on servers you are authorised to run bots on.
 */
public final class ActionApplier {
    private static final double MOVE_DEAD_ZONE = 0.25;
    private static final double FACING_DOT = Math.cos(Math.toRadians(14.0));

    private ActionApplier() {}

    /**
     * Applies one action. Returns {@code true} if the held item's identity
     * changed this tick (a hotbar re-select or a buried-item swap) - the
     * caller folds that into the reconstructed {@code self_swap_lockout}
     * observation timer, mirroring the modern input-order lock-out.
     */
    public static boolean apply(MinecraftClient client, Action a, Entity target, RlConfig cfg,
                                 ClientGuard guard, long tick) {
        ClientPlayerEntity self = client.player;
        if (self == null) return false;

        // --- look: plan this tick's virtual-mouse turn (applied per-frame by the guard) ---
        guard.planLook(self, Math.toDegrees(a.yawDelta()), Math.toDegrees(a.pitchDelta()));

        // --- movement keys ---
        boolean forward = a.moveZ() > MOVE_DEAD_ZONE;
        boolean back = a.moveZ() < -MOVE_DEAD_ZONE;
        boolean right = a.moveX() > MOVE_DEAD_ZONE;
        boolean left = a.moveX() < -MOVE_DEAD_ZONE;
        press(client.options.forwardKey, forward);
        press(client.options.backKey, back);
        press(client.options.leftKey, left);
        press(client.options.rightKey, right);
        press(client.options.jumpKey, a.jump());
        boolean sneak = guard.resolveSneak(a.sneak());
        press(client.options.sneakKey, sneak);

        boolean canSprint = a.sprint() && forward && !sneak
            && self.getHungerManager().getFoodLevel() > 6;
        press(client.options.sprintKey, canSprint);
        self.setSprinting(canSprint);

        // --- hotbar selection (including a buried-item swap) ---
        boolean hotbarChanged = applyHeldSlot(client, self, a.heldSlot(), guard, tick);

        // --- attack vs use (mutually exclusive, like the sim) ---
        boolean attacked = false;
        if (a.attack() && target != null && client.interactionManager != null && guard.aimSettled()) {
            double reachSq = cfg.reach * cfg.reach;
            if (self.squaredDistanceTo(target) <= reachSq
                && self.getAttackCooldownProgress(0.5f) >= 1.0f
                && facing(self, target)
                && (!cfg.requireLineOfSight || hasLineOfSight(client, self, target))
                && guard.tryAttack()) {
                client.interactionManager.attackEntity(self, target);
                self.swingHand(Hand.MAIN_HAND);
                attacked = true;
            }
        }
        press(client.options.attackKey, false);
        press(client.options.useKey, !attacked && a.useItem());
        return hotbarChanged;
    }

    /** Release every input the mod drives - call when the controller stops. */
    public static void releaseAll(MinecraftClient client) {
        if (client.options == null) return;
        for (KeyBinding k : new KeyBinding[] {
            client.options.forwardKey, client.options.backKey, client.options.leftKey,
            client.options.rightKey, client.options.jumpKey, client.options.sneakKey,
            client.options.sprintKey, client.options.attackKey, client.options.useKey
        }) {
            press(k, false);
        }
        if (client.player != null) client.player.setSprinting(false);
    }

    /**
     * Resolves the policy's {@code held_slot} action, mirroring
     * {@code azalea_bot/src/main.rs::apply_held_slot}: {@code 0..hotbarSlots}
     * just selects that hotbar slot; higher values are a {@code KitItem}
     * hotkey, resolved the least-invasive way possible - select it if it's
     * already on the hotbar, otherwise swap it up from the main inventory
     * (rate-limited by {@code hotkey_swap_min_gap_ticks}, and only while no
     * other container screen is open). Returns whether the held item's
     * identity changed this tick.
     */
    private static boolean applyHeldSlot(MinecraftClient client, ClientPlayerEntity self, int heldSlot,
                                          ClientGuard guard, long tick) {
        int hotbarSlots = 9;
        if (heldSlot < 0) return false;
        var inv = self.getInventory();
        int selectedBefore = inv.getSelectedSlot();

        if (heldSlot < hotbarSlots) {
            if (selectedBefore != heldSlot) inv.setSelectedSlot(heldSlot);
            return selectedBefore != heldSlot;
        }

        int wantId = heldSlot - hotbarSlots; // KitItem ordinal
        if (wantId <= 0 || wantId >= KitItem.values().length) return false;
        KitItem want = KitItem.values()[wantId];

        for (int i = 0; i < hotbarSlots; i++) {
            if (KitItem.of(inv.getStack(i)) == want) {
                if (selectedBefore != i) inv.setSelectedSlot(i);
                return selectedBefore != i;
            }
        }

        // Buried in the main inventory: swap it into the currently selected
        // hotbar slot via the player's own (always-open, syncId 0) screen
        // handler - the same trick a vanilla client uses to drag an item
        // onto a number key, just without an InventoryScreen actually shown.
        if (self.currentScreenHandler != self.playerScreenHandler) return false; // some other GUI is open
        if (!guard.tryHotkeySwap(tick)) return false;
        // Main-inventory PlayerInventory indices (9..36) line up 1:1 with
        // PlayerScreenHandler slot ids; the hotbar range was already checked above.
        for (int i = hotbarSlots; i < inv.size(); i++) {
            if (KitItem.of(inv.getStack(i)) != want) continue;
            client.interactionManager.clickSlot(self.playerScreenHandler.syncId, i,
                selectedBefore, SlotActionType.SWAP, self);
            return true;
        }
        return false;
    }

    private static boolean facing(ClientPlayerEntity self, Entity target) {
        Vec3d look = self.getRotationVec(1.0f);
        Vec3d to = target.getEyePos().subtract(self.getEyePos());
        if (to.lengthSquared() < 1.0e-6) return true;
        return look.normalize().dotProduct(to.normalize()) >= FACING_DOT;
    }

    /** Blocks-only line of sight, eye to eye - vanilla attacks don't see through walls either. */
    private static boolean hasLineOfSight(MinecraftClient client, ClientPlayerEntity self, Entity target) {
        if (client.world == null) return true;
        BlockHitResult hit = client.world.raycast(new RaycastContext(
            self.getEyePos(), target.getEyePos(),
            RaycastContext.ShapeType.COLLIDER, RaycastContext.FluidHandling.NONE, self));
        return hit.getType() == HitResult.Type.MISS;
    }

    private static void press(KeyBinding key, boolean pressed) {
        if (key != null) key.setPressed(pressed);
    }
}
