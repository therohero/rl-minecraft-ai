package rl.minecraft.ai.client.combat;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.AbstractClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.math.Box;

/** Finds targetable players for {@link TargetLock}. */
public final class TargetSelector {
    private TargetSelector() {}

    /** Alive, non-spectating, not on {@code self}'s team, and not {@code self}. */
    public static boolean targetable(PlayerEntity self, Entity e) {
        if (e == self || !(e instanceof PlayerEntity p)) return false;
        return p.isAlive() && !p.isSpectator() && !self.isTeammate(p);
    }

    /**
     * Nearest targetable player to {@code self}, within {@code maxRange} blocks
     * ({@code <= 0} = unlimited). {@code null} when no one qualifies.
     */
    public static PlayerEntity nearest(ClientWorld world, PlayerEntity self, double maxRange) {
        if (world == null || self == null) return null;
        double limitSq = maxRange > 0 ? maxRange * maxRange : Double.MAX_VALUE;
        PlayerEntity best = null;
        double bestSq = limitSq;
        for (AbstractClientPlayerEntity p : world.getPlayers()) {
            if (!targetable(self, p)) continue;
            double d = self.squaredDistanceTo(p);
            if (d < bestSq) {
                bestSq = d;
                best = p;
            }
        }
        return best;
    }

    /** Nearest targetable player whose feet are inside {@code box}. */
    public static PlayerEntity nearestInBox(ClientWorld world, PlayerEntity self, Box box) {
        if (world == null || self == null || box == null) return null;
        PlayerEntity best = null;
        double bestSq = Double.MAX_VALUE;
        for (AbstractClientPlayerEntity p : world.getPlayers()) {
            if (!targetable(self, p) || !box.contains(p.getEntityPos())) continue;
            double d = self.squaredDistanceTo(p);
            if (d < bestSq) {
                bestSq = d;
                best = p;
            }
        }
        return best;
    }

    /** Targetable player whose name matches {@code name} (case-insensitive). */
    public static PlayerEntity byName(ClientWorld world, PlayerEntity self, String name) {
        if (world == null || self == null || name == null) return null;
        for (AbstractClientPlayerEntity p : world.getPlayers()) {
            if (targetable(self, p) && p.getName().getString().equalsIgnoreCase(name)) return p;
        }
        return null;
    }

    /** The player currently under the crosshair, if any (uses vanilla's pick result). */
    public static PlayerEntity underCrosshair(MinecraftClient mc) {
        if (mc.crosshairTarget instanceof EntityHitResult hit
            && mc.player != null && targetable(mc.player, hit.getEntity())) {
            return (PlayerEntity) hit.getEntity();
        }
        return null;
    }
}
