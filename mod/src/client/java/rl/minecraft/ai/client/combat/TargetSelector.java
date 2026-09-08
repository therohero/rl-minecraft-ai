package rl.minecraft.ai.client.combat;

import net.minecraft.client.network.AbstractClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.player.PlayerEntity;

/** Picks the nearest live, targetable player to a reference player. */
public final class TargetSelector {
    private TargetSelector() {}

    /**
     * Nearest other player to {@code self}: alive, not spectating, not on
     * {@code self}'s scoreboard team. Returns {@code null} when no one
     * qualifies (or the world is gone).
     */
    public static PlayerEntity nearest(ClientWorld world, PlayerEntity self) {
        if (world == null || self == null) return null;
        PlayerEntity best = null;
        double bestSq = Double.MAX_VALUE;
        for (AbstractClientPlayerEntity p : world.getPlayers()) {
            if (p == self || !p.isAlive() || p.isSpectator()) continue;
            if (self.isTeammate(p)) continue;
            double d = self.squaredDistanceTo(p);
            if (d < bestSq) {
                bestSq = d;
                best = p;
            }
        }
        return best;
    }
}
