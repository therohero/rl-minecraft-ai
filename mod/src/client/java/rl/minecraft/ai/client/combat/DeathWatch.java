package rl.minecraft.ai.client.combat;

import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.GameMode;

import rl.minecraft.ai.client.RlConfig;

/**
 * Detects "we just died" from per-tick self state - including the cases where
 * a server plugin <em>cancels</em> the vanilla death (no death screen, HP
 * bounced back, and the player was silently teleported to spawn or dropped
 * into spectator). Getting this right is what lets {@link FightController}
 * cut one recorded episode per real fight on a live server.
 *
 * <p>A vanilla death (HP hits 0, death screen, respawn) is unambiguous. A
 * plugin-blocked death is inferred from a <em>one-tick</em> HP spike off a
 * near-dead floor that coincides with a hard context change (a long teleport,
 * a dimension change, or a flip into spectator/adventure) - none of which a
 * normal fight produces.
 *
 * <p>Once a death is reported the watch latches; {@link #rearm()} clears it
 * after {@link FightController} has decided whether to continue.
 */
public final class DeathWatch {
    private final RlConfig cfg;

    private boolean armed = true;
    private Double prevHp;
    private Vec3d prevPos;
    private GameMode prevMode;
    private String prevDim;

    public DeathWatch(RlConfig cfg) {
        this.cfg = cfg;
    }

    /**
     * Feed the current live self state. Returns a short human reason string
     * the tick a death is detected, otherwise {@code null}. Only fires once
     * per {@link #rearm()}.
     */
    public String poll(ClientPlayerEntity self, ClientWorld world, GameMode gameMode) {
        double hp = self.getHealth();
        double maxHp = self.getMaxHealth();
        Vec3d pos = self.getEntityPos();
        String dim = world.getRegistryKey().getValue().toString();

        Double lastHp = prevHp;
        Vec3d lastPos = prevPos;
        GameMode lastMode = prevMode;
        String lastDim = prevDim;

        prevHp = hp;
        prevPos = pos;
        prevMode = gameMode;
        prevDim = dim;

        if (!armed) return null;

        // --- unambiguous vanilla death ---
        if (!self.isAlive() || hp <= 0.5) {
            armed = false;
            return "hp reached 0";
        }

        if (lastHp == null) return null;

        boolean teleported = lastPos != null
            && lastPos.squaredDistanceTo(pos) > cfg.deathTeleportBlocks * cfg.deathTeleportBlocks;
        boolean dimChanged = lastDim != null && !lastDim.equals(dim);
        boolean toSpectator = gameMode != lastMode
            && (gameMode == GameMode.SPECTATOR || gameMode == GameMode.ADVENTURE);
        // Was on death's door last tick, at (near) full this tick - no natural
        // regen or single potion moves HP that far in one tick.
        boolean hpSpike = lastHp <= cfg.deathHpFloor && hp >= maxHp - 1.0 && (hp - lastHp) > 6.0;

        if (hpSpike && (teleported || dimChanged || toSpectator)) {
            armed = false;
            return "plugin-blocked death (hp restored + "
                + (dimChanged ? "dimension change" : teleported ? "teleport" : "spectator") + ")";
        }
        // Dropped into spectator straight off a low HP bar even without the
        // instant restore (some plugins set gamemode first).
        if (toSpectator && lastHp <= maxHp * 0.5) {
            armed = false;
            return "forced into " + gameMode.name().toLowerCase() + " at low hp";
        }
        return null;
    }

    /** A respawn / new player-entity instance while the fight is live is itself a death. */
    public String onPlayerEntitySwapped() {
        if (!armed) return null;
        armed = false;
        return "respawned";
    }

    /** Clear the latch so the next death can be detected again. */
    public void rearm() {
        armed = true;
    }

    /** Reset all history (call on fight start / after a legitimate context change). */
    public void reset() {
        armed = true;
        prevHp = null;
        prevPos = null;
        prevMode = null;
        prevDim = null;
    }
}
