package rl.minecraft.ai.client.combat;

import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Box;

import rl.minecraft.ai.client.RlConfig;

/**
 * Who the bot is allowed to fight. On a populated server "swing at the
 * nearest player" hits bystanders and gets you banned, so the default is
 * {@link Kind#PASSIVE}: attack no one until someone hits you (then only them)
 * or you name a target explicitly.
 *
 * <ul>
 *   <li>{@code PASSIVE}  - fight back only. The current attacker (fed in by
 *       {@link FightController}) is the target; nobody = no target.</li>
 *   <li>{@code NAME}     - a named player; no auto-reacquire when they die/leave.</li>
 *   <li>{@code ENTITY}   - one pinned entity id (from {@code /fight target
 *       look} or {@code nearest}); also no reacquire.</li>
 *   <li>{@code REGION}   - anyone inside an axis-aligned box you set with two
 *       {@code /fight target region} calls (arena mode).</li>
 *   <li>{@code AUTO}     - continuous nearest player, the old behaviour;
 *       opt-in only.</li>
 * </ul>
 */
public final class TargetLock {
    public enum Kind { PASSIVE, NAME, ENTITY, REGION, AUTO }

    private Kind kind = Kind.PASSIVE;
    private String name;
    private int entityId = -1;
    private Box region;
    private BlockPos regionCorner;

    public Kind kind() {
        return kind;
    }

    public void passive() {
        kind = Kind.PASSIVE;
        clearPins();
    }

    public void byName(String n) {
        kind = Kind.NAME;
        clearPins();
        name = n;
    }

    public void byEntity(int id) {
        kind = Kind.ENTITY;
        clearPins();
        entityId = id;
    }

    public void auto() {
        kind = Kind.AUTO;
        clearPins();
    }

    /**
     * Feed a region corner. The first call arms it, the second completes the
     * box and switches to {@link Kind#REGION}. Returns {@code true} once the
     * region is active.
     */
    public boolean regionCorner(BlockPos p) {
        if (regionCorner == null) {
            regionCorner = p;
            return false;
        }
        region = new Box(regionCorner).union(new Box(p)).expand(1.0);
        regionCorner = null;
        kind = Kind.REGION;
        entityId = -1;
        name = null;
        return true;
    }

    public boolean awaitingRegionCorner() {
        return regionCorner != null;
    }

    private void clearPins() {
        name = null;
        entityId = -1;
        region = null;
        regionCorner = null;
    }

    /**
     * The player to fight this tick, or {@code null}. {@code attacker} is
     * whoever last damaged us (only used in {@link Kind#PASSIVE}).
     */
    public PlayerEntity resolve(ClientWorld world, ClientPlayerEntity self, RlConfig cfg, PlayerEntity attacker) {
        return switch (kind) {
            case PASSIVE -> TargetSelector.targetable(self, attacker) ? attacker : null;
            case NAME -> TargetSelector.byName(world, self, name);
            case ENTITY -> {
                Entity e = world.getEntityById(entityId);
                yield TargetSelector.targetable(self, e) ? (PlayerEntity) e : null;
            }
            case REGION -> TargetSelector.nearestInBox(world, self, region);
            case AUTO -> TargetSelector.nearest(world, self, cfg.engageRange);
        };
    }

    public String describe() {
        return switch (kind) {
            case PASSIVE -> "passive (fight back only)";
            case NAME -> "name=" + name;
            case ENTITY -> "pinned entity #" + entityId;
            case REGION -> "region " + (region == null ? "?" : fmt(region));
            case AUTO -> "auto (nearest player)";
        };
    }

    private static String fmt(Box b) {
        return String.format("[%.0f,%.0f,%.0f]..[%.0f,%.0f,%.0f]",
            b.minX, b.minY, b.minZ, b.maxX, b.maxY, b.maxZ);
    }
}
