package rl.minecraft.ai.client.debug;

import com.mojang.authlib.GameProfile;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.network.OtherClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.Entity;
import net.minecraft.text.Text;

import rl.minecraft.ai.client.RlMinecraftAiClient;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.function.Consumer;

/**
 * Client-side stand-in opponents for solo testing. A {@link OtherClientPlayerEntity}
 * added straight into the {@link ClientWorld} shows up in
 * {@code world.getPlayers()}, so {@code TargetSelector}, the observation
 * {@code enemies} rows and the aim pipeline all treat it as a real opponent -
 * enough to exercise obs -> target -> rotation end to end without a second
 * player.
 *
 * <p><b>Limitation:</b> the server knows nothing about these entities, so
 * attacks against them deal no damage and never produce a kill / {@code win}
 * outcome. Use {@code /rldebug dummy mob} (a real summoned mob) plus
 * {@code /damage} / {@code /kill} on yourself for the damage / death / episode
 * boundary paths.
 */
public final class Dummy {
    private Dummy() {}

    private static final int ID_BASE = 900_000;
    private static final List<Integer> SPAWNED = new ArrayList<>();

    /** Spawn {@code count} fake players {@code dist} blocks ahead of the player. */
    public static void spawnPlayers(MinecraftClient mc, int count, double dist, Consumer<Text> fb) {
        ClientPlayerEntity self = mc.player;
        ClientWorld world = mc.world;
        if (self == null || world == null) {
            fb.accept(Text.literal("§cnot in a world"));
            return;
        }
        double yawRad = Math.toRadians(self.getYaw());
        double fx = -Math.sin(yawRad);
        double fz = Math.cos(yawRad);
        int spawned = 0;
        for (int i = 0; i < count; i++) {
            int id = ID_BASE + SPAWNED.size();
            try {
                GameProfile profile = new GameProfile(UUID.randomUUID(), "rl_dummy" + id);
                OtherClientPlayerEntity e = new OtherClientPlayerEntity(world, profile);
                e.setId(id);
                double off = 1.0 + i;                       // fan them out along the facing line
                double x = self.getX() + fx * (dist + off);
                double z = self.getZ() + fz * (dist + off);
                e.refreshPositionAndAngles(x, self.getY(), z, (self.getYaw() + 180f) % 360f, 0f);
                e.setHeadYaw(e.getYaw());
                e.setBodyYaw(e.getYaw());
                world.addEntity(e);
                SPAWNED.add(id);
                spawned++;
            } catch (Throwable t) {
                RlMinecraftAiClient.LOGGER.warn("dummy spawn failed", t);
                fb.accept(Text.literal("§cdummy spawn failed: " + t.getMessage()));
                break;
            }
        }
        fb.accept(Text.literal("§aspawned §f" + spawned + "§a client-side dummy player(s) §7(no "
            + "server entity - damage/kills won't register; use 'dummy mob' for those)"));
    }

    /** Summon a real server-side mob {@code dist} blocks ahead (damage / knockback paths). */
    public static void spawnMob(MinecraftClient mc, String type, double dist, Consumer<Text> fb) {
        ClientPlayerEntity self = mc.player;
        if (self == null) {
            fb.accept(Text.literal("§cnot in a world"));
            return;
        }
        String id = type == null || type.isBlank() ? "minecraft:zombie" : type;
        if (!id.contains(":")) id = "minecraft:" + id;
        double yawRad = Math.toRadians(self.getYaw());
        double x = self.getX() - Math.sin(yawRad) * dist;
        double z = self.getZ() + Math.cos(yawRad) * dist;
        String cmd = String.format(java.util.Locale.ROOT,
            "summon %s %.2f %.2f %.2f {PersistenceRequired:1b,NoAI:1b}", id, x, self.getY(), z);
        ServerCmd.run(mc, cmd, fb);
        fb.accept(Text.literal("§asummoned §f" + id + "§a (NoAI) §7" + dist + "m ahead"));
    }

    /** Remove all client-side dummy players (and mobs tagged by 'dummy mob'). */
    public static void clear(MinecraftClient mc, Consumer<Text> fb) {
        ClientWorld world = mc.world;
        int n = 0;
        if (world != null) {
            for (int id : SPAWNED) {
                Entity e = world.getEntityById(id);
                if (e != null) {
                    world.removeEntity(id, Entity.RemovalReason.DISCARDED);
                    n++;
                }
            }
        }
        SPAWNED.clear();
        // Best-effort server-side cleanup of summoned mobs standing near the player.
        ServerCmd.run(mc, "kill @e[type=!player,distance=..40]", null);
        fb.accept(Text.literal("§eremoved §f" + n + "§e client dummies + killed nearby mobs"));
    }

    public static int count() {
        return SPAWNED.size();
    }
}
