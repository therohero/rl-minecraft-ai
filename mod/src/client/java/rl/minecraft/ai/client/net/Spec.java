package rl.minecraft.ai.client.net;

import com.google.gson.JsonObject;

/**
 * The bits of {@code model/spec.json} (served at {@code GET /spec} by
 * {@code azalea-bot/inference_server.py}) the observation builder needs so it
 * normalises exactly the way the policy was trained. Missing fields fall back
 * to the sim defaults baked in here.
 */
public final class Spec {
    public final double maxHp;
    public final double matchTimeSeconds;
    public final double arenaRadius;
    public final double terrainMaxAmplitude;
    public final double maxPingMs;
    public final int maxObservedEnemies;
    public final int maxObservedTeammates;
    public final int maxObservedProjectiles;
    public final int blockViewSize;
    public final int hotbarSlots;
    public final double bowMaxDrawSeconds;
    public final double hurtInvulnerabilitySeconds;
    public final double binaryActionThreshold;

    public static final Spec DEFAULT = new Spec(new JsonObject());

    public Spec(JsonObject root) {
        JsonObject sc = obj(root, "sim_constants");
        JsonObject cc = obj(root, "combat_constants");
        this.maxHp = d(sc, "max_hp", 20.0);
        this.matchTimeSeconds = d(sc, "match_time_seconds", 90.0);
        this.arenaRadius = d(sc, "arena_radius", 12.0);
        this.terrainMaxAmplitude = d(sc, "terrain_max_amplitude", 3.0);
        this.maxPingMs = d(sc, "max_ping_ms", 100.0);
        this.maxObservedEnemies = i(root, "max_observed_enemies", (int) d(sc, "max_observed_enemies", 3));
        this.maxObservedTeammates = i(root, "max_observed_teammates", (int) d(sc, "max_observed_teammates", 2));
        this.maxObservedProjectiles = i(root, "max_observed_projectiles", (int) d(sc, "max_observed_projectiles", 2));
        this.blockViewSize = i(root, "block_view_size", (int) d(sc, "block_view_size", 5));
        this.hotbarSlots = i(root, "hotbar_slots", 9);
        this.bowMaxDrawSeconds = d(cc, "bow_max_draw_seconds", 1.0);
        this.hurtInvulnerabilitySeconds = d(cc, "hurt_invulnerability_seconds", 1.0);
        this.binaryActionThreshold = d(root, "binary_action_threshold", 0.5);
    }

    private static JsonObject obj(JsonObject o, String k) {
        return o.has(k) && o.get(k).isJsonObject() ? o.getAsJsonObject(k) : new JsonObject();
    }

    private static double d(JsonObject o, String k, double def) {
        try {
            return o.has(k) ? o.get(k).getAsDouble() : def;
        } catch (RuntimeException e) {
            return def;
        }
    }

    private static int i(JsonObject o, String k, int def) {
        try {
            return o.has(k) ? o.get(k).getAsInt() : def;
        } catch (RuntimeException e) {
            return def;
        }
    }
}
