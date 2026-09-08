package rl.minecraft.ai.client;

import net.fabricmc.loader.api.FabricLoader;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

/**
 * Runtime knobs for the mod, resolved once at startup. Every value can be
 * overridden from {@code <game dir>/config/rl-minecraft-ai.properties} or a
 * matching {@code -Drl.minecraft.ai.<key>} JVM system property (the property
 * wins). Nothing here needs a restart-safe default beyond the constants below.
 *
 * <pre>
 * inference_url = http://127.0.0.1:8800/act   # POST endpoint of azalea-bot/inference_server.py
 * dataset_dir   = &lt;game dir&gt;/rl-datasets       # where /fight train writes its JSONL episodes
 * max_yaw_deg_per_tick   = 80                 # client-side aim-rate clamp (anti-cheat friendliness)
 * max_pitch_deg_per_tick = 60
 * reach                  = 3.0                # only swing at a target within this many blocks
 * rotation_smoothing     = 0.7                # 1.0 = no smoothing, lower = softer accel onto target
 * aim_jitter_deg         = 0.4                # per-tick gaussian "hand tremor" mixed into rotation
 * aim_settle_deg         = 50                 # hold the attack one tick after a turn bigger than this
 * min_sneak_hold_ticks   = 3                  # debounce: min ticks a sneak state is held before flipping
 * require_line_of_sight    = true              # don't attack through a wall even if in reach cone
 * hotkey_swap_min_gap_ticks = 10              # min ticks between buried-item inventory swaps
 * </pre>
 */
public final class RlConfig {
    public final String inferenceUrl;
    public final String specUrl;
    public final Path datasetDir;
    public final double maxYawDegPerTick;
    public final double maxPitchDegPerTick;
    public final double reach;
    public final double rotationSmoothing;
    public final double aimJitterDeg;
    public final double aimSettleDeg;
    public final int minSneakHoldTicks;
    public final boolean requireLineOfSight;
    public final int hotkeySwapMinGapTicks;

    private RlConfig(Properties p, Path gameDir) {
        this.inferenceUrl = get(p, "inference_url", "http://127.0.0.1:8800/act");
        // /spec lives next to /act on the same server.
        this.specUrl = inferenceUrl.endsWith("/act")
            ? inferenceUrl.substring(0, inferenceUrl.length() - 4) + "/spec"
            : inferenceUrl + "/spec";
        String ds = get(p, "dataset_dir", "");
        this.datasetDir = ds.isBlank() ? gameDir.resolve("rl-datasets") : Path.of(ds);
        this.maxYawDegPerTick = getDouble(p, "max_yaw_deg_per_tick", 80.0);
        this.maxPitchDegPerTick = getDouble(p, "max_pitch_deg_per_tick", 60.0);
        this.reach = getDouble(p, "reach", 3.0);
        this.rotationSmoothing = getDouble(p, "rotation_smoothing", 0.7);
        this.aimJitterDeg = getDouble(p, "aim_jitter_deg", 0.4);
        this.aimSettleDeg = getDouble(p, "aim_settle_deg", 50.0);
        this.minSneakHoldTicks = (int) getDouble(p, "min_sneak_hold_ticks", 3);
        this.requireLineOfSight = getBoolean(p, "require_line_of_sight", true);
        this.hotkeySwapMinGapTicks = (int) getDouble(p, "hotkey_swap_min_gap_ticks", 10);
    }

    public static RlConfig load() {
        Path gameDir = FabricLoader.getInstance().getGameDir();
        Path file = FabricLoader.getInstance().getConfigDir().resolve("rl-minecraft-ai.properties");
        Properties p = new Properties();
        if (Files.isReadable(file)) {
            try (var in = Files.newInputStream(file)) {
                p.load(in);
            } catch (Exception e) {
                RlMinecraftAiClient.LOGGER.warn("could not read {} - using defaults", file, e);
            }
        }
        return new RlConfig(p, gameDir);
    }

    private static String get(Properties p, String key, String def) {
        String sys = System.getProperty("rl.minecraft.ai." + key);
        if (sys != null) return sys;
        return p.getProperty(key, def);
    }

    private static double getDouble(Properties p, String key, double def) {
        try {
            return Double.parseDouble(get(p, key, Double.toString(def)));
        } catch (NumberFormatException e) {
            return def;
        }
    }

    private static boolean getBoolean(Properties p, String key, boolean def) {
        String v = get(p, key, Boolean.toString(def)).trim().toLowerCase();
        return switch (v) {
            case "1", "true", "yes", "on" -> true;
            case "0", "false", "no", "off" -> false;
            default -> def;
        };
    }
}
