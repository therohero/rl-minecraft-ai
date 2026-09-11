package rl.minecraft.ai.client.debug;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;

import rl.minecraft.ai.client.RlMinecraftAiClient;
import rl.minecraft.ai.client.combat.ClientGuard;
import rl.minecraft.ai.client.combat.FightController;
import rl.minecraft.ai.client.net.Action;
import rl.minecraft.ai.client.net.InferenceClient;

import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Structured, non-blocking debug log for the mod. One JSON object per line
 * (JSONL) written to {@code <game dir>/logs/rl-debug-<stamp>.jsonl}:
 *
 * <ul>
 *   <li>{@code basic} - one {@code tick} record per active fight tick: mode,
 *       kit, target, pause state, inference latency, the applied {@link Action},
 *       and a {@link ClientGuard} snapshot (cps, raw vs smoothed vs applied
 *       rotation, quantum).</li>
 *   <li>{@code verbose} - as basic, plus the full {@code obs} object, i.e. the
 *       exact body POSTed to {@code /act}.</li>
 * </ul>
 *
 * Lines are enqueued on the game thread and flushed by a daemon thread, so the
 * tick loop never blocks on IO. Dev-only (see {@link DebugHarness}).
 */
public final class DebugLog {
    public enum Level { OFF, BASIC, VERBOSE }

    private static final Gson GSON = new Gson();
    private static final DateTimeFormatter STAMP =
        DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss").withZone(ZoneId.systemDefault());

    private final Path file;
    private final ConcurrentLinkedQueue<String> queue = new ConcurrentLinkedQueue<>();
    private final Thread writerThread;
    private volatile boolean running = true;
    private volatile Level level;
    private final AtomicLong written = new AtomicLong();

    public DebugLog(Level initial) {
        this.level = initial;
        Path logs = FabricLoader.getInstance().getGameDir().resolve("logs");
        this.file = logs.resolve("rl-debug-" + STAMP.format(Instant.now()) + ".jsonl");
        try {
            Files.createDirectories(logs);
        } catch (Exception e) {
            RlMinecraftAiClient.LOGGER.warn("debug log: could not create {}", logs, e);
        }
        this.writerThread = new Thread(this::drainLoop, "rl-debug-log");
        this.writerThread.setDaemon(true);
        this.writerThread.start();
        event("log_open", o -> {
            o.addProperty("level", level.name().toLowerCase());
            o.addProperty("file", file.toString());
        });
        RlMinecraftAiClient.LOGGER.info("debug log -> {} (level={})", file, level);
    }

    public Level level() {
        return level;
    }

    public void setLevel(Level l) {
        this.level = l;
        event("log_level", o -> o.addProperty("level", l.name().toLowerCase()));
    }

    public Path path() {
        return file;
    }

    public long linesWritten() {
        return written.get();
    }

    /** Append a free-form event record ({@code type} + whatever {@code fill} adds). */
    public void event(String type, java.util.function.Consumer<JsonObject> fill) {
        JsonObject o = new JsonObject();
        o.addProperty("ts", Instant.now().toString());
        o.addProperty("kind", "event");
        o.addProperty("type", type);
        if (fill != null) fill.accept(o);
        queue.add(GSON.toJson(o));
    }

    /** One active-tick record. {@code action} may be {@code null} (no fresh reply). */
    public void onTick(FightController controller, JsonObject obs, Action action) {
        Level lvl = level;
        if (lvl == Level.OFF) return;
        MinecraftClient mc = MinecraftClient.getInstance();
        ClientPlayerEntity self = mc.player;

        JsonObject o = new JsonObject();
        o.addProperty("kind", "tick");
        o.addProperty("t", controller.tick());
        o.addProperty("mode", controller.mode().name());
        o.addProperty("kit", controller.kitId());
        o.addProperty("target", controller.targetLabel());
        o.addProperty("paused", controller.paused());
        if (controller.pauseReason() != null) o.addProperty("pause_reason", controller.pauseReason());
        o.addProperty("episode", controller.episodeIndex());
        o.addProperty("recorded_ticks", controller.recordedTicks());
        o.addProperty("outcome", controller.outcome());

        InferenceClient inf = controller.inference();
        if (inf != null) {
            o.addProperty("seen_server", inf.seenServer());
            o.addProperty("latency_ms", inf.latencyMs());
            if (inf.lastError() != null && !inf.lastError().isEmpty()) {
                o.addProperty("last_error", inf.lastError());
            }
        }

        if (action != null) o.add("action", GSON.toJsonTree(action));

        ClientGuard g = controller.guard();
        if (g != null) {
            JsonObject gj = new JsonObject();
            gj.addProperty("cps", g.cps());
            gj.addProperty("quantum_deg", g.rotationQuantumDeg());
            gj.addProperty("last_yaw_turn_deg", g.lastYawTurnDeg());
            gj.addProperty("aim_settled", g.aimSettled());
            double[] p = g.lastPlan();
            gj.addProperty("raw_yaw_deg", p[0]);
            gj.addProperty("raw_pitch_deg", p[1]);
            gj.addProperty("smoothed_yaw_rate", p[2]);
            gj.addProperty("smoothed_pitch_rate", p[3]);
            gj.addProperty("mouse_vel_yaw", p[4]);
            gj.addProperty("mouse_vel_pitch", p[5]);
            gj.addProperty("plan_counts_x", p[6]);
            gj.addProperty("plan_counts_y", p[7]);
            gj.addProperty("applied_yaw_deg", p[6] * g.rotationQuantumDeg());
            gj.addProperty("applied_pitch_deg", p[7] * g.rotationQuantumDeg());
            o.add("guard", gj);
        }

        if (self != null) {
            JsonObject s = new JsonObject();
            s.addProperty("hp", self.getHealth());
            s.addProperty("x", self.getX());
            s.addProperty("y", self.getY());
            s.addProperty("z", self.getZ());
            s.addProperty("yaw", self.getYaw());
            s.addProperty("pitch", self.getPitch());
            o.add("self", s);
        }

        if (lvl == Level.VERBOSE && obs != null) o.add("obs", obs);

        queue.add(GSON.toJson(o));
    }

    private void drainLoop() {
        try (BufferedWriter w = Files.newBufferedWriter(file, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            while (running || !queue.isEmpty()) {
                String line = queue.poll();
                if (line == null) {
                    Thread.sleep(20);
                    continue;
                }
                w.write(line);
                w.write('\n');
                written.incrementAndGet();
                if (queue.isEmpty()) w.flush();
            }
            w.flush();
        } catch (Exception e) {
            RlMinecraftAiClient.LOGGER.warn("debug log writer stopped", e);
        }
    }

    /** Flush everything queued and stop the writer thread (best effort). */
    public void flushAndClose() {
        running = false;
        try {
            writerThread.join(2000);
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        }
    }
}
