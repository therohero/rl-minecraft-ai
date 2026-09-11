package rl.minecraft.ai.client.data;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import rl.minecraft.ai.client.RlMinecraftAiClient;
import rl.minecraft.ai.client.combat.Kit;
import rl.minecraft.ai.client.net.Action;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

/**
 * Records {@code /fight train} to disk, <b>one file per fight</b>. A single
 * {@code /fight train} session rotates through
 * {@code <dataset dir>/<kit>/session-<timestamp>-e<N>.jsonl}: each death or
 * kill closes the current episode file and the next engagement opens the
 * next. Every file is one JSON object per tick
 * ({@code {t, kit, target, obs, action}}, obs raw / un-normalised) followed by
 * a trailing outcome record with no {@code obs} key:
 *
 * <pre>{t, outcome, reason, self_hp_end, enemy_hp_end, opponent, server, match}</pre>
 *
 * so {@code training/python/train_from_episodes.py} gets a clean per-fight
 * return and win/loss instead of guessing from the observation stream. A
 * matching line lands in {@code <dataset dir>/manifest.jsonl} per episode.
 */
public final class EpisodeRecorder implements AutoCloseable {
    private static final Gson GSON = new Gson();
    private static final DateTimeFormatter STAMP =
        DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss").withZone(ZoneId.systemDefault());

    private final Path datasetDir;
    private final Path kitDir;
    private final String sessionId;
    private final Kit kit;
    private final String server;
    private final String match;

    private int episodeIndex = 0;
    private int episodeTicks = 0;
    private int totalTicks = 0;
    private String opponent = "";
    private Instant episodeStartedAt;
    private BufferedWriter writer;

    private EpisodeRecorder(Path datasetDir, Path kitDir, String sessionId, Kit kit,
                            String server, String match) {
        this.datasetDir = datasetDir;
        this.kitDir = kitDir;
        this.sessionId = sessionId;
        this.kit = kit;
        this.server = server == null || server.isBlank() ? "unknown" : server;
        this.match = match == null || match.isBlank() ? "real" : match;
    }

    public static EpisodeRecorder start(Path datasetDir, Kit kit, String server, String match)
            throws IOException {
        Path kitDir = datasetDir.resolve(kit.id);
        Files.createDirectories(kitDir);
        String sessionId = "session-" + STAMP.format(Instant.now());
        RlMinecraftAiClient.LOGGER.info("recording training session {} (kit={}, server={}, match={})",
            sessionId, kit.id, server, match);
        return new EpisodeRecorder(datasetDir, kitDir, sessionId, kit, server, match);
    }

    private Path episodeFile() {
        return kitDir.resolve(sessionId + "-e" + episodeIndex + ".jsonl");
    }

    private void openEpisode() throws IOException {
        writer = Files.newBufferedWriter(episodeFile(), StandardCharsets.UTF_8,
            StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING);
        episodeStartedAt = Instant.now();
        episodeTicks = 0;
        opponent = "";
        RlMinecraftAiClient.LOGGER.info("  -> episode {} -> {}", episodeIndex, episodeFile());
    }

    public void record(JsonObject obs, Action action, String targetName) {
        try {
            if (writer == null) openEpisode();
        } catch (IOException e) {
            RlMinecraftAiClient.LOGGER.warn("could not open episode file", e);
            return;
        }
        if (targetName != null && !targetName.isBlank() && opponent.isBlank()) opponent = targetName;
        JsonObject line = new JsonObject();
        line.addProperty("t", episodeTicks++);
        totalTicks++;
        line.addProperty("kit", kit.id);
        line.addProperty("target", targetName == null ? "" : targetName);
        line.add("obs", obs);
        line.add("action", GSON.toJsonTree(action));
        try {
            writer.write(GSON.toJson(line));
            writer.write('\n');
        } catch (IOException e) {
            RlMinecraftAiClient.LOGGER.warn("failed to write training sample", e);
        }
    }

    /**
     * Close the current episode file with its outcome and roll to the next.
     * A no-op if no episode has any ticks yet (no empty files). {@code outcome}
     * is {@code "win"} / {@code "loss"} / {@code "unknown"}; pass {@code NaN}
     * for an HP that wasn't observed.
     */
    public void endEpisode(String outcome, String reason, double selfHpEnd, double enemyHpEnd) {
        if (writer == null) return;
        Path file = episodeFile();
        try {
            JsonObject end = new JsonObject();
            end.addProperty("t", episodeTicks);
            end.addProperty("outcome", outcome);
            end.addProperty("reason", reason);
            addHp(end, "self_hp_end", selfHpEnd);
            addHp(end, "enemy_hp_end", enemyHpEnd);
            end.addProperty("opponent", opponent);
            end.addProperty("server", server);
            end.addProperty("match", match);
            writer.write(GSON.toJson(end));
            writer.write('\n');
            writer.flush();
            writer.close();
        } catch (IOException e) {
            RlMinecraftAiClient.LOGGER.warn("failed to close episode file", e);
        }
        writer = null;
        try {
            JsonObject entry = new JsonObject();
            entry.addProperty("session", sessionId);
            entry.addProperty("episode", episodeIndex);
            entry.addProperty("file", file.getFileName().toString());
            entry.addProperty("kit", kit.id);
            entry.addProperty("ticks", episodeTicks);
            entry.addProperty("outcome", outcome);
            entry.addProperty("reason", reason);
            addHp(entry, "self_hp_end", selfHpEnd);
            addHp(entry, "enemy_hp_end", enemyHpEnd);
            entry.addProperty("opponent", opponent);
            entry.addProperty("server", server);
            entry.addProperty("match", match);
            entry.addProperty("started_at", String.valueOf(episodeStartedAt));
            entry.addProperty("ended_at", Instant.now().toString());
            Files.writeString(datasetDir.resolve("manifest.jsonl"),
                GSON.toJson(entry) + "\n", StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            RlMinecraftAiClient.LOGGER.warn("failed to append to manifest.jsonl", e);
        }
        RlMinecraftAiClient.LOGGER.info("episode {} closed: {} ticks, outcome={} ({})",
            episodeIndex, episodeTicks, outcome, reason);
        episodeIndex++;
        episodeTicks = 0;
        opponent = "";
    }

    /** Ticks in the episode currently being recorded. */
    public int ticks() {
        return episodeTicks;
    }

    public int totalTicks() {
        return totalTicks;
    }

    public int episodeIndex() {
        return episodeIndex;
    }

    /** The file the current (or next) episode writes to. */
    public Path file() {
        return episodeFile();
    }

    public Path sessionDir() {
        return kitDir;
    }

    @Override
    public void close() {
        endEpisode("unknown", "session ended", Double.NaN, Double.NaN);
    }

    private static void addHp(JsonObject o, String key, double hp) {
        if (Double.isNaN(hp)) o.add(key, null);
        else o.addProperty(key, hp);
    }
}
