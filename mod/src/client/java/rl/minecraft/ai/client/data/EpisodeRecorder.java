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
 * Streams one JSON line per tick during {@code /fight train} to
 * {@code <dataset dir>/<kit>/session-<timestamp>.jsonl}, and appends a
 * one-line summary to {@code <dataset dir>/manifest.jsonl} when the session
 * ends. Each line is {@code {t, kit, target, obs, action}} - obs is the raw
 * (un-normalised) observation dict, ready for the trainer to consume the same
 * way {@code training/python/features.py::observation_to_row} does.
 */
public final class EpisodeRecorder implements AutoCloseable {
    private static final Gson GSON = new Gson();
    private static final DateTimeFormatter STAMP =
        DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss").withZone(ZoneId.systemDefault());

    private final Path datasetDir;
    private final Path sessionFile;
    private final Kit kit;
    private final Instant startedAt;
    private BufferedWriter writer;
    private int ticks = 0;

    private EpisodeRecorder(Path datasetDir, Path sessionFile, Kit kit) {
        this.datasetDir = datasetDir;
        this.sessionFile = sessionFile;
        this.kit = kit;
        this.startedAt = Instant.now();
    }

    public static EpisodeRecorder start(Path datasetDir, Kit kit) throws IOException {
        Path dir = datasetDir.resolve(kit.id);
        Files.createDirectories(dir);
        Path file = dir.resolve("session-" + STAMP.format(Instant.now()) + ".jsonl");
        EpisodeRecorder rec = new EpisodeRecorder(datasetDir, file, kit);
        rec.writer = Files.newBufferedWriter(file, StandardCharsets.UTF_8,
            StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING);
        RlMinecraftAiClient.LOGGER.info("recording training episode -> {}", file);
        return rec;
    }

    public void record(JsonObject obs, Action action, String targetName) {
        if (writer == null) return;
        JsonObject line = new JsonObject();
        line.addProperty("t", ticks++);
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

    public int ticks() {
        return ticks;
    }

    public Path file() {
        return sessionFile;
    }

    @Override
    public void close() {
        if (writer == null) return;
        try {
            writer.flush();
            writer.close();
        } catch (IOException e) {
            RlMinecraftAiClient.LOGGER.warn("failed to close training file", e);
        }
        writer = null;
        try {
            JsonObject entry = new JsonObject();
            entry.addProperty("session", sessionFile.getFileName().toString());
            entry.addProperty("kit", kit.id);
            entry.addProperty("ticks", ticks);
            entry.addProperty("started_at", startedAt.toString());
            entry.addProperty("ended_at", Instant.now().toString());
            Files.writeString(datasetDir.resolve("manifest.jsonl"),
                GSON.toJson(entry) + "\n", StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            RlMinecraftAiClient.LOGGER.warn("failed to append to manifest.jsonl", e);
        }
        RlMinecraftAiClient.LOGGER.info("training episode closed: {} ticks -> {}", ticks, sessionFile);
    }
}
