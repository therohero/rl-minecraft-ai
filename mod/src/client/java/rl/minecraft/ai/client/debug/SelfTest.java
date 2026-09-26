package rl.minecraft.ai.client.debug;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;
import net.minecraft.text.Text;

import rl.minecraft.ai.client.RlConfig;
import rl.minecraft.ai.client.RlMinecraftAiClient;
import rl.minecraft.ai.client.combat.FightController;
import rl.minecraft.ai.client.combat.Kit;
import rl.minecraft.ai.client.net.Action;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Scripted, tick-driven end-to-end check of the fight loop - the mod's answer
 * to {@code smoke_train.py}. Builds the test world, equips the sword kit,
 * spawns a client-side dummy, runs {@code /fight train} against it for a couple
 * hundred ticks, then asserts on the observation stream, the applied actions,
 * the {@link rl.minecraft.ai.client.combat.ClientGuard} invariants and the
 * dataset the recorder wrote. Prints PASS/FAIL per assertion and writes
 * {@code <game dir>/rl-debug/selftest-<stamp>.json}.
 *
 * <p>Runs from {@code /rldebug selftest} or, unattended, from
 * {@code -Drl.minecraft.ai.debug.autorun=selftest} (which also closes the
 * client on completion).
 */
public final class SelfTest {
    private static final DateTimeFormatter STAMP =
        DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss").withZone(ZoneId.systemDefault());
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private enum Phase { SETUP, EQUIP, ENGAGE, RUN, STOP, VERIFY, DONE }

    private final RlConfig cfg;
    private final FightController controller;
    private final DebugLog log;
    private final Consumer<Text> chat;
    private final boolean autorun;

    private MockInferenceServer mock;

    private Phase phase = Phase.SETUP;
    private int phaseTicks = 0;
    private int runTicks = 0;
    private static final int RUN_TICKS = 200;

    // observed during RUN
    private int tapCalls = 0;
    private int obsSeen = 0;
    private int obsBadKeys = 0;
    private int actionsApplied = 0;
    private double cpsMax = 0;
    private int quantiseViolations = 0;
    private double aimErrStart = Double.NaN;
    private double aimErrEnd = Double.NaN;
    private int dummyId = -1;
    // lstm_state continuity (MockInferenceServer always echoes one - see its
    // class doc comment): first/last h[0] seen and whether the sequence ever
    // went backwards, i.e. InferenceClient actually carries the state
    // between ticks instead of dropping it. Does NOT cover the
    // reset-at-episode-boundary half of the contract - that needs a real
    // in-fight episode boundary (kill/death/disengage), which this solo
    // scenario never reaches (see the "extend the debug-harness selftest"
    // TODO item).
    private int lstmFirst = -1;
    private int lstmLast = -1;
    private boolean lstmMonotonic = true;

    private final List<JsonObject> results = new ArrayList<>();
    private boolean finished = false;

    public SelfTest(RlConfig cfg, FightController controller, DebugLog log,
                    Consumer<Text> chat, boolean autorun) {
        this.cfg = cfg;
        this.controller = controller;
        this.log = log;
        this.chat = chat;
        this.autorun = autorun;
    }

    public boolean finished() {
        return finished;
    }

    /** Release resources if the run is torn down early (e.g. a tick error). */
    public void abort() {
        if (mock != null) {
            mock.stop();
            mock = null;
        }
        finished = true;
    }

    /** Forwarded from the tick tap while the test is in its RUN phase. */
    public void onTap(JsonObject obs, Action action) {
        if (phase != Phase.RUN) return;
        tapCalls++;
        if (obs != null) {
            obsSeen++;
            if (hasBadNumbers(obs)) obsBadKeys++;
        }
        if (action != null) actionsApplied++;
        if (action != null && action.lstmState() != null) {
            int h0 = firstArrayInt(action.lstmState(), "h", -1);
            if (h0 >= 0) {
                if (lstmFirst < 0) lstmFirst = h0;
                if (h0 < lstmLast) lstmMonotonic = false;
                lstmLast = h0;
            }
        }
        var g = controller.guard();
        if (g != null) {
            cpsMax = Math.max(cpsMax, g.cps());
            double q = g.rotationQuantumDeg();
            double[] p = g.lastPlan();
            double appliedYaw = p[6] * q;
            if (q > 0 && Math.abs(appliedYaw / q - Math.rint(appliedYaw / q)) > 1e-6) {
                quantiseViolations++;
            }
        }
    }

    public void tick(MinecraftClient mc) {
        if (finished) return;
        phaseTicks++;
        ClientPlayerEntity self = mc.player;
        if (self == null || mc.world == null) return;

        switch (phase) {
            case SETUP -> {
                if (phaseTicks == 1) {
                    say("§7selftest: applying world setup");
                    TestWorld.applySetup(mc, chat);
                    mock = MockInferenceServer.startIfFree(cfg);
                    say(mock != null
                        ? "§7selftest: mock inference server started on " + cfg.inferenceUrl
                        : "§7selftest: using the inference server already at " + cfg.inferenceUrl);
                }
                if (phaseTicks >= 20) advance(Phase.EQUIP);
            }
            case EQUIP -> {
                if (phaseTicks == 1) {
                    say("§7selftest: equipping sword kit");
                    Scenario.give(mc, "sword", chat);
                }
                if (phaseTicks >= 20) {
                    record("kit_detected", Kit.detect(self) == Kit.SWORD,
                        "detected " + Kit.detect(self).id + ", expected sword");
                    advance(Phase.ENGAGE);
                }
            }
            case ENGAGE -> {
                if (phaseTicks == 1) {
                    say("§7selftest: spawning dummy + starting /fight train");
                    Dummy.spawnPlayers(mc, 1, 3.0, chat);   // ~4m ahead, stationary aim target
                }
                if (phaseTicks == 5) {
                    dummyId = firstDummyId(mc);
                    controller.setTarget("nearest", chat);
                    controller.start(true, true, chat);
                }
                if (phaseTicks >= 15) {
                    aimErrStart = aimError(mc);
                    advance(Phase.RUN);
                }
            }
            case RUN -> {
                runTicks++;
                if (runTicks >= RUN_TICKS) {
                    aimErrEnd = aimError(mc);
                    advance(Phase.STOP);
                }
            }
            case STOP -> {
                if (phaseTicks == 1) {
                    say("§7selftest: stopping");
                    finishRunAssertions();
                    controller.stop(chat);
                }
                if (phaseTicks >= 10) advance(Phase.VERIFY);
            }
            case VERIFY -> {
                verifyDataset();
                writeResult();
                advance(Phase.DONE);
            }
            case DONE -> {
                Dummy.clear(mc, chat);
                if (mock != null) {
                    mock.stop();
                    mock = null;
                }
                finished = true;
                if (autorun) {
                    boolean pass = allPassed();
                    say(pass ? "§aselftest PASS - closing client" : "§cselftest FAIL - closing client");
                    log.flushAndClose();
                    new Thread(() -> {
                        try {
                            Thread.sleep(500);
                        } catch (InterruptedException ignored) {
                            Thread.currentThread().interrupt();
                        }
                        Runtime.getRuntime().halt(pass ? 0 : 1);
                    }, "rl-selftest-exit").start();
                    mc.scheduleStop();
                }
            }
        }
    }

    // ---------------------------------------------------------------- assertions

    private void finishRunAssertions() {
        record("obs_every_tick", tapCalls > 0 && obsSeen >= tapCalls - 2,
            obsSeen + "/" + tapCalls + " ticks built an observation");
        record("obs_no_nan", obsBadKeys == 0,
            obsBadKeys + " observations had a NaN / non-finite number");

        boolean seen = controller.inference() != null && controller.inference().seenServer();
        String err = controller.inference() == null ? null : controller.inference().lastError();
        if (seen) {
            record("inference_reachable", true, "inference server answered /act");
        } else if (err != null && (err.contains("cannot reach") || err.contains("Connection refused"))) {
            skip("inference_reachable", "no inference server at " + cfg.inferenceUrl
                + " - start azalea-bot/inference_server.py for a real check");
        } else {
            record("inference_reachable", false,
                "no /act reply; lastError=" + err);
        }

        record("actions_applied", actionsApplied > 0,
            actionsApplied + " ticks applied an action");
        if (!Double.isNaN(aimErrStart) && !Double.isNaN(aimErrEnd)) {
            record("aim_converges", aimErrEnd <= aimErrStart + 2.0 || aimErrEnd < 15.0,
                String.format("angle-to-target %.1f° -> %.1f°", aimErrStart, aimErrEnd));
        } else {
            skip("aim_converges", "no dummy target to measure against");
        }
        record("cps_within_cap", cpsMax <= cfg.maxCps + 1e-6,
            String.format("peak cps %.1f vs cap %.1f", cpsMax, cfg.maxCps));
        record("rotation_quantised", quantiseViolations == 0,
            quantiseViolations + " applied rotations were not a mouse-count multiple");

        // See MockInferenceServer's class doc comment: it always echoes
        // lstm_state = {"h": [incoming h[0] + 1]}, so a rising sequence here
        // is only possible if InferenceClient is actually round-tripping
        // the carried state on every request rather than dropping it.
        record("lstm_state_carried", lstmFirst >= 0 && lstmMonotonic && lstmLast > lstmFirst,
            "lstm_state h[0] " + lstmFirst + " -> " + lstmLast + " (monotonic=" + lstmMonotonic + ")");
    }

    private void verifyDataset() {
        try {
            Path kitDir = cfg.datasetDir.resolve("sword");
            Path newest = null;
            long newestMs = Long.MIN_VALUE;
            if (Files.isDirectory(kitDir)) {
                try (var s = Files.list(kitDir)) {
                    for (Path p : s.filter(p -> p.getFileName().toString().endsWith(".jsonl")).toList()) {
                        long m = Files.getLastModifiedTime(p).toMillis();
                        if (m > newestMs) {
                            newestMs = m;
                            newest = p;
                        }
                    }
                }
            }
            if (newest == null) {
                record("dataset_written", false, "no .jsonl under " + kitDir);
                return;
            }
            List<String> lines = Files.readAllLines(newest, StandardCharsets.UTF_8);
            boolean enough = lines.size() >= 10;
            boolean outcomeRow = !lines.isEmpty()
                && lines.get(lines.size() - 1).contains("\"outcome\"");
            record("dataset_written", enough && outcomeRow,
                newest.getFileName() + ": " + lines.size() + " lines, trailing outcome record="
                    + outcomeRow);

            Path manifest = cfg.datasetDir.resolve("manifest.jsonl");
            record("manifest_appended", Files.isReadable(manifest)
                    && !Files.readAllLines(manifest).isEmpty(),
                "manifest.jsonl present and non-empty");
        } catch (IOException e) {
            record("dataset_written", false, "read error: " + e.getMessage());
        }
    }

    // ---------------------------------------------------------------- helpers

    private void advance(Phase next) {
        phase = next;
        phaseTicks = 0;
    }

    private int firstDummyId(MinecraftClient mc) {
        for (var p : mc.world.getPlayers()) {
            if (p.getName().getString().startsWith("rl_dummy")) return p.getId();
        }
        return -1;
    }

    private double aimError(MinecraftClient mc) {
        if (dummyId < 0) return Double.NaN;
        Entity t = mc.world.getEntityById(dummyId);
        ClientPlayerEntity self = mc.player;
        if (t == null || self == null) return Double.NaN;
        double dx = t.getX() - self.getX();
        double dz = t.getZ() - self.getZ();
        double wantYaw = Math.toDegrees(Math.atan2(-dx, dz));
        double diff = Math.abs(wrapDeg(wantYaw - self.getYaw()));
        return diff;
    }

    private static double wrapDeg(double d) {
        d %= 360.0;
        if (d >= 180.0) d -= 360.0;
        if (d < -180.0) d += 360.0;
        return d;
    }

    /** `obj[key][0]` as an int, or `def` if that path doesn't parse. */
    private static int firstArrayInt(JsonObject obj, String key, int def) {
        try {
            return obj.getAsJsonArray(key).get(0).getAsInt();
        } catch (RuntimeException e) {
            return def;
        }
    }

    private static boolean hasBadNumbers(JsonObject obs) {
        for (var entry : obs.entrySet()) {
            var v = entry.getValue();
            if (v.isJsonPrimitive() && v.getAsJsonPrimitive().isNumber()) {
                double d = v.getAsDouble();
                if (Double.isNaN(d) || Double.isInfinite(d)) return true;
            } else if (v.isJsonArray()) {
                for (var el : v.getAsJsonArray()) {
                    if (el.isJsonPrimitive() && el.getAsJsonPrimitive().isNumber()) {
                        double d = el.getAsDouble();
                        if (Double.isNaN(d) || Double.isInfinite(d)) return true;
                    }
                }
            }
        }
        return false;
    }

    private void record(String name, boolean pass, String detail) {
        JsonObject o = new JsonObject();
        o.addProperty("name", name);
        o.addProperty("status", pass ? "PASS" : "FAIL");
        o.addProperty("detail", detail);
        results.add(o);
        say((pass ? "§a[PASS] " : "§c[FAIL] ") + name + " §7- " + detail);
        log.event("selftest_assert", j -> {
            j.addProperty("name", name);
            j.addProperty("status", pass ? "PASS" : "FAIL");
            j.addProperty("detail", detail);
        });
    }

    private void skip(String name, String detail) {
        JsonObject o = new JsonObject();
        o.addProperty("name", name);
        o.addProperty("status", "SKIP");
        o.addProperty("detail", detail);
        results.add(o);
        say("§e[SKIP] " + name + " §7- " + detail);
    }

    private boolean allPassed() {
        for (JsonObject o : results) {
            if ("FAIL".equals(o.get("status").getAsString())) return false;
        }
        return true;
    }

    private void writeResult() {
        JsonObject root = new JsonObject();
        root.addProperty("ts", Instant.now().toString());
        root.addProperty("pass", allPassed());
        root.addProperty("run_ticks", runTicks);
        root.addProperty("inference_url", cfg.inferenceUrl);
        JsonArray arr = new JsonArray();
        results.forEach(arr::add);
        root.add("assertions", arr);

        Path dir = FabricLoader.getInstance().getGameDir().resolve("rl-debug");
        Path out = dir.resolve("selftest-" + STAMP.format(Instant.now()) + ".json");
        try {
            Files.createDirectories(dir);
            Files.writeString(out, GSON.toJson(root), StandardCharsets.UTF_8);
            say("§7selftest result -> " + out);
            RlMinecraftAiClient.LOGGER.info("selftest {} -> {}",
                allPassed() ? "PASS" : "FAIL", out);
        } catch (IOException e) {
            RlMinecraftAiClient.LOGGER.warn("could not write selftest result", e);
        }
        log.event("selftest_result", j -> {
            j.addProperty("pass", allPassed());
            j.addProperty("file", out.toString());
        });
    }

    private void say(String msg) {
        if (chat != null) chat.accept(Text.literal(msg));
        RlMinecraftAiClient.LOGGER.info("[selftest] {}", msg.replaceAll("§.", ""));
    }
}
