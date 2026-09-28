package rl.minecraft.ai.client.debug;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.text.Text;

import rl.minecraft.ai.client.RlConfig;
import rl.minecraft.ai.client.RlMinecraftAiClient;
import rl.minecraft.ai.client.combat.FightController;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.function.Consumer;

/**
 * Unattended {@code /fight} run against a <b>real</b> server the client has
 * connected to (see {@code -Drl.minecraft.ai.debug.autorun=realserver} in
 * {@link DebugHarness}) - the "test the mod yourself" counterpart to
 * {@link SelfTest}, which only ever runs against the singleplayer test world
 * and a mock inference server.
 *
 * <p>This does <b>not</b> assert anything about anticheat detection itself -
 * that verdict lives in the real server's own plugin logs (GrimAC's console
 * alerts / violation history), which this harness has no way to read from
 * inside the client. It only drives a real fight against a real vanilla
 * server for a configurable duration and then exits cleanly, so a human (or
 * a script tailing the server log) can watch for flags while it runs.
 *
 * <p>Needs the server to already have this client's account opped (so
 * {@code /give}/{@code /summon} land - see {@code mod/test-server/README.md})
 * and {@code inference_server.py} reachable at {@code cfg.inferenceUrl}.
 */
public final class RealServerRun {
    private static final DateTimeFormatter STAMP =
        DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss").withZone(ZoneId.systemDefault());

    private enum Phase { SETTLE, EQUIP, ENGAGE, RUN, STOP, DONE }

    private final RlConfig cfg;
    private final FightController controller;
    private final Consumer<Text> chat;
    private final String kitId;
    private final int runTicks;
    private final boolean train;

    private Phase phase = Phase.SETTLE;
    private int phaseTicks = 0;
    private int runTicksDone = 0;
    private boolean finished = false;

    public RealServerRun(RlConfig cfg, FightController controller, Consumer<Text> chat,
                          String kitId, int runTicks, boolean train) {
        this.cfg = cfg;
        this.controller = controller;
        this.chat = chat;
        this.kitId = kitId;
        this.runTicks = Math.max(runTicks, 20);
        this.train = train;
    }

    public boolean finished() {
        return finished;
    }

    public void abort() {
        finished = true;
    }

    public void tick(MinecraftClient mc) {
        if (finished) return;
        phaseTicks++;
        ClientPlayerEntity self = mc.player;
        if (self == null || mc.world == null) return;

        switch (phase) {
            case SETTLE -> {
                // Let the join settle (chunks, inventory sync) before issuing commands -
                // mirrors TestWorld/SelfTest's own settle windows.
                if (phaseTicks >= 40) advance(Phase.EQUIP);
            }
            case EQUIP -> {
                if (phaseTicks == 1) {
                    say("§7realserver: equipping " + kitId + " kit");
                    Scenario.give(mc, kitId, chat);
                }
                if (phaseTicks >= 20) advance(Phase.ENGAGE);
            }
            case ENGAGE -> {
                if (phaseTicks == 1) {
                    // A real summoned mob doesn't work here: FightController /
                    // TargetSelector only ever locks onto a PlayerEntity (this mod is
                    // PvP-only), so the target has to be a fake client-side dummy
                    // player - exactly what SelfTest uses. It has no server-side
                    // entity (no real hit registration), but the rotation, click-
                    // cadence and attack-swing *packets* it drives are real network
                    // traffic a real server anticheat inspects - the thing this run
                    // exists to check.
                    say("§7realserver: spawning a dummy target + starting /fight"
                        + (train ? " train" : ""));
                    Dummy.spawnPlayers(mc, 1, 3.0, chat);
                }
                if (phaseTicks == 15) {
                    // start() resets to a passive target lock internally, so it must run
                    // before setTarget, not after (the other order silently no-ops:
                    // setTarget bails out with "not fighting" while mode is still IDLE).
                    controller.start(train, true, chat);
                    controller.setTarget("nearest", chat);
                    say(String.format(
                        "§7realserver: fighting for %d ticks (~%.0fs) - watch the server console "
                            + "for GrimAC alerts", runTicks, runTicks / 20.0));
                }
                if (phaseTicks >= 20) advance(Phase.RUN);
            }
            case RUN -> {
                runTicksDone++;
                if (runTicksDone >= runTicks) advance(Phase.STOP);
            }
            case STOP -> {
                if (phaseTicks == 1) {
                    say("§7realserver: stopping");
                    controller.stop(chat);
                    Dummy.clear(mc, chat);
                }
                if (phaseTicks >= 10) advance(Phase.DONE);
            }
            case DONE -> {
                finished = true;
                say(String.format("§arealserver run complete - %d ticks (kit=%s)",
                    runTicksDone, kitId));
                RlMinecraftAiClient.LOGGER.info("[realserver] done: {} ticks, kit={}, ts={}",
                    runTicksDone, kitId, STAMP.format(Instant.now()));
            }
        }
    }

    private void advance(Phase next) {
        phase = next;
        phaseTicks = 0;
    }

    private void say(String msg) {
        if (chat != null) chat.accept(Text.literal(msg));
        RlMinecraftAiClient.LOGGER.info("[realserver] {}", msg.replaceAll("§.", ""));
    }
}
