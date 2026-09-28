package rl.minecraft.ai.client.debug;

import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.TitleScreen;
import net.minecraft.client.gui.screen.multiplayer.ConnectScreen;
import net.minecraft.client.network.ServerAddress;
import net.minecraft.client.network.ServerInfo;
import net.minecraft.text.Text;

import rl.minecraft.ai.client.RlConfig;
import rl.minecraft.ai.client.RlMinecraftAiClient;
import rl.minecraft.ai.client.combat.FightController;

import java.util.Locale;
import java.util.function.Consumer;

/**
 * Dev-only debug harness entrypoint. Loaded reflectively by
 * {@link rl.minecraft.ai.client.RlMinecraftAiClient} only in a development
 * environment; the whole {@code rl.minecraft.ai.client.debug} package is
 * stripped from every released jar (see {@code build.gradle}).
 *
 * <p>Wires up:
 * <ul>
 *   <li>a per-active-tick tap into {@link DebugLog} (level from
 *       {@code -Drl.minecraft.ai.debug.log}, default {@code off}),</li>
 *   <li>the {@code /rldebug} command tree ({@link DebugCommands}),</li>
 *   <li>the {@link SelfTest} state machine, and</li>
 *   <li>{@code -Drl.minecraft.ai.debug.autorun=selftest}: open vanilla's
 *       test-world screen and click <i>Create New World</i>, wait for the
 *       join, run the selftest, write its result and close the client.</li>
 * </ul>
 */
public final class DebugHarness {
    private DebugHarness() {}

    private static DebugLog log;
    private static FightController controller;
    private static RlConfig cfg;
    private static volatile SelfTest activeTest;

    private static boolean autorunSelftest;
    private static boolean autorunWorldTriggered;
    private static boolean autorunCreatePressed;
    private static boolean autorunJoined;
    private static boolean autorunTestLaunched;
    private static int lifeTicks;
    private static int joinedTicks;

    private static boolean autorunRealserver;
    private static String realserverAddress;
    private static String realserverKit;
    private static int realserverRunTicks;
    private static boolean realserverTrain;
    private static boolean realserverConnectTriggered;
    private static boolean realserverLaunched;
    private static volatile RealServerRun activeRealServerRun;

    /** Reflection entrypoint - signature must stay {@code (RlConfig, FightController)}. */
    public static void init(RlConfig config, FightController fightController) {
        cfg = config;
        controller = fightController;
        log = new DebugLog(parseLevel(prop("rl.minecraft.ai.debug.log", "RL_DEBUG_LOG", "off")));

        String autorun = prop("rl.minecraft.ai.debug.autorun", "RL_DEBUG_AUTORUN", "");
        autorunSelftest = autorun.equalsIgnoreCase("selftest");
        autorunRealserver = autorun.equalsIgnoreCase("realserver");
        if (autorunRealserver) {
            realserverAddress = prop("rl.minecraft.ai.debug.server", "RL_DEBUG_SERVER", "");
            realserverKit = prop("rl.minecraft.ai.debug.kit", "RL_DEBUG_KIT", "sword");
            realserverRunTicks = Integer.parseInt(prop("rl.minecraft.ai.debug.run_ticks", "RL_DEBUG_RUN_TICKS", "1200"));
            realserverTrain = Boolean.parseBoolean(prop("rl.minecraft.ai.debug.train", "RL_DEBUG_TRAIN", "false"));
            if (realserverAddress.isBlank()) {
                RlMinecraftAiClient.LOGGER.error(
                    "autorun=realserver needs -Drl.minecraft.ai.debug.server / RL_DEBUG_SERVER=<host:port>");
                autorunRealserver = false;
            }
        }

        controller.setTickTap((obs, action) -> {
            try {
                log.onTick(controller, obs, action);
                SelfTest t = activeTest;
                if (t != null) t.onTap(obs, action);
            } catch (Throwable e) {
                RlMinecraftAiClient.LOGGER.warn("debug tick tap error", e);
            }
        });

        ClientCommandRegistrationCallback.EVENT.register((dispatcher, access) ->
            DebugCommands.register(dispatcher, controller));

        ClientTickEvents.END_CLIENT_TICK.register(DebugHarness::onEndTick);

        if (autorunSelftest || autorunRealserver) {
            ClientPlayConnectionEvents.JOIN.register((handler, sender, client) -> {
                autorunJoined = true;
                joinedTicks = 0;
                RlMinecraftAiClient.LOGGER.info("autorun: joined world");
            });
        }

        RlMinecraftAiClient.LOGGER.info("debug harness ready (log={}, autorun={})",
            log.level(), autorunSelftest ? "selftest" : (autorunRealserver ? "realserver" : "none"));
    }

    public static DebugLog log() {
        return log;
    }

    public static void startSelfTest(Consumer<Text> chat, boolean autorun) {
        if (activeTest != null && !activeTest.finished()) {
            chat.accept(Text.literal("§eselftest already running"));
            return;
        }
        activeTest = new SelfTest(cfg, controller, log, chat, autorun);
    }

    private static void onEndTick(MinecraftClient mc) {
        lifeTicks++;

        SelfTest t = activeTest;
        if (t != null) {
            if (!t.finished()) {
                try {
                    t.tick(mc);
                } catch (Throwable e) {
                    RlMinecraftAiClient.LOGGER.warn("selftest tick error", e);
                    t.abort();
                    activeTest = null;
                }
            } else {
                activeTest = null;
            }
        }

        RealServerRun r = activeRealServerRun;
        if (r != null) {
            if (!r.finished()) {
                try {
                    r.tick(mc);
                } catch (Throwable e) {
                    RlMinecraftAiClient.LOGGER.warn("realserver run tick error - aborting and closing client", e);
                    r.abort();
                    activeRealServerRun = null;
                    log.flushAndClose();
                    mc.scheduleStop();
                }
            } else {
                activeRealServerRun = null;
                RlMinecraftAiClient.LOGGER.info("autorun: realserver run finished - closing client");
                log.flushAndClose();
                mc.scheduleStop();
            }
        }

        if (autorunRealserver) {
            onRealserverAutorunTick(mc);
            return;
        }

        if (!autorunSelftest || autorunTestLaunched) return;

        // 1. no world yet: open the vanilla test-world screen once the title
        //    screen settles, then click its "Create New World" button for it.
        if (mc.world == null) {
            if (!autorunWorldTriggered && lifeTicks > 60 && mc.currentScreen instanceof TitleScreen) {
                autorunWorldTriggered = true;
                RlMinecraftAiClient.LOGGER.info("autorun: opening test-world screen");
                TestWorld.createOrLoad(mc, m ->
                    RlMinecraftAiClient.LOGGER.info("[autorun] {}", m.getString()));
            }
            if (autorunWorldTriggered && !autorunCreatePressed && TestWorld.pressCreateButton(mc)) {
                autorunCreatePressed = true;
            }
            return;
        }

        // 2. joined and settled: launch the selftest exactly once.
        if (autorunJoined && ++joinedTicks > 40) {
            autorunTestLaunched = true;
            RlMinecraftAiClient.LOGGER.info("autorun: starting selftest");
            startSelfTest(m ->
                RlMinecraftAiClient.LOGGER.info("[autorun] {}", m.getString().replaceAll("§.", "")),
                true);
        }
    }

    /** Drives the -Drl.minecraft.ai.debug.autorun=realserver flow: connect once the title
     * screen settles, then launch {@link RealServerRun} once the join has had time to settle. */
    private static void onRealserverAutorunTick(MinecraftClient mc) {
        if (realserverLaunched) return;  // launched exactly once - even after it finishes

        if (mc.world == null) {
            if (!realserverConnectTriggered && lifeTicks > 60 && mc.currentScreen instanceof TitleScreen) {
                realserverConnectTriggered = true;
                RlMinecraftAiClient.LOGGER.info("autorun: connecting to {}", realserverAddress);
                mc.execute(() -> {
                    ServerAddress addr = ServerAddress.parse(realserverAddress);
                    ServerInfo info = new ServerInfo("rl-debug", realserverAddress, ServerInfo.ServerType.OTHER);
                    ConnectScreen.connect(mc.currentScreen, mc, addr, info, false, null);
                });
            }
            return;
        }

        if (autorunJoined && ++joinedTicks > 40) {
            realserverLaunched = true;
            RlMinecraftAiClient.LOGGER.info(
                "autorun: starting realserver run (kit={}, run_ticks={}, train={})",
                realserverKit, realserverRunTicks, realserverTrain);
            activeRealServerRun = new RealServerRun(cfg, controller,
                m -> RlMinecraftAiClient.LOGGER.info("[autorun] {}", m.getString().replaceAll("§.", "")),
                realserverKit, realserverRunTicks, realserverTrain);
        }
    }

    /** System property, falling back to an environment variable, then a default. */
    private static String prop(String sysKey, String envKey, String def) {
        String v = System.getProperty(sysKey);
        if (v == null || v.isBlank()) v = System.getenv(envKey);
        return v == null || v.isBlank() ? def : v;
    }

    private static DebugLog.Level parseLevel(String s) {
        return switch (s.toLowerCase(Locale.ROOT)) {
            case "basic", "on", "true", "1" -> DebugLog.Level.BASIC;
            case "verbose", "full", "2" -> DebugLog.Level.VERBOSE;
            default -> DebugLog.Level.OFF;
        };
    }
}
