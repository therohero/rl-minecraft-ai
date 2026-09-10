package rl.minecraft.ai.client.debug;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.DoubleArgumentType;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;

import net.fabricmc.fabric.api.client.command.v2.ClientCommandManager;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.MinecraftClient;
import net.minecraft.text.Text;

import rl.minecraft.ai.client.RlMinecraftAiClient;
import rl.minecraft.ai.client.combat.ClientGuard;
import rl.minecraft.ai.client.combat.FightController;
import rl.minecraft.ai.client.combat.Kit;
import rl.minecraft.ai.client.net.InferenceClient;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;

/**
 * The {@code /rldebug} command tree (dev-only). See {@code mod/README.md}.
 */
final class DebugCommands {
    private DebugCommands() {}

    static void register(CommandDispatcher<FabricClientCommandSource> d, FightController controller) {
        d.register(ClientCommandManager.literal("rldebug")
            .executes(ctx -> {
                status(controller, ctx.getSource()::sendFeedback);
                return 1;
            })
            .then(ClientCommandManager.literal("status").executes(ctx -> {
                status(controller, ctx.getSource()::sendFeedback);
                return 1;
            }))

            // --- log level --------------------------------------------------
            .then(ClientCommandManager.literal("log")
                .executes(ctx -> {
                    DebugLog l = DebugHarness.log();
                    ctx.getSource().sendFeedback(Text.literal("§7debug log level=§f"
                        + l.level().name().toLowerCase() + " §7(" + l.linesWritten() + " lines) -> §f"
                        + l.path()));
                    return 1;
                })
                .then(ClientCommandManager.literal("off").executes(ctx -> setLevel(ctx.getSource(), DebugLog.Level.OFF)))
                .then(ClientCommandManager.literal("basic").executes(ctx -> setLevel(ctx.getSource(), DebugLog.Level.BASIC)))
                .then(ClientCommandManager.literal("verbose").executes(ctx -> setLevel(ctx.getSource(), DebugLog.Level.VERBOSE)))
                .then(ClientCommandManager.literal("path").executes(ctx -> {
                    ctx.getSource().sendFeedback(Text.literal("§7" + DebugHarness.log().path()));
                    return 1;
                })))

            // --- test world ------------------------------------------------
            .then(ClientCommandManager.literal("world")
                .executes(ctx -> {
                    TestWorld.createOrLoad(MinecraftClient.getInstance(), ctx.getSource()::sendFeedback);
                    return 1;
                })
                .then(ClientCommandManager.literal("setup").executes(ctx -> {
                    TestWorld.applySetup(MinecraftClient.getInstance(), ctx.getSource()::sendFeedback);
                    return 1;
                })))

            // --- run commands --------------------------------------------
            .then(ClientCommandManager.literal("cmd")
                .then(ClientCommandManager.argument("command", StringArgumentType.greedyString())
                    .executes(ctx -> {
                        String c = StringArgumentType.getString(ctx, "command");
                        boolean ok = ServerCmd.run(MinecraftClient.getInstance(), c, ctx.getSource()::sendFeedback);
                        ctx.getSource().sendFeedback(Text.literal(ok ? "§7ran: §f/" + c : "§ccould not run /" + c));
                        return ok ? 1 : 0;
                    })))
            .then(ClientCommandManager.literal("script")
                .then(ClientCommandManager.argument("name", StringArgumentType.greedyString())
                    .executes(ctx -> runScript(ctx.getSource(), StringArgumentType.getString(ctx, "name")))))

            // --- kit ------------------------------------------------------
            .then(ClientCommandManager.literal("kit")
                .then(ClientCommandManager.argument("kit", StringArgumentType.word())
                    .executes(ctx -> {
                        Scenario.give(MinecraftClient.getInstance(),
                            StringArgumentType.getString(ctx, "kit"), ctx.getSource()::sendFeedback);
                        return 1;
                    })))

            // --- dummies ------------------------------------------------
            .then(ClientCommandManager.literal("dummy")
                .then(ClientCommandManager.literal("player")
                    .executes(ctx -> { Dummy.spawnPlayers(MinecraftClient.getInstance(), 1, 4.0, ctx.getSource()::sendFeedback); return 1; })
                    .then(ClientCommandManager.argument("count", IntegerArgumentType.integer(1, 8))
                        .executes(ctx -> { Dummy.spawnPlayers(MinecraftClient.getInstance(),
                            IntegerArgumentType.getInteger(ctx, "count"), 4.0, ctx.getSource()::sendFeedback); return 1; })
                        .then(ClientCommandManager.argument("dist", DoubleArgumentType.doubleArg(1.0, 30.0))
                            .executes(ctx -> { Dummy.spawnPlayers(MinecraftClient.getInstance(),
                                IntegerArgumentType.getInteger(ctx, "count"),
                                DoubleArgumentType.getDouble(ctx, "dist"), ctx.getSource()::sendFeedback); return 1; }))))
                .then(ClientCommandManager.literal("mob")
                    .executes(ctx -> { Dummy.spawnMob(MinecraftClient.getInstance(), "zombie", 4.0, ctx.getSource()::sendFeedback); return 1; })
                    .then(ClientCommandManager.argument("type", StringArgumentType.word())
                        .executes(ctx -> { Dummy.spawnMob(MinecraftClient.getInstance(),
                            StringArgumentType.getString(ctx, "type"), 4.0, ctx.getSource()::sendFeedback); return 1; })
                        .then(ClientCommandManager.argument("dist", DoubleArgumentType.doubleArg(1.0, 30.0))
                            .executes(ctx -> { Dummy.spawnMob(MinecraftClient.getInstance(),
                                StringArgumentType.getString(ctx, "type"),
                                DoubleArgumentType.getDouble(ctx, "dist"), ctx.getSource()::sendFeedback); return 1; }))))
                .then(ClientCommandManager.literal("clear")
                    .executes(ctx -> { Dummy.clear(MinecraftClient.getInstance(), ctx.getSource()::sendFeedback); return 1; })))

            // --- selftest ------------------------------------------------
            .then(ClientCommandManager.literal("selftest").executes(ctx -> {
                if (MinecraftClient.getInstance().world == null) {
                    ctx.getSource().sendFeedback(Text.literal("§cnot in a world - run /rldebug world first"));
                    return 0;
                }
                DebugHarness.startSelfTest(ctx.getSource()::sendFeedback, false);
                ctx.getSource().sendFeedback(Text.literal("§7selftest started…"));
                return 1;
            })));
    }

    private static int setLevel(FabricClientCommandSource src, DebugLog.Level level) {
        DebugHarness.log().setLevel(level);
        src.sendFeedback(Text.literal("§adebug log level = §f" + level.name().toLowerCase()
            + " §7-> " + DebugHarness.log().path()));
        return 1;
    }

    private static int runScript(FabricClientCommandSource src, String name) {
        Path file = FabricLoader.getInstance().getGameDir().resolve("rl-debug").resolve(name + ".txt");
        if (!Files.isReadable(file)) {
            src.sendFeedback(Text.literal("§cno script at " + file));
            return 0;
        }
        try {
            List<String> lines = Files.readAllLines(file);
            ServerCmd.runAll(MinecraftClient.getInstance(), lines, src::sendFeedback);
            src.sendFeedback(Text.literal("§aran script §f" + name + "§a (" + lines.size() + " lines)"));
            return 1;
        } catch (Exception e) {
            RlMinecraftAiClient.LOGGER.warn("script {} failed", name, e);
            src.sendFeedback(Text.literal("§cscript failed: " + e.getMessage()));
            return 0;
        }
    }

    private static void status(FightController c, Consumer<Text> fb) {
        MinecraftClient mc = MinecraftClient.getInstance();
        fb.accept(Text.literal(String.format(
            "§7mode=§f%s §7kit=§f%s §7target=§f%s §7paused=§f%s §7ep=§f%d §7tick=§f%d",
            c.mode(), c.kitId(), c.targetLabel(), c.paused(), c.episodeIndex(), c.tick())));
        ClientGuard g = c.guard();
        if (g != null) {
            fb.accept(Text.literal(String.format(
                "§7guard: cps=§f%.1f§7 quantum=§f%.3f°§7 lastYawTurn=§f%.1f°§7 settled=§f%s",
                g.cps(), g.rotationQuantumDeg(), g.lastYawTurnDeg(), g.aimSettled())));
        }
        InferenceClient inf = c.inference();
        if (inf != null) {
            fb.accept(Text.literal(String.format("§7inference: seen=§f%s§7 latency=§f%.0fms§7 err=§f%s",
                inf.seenServer(), inf.latencyMs(), inf.lastError() == null ? "-" : inf.lastError())));
        }
        DebugLog l = DebugHarness.log();
        fb.accept(Text.literal("§7log=§f" + l.level().name().toLowerCase() + "§7 (" + l.linesWritten()
            + " lines) §7dummies=§f" + Dummy.count()
            + "§7 world=§f" + (mc.world == null ? "none" : (mc.isInSingleplayer() ? "singleplayer" : "server"))));
        if (mc.player != null) {
            fb.accept(Text.literal("§7detected kit for current inventory: §f" + Kit.detect(mc.player).id));
        }
    }
}
