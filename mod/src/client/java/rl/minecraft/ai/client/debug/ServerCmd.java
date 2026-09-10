package rl.minecraft.ai.client.debug;

import net.minecraft.client.MinecraftClient;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.command.CommandManager;
import net.minecraft.text.Text;

import rl.minecraft.ai.client.RlMinecraftAiClient;

import java.util.List;
import java.util.function.Consumer;

/**
 * Runs vanilla commands from the debug harness. On an integrated (singleplayer)
 * server the command is executed on the server thread with the server's own
 * op-level-4 {@code ServerCommandSource}, so {@code /gamerule}, {@code /summon},
 * {@code /give}, {@code /tp} etc. all work regardless of the player's own perms.
 * On a real server it falls back to sending the command as the player (subject
 * to that server's permissions).
 */
public final class ServerCmd {
    private ServerCmd() {}

    /** Run one command (with or without a leading '/'). Returns false if it could not be dispatched. */
    public static boolean run(MinecraftClient mc, String command, Consumer<Text> fb) {
        String cmd = CommandManager.stripLeadingSlash(command.trim());
        if (cmd.isEmpty()) return false;

        MinecraftServer server = mc.getServer();
        if (server != null) {
            server.execute(() -> {
                try {
                    server.getCommandManager().parseAndExecute(server.getCommandSource(), cmd);
                } catch (Exception e) {
                    RlMinecraftAiClient.LOGGER.warn("debug cmd failed: /{} - {}", cmd, e.toString());
                    if (fb != null) fb.accept(Text.literal("§ccmd failed: " + e.getMessage()));
                }
            });
            return true;
        }

        if (mc.player != null) {
            mc.player.networkHandler.sendChatCommand(cmd);
            return true;
        }

        if (fb != null) fb.accept(Text.literal("§cnot in a world"));
        return false;
    }

    /** Run several commands in order (integrated server only keeps their order deterministic). */
    public static void runAll(MinecraftClient mc, List<String> commands, Consumer<Text> fb) {
        for (String c : commands) {
            String t = c.strip();
            if (t.isEmpty() || t.startsWith("#")) continue;
            run(mc, t, fb);
        }
    }
}
