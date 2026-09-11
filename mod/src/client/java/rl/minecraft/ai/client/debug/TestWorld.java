package rl.minecraft.ai.client.debug;

import net.fabricmc.fabric.api.client.screen.v1.Screens;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.TitleScreen;
import net.minecraft.client.gui.screen.world.CreateWorldScreen;
import net.minecraft.client.gui.widget.ButtonWidget;
import net.minecraft.client.gui.widget.ClickableWidget;
import net.minecraft.client.input.MouseInput;
import net.minecraft.text.Text;

import rl.minecraft.ai.client.RlMinecraftAiClient;

import java.util.List;
import java.util.function.Consumer;

/**
 * Spins up a throwaway singleplayer world for testing the fight loop. Uses
 * vanilla's built-in test world ({@link CreateWorldScreen#showTestWorld} -
 * superflat, creative, cheats on) and then applies a deterministic gamerule
 * set so runs are repeatable:
 *
 * <pre>
 * doDaylightCycle=false, time=noon, doWeatherCycle=false, weather clear,
 * doMobSpawning=false, mobGriefing=false, keepInventory=true,
 * doImmediateRespawn=true, difficulty=normal, gamemode=survival
 * </pre>
 */
public final class TestWorld {
    private TestWorld() {}

    /** Gamerules applied on entry so a debug fight behaves the same every run. */
    public static final List<String> SETUP_COMMANDS = List.of(
        "gamerule doDaylightCycle false",
        "time set noon",
        "gamerule doWeatherCycle false",
        "weather clear",
        "gamerule doMobSpawning false",
        "gamerule mobGriefing false",
        "gamerule keepInventory true",
        "gamerule doImmediateRespawn true",
        "gamerule showDeathMessages false",
        "difficulty normal",
        "gamemode survival @s"
    );

    /**
     * Create and load the vanilla test world. No-op with a message if already
     * in a world - quit to the title screen first, or use the running world.
     */
    public static void createOrLoad(MinecraftClient mc, Consumer<Text> fb) {
        if (mc.world != null) {
            fb.accept(Text.literal("§eAlready in a world. §7/rldebug world setup§e to (re)apply the "
                + "gamerules, or disconnect to the title screen first for a fresh one."));
            return;
        }
        fb.accept(Text.literal("§7Creating vanilla test world…"));
        mc.execute(() -> {
            try {
                CreateWorldScreen.showTestWorld(mc, () -> mc.setScreen(new TitleScreen()));
            } catch (Throwable t) {
                RlMinecraftAiClient.LOGGER.warn("could not create test world", t);
                fb.accept(Text.literal("§ccould not create the test world (" + t.getMessage()
                    + ") - create one by hand and run /rldebug world setup"));
            }
        });
    }

    /**
     * {@link CreateWorldScreen#showTestWorld} only opens the create-world GUI;
     * a human still has to click "Create New World". For the unattended autorun
     * we press that button ourselves. Returns {@code true} once the click has
     * been dispatched (the screen is the test-world {@link CreateWorldScreen}
     * and its create button was found), {@code false} while we're not there yet.
     */
    public static boolean pressCreateButton(MinecraftClient mc) {
        if (!(mc.currentScreen instanceof CreateWorldScreen screen)) return false;
        String want = Text.translatable("selectWorld.create").getString();
        for (ClickableWidget w : Screens.getButtons(screen)) {
            if (w instanceof ButtonWidget b && b.getMessage() != null
                    && want.equals(b.getMessage().getString())) {
                if (!b.active) return false;   // world data still loading
                b.onPress(new MouseInput(0, 0));
                RlMinecraftAiClient.LOGGER.info("autorun: pressed '{}'", want);
                return true;
            }
        }
        return false;
    }

    /** (Re)apply the deterministic gamerule set to the world we're in. */
    public static void applySetup(MinecraftClient mc, Consumer<Text> fb) {
        if (mc.world == null) {
            fb.accept(Text.literal("§cnot in a world"));
            return;
        }
        ServerCmd.runAll(mc, SETUP_COMMANDS, fb);
        fb.accept(Text.literal("§aapplied test-world gamerules (" + SETUP_COMMANDS.size() + " commands)"));
    }
}
