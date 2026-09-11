package rl.minecraft.ai.client.debug;

import net.minecraft.client.MinecraftClient;
import net.minecraft.text.Text;

import rl.minecraft.ai.client.combat.Kit;

import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;

/**
 * Kit loadouts for the debug harness. Each list of {@code /give} commands
 * produces an inventory that {@link Kit#detect} classifies as the matching
 * training kit (mirrors {@code training/sim/src/kit.rs} item signatures).
 */
public final class Scenario {
    private Scenario() {}

    public static List<String> giveCommands(String kitId) {
        return switch (kitId.toLowerCase(Locale.ROOT)) {
            case "sword" -> List.of(
                "clear @s",
                "give @s minecraft:diamond_sword",
                "give @s minecraft:cooked_beef 16");
            case "axe" -> List.of(
                "clear @s",
                "give @s minecraft:diamond_sword",
                "give @s minecraft:diamond_axe",
                "give @s minecraft:bow",
                "give @s minecraft:arrow 64",
                "give @s minecraft:cooked_beef 16");
            case "uhc" -> List.of(
                "clear @s",
                "give @s minecraft:diamond_sword",
                "give @s minecraft:diamond_axe",
                "give @s minecraft:diamond_pickaxe",
                "give @s minecraft:bow",
                "give @s minecraft:crossbow",
                "give @s minecraft:arrow 64",
                "give @s minecraft:oak_planks 64",
                "give @s minecraft:cobweb 8",
                "give @s minecraft:water_bucket",
                "give @s minecraft:golden_apple 8");
            default -> List.of();
        };
    }

    public static Kit expected(String kitId) {
        return switch (kitId.toLowerCase(Locale.ROOT)) {
            case "axe" -> Kit.AXE;
            case "uhc" -> Kit.UHC;
            default -> Kit.SWORD;
        };
    }

    /** Issue the /give commands for {@code kitId}. Returns false for an unknown kit. */
    public static boolean give(MinecraftClient mc, String kitId, Consumer<Text> fb) {
        List<String> cmds = giveCommands(kitId);
        if (cmds.isEmpty()) {
            fb.accept(Text.literal("§cunknown kit '" + kitId + "' (sword|axe|uhc)"));
            return false;
        }
        ServerCmd.runAll(mc, cmds, fb);
        fb.accept(Text.literal("§agave §e" + kitId + "§a loadout - /rldebug status to confirm the "
            + "detected kit once the items land"));
        return true;
    }
}
