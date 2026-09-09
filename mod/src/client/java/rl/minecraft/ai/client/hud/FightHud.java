package rl.minecraft.ai.client.hud;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.render.RenderTickCounter;
import net.minecraft.text.Text;
import net.minecraft.util.Formatting;

import rl.minecraft.ai.client.combat.FightController;
import rl.minecraft.ai.client.net.InferenceClient;

import java.util.ArrayList;
import java.util.List;

/**
 * A small top-left readout shown while {@code /fight} is active: mode
 * (fight / train), the detected {@link rl.minecraft.ai.client.combat.Kit},
 * the current target and the inference-server round-trip. It reads the live
 * {@link FightController} state - nothing here drives the fight.
 *
 * <p>Registered once as a Fabric HUD element (see
 * {@code RlMinecraftAiClient}); it draws nothing while idle, while the
 * vanilla HUD is hidden (F1), or while the F3 debug screen is up.
 */
public final class FightHud {
    private static final int MARGIN = 4;   // gap from the screen edge
    private static final int PAD = 3;      // panel inner padding
    private static final int LINE_GAP = 2; // extra px between lines
    private static final int PANEL_ARGB = 0xA0000000;
    private static final int TEXT_ARGB = 0xFFFFFFFF;

    private final FightController controller;

    public FightHud(FightController controller) {
        this.controller = controller;
    }

    /** Fabric {@code HudElement} entrypoint. */
    public void render(DrawContext ctx, RenderTickCounter tickCounter) {
        FightController.Mode mode = controller.mode();
        if (mode == FightController.Mode.IDLE) return;

        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc.player == null || mc.options.hudHidden) return;
        if (mc.inGameHud.getDebugHud().shouldShowDebugHud()) return;

        boolean training = mode == FightController.Mode.TRAINING;
        List<Text> lines = new ArrayList<>(5);
        lines.add(Text.literal(training ? "● RL TRAIN" : "● RL FIGHT")
            .formatted(training ? Formatting.LIGHT_PURPLE : Formatting.GREEN));
        lines.add(kv("kit", controller.kitId()));
        String target = controller.targetLabel();
        lines.add(kv("target", target == null ? "none" : target));
        lines.add(serverLine(controller.inference()));
        if (training) {
            int t = controller.recordedTicks();
            if (t >= 0) lines.add(kv("recorded", t + "t"));
        }

        int textW = 0;
        for (Text line : lines) textW = Math.max(textW, mc.textRenderer.getWidth(line));
        int lineH = mc.textRenderer.fontHeight + LINE_GAP;
        int panelW = textW + PAD * 2;
        int panelH = lineH * lines.size() - LINE_GAP + PAD * 2;

        int x = MARGIN;
        int y = MARGIN;
        ctx.fill(x, y, x + panelW, y + panelH, PANEL_ARGB);
        int ty = y + PAD;
        for (Text line : lines) {
            ctx.drawTextWithShadow(mc.textRenderer, line, x + PAD, ty, TEXT_ARGB);
            ty += lineH;
        }
    }

    private static Text kv(String key, String value) {
        return Text.literal(key + " ").formatted(Formatting.GRAY)
            .append(Text.literal(value).formatted(Formatting.WHITE));
    }

    private static Text serverLine(InferenceClient inference) {
        if (inference == null) return kv("server", "-");
        if (!inference.seenServer()) {
            String err = inference.lastError();
            String what = (err == null || err.isEmpty()) ? "connecting…" : "unreachable";
            return Text.literal("server ").formatted(Formatting.GRAY)
                .append(Text.literal(what).formatted(Formatting.RED));
        }
        double ms = inference.latencyMs();
        if (ms < 0) {
            return Text.literal("server ").formatted(Formatting.GRAY)
                .append(Text.literal("ok").formatted(Formatting.GREEN));
        }
        Formatting colour = ms < 40 ? Formatting.GREEN : ms < 90 ? Formatting.YELLOW : Formatting.RED;
        return Text.literal("server ").formatted(Formatting.GRAY)
            .append(Text.literal(String.format("%.0fms", ms)).formatted(colour));
    }
}
