package rl.minecraft.ai.client;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandManager;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;

import com.mojang.brigadier.arguments.StringArgumentType;

import net.minecraft.util.Identifier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import rl.minecraft.ai.client.combat.FightController;
import rl.minecraft.ai.client.hud.FightHud;

/**
 * Client entrypoint. Wires up the {@code /fight} command family and the
 * per-tick {@link FightController} loop:
 *
 * <ul>
 *   <li>{@code /fight} - fight the nearest player using the trained policy
 *       (served by {@code azalea-bot/inference_server.py}).</li>
 *   <li>{@code /fight train} - same, but also record every
 *       {@code (observation, action)} pair to a dataset, tagged with the kit
 *       auto-detected from the current inventory (uhc / sword / axe, default
 *       sword).</li>
 *   <li>{@code /fight stop} - hand control back to the player.</li>
 *   <li>{@code /fight status} - show mode / kit / current target / server.</li>
 * </ul>
 */
public class RlMinecraftAiClient implements ClientModInitializer {
    public static final String MOD_ID = "rl-minecraft-ai";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    private FightController controller;

    @Override
    public void onInitializeClient() {
        RlConfig config = RlConfig.load();
        this.controller = new FightController(config);
        LOGGER.info("rl-minecraft-ai ready - inference at {}, datasets in {}",
            config.inferenceUrl, config.datasetDir);

        // Drive input at the very start of the tick so the keybinding state we
        // set is picked up by vanilla's input polling this same tick.
        ClientTickEvents.START_CLIENT_TICK.register(controller::onClientTick);

        // Per-frame: glide the aim rotation to the framerate (the per-tick
        // plan is doled out here) so turns don't step 20x/s.
        HudRenderCallback.EVENT.register((ctx, tickCounter) -> controller.onRenderFrame());

        // Small on-screen mode / kit / target / latency readout while fighting.
        if (config.hudEnabled) {
            HudElementRegistry.addLast(
                Identifier.of(MOD_ID, "fight_hud"), new FightHud(controller)::render);
        }

        // Never keep mind-controlling a player across a disconnect.
        ClientPlayConnectionEvents.DISCONNECT.register((handler, client) ->
            controller.stop(t -> LOGGER.info("fight stopped: {}", t.getString())));

        ClientCommandRegistrationCallback.EVENT.register((dispatcher, access) ->
            dispatcher.register(ClientCommandManager.literal("fight")
                .executes(ctx -> {
                    controller.start(false, false, ctx.getSource()::sendFeedback);
                    return 1;
                })
                .then(ClientCommandManager.literal("train")
                    .executes(ctx -> {
                        controller.start(true, false, ctx.getSource()::sendFeedback);
                        return 1;
                    })
                    .then(ClientCommandManager.literal("practice").executes(ctx -> {
                        controller.start(true, true, ctx.getSource()::sendFeedback);
                        return 1;
                    })))
                .then(ClientCommandManager.literal("stop").executes(ctx -> {
                    controller.stop(ctx.getSource()::sendFeedback);
                    return 1;
                }))
                .then(ClientCommandManager.literal("status").executes(ctx -> {
                    controller.status(ctx.getSource()::sendFeedback);
                    return 1;
                }))
                .then(ClientCommandManager.literal("stopfightingtoggle").executes(ctx -> {
                    controller.toggleFightThroughDeath(ctx.getSource()::sendFeedback);
                    return 1;
                }))
                .then(ClientCommandManager.literal("target")
                    .executes(ctx -> {
                        controller.describeTarget(ctx.getSource()::sendFeedback);
                        return 1;
                    })
                    .then(ClientCommandManager.argument("spec", StringArgumentType.greedyString())
                        .executes(ctx -> {
                            controller.setTarget(StringArgumentType.getString(ctx, "spec"),
                                ctx.getSource()::sendFeedback);
                            return 1;
                        })))));
    }
}
