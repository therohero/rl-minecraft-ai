package rl.minecraft.ai.client.combat;

import com.google.gson.JsonObject;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.text.Text;

import rl.minecraft.ai.client.RlConfig;
import rl.minecraft.ai.client.RlMinecraftAiClient;
import rl.minecraft.ai.client.data.EpisodeRecorder;
import rl.minecraft.ai.client.net.Action;
import rl.minecraft.ai.client.net.InferenceClient;
import rl.minecraft.ai.client.net.Spec;
import rl.minecraft.ai.client.obs.ActionApplier;
import rl.minecraft.ai.client.obs.ObservationBuilder;

import java.util.function.Consumer;

/**
 * The brain behind {@code /fight}. Each client tick it finds the nearest
 * player, builds the trained observation, asks the inference server for an
 * action (off-thread), and applies the freshest answer. In {@code train} mode
 * it also streams every {@code (observation, action)} pair to disk via
 * {@link EpisodeRecorder}, tagged with the {@link Kit} it detected from the
 * current inventory.
 */
public final class FightController {
    public enum Mode { IDLE, FIGHTING, TRAINING }

    private final RlConfig cfg;

    private volatile Mode mode = Mode.IDLE;
    private volatile Spec spec = Spec.DEFAULT;
    private InferenceClient inference;
    private EpisodeRecorder recorder;
    private ClientGuard guard;
    private Kit kit = Kit.SWORD;

    // reconstructed self-timers (see ObservationBuilder)
    private int bowDrawTicks = 0;
    private int ticksSinceSwap = 99;
    private int lastSelectedSlot = -1;
    private long lastLogTick = 0;
    private long tick = 0;
    private ClientPlayerEntity lastSelfSeen;

    // match-outcome tracking for the recorded episode (see EpisodeRecorder)
    private boolean sawEnemy = false;
    private double lastSelfHp = Double.NaN;
    private double lastEnemyHp = Double.NaN;
    private PlayerEntity lastTarget = null;
    private String outcome = "unknown";
    private String outcomeReason = "manual stop";

    // Live snapshot for the HUD overlay (see hud/FightHud). Written from the
    // client thread each tick, read from the render thread.
    private volatile String targetLabel = null;

    public FightController(RlConfig cfg) {
        this.cfg = cfg;
    }

    public Mode mode() {
        return mode;
    }

    /** Detected kit id ({@code sword} / {@code axe} / {@code uhc}). */
    public String kitId() {
        return kit.id;
    }

    /** {@code "<name> <dist>m"} of the current target, or {@code null}. */
    public String targetLabel() {
        return targetLabel;
    }

    /** The inference client for this fight, or {@code null} while idle. */
    public InferenceClient inference() {
        return inference;
    }

    /** Ticks recorded so far in {@code train} mode, or -1 when not recording. */
    public int recordedTicks() {
        EpisodeRecorder r = recorder;
        return r == null ? -1 : r.ticks();
    }

    /** Start (or restart) fighting. {@code train} also records a dataset. */
    public synchronized void start(boolean train, Consumer<Text> feedback) {
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc.player == null || mc.world == null) {
            feedback.accept(Text.literal("§cnot in a world"));
            return;
        }
        stopInternal(mc);

        this.kit = Kit.detect(mc.player);
        this.inference = new InferenceClient(cfg.inferenceUrl, cfg.specUrl);
        this.guard = new ClientGuard(cfg);
        this.spec = Spec.DEFAULT;
        fetchSpecAsync();

        if (train) {
            try {
                this.recorder = EpisodeRecorder.start(cfg.datasetDir, kit);
            } catch (Exception e) {
                feedback.accept(Text.literal("§ccould not open dataset file: " + e.getMessage()));
                return;
            }
            this.mode = Mode.TRAINING;
            feedback.accept(Text.literal("§arecording a §e" + kit.id
                + "§a training episode (obs + action per tick) - /fight stop to end."));
            feedback.accept(Text.literal("§7 -> " + recorder.file()
                + " §8(offline-train later: ./run_train_mod.sh)"));
        } else {
            this.mode = Mode.FIGHTING;
            feedback.accept(Text.literal("§afighting (kit: §e" + kit.id
                + "§a) - /fight stop to end"));
        }
        RlMinecraftAiClient.LOGGER.info("fight start: mode={} kit={} inference={}",
            mode, kit.id, cfg.inferenceUrl);
    }

    public synchronized void stop(Consumer<Text> feedback) {
        MinecraftClient mc = MinecraftClient.getInstance();
        boolean wasActive = mode != Mode.IDLE;
        boolean wasTraining = mode == Mode.TRAINING && recorder != null;
        int recordedTicks = wasTraining ? recorder.ticks() : 0;
        String finalOutcome = outcome;
        stopInternal(mc);
        if (wasTraining) {
            feedback.accept(Text.literal("§estopped - recorded §f" + recordedTicks
                + "§e ticks (outcome: §f" + finalOutcome
                + "§e). Offline-train on it with §f./run_train_mod.sh"));
        } else {
            feedback.accept(Text.literal(wasActive ? "§estopped" : "§7not fighting"));
        }
    }

    public void status(Consumer<Text> feedback) {
        MinecraftClient mc = MinecraftClient.getInstance();
        String target = "none";
        if (mode != Mode.IDLE && mc.player != null && mc.world != null) {
            PlayerEntity t = TargetSelector.nearest(mc.world, mc.player);
            if (t != null) target = t.getName().getString()
                + String.format(" (%.1fm)", mc.player.distanceTo(t));
        }
        String server;
        if (inference == null) {
            server = "§7-";
        } else if (inference.seenServer()) {
            double ms = inference.latencyMs();
            server = ms < 0 ? "§aok" : String.format("§aok §7(%.0fms)", ms);
        } else {
            String err = inference.lastError();
            server = (err == null || err.isEmpty()) ? "§eno reply yet" : "§c" + err;
        }
        feedback.accept(Text.literal(String.format(
            "§7mode=§f%s §7kit=§f%s §7target=§f%s §7inference=%s", mode, kit.id, target, server)));
    }

    /** Called from a client-tick event before vanilla input is polled. */
    public void onClientTick(MinecraftClient mc) {
        if (mode == Mode.IDLE) return;
        tick++;
        ClientPlayerEntity self = mc.player;
        ClientWorld world = mc.world;
        if (self == null || world == null || !self.isAlive()) {
            if (self != null && !self.isAlive() && sawEnemy && "unknown".equals(outcome)) {
                outcome = "loss";
                outcomeReason = "self died";
                lastSelfHp = 0.0;
            }
            lastSelfSeen = null; // dead or gone - next live tick is a fresh respawn either way
            return;
        }
        // A respawn hands the client a brand-new player entity instance; the
        // guard's rotation low-pass and our reconstructed timers only make
        // sense for the entity they were built from.
        if (self != lastSelfSeen) {
            lastSelfSeen = self;
            if (guard != null) guard.onRespawn();
            bowDrawTicks = 0;
            ticksSinceSwap = 99;
            lastSelectedSlot = -1;
        }

        PlayerEntity target = TargetSelector.nearest(world, self);
        targetLabel = target == null ? null
            : target.getName().getString() + String.format(" %.1fm", self.distanceTo(target));

        // outcome tracking: a target that was there and is now un-targetable
        // because it died (or vanished at ~0 hp) is a win; anything else that
        // ends the fight with us alive stays "unknown" (manual stop, ran off).
        lastSelfHp = self.getHealth();
        if (target != null) {
            sawEnemy = true;
            lastTarget = target;
            lastEnemyHp = target.getHealth();
        } else if (sawEnemy && "unknown".equals(outcome) && lastTarget != null) {
            boolean enemyDead = !lastTarget.isAlive()
                || (!Double.isNaN(lastEnemyHp) && lastEnemyHp <= 0.5f);
            if (enemyDead) {
                outcome = "win";
                outcomeReason = "enemy down";
                lastEnemyHp = 0.0;
            } else {
                outcomeReason = "enemy left range";
            }
        }

        // advance reconstructed timers from the state we're about to observe
        int selected = self.getInventory().getSelectedSlot();
        ticksSinceSwap = (lastSelectedSlot >= 0 && selected != lastSelectedSlot) ? 0 : ticksSinceSwap + 1;
        lastSelectedSlot = selected;
        boolean drawingBow = self.isUsingItem()
            && KitItem.of(self.getActiveItem()) == KitItem.BOW;
        bowDrawTicks = drawingBow ? bowDrawTicks + 1 : 0;

        double bowDraw = bowDrawTicks / Math.max(1.0, spec.bowMaxDrawSeconds * 20.0);
        double swapLockout = ticksSinceSwap == 0 ? 1.0 : 0.0;

        JsonObject obs = ObservationBuilder.build(mc, self, world, spec, bowDraw, swapLockout);
        inference.requestAsync(obs);

        Action action = inference.latestAction();
        if (action != null) {
            if (ActionApplier.apply(mc, action, target, cfg, guard, tick)) {
                ticksSinceSwap = 0; // a buried-item swap changed the held item without changing the index
            }
            if (mode == Mode.TRAINING && recorder != null) {
                recorder.record(obs, action, target == null ? null : target.getName().getString());
            }
        } else if (tick - lastLogTick > 60 && !inference.seenServer()) {
            lastLogTick = tick;
            String err = inference.lastError();
            if (err == null || err.isEmpty()) {
                self.sendMessage(Text.literal("§6[rl] waiting for the inference server at "
                    + cfg.inferenceUrl + " - run: cd azalea-bot && python inference_server.py "
                    + "--model-dir ./model --port 8800"), false);
            } else {
                self.sendMessage(Text.literal("§c[rl] " + err
                    + " §7- if the shapes mismatch, re-export a current checkpoint "
                    + "(training/python/export_model.py)"), false);
            }
        }
    }

    private void stopInternal(MinecraftClient mc) {
        mode = Mode.IDLE;
        ActionApplier.releaseAll(mc);
        if (recorder != null) {
            recorder.finish(outcome, outcomeReason, lastSelfHp,
                sawEnemy ? lastEnemyHp : Double.NaN);
            recorder.close();
            recorder = null;
        }
        if (inference != null) {
            inference.reset();
        }
        guard = null;
        bowDrawTicks = 0;
        ticksSinceSwap = 99;
        lastSelectedSlot = -1;
        lastSelfSeen = null;
        targetLabel = null;
        sawEnemy = false;
        lastSelfHp = Double.NaN;
        lastEnemyHp = Double.NaN;
        lastTarget = null;
        outcome = "unknown";
        outcomeReason = "manual stop";
    }

    private void fetchSpecAsync() {
        InferenceClient client = this.inference;
        Thread t = new Thread(() -> {
            try {
                Spec fetched = client.fetchSpec();
                if (client == this.inference) {
                    this.spec = fetched;
                    RlMinecraftAiClient.LOGGER.info("loaded spec from server (max_hp={}, arena_radius={})",
                        fetched.maxHp, fetched.arenaRadius);
                }
            } catch (Exception e) {
                RlMinecraftAiClient.LOGGER.warn("could not fetch /spec ({}), using sim defaults",
                    e.getMessage());
            }
        }, "rl-spec-fetch");
        t.setDaemon(true);
        t.start();
    }
}
