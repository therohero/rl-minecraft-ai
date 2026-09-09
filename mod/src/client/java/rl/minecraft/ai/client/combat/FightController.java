package rl.minecraft.ai.client.combat;

import com.google.gson.JsonObject;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.ChatScreen;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.network.ServerInfo;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.text.Text;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.GameMode;

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
 * The brain behind {@code /fight}. Each client tick it resolves a target
 * (passive by default - see {@link TargetLock}), builds the trained
 * observation, asks the inference server for an action (off-thread), and
 * applies the freshest answer through the {@link ClientGuard}. In
 * {@code train} mode it streams {@code (observation, action)} to disk via
 * {@link EpisodeRecorder}, <b>one file per fight</b>: {@link DeathWatch} and
 * the kill detector cut a new episode on every death / kill so the offline
 * trainer gets clean per-fight returns even on a live server.
 *
 * <p>Server-safety features: passive targeting, {@link #setTarget} locks,
 * auto-pause while a GUI is open or you're a spectator, robust death
 * detection (including a plugin that cancels the vanilla death), and
 * {@code /fight stopfightingtoggle} to choose whether a death hands control
 * back or just rolls to the next episode.
 */
public final class FightController {
    public enum Mode { IDLE, FIGHTING, TRAINING }

    private final RlConfig cfg;
    private final TargetLock lock = new TargetLock();

    private volatile Mode mode = Mode.IDLE;
    private volatile Spec spec = Spec.DEFAULT;
    private InferenceClient inference;
    private EpisodeRecorder recorder;
    private ClientGuard guard;
    private DeathWatch deathWatch;
    private Kit kit = Kit.SWORD;
    private boolean fightThroughDeath;
    private String matchTag = "real";

    // reconstructed self-timers (see ObservationBuilder)
    private int bowDrawTicks = 0;
    private int ticksSinceSwap = 99;
    private int lastSelectedSlot = -1;
    private long lastLogTick = 0;
    private long tick = 0;
    private ClientPlayerEntity lastSelfSeen;
    private String lastDim;
    private int prevHurtTime = 0;
    private boolean pendingRearm = false;

    // pause state (GUI open / spectator)
    private boolean paused = false;

    // who last damaged us - the target while passive, and /fight target last
    private PlayerEntity lastAttacker;

    // per-episode outcome tracking
    private boolean sawEnemy = false;
    private double lastSelfHp = Double.NaN;
    private double lastEnemyHp = Double.NaN;
    private PlayerEntity lastTarget = null;
    private String outcome = "unknown";
    private String outcomeReason = "in progress";
    private int disengageCounter = 0;

    // Live snapshot for the HUD overlay (see hud/FightHud).
    private volatile String targetLabel = null;

    public FightController(RlConfig cfg) {
        this.cfg = cfg;
        this.fightThroughDeath = cfg.fightThroughDeath;
    }

    public Mode mode() {
        return mode;
    }

    public String kitId() {
        return kit.id;
    }

    public String targetLabel() {
        return targetLabel;
    }

    public InferenceClient inference() {
        return inference;
    }

    /** Ticks recorded in the current episode, or -1 when not recording. */
    public int recordedTicks() {
        EpisodeRecorder r = recorder;
        return r == null ? -1 : r.ticks();
    }

    // ------------------------------------------------------------------ commands

    /** Start (or restart) fighting. {@code train} records; {@code practice} tags the data. */
    public synchronized void start(boolean train, boolean practice, Consumer<Text> feedback) {
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc.player == null || mc.world == null) {
            feedback.accept(Text.literal("§cnot in a world"));
            return;
        }
        stopInternal(mc, "unknown", "restarted");

        this.kit = Kit.detect(mc.player);
        this.inference = new InferenceClient(cfg.inferenceUrl, cfg.specUrl);
        this.guard = new ClientGuard(cfg, resolveSensitivity(mc));
        this.deathWatch = new DeathWatch(cfg);
        this.spec = Spec.DEFAULT;
        this.matchTag = practice ? "practice" : "real";
        this.lastDim = mc.world.getRegistryKey().getValue().toString();
        lock.passive();
        fetchSpecAsync();

        if (train) {
            try {
                this.recorder = EpisodeRecorder.start(cfg.datasetDir, kit, serverTag(mc), matchTag);
            } catch (Exception e) {
                feedback.accept(Text.literal("§ccould not open dataset dir: " + e.getMessage()));
                return;
            }
            this.mode = Mode.TRAINING;
            feedback.accept(Text.literal("§arecording §e" + kit.id + "§a training ("
                + matchTag + ") - one file per fight, /fight stop to end."));
            feedback.accept(Text.literal("§7 -> " + recorder.sessionDir()
                + " §8(offline-train later: ./run_train_mod.sh)"));
        } else {
            this.mode = Mode.FIGHTING;
            feedback.accept(Text.literal("§afighting (kit: §e" + kit.id + "§a) - /fight stop to end"));
        }
        feedback.accept(Text.literal("§7target: §fpassive§7 - it won't attack anyone until they hit "
            + "you or you run §f/fight target <name>"));
        RlMinecraftAiClient.LOGGER.info("fight start: mode={} kit={} match={} inference={}",
            mode, kit.id, matchTag, cfg.inferenceUrl);
    }

    public synchronized void stop(Consumer<Text> feedback) {
        MinecraftClient mc = MinecraftClient.getInstance();
        boolean wasActive = mode != Mode.IDLE;
        boolean wasTraining = mode == Mode.TRAINING && recorder != null;
        int episodes = wasTraining ? recorder.episodeIndex() : 0;
        int total = wasTraining ? recorder.totalTicks() : 0;
        stopInternal(mc, sawEnemy ? outcome : "unknown", "manual stop");
        if (wasTraining) {
            feedback.accept(Text.literal("§estopped - recorded §f" + total + "§e ticks across §f"
                + (episodes + 1) + "§e episode(s). Offline-train: §f./run_train_mod.sh"));
        } else {
            feedback.accept(Text.literal(wasActive ? "§estopped" : "§7not fighting"));
        }
    }

    public synchronized void toggleFightThroughDeath(Consumer<Text> feedback) {
        fightThroughDeath = !fightThroughDeath;
        feedback.accept(Text.literal("§efight-through-death: §f" + (fightThroughDeath ? "ON" : "OFF")
            + "§7 - " + (fightThroughDeath
                ? "a death just rolls to the next episode, control stays with the bot"
                : "a death hands control back to you")));
    }

    /** {@code /fight target <spec>}: passive | auto | nearest | look | last | region | <name>. */
    public synchronized void setTarget(String spec, Consumer<Text> fb) {
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mode == Mode.IDLE || mc.player == null || mc.world == null) {
            fb.accept(Text.literal("§7not fighting"));
            return;
        }
        String s = spec == null ? "" : spec.trim();
        switch (s.toLowerCase()) {
            case "", "?", "status" -> describeTarget(fb);
            case "clear", "none", "passive" -> {
                lock.passive();
                lastAttacker = null;
                fb.accept(Text.literal("§etarget: §fpassive§e (fight back only)"));
            }
            case "auto" -> {
                lock.auto();
                fb.accept(Text.literal("§6target: §fauto§6 - nearest player. Use only where "
                    + "you're authorised to; it will swing at bystanders."));
            }
            case "nearest" -> {
                PlayerEntity n = TargetSelector.nearest(mc.world, mc.player, 0);
                if (n == null) {
                    fb.accept(Text.literal("§7no targetable player nearby"));
                } else {
                    lock.byEntity(n.getId());
                    fb.accept(Text.literal("§etarget locked: §f" + n.getName().getString()));
                }
            }
            case "look", "crosshair" -> {
                PlayerEntity n = TargetSelector.underCrosshair(mc);
                if (n == null) {
                    fb.accept(Text.literal("§7you're not looking at a player"));
                } else {
                    lock.byEntity(n.getId());
                    fb.accept(Text.literal("§etarget locked: §f" + n.getName().getString()));
                }
            }
            case "last", "me" -> {
                if (lastAttacker == null) {
                    fb.accept(Text.literal("§7nobody has hit you yet"));
                } else {
                    lock.byName(lastAttacker.getName().getString());
                    fb.accept(Text.literal("§etarget locked: §f" + lastAttacker.getName().getString()));
                }
            }
            case "region" -> {
                BlockPos p = mc.player.getBlockPos();
                if (lock.regionCorner(p)) {
                    fb.accept(Text.literal("§eregion set §7- engaging any player inside it"));
                } else {
                    fb.accept(Text.literal("§7region corner 1 set - run §f/fight target region§7 "
                        + "again at the opposite corner"));
                }
            }
            default -> {
                lock.byName(s);
                PlayerEntity n = TargetSelector.byName(mc.world, mc.player, s);
                fb.accept(Text.literal("§etarget: §fname=" + s
                    + (n == null ? "§7 (not visible right now)" : "")));
            }
        }
    }

    public void describeTarget(Consumer<Text> fb) {
        String extra = lastAttacker != null ? " §8(last hit by " + lastAttacker.getName().getString() + ")" : "";
        fb.accept(Text.literal("§7target lock: §f" + lock.describe() + extra));
    }

    public void status(Consumer<Text> feedback) {
        MinecraftClient mc = MinecraftClient.getInstance();
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
        int ep = recorder == null ? -1 : recorder.episodeIndex();
        feedback.accept(Text.literal(String.format(
            "§7mode=§f%s §7kit=§f%s §7target=§f%s%s §7inference=%s%s",
            mode, kit.id, lock.describe(), paused ? " §c[paused]" : "",
            server, ep >= 0 ? " §7episode=§f" + ep : "")));
        feedback.accept(Text.literal("§7fight-through-death=§f" + (fightThroughDeath ? "on" : "off")
            + "§7  (toggle: §f/fight stopfightingtoggle§7)"));
    }

    // ------------------------------------------------------------------ tick loop

    /** Called from a client-tick event before vanilla input is polled. */
    public void onClientTick(MinecraftClient mc) {
        if (mode == Mode.IDLE) return;
        tick++;
        ClientPlayerEntity self = mc.player;
        ClientWorld world = mc.world;
        if (self == null || world == null) {
            ActionApplier.releaseAll(mc);
            lastSelfSeen = null;
            return;
        }

        // --- pause gate ---
        String pause = pauseReason(mc);
        if (pause != null) {
            enterPause(mc, pause);
            return;
        }
        leavePause();

        // --- respawn / new player-entity instance (== a death mid-fight) ---
        if (self != lastSelfSeen) {
            boolean hadPrevious = lastSelfSeen != null;
            lastSelfSeen = self;
            if (guard != null) guard.onRespawn();
            bowDrawTicks = 0;
            ticksSinceSwap = 99;
            lastSelectedSlot = -1;
            prevHurtTime = 0;
            lastAttacker = null;
            if (hadPrevious) {
                String r = deathWatch.onPlayerEntitySwapped();
                if (r != null && handleDeath(mc, r)) return;
            }
            deathWatch.reset();
            lastDim = world.getRegistryKey().getValue().toString();
        }

        // --- dimension change (portal etc.) - an episode boundary, not a stop ---
        String dim = world.getRegistryKey().getValue().toString();
        if (!dim.equals(lastDim)) {
            lastDim = dim;
            if (mode == Mode.TRAINING && recorder != null) {
                recorder.endEpisode("unknown", "dimension change", lastSelfHp,
                    sawEnemy ? lastEnemyHp : Double.NaN);
            }
            resetEpisodeOutcome();
            deathWatch.reset();
            if (guard != null) guard.onRespawn();
        }

        GameMode gm = gameMode(mc);
        if (!self.isAlive()) {
            String r = deathWatch.poll(self, world, gm);
            if (r != null && handleDeath(mc, r)) return;
            ActionApplier.releaseAll(mc);
            return;
        }

        // re-arm death detection once we're alive and off the floor again
        if (pendingRearm && self.getHealth() > cfg.deathHpFloor) {
            deathWatch.rearm();
            pendingRearm = false;
        }

        // --- death detection (vanilla + plugin-blocked) ---
        String death = deathWatch.poll(self, world, gm);
        if (death != null && handleDeath(mc, death)) return;

        // --- who last hit us ---
        trackAttacker(self, world);

        // --- resolve the target ---
        PlayerEntity target = lock.resolve(world, self, cfg, lastAttacker);
        targetLabel = target == null ? (lock.kind() == TargetLock.Kind.PASSIVE ? "passive" : null)
            : target.getName().getString() + String.format(" %.1fm", self.distanceTo(target));

        // --- per-episode outcome tracking + kill/disengage episode boundaries ---
        trackOutcome(self, target);

        // --- reconstructed timers (self_bow_draw / self_swap_lockout) ---
        int selected = self.getInventory().getSelectedSlot();
        ticksSinceSwap = (lastSelectedSlot >= 0 && selected != lastSelectedSlot) ? 0 : ticksSinceSwap + 1;
        lastSelectedSlot = selected;
        boolean drawingBow = self.isUsingItem() && KitItem.of(self.getActiveItem()) == KitItem.BOW;
        bowDrawTicks = drawingBow ? bowDrawTicks + 1 : 0;
        double bowDraw = bowDrawTicks / Math.max(1.0, spec.bowMaxDrawSeconds * 20.0);
        double swapLockout = ticksSinceSwap == 0 ? 1.0 : 0.0;

        JsonObject obs = ObservationBuilder.build(mc, self, world, spec, bowDraw, swapLockout);
        inference.requestAsync(obs);

        Action action = inference.latestAction();
        if (action != null) {
            if (ActionApplier.apply(mc, action, target, cfg, guard, tick)) {
                ticksSinceSwap = 0;
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

    // ------------------------------------------------------------------ helpers

    private String pauseReason(MinecraftClient mc) {
        if (cfg.pauseOnScreen && mc.currentScreen != null && !(mc.currentScreen instanceof ChatScreen)) {
            return "screen open";
        }
        if (gameMode(mc) == GameMode.SPECTATOR) return "spectator";
        return null;
    }

    private void enterPause(MinecraftClient mc, String reason) {
        if (!paused) {
            paused = true;
            ActionApplier.releaseAll(mc);
            if (guard != null) guard.onRespawn();
            RlMinecraftAiClient.LOGGER.info("fight paused ({})", reason);
        }
        targetLabel = "paused: " + reason;
    }

    private void leavePause() {
        if (paused) {
            paused = false;
            if (guard != null) guard.onRespawn();
            RlMinecraftAiClient.LOGGER.info("fight resumed");
        }
    }

    private void trackAttacker(ClientPlayerEntity self, ClientWorld world) {
        int hurt = self.hurtTime;
        if (hurt > 0 && prevHurtTime == 0) {
            PlayerEntity a = null;
            if (self.getAttacker() instanceof PlayerEntity p && TargetSelector.targetable(self, p)) {
                a = p;
            }
            if (a == null) a = TargetSelector.nearest(world, self, 6.0);
            if (a != null) lastAttacker = a;
        }
        prevHurtTime = hurt;
        if (lastAttacker != null
            && (!lastAttacker.isAlive() || self.squaredDistanceTo(lastAttacker) > 40.0 * 40.0)) {
            lastAttacker = null;
        }
    }

    private void trackOutcome(ClientPlayerEntity self, PlayerEntity target) {
        lastSelfHp = self.getHealth();
        if (target != null) {
            sawEnemy = true;
            lastTarget = target;
            lastEnemyHp = target.getHealth();
            disengageCounter = 0;
            if ("in progress".equals(outcomeReason)) outcome = "unknown";
            return;
        }
        if (!sawEnemy) return;

        boolean killed = lastTarget != null
            && (!lastTarget.isAlive() || (!Double.isNaN(lastEnemyHp) && lastEnemyHp <= 0.5));
        if (killed) {
            RlMinecraftAiClient.LOGGER.info("kill detected -> episode boundary");
            if (mode == Mode.TRAINING && recorder != null) {
                recorder.endEpisode("win", "enemy down", lastSelfHp, 0.0);
            }
            resetEpisodeOutcome();
            lastAttacker = null;
        } else if (++disengageCounter >= cfg.disengageTicks) {
            RlMinecraftAiClient.LOGGER.info("target gone {} ticks -> episode boundary", disengageCounter);
            if (mode == Mode.TRAINING && recorder != null) {
                recorder.endEpisode("unknown", "disengaged", lastSelfHp,
                    Double.isNaN(lastEnemyHp) ? Double.NaN : lastEnemyHp);
            }
            resetEpisodeOutcome();
            lastAttacker = null;
        }
    }

    /** @return true if this tick's processing should stop (control handed back or dead). */
    private boolean handleDeath(MinecraftClient mc, String reason) {
        RlMinecraftAiClient.LOGGER.info("death detected: {}", reason);
        if (fightThroughDeath) {
            if (mode == Mode.TRAINING && recorder != null) {
                recorder.endEpisode("loss", reason, 0.0, sawEnemy ? lastEnemyHp : Double.NaN);
            }
            resetEpisodeOutcome();
            lastAttacker = null;
            pendingRearm = true;
            ActionApplier.releaseAll(mc);
            if (mc.player != null) {
                mc.player.sendMessage(Text.literal("§e[rl] died (" + reason
                    + ") - continuing to the next episode. §7/fight stopfightingtoggle to change"), false);
            }
            return true;
        }
        Consumer<Text> chat = t -> {
            if (mc.player != null) mc.player.sendMessage(t, false);
        };
        stopInternal(mc, "loss", reason);
        chat.accept(Text.literal("§e[rl] died (" + reason + ") - control handed back. "
            + "§7/fight stopfightingtoggle to fight through deaths"));
        return true;
    }

    private void resetEpisodeOutcome() {
        sawEnemy = false;
        lastEnemyHp = Double.NaN;
        lastTarget = null;
        outcome = "unknown";
        outcomeReason = "in progress";
        disengageCounter = 0;
    }

    private void stopInternal(MinecraftClient mc, String episodeOutcome, String episodeReason) {
        mode = Mode.IDLE;
        ActionApplier.releaseAll(mc);
        if (recorder != null) {
            recorder.endEpisode(episodeOutcome, episodeReason, lastSelfHp,
                sawEnemy ? lastEnemyHp : Double.NaN);
            recorder = null;
        }
        if (inference != null) inference.reset();
        guard = null;
        deathWatch = null;
        lock.passive();
        bowDrawTicks = 0;
        ticksSinceSwap = 99;
        lastSelectedSlot = -1;
        lastSelfSeen = null;
        lastDim = null;
        prevHurtTime = 0;
        pendingRearm = false;
        paused = false;
        lastAttacker = null;
        targetLabel = null;
        resetEpisodeOutcome();
    }

    private double resolveSensitivity(MinecraftClient mc) {
        if (cfg.mouseSensitivity >= 0.0) return cfg.mouseSensitivity;
        try {
            return mc.options.getMouseSensitivity().getValue();
        } catch (Exception e) {
            return 0.5;
        }
    }

    private GameMode gameMode(MinecraftClient mc) {
        return mc.interactionManager != null ? mc.interactionManager.getCurrentGameMode() : GameMode.DEFAULT;
    }

    private static String serverTag(MinecraftClient mc) {
        if (mc.isInSingleplayer()) return "singleplayer";
        ServerInfo s = mc.getCurrentServerEntry();
        return s != null && s.address != null && !s.address.isBlank() ? s.address : "unknown";
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
