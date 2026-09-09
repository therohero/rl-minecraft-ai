package rl.minecraft.ai.client.combat;

import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.util.math.MathHelper;

import rl.minecraft.ai.client.RlConfig;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Per-fight "personal anticheat" between the policy's raw action and the real
 * client input - the client-mod counterpart of {@code azalea_bot/src/guard.rs}.
 * A real vanilla client already enforces jump-on-ground, the hunger cost of
 * sprinting and the attack cooldown, so this concentrates on the two things a
 * client can still give away: <b>rotation that doesn't look mouse-driven</b>
 * and <b>a click stream that's too regular</b>.
 *
 * <h2>Virtual mouse</h2>
 * The policy emits a per-tick yaw/pitch <em>delta</em>. Instead of adding that
 * straight onto the rotation, the guard drives a model of an actual mouse:
 *
 * <ol>
 *   <li>the desired turn rate is delayed by {@code aim_latency_ticks} (a
 *       human reaction lag) and low-pass smoothed, so it's never a perfectly
 *       linear ramp,</li>
 *   <li>the mouse's angular velocity chases that rate under an
 *       <em>acceleration</em> cap ({@code max_yaw_accel_deg}) and a top-speed
 *       cap - so it can't reverse or snap instantly, and naturally overshoots
 *       and settles,</li>
 *   <li>a sub-degree gaussian tremor is mixed in,</li>
 *   <li>the result is converted to a whole number of <em>mouse counts</em>
 *       through the exact vanilla sensitivity curve
 *       ({@code (s*0.6+0.2)^3 * 8}, then {@code * 0.15}) with the leftover
 *       fraction carried to the next tick, and applied via
 *       {@link ClientPlayerEntity#changeLookDirection} - the same call path a
 *       real mouse takes.</li>
 * </ol>
 *
 * Every rotation the server sees is therefore an integer multiple of this
 * client's mouse-count quantum with bounded velocity and acceleration and
 * per-tick noise - which is what rotation-analysis anticheats (Grim, Vulcan,
 * ...) actually score.
 *
 * <h2>Click cadence</h2>
 * On top of the real attack cooldown, {@link #tryAttack} enforces a
 * randomised minimum gap between clicks and a hard clicks-per-second ceiling,
 * so the inter-click series has a human spread instead of a fixed period.
 *
 * <p>One instance lives for a single {@code /fight} session.
 */
public final class ClientGuard {
    private final RlConfig cfg;

    /** Vanilla mouse-count quantum (degrees): {@code sensitivityFactor * 0.15}. */
    private final double quantum;
    /** Vanilla sensitivity factor {@code (s*0.6+0.2)^3 * 8} for this client. */
    private final double sensFactor;

    // virtual-mouse state
    private double smoothedYawRate;
    private double smoothedPitchRate;
    private double mouseVelYaw;
    private double mouseVelPitch;
    private double residualCountsX;
    private double residualCountsY;
    private final Deque<double[]> reactionBuf = new ArrayDeque<>();
    /** Actual applied yaw turn last tick (deg) - feeds the aim-settle gate. */
    private double lastYawTurnDeg;

    // sneak debounce
    private boolean sneakState;
    private int sneakHeldTicks = Integer.MAX_VALUE / 2;

    // click cadence
    private final Deque<Long> clickTimes = new ArrayDeque<>();
    private long nextClickAtNanos = System.nanoTime();

    private long lastHotkeySwapTick = Long.MIN_VALUE / 2;

    public ClientGuard(RlConfig cfg, double mouseSensitivity) {
        this.cfg = cfg;
        double s = MathHelper.clamp(mouseSensitivity, 0.0, 1.0);
        double d = s * 0.6 + 0.2;
        this.sensFactor = d * d * d * 8.0;
        this.quantum = this.sensFactor * 0.15;
    }

    /** The vanilla mouse-count quantum (degrees per mouse count) for this client. */
    public double rotationQuantumDeg() {
        return quantum;
    }

    /**
     * Drive the virtual mouse one tick toward the policy's requested yaw/pitch
     * delta (degrees) and apply the resulting whole-mouse-count turn to
     * {@code self} via the vanilla look path.
     */
    public void applyLook(ClientPlayerEntity self, double rawYawDeltaDeg, double rawPitchDeltaDeg) {
        // 1. reaction delay: act on the turn rate we wanted N ticks ago.
        reactionBuf.addLast(new double[] { rawYawDeltaDeg, rawPitchDeltaDeg });
        double wantYawRate = 0.0;
        double wantPitchRate = 0.0;
        if (reactionBuf.size() > cfg.aimLatencyTicks) {
            double[] delayed = reactionBuf.removeFirst();
            wantYawRate = delayed[0];
            wantPitchRate = delayed[1];
        }

        // 2. low-pass so the aim isn't a constant-delta ramp.
        double a = clamp01(cfg.rotationSmoothing);
        smoothedYawRate += a * (wantYawRate - smoothedYawRate);
        smoothedPitchRate += a * (wantPitchRate - smoothedPitchRate);

        // 3. the hand's angular velocity chases that rate under an accel cap,
        //    then a top-speed cap.
        mouseVelYaw += clampAbs(smoothedYawRate - mouseVelYaw, cfg.maxYawAccelDeg);
        mouseVelPitch += clampAbs(smoothedPitchRate - mouseVelPitch, cfg.maxPitchAccelDeg);
        mouseVelYaw = clampAbs(mouseVelYaw, cfg.maxYawDegPerTick);
        mouseVelPitch = clampAbs(mouseVelPitch, cfg.maxPitchDegPerTick);

        // 4. sub-degree tremor.
        double turnYaw = mouseVelYaw + gaussian() * cfg.aimJitterDeg;
        double turnPitch = mouseVelPitch + gaussian() * cfg.aimJitterDeg;

        // 5. quantise to whole mouse counts, carrying the leftover fraction.
        double countsXf = turnYaw / quantum + residualCountsX;
        double countsYf = turnPitch / quantum + residualCountsY;
        long countsX = Math.round(countsXf);
        long countsY = Math.round(countsYf);
        residualCountsX = countsXf - countsX;
        residualCountsY = countsYf - countsY;

        double appliedYaw = countsX * quantum;
        double appliedPitch = countsY * quantum;

        // clamp pitch the way vanilla does before sending.
        float curPitch = self.getPitch();
        double clampedPitch = MathHelper.clamp(curPitch + appliedPitch, -90.0, 90.0);
        double pitchCounts = countsY;
        if (clampedPitch != curPitch + appliedPitch) {
            pitchCounts = (clampedPitch - curPitch) / quantum;
        }

        self.lastYaw = self.getYaw();
        self.lastPitch = self.getPitch();
        if (countsX != 0 || pitchCounts != 0) {
            // changeLookDirection multiplies by 0.15 only; pre-scale by the
            // sensitivity factor so total = counts * quantum.
            self.changeLookDirection(countsX * sensFactor, pitchCounts * sensFactor);
        }
        self.setHeadYaw(self.getYaw());
        self.setBodyYaw(self.getYaw());

        lastYawTurnDeg = Math.abs(appliedYaw);
    }

    /** Reset the mouse model after a respawn / teleport / pause. */
    public void onRespawn() {
        smoothedYawRate = 0.0;
        smoothedPitchRate = 0.0;
        mouseVelYaw = 0.0;
        mouseVelPitch = 0.0;
        residualCountsX = 0.0;
        residualCountsY = 0.0;
        reactionBuf.clear();
        lastYawTurnDeg = 0.0;
    }

    /** No hit the same tick as a big turn (a classic aim-assist signature). */
    public boolean aimSettled() {
        return lastYawTurnDeg <= cfg.aimSettleDeg;
    }

    /**
     * Randomised click gate on top of the real attack cooldown: a jittered
     * minimum gap between clicks and a hard {@code max_cps} ceiling. Consumes
     * the budget (records a click) only when it returns {@code true}, so call
     * it last, after every other attack precondition has passed.
     */
    public boolean tryAttack() {
        long now = System.nanoTime();
        if (now < nextClickAtNanos) return false;
        long cutoff = now - 1_000_000_000L;
        while (!clickTimes.isEmpty() && clickTimes.peekFirst() < cutoff) clickTimes.removeFirst();
        if (clickTimes.size() >= cfg.maxCps) return false;

        clickTimes.addLast(now);
        double baseMs = 1000.0 / Math.max(0.1, cfg.maxCps);
        double gapMs = Math.max(baseMs * 0.5, baseMs + gaussian() * cfg.clickJitterMs);
        nextClickAtNanos = now + (long) (gapMs * 1_000_000.0);
        return true;
    }

    /**
     * Debounces the sneak toggle: once committed, a state is held for
     * {@code min_sneak_hold_ticks} before it can flip (crouch spam is its own
     * anticheat flag).
     */
    public boolean resolveSneak(boolean wantSneak) {
        sneakHeldTicks++;
        if (wantSneak != sneakState && sneakHeldTicks >= cfg.minSneakHoldTicks) {
            sneakState = wantSneak;
            sneakHeldTicks = 0;
        }
        return sneakState;
    }

    /** True if a buried-inventory hotkey swap is allowed this tick; consumes the gap on yes. */
    public boolean tryHotkeySwap(long tick) {
        if (tick - lastHotkeySwapTick < cfg.hotkeySwapMinGapTicks) return false;
        lastHotkeySwapTick = tick;
        return true;
    }

    private static double clamp01(double v) {
        return Math.max(0.0, Math.min(1.0, v));
    }

    private static double clampAbs(double v, double limit) {
        return Math.max(-limit, Math.min(limit, v));
    }

    /** One draw from an approx. standard normal (sum of 3 uniforms). */
    private static double gaussian() {
        return (Math.random() + Math.random() + Math.random() - 1.5) * 2.0;
    }
}
