package rl.minecraft.ai.client.combat;

import rl.minecraft.ai.client.RlConfig;

/**
 * Per-fight "personal anticheat" state sitting between the policy's raw
 * action and the real client input it drives - the client-mod counterpart of
 * {@code azalea_bot/src/guard.rs}. Unlike a headless bot, this mod moves an
 * actual vanilla player: real physics already refuses an illegal jump or an
 * over-hunger sprint, and the real attack cooldown already bounds click
 * rate. What a real client can still give away is a *robotic* rotation
 * (perfectly linear turns, always on the same fraction of a degree) and a
 * hit landed the same tick as a big snap-turn, so that is what this guards.
 *
 * <p>One instance lives for the duration of a single {@code /fight} session
 * (owned by {@code FightController}) so its rotation low-pass and debounce
 * timers carry across ticks.
 */
public final class ClientGuard {
    /** The rotation quantum (degrees) vanilla's default 50% sensitivity snaps every look to. */
    private static final double ROTATION_GCD_DEG = 0.15;

    private final RlConfig cfg;

    private double smoothedYawDeltaDeg;
    private double smoothedPitchDeltaDeg;
    /** Actual applied yaw turn last tick (deg) - feeds the aim-settle gate. */
    private double lastYawTurnDeg;

    private boolean sneakState;
    private int sneakHeldTicks = Integer.MAX_VALUE / 2;

    private long lastHotkeySwapTick = Long.MIN_VALUE / 2;

    public ClientGuard(RlConfig cfg) {
        this.cfg = cfg;
    }

    /**
     * Turns the policy's raw per-tick yaw/pitch delta (degrees) into what a
     * human-sensitivity client would actually send: low-pass smoothed, rate
     * clamped, given a touch of gaussian tremor, then snapped to the vanilla
     * mouse-sensitivity grid. Returns the applied {@code {yawDelta, pitchDelta}}.
     */
    public double[] resolveRotation(double rawYawDeltaDeg, double rawPitchDeltaDeg) {
        double a = clamp01(cfg.rotationSmoothing);
        smoothedYawDeltaDeg += a * (rawYawDeltaDeg - smoothedYawDeltaDeg);
        smoothedPitchDeltaDeg += a * (rawPitchDeltaDeg - smoothedPitchDeltaDeg);

        double dy = clamp(smoothedYawDeltaDeg, cfg.maxYawDegPerTick);
        double dp = clamp(smoothedPitchDeltaDeg, cfg.maxPitchDegPerTick);
        // Bleed the clamped-off part out of the low-pass memory so it doesn't
        // accumulate into a lasting turn-rate bias.
        smoothedYawDeltaDeg = dy;
        smoothedPitchDeltaDeg = dp;

        if (cfg.aimJitterDeg > 0.0) {
            dy += gaussian() * cfg.aimJitterDeg;
            dp += gaussian() * cfg.aimJitterDeg;
        }
        dy = snapToGrid(dy);
        dp = snapToGrid(dp);

        lastYawTurnDeg = Math.abs(dy);
        return new double[] { dy, dp };
    }

    /** Reset the rotation low-pass after a respawn/teleport (carried-over deltas would be meaningless). */
    public void onRespawn() {
        smoothedYawDeltaDeg = 0.0;
        smoothedPitchDeltaDeg = 0.0;
        lastYawTurnDeg = 0.0;
    }

    /**
     * Whether an attack is allowed to land *this* tick given how much the
     * aim just moved - a big snap-turn landing a hit on the same tick is a
     * classic aim-assist signature, so the guard holds fire one tick for the
     * aim to settle.
     */
    public boolean aimSettled() {
        return lastYawTurnDeg <= cfg.aimSettleDeg;
    }

    /**
     * Debounces the sneak toggle: once a state is committed it must be held
     * for {@code min_sneak_hold_ticks} before it can flip again (rapid crouch
     * spam is its own anticheat flag).
     */
    public boolean resolveSneak(boolean wantSneak) {
        sneakHeldTicks++;
        if (wantSneak != sneakState && sneakHeldTicks >= cfg.minSneakHoldTicks) {
            sneakState = wantSneak;
            sneakHeldTicks = 0;
        }
        return sneakState;
    }

    /** True if a buried-inventory hotkey swap is allowed this tick (rate-limited); consumes the gap on yes. */
    public boolean tryHotkeySwap(long tick) {
        if (tick - lastHotkeySwapTick < cfg.hotkeySwapMinGapTicks) return false;
        lastHotkeySwapTick = tick;
        return true;
    }

    private double clamp(double v, double limit) {
        return Math.max(-limit, Math.min(limit, v));
    }

    private double clamp01(double v) {
        return Math.max(0.0, Math.min(1.0, v));
    }

    private static double snapToGrid(double deg) {
        return Math.round(deg / ROTATION_GCD_DEG) * ROTATION_GCD_DEG;
    }

    /** One draw from an approx. standard normal (sum of 3 uniforms), plenty for sub-degree aim noise. */
    private static double gaussian() {
        return (Math.random() + Math.random() + Math.random() - 1.5) * 2.0;
    }
}
