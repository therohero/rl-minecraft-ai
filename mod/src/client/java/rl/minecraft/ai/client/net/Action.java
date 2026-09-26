package rl.minecraft.ai.client.net;

import com.google.gson.JsonObject;

/**
 * One decoded {@code /act} response. The continuous fields match
 * spec.json's {@code continuous_action_order}; {@code yawDelta}/{@code pitchDelta}
 * arrive already in <b>radians</b> (the server multiplies by
 * {@code yaw_pitch_delta_scale_radians} before replying). {@code heldSlot}
 * follows spec.json's {@code held_slot_semantics}: {@code 0..hotbarSlots}
 * selects that hotbar slot, higher values are an item-id hotkey.
 *
 * <p>{@code lstmState} is only present for a recurrent ({@code --lstm})
 * checkpoint ({@code spec.lstmHidden > 0}) - opaque to this bot, it's just
 * round-tripped into the next {@code /act} request's {@code lstm_state} by
 * {@link InferenceClient}. {@code null} for the default memoryless MLP
 * policy.
 */
public record Action(
    double moveX, double moveZ, double yawDelta, double pitchDelta,
    boolean jump, boolean attack, boolean sprint, boolean useItem, boolean sneak,
    int heldSlot, JsonObject lstmState
) {
    public static Action fromJson(JsonObject o) {
        return new Action(
            d(o, "move_x"), d(o, "move_z"), d(o, "yaw_delta"), d(o, "pitch_delta"),
            b(o, "jump"), b(o, "attack"), b(o, "sprint"), b(o, "use_item"), b(o, "sneak"),
            o.has("held_slot") ? o.get("held_slot").getAsInt() : -1,
            o.has("lstm_state") && o.get("lstm_state").isJsonObject() ? o.getAsJsonObject("lstm_state") : null
        );
    }

    private static double d(JsonObject o, String k) {
        return o.has(k) ? o.get(k).getAsDouble() : 0.0;
    }

    private static boolean b(JsonObject o, String k) {
        return o.has(k) && o.get(k).getAsBoolean();
    }
}
