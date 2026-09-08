package rl.minecraft.ai.client.net;

import com.google.gson.JsonObject;

/**
 * One decoded {@code /act} response. The continuous fields match
 * spec.json's {@code continuous_action_order}; {@code yawDelta}/{@code pitchDelta}
 * arrive already in <b>radians</b> (the server multiplies by
 * {@code yaw_pitch_delta_scale_radians} before replying). {@code heldSlot}
 * follows spec.json's {@code held_slot_semantics}: {@code 0..hotbarSlots}
 * selects that hotbar slot, higher values are an item-id hotkey.
 */
public record Action(
    double moveX, double moveZ, double yawDelta, double pitchDelta,
    boolean jump, boolean attack, boolean sprint, boolean useItem, boolean sneak,
    int heldSlot
) {
    public static Action fromJson(JsonObject o) {
        return new Action(
            d(o, "move_x"), d(o, "move_z"), d(o, "yaw_delta"), d(o, "pitch_delta"),
            b(o, "jump"), b(o, "attack"), b(o, "sprint"), b(o, "use_item"), b(o, "sneak"),
            o.has("held_slot") ? o.get("held_slot").getAsInt() : -1
        );
    }

    private static double d(JsonObject o, String k) {
        return o.has(k) ? o.get(k).getAsDouble() : 0.0;
    }

    private static boolean b(JsonObject o, String k) {
        return o.has(k) && o.get(k).getAsBoolean();
    }
}
