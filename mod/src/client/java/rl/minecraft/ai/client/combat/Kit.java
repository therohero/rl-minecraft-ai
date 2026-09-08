package rl.minecraft.ai.client.combat;

import net.minecraft.entity.player.PlayerEntity;

import java.util.EnumSet;
import java.util.Set;

/**
 * The training kits the policy is exported for, matching
 * {@code training/sim/src/kit.rs} and {@code export_model.py}'s
 * {@code hotbar_layouts}. {@link #detect} classifies a live inventory as the
 * <em>nearest</em> of these; when nothing recognisable is carried it falls
 * back to {@link #SWORD}.
 */
public enum Kit {
    SWORD(EnumSet.of(KitItem.SWORD)),
    AXE(EnumSet.of(KitItem.SWORD, KitItem.AXE, KitItem.BOW, KitItem.CROSSBOW)),
    UHC(EnumSet.of(KitItem.SWORD, KitItem.AXE, KitItem.PICKAXE, KitItem.BOW, KitItem.CROSSBOW,
        KitItem.PLANKS, KitItem.COBWEB, KitItem.WATER_BUCKET, KitItem.GOLDEN_APPLE));

    /** Lower-case name the trainer's {@code --kit} flag and spec.json use. */
    public final String id;
    private final Set<KitItem> signature;

    Kit(Set<KitItem> signature) {
        this.id = name().toLowerCase();
        this.signature = signature;
    }

    /**
     * Picks the kit whose item signature best matches what the player is
     * carrying. Score = (signature items actually present) - 0.5 * (carried
     * modelled items outside the signature); the richer kits therefore only
     * win when their extra items are really there. Ties and an empty
     * inventory resolve to {@link #SWORD}.
     */
    public static Kit detect(PlayerEntity player) {
        Set<KitItem> carried = EnumSet.noneOf(KitItem.class);
        var inv = player.getInventory();
        for (int i = 0; i < inv.size(); i++) {
            KitItem it = KitItem.of(inv.getStack(i));
            if (it != KitItem.EMPTY) carried.add(it);
        }
        if (carried.isEmpty()) return SWORD;

        Kit best = SWORD;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (Kit kit : values()) {
            int match = 0;
            for (KitItem it : kit.signature) if (carried.contains(it)) match++;
            int extra = 0;
            for (KitItem it : carried) if (!kit.signature.contains(it)) extra++;
            double score = match - 0.5 * extra;
            // Prefer the simpler kit on a tie: values() is ordered simplest-first
            // and we only replace on a strict improvement.
            if (score > bestScore) {
                bestScore = score;
                best = kit;
            }
        }
        return best;
    }
}
