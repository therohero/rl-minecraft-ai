package rl.minecraft.ai.client.obs;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.effect.StatusEffectInstance;
import net.minecraft.entity.effect.StatusEffect;
import net.minecraft.entity.effect.StatusEffects;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.entity.projectile.PersistentProjectileEntity;
import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.item.ItemStack;
import net.minecraft.item.Items;
import net.minecraft.registry.tag.FluidTags;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;

import rl.minecraft.ai.client.combat.KitItem;
import rl.minecraft.ai.client.net.Spec;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Builds the exact observation dict the policy was trained on, straight from
 * live client state. This is a Java port of
 * {@code azalea-bot/azalea_bot/src/main.rs::build_observation} (which in turn
 * mirrors {@code training/sim/src/observation.rs}); keep the three in sync.
 *
 * <p>Fields the real protocol doesn't hand a client as one ready value are
 * reconstructed by the caller and passed in ({@code bowDraw},
 * {@code swapLockout}); {@code shield_disabled} and {@code mining} are read
 * straight off client-local vanilla state (the item-cooldown manager and the
 * interaction manager's own mining flag - no packet tracking needed, unlike
 * {@code azalea_bot}'s headless {@code tracker.rs}). Sim-only geometry
 * ({@code dist_from_center}, {@code time_left}) is filled from spec so the
 * normalised feature lands in the trained range.
 */
public final class ObservationBuilder {
    private static final ItemStack SHIELD_PROBE = new ItemStack(Items.SHIELD);

    private ObservationBuilder() {}

    /** Sim look convention == Minecraft's, only radians vs degrees. */
    private static double simYaw(float mcYawDeg) {
        return Math.toRadians(mcYawDeg);
    }

    public static JsonObject build(MinecraftClient client, ClientPlayerEntity self, ClientWorld world, Spec spec,
                                   double bowDraw, double swapLockout) {
        JsonObject o = new JsonObject();

        Vec3d pos = self.getEntityPos();
        Vec3d vel = self.getVelocity();
        double yaw = simYaw(self.getYaw());
        double pitch = Math.toRadians(self.getPitch());
        double sinY = Math.sin(yaw);
        double cosY = Math.cos(yaw);

        // --- self scalars ---
        o.addProperty("self_hp", self.getHealth());
        o.addProperty("self_vel_x", vel.x);
        o.addProperty("self_vel_y", vel.y);
        o.addProperty("self_vel_z", vel.z);
        o.addProperty("self_yaw", yaw);
        o.addProperty("self_pitch", pitch);
        o.addProperty("self_on_ground", self.isOnGround());
        o.addProperty("self_attack_cooldown", clamp01(self.getAttackCooldownProgress(0.0f)));
        o.addProperty("self_ping_ms", pingMs(self));
        boolean usingItem = self.isUsingItem();
        KitItem held = KitItem.of(self.getMainHandStack());
        o.addProperty("self_shield", usingItem && held.isShieldish() ? 1.0 : 0.0);

        double dist = Math.sqrt(pos.x * pos.x + pos.z * pos.z);
        o.addProperty("self_dist_from_center", dist / spec.arenaRadius);

        double selfGround = surfaceY(world, pos.x, pos.z, pos.y);
        // sim convention: forward = (-sin, cos), right = (cos, sin); sample 1 block out.
        double groundFwd = surfaceY(world, pos.x - sinY, pos.z + cosY, pos.y);
        double groundRight = surfaceY(world, pos.x + cosY, pos.z + sinY, pos.y);
        o.addProperty("self_ground_height", selfGround);
        o.addProperty("self_slope_forward", groundFwd - selfGround);
        o.addProperty("self_slope_right", groundRight - selfGround);

        o.addProperty("self_hurt", clamp01(self.hurtTime / 10.0));
        o.addProperty("self_held", KitItem.of(self.getMainHandStack()).ordinal());
        o.addProperty("self_absorption", self.getAbsorptionAmount());
        o.addProperty("self_eating", usingItem && (held.isFood() || held.isSplashPotion()) ? 1.0 : 0.0);
        o.addProperty("self_bow_draw", clamp01(bowDraw));
        o.addProperty("self_burning", self.isOnFire() ? 1.0 : 0.0);
        o.addProperty("self_shield_disabled", clamp01(self.getItemCooldownManager().getCooldownProgress(SHIELD_PROBE, 0.0f)));
        o.addProperty("self_swap_lockout", clamp01(swapLockout));
        o.addProperty("self_mining", client.interactionManager != null && client.interactionManager.isBreakingBlock() ? 1.0 : 0.0);
        o.addProperty("self_sneaking", self.isSneaking() ? 1.0 : 0.0);
        o.addProperty("self_food", self.getHungerManager().getFoodLevel());

        // self_effects: one entry per effects.rs::Effect, in that order;
        // amplifier + 1 while active, else 0. The two instant effects never
        // persist, so they're always 0.
        JsonArray eff = new JsonArray();
        eff.add(effLevel(self, StatusEffects.SPEED));
        eff.add(effLevel(self, StatusEffects.SLOWNESS));
        eff.add(effLevel(self, StatusEffects.STRENGTH));
        eff.add(effLevel(self, StatusEffects.WEAKNESS));
        eff.add(effLevel(self, StatusEffects.REGENERATION));
        eff.add(effLevel(self, StatusEffects.POISON));
        eff.add(0.0); // instant_health
        eff.add(0.0); // instant_damage
        eff.add(effLevel(self, StatusEffects.FIRE_RESISTANCE));
        o.add("self_effects", eff);

        // --- inventory counts + hotbar layout + arrows + selected slot ---
        int hotbarSlots = spec.hotbarSlots;
        double[] counts = new double[KitItem.COUNT];
        int arrows = 0;
        for (int i = 0; i < self.getInventory().size(); i++) {
            ItemStack st = self.getInventory().getStack(i);
            if (st.isEmpty()) continue;
            if (KitItem.isArrow(st)) arrows += st.getCount();
            KitItem it = KitItem.of(st);
            if (it != KitItem.EMPTY) counts[it.ordinal() - 1] += st.getCount();
        }
        JsonArray inv = new JsonArray();
        for (double c : counts) inv.add(c);
        o.add("inventory", inv);

        int selected = self.getInventory().getSelectedSlot();
        JsonArray hotbar = new JsonArray();
        for (int i = 0; i < hotbarSlots; i++) {
            hotbar.add(KitItem.of(self.getInventory().getStack(i)).ordinal());
        }
        o.add("hotbar", hotbar);
        o.addProperty("self_arrows", arrows);
        o.addProperty("self_slot", Math.min(selected, hotbarSlots - 1));

        // --- other players, nearest first, split friend/foe ---
        List<PlayerEntity> others = new ArrayList<>();
        for (PlayerEntity p : world.getPlayers()) {
            if (p == self || !p.isAlive() || p.isSpectator()) continue;
            others.add(p);
        }
        others.sort(Comparator.comparingDouble(self::squaredDistanceTo));

        JsonArray enemies = new JsonArray();
        JsonArray teammates = new JsonArray();
        int nEnemies = 0, nTeammates = 0;
        for (PlayerEntity p : others) {
            boolean ally = self.isTeammate(p);
            if (ally && nTeammates >= spec.maxObservedTeammates) continue;
            if (!ally && nEnemies >= spec.maxObservedEnemies) continue;
            JsonObject row = otherRow(p, self, world, sinY, cosY);
            if (ally) { teammates.add(row); nTeammates++; }
            else { enemies.add(row); nEnemies++; }
        }
        padPresentRows(enemies, spec.maxObservedEnemies);
        padPresentRows(teammates, spec.maxObservedTeammates);
        o.add("enemies", enemies);
        o.add("teammates", teammates);

        // --- nearest arrows ---
        JsonArray projectiles = new JsonArray();
        List<PersistentProjectileEntity> arrowsList = new ArrayList<>();
        for (var e : world.getEntities()) {
            if (e instanceof PersistentProjectileEntity ppe && ppe.isAlive()) arrowsList.add(ppe);
        }
        arrowsList.sort(Comparator.comparingDouble(self::squaredDistanceTo));
        for (int i = 0; i < spec.maxObservedProjectiles; i++) {
            JsonObject row = new JsonObject();
            if (i < arrowsList.size()) {
                PersistentProjectileEntity a = arrowsList.get(i);
                double[] rel = rot(a.getX() - pos.x, a.getZ() - pos.z, sinY, cosY);
                double[] rv = rot(a.getVelocity().x, a.getVelocity().z, sinY, cosY);
                row.addProperty("present", 1.0);
                row.addProperty("rel_x", rel[0]);
                row.addProperty("rel_y", a.getY() - pos.y);
                row.addProperty("rel_z", rel[1]);
                row.addProperty("vel_x", rv[0]);
                row.addProperty("vel_y", a.getVelocity().y);
                row.addProperty("vel_z", rv[1]);
            } else {
                row.addProperty("present", 0.0);
            }
            projectiles.add(row);
        }
        o.add("projectiles", projectiles);

        // --- yaw-rotated block-column grid ---
        o.add("block_view", blockColumnView(world, pos, sinY, cosY, spec.blockViewSize,
            spec.terrainMaxAmplitude));

        o.addProperty("time_left", spec.matchTimeSeconds); // no match clock on a live server
        o.addProperty("enemies_alive", nEnemies);
        o.addProperty("teammates_alive", nTeammates);
        return o;
    }

    private static JsonObject otherRow(PlayerEntity p, PlayerEntity self, ClientWorld world,
                                       double sinY, double cosY) {
        double[] rel = rot(p.getX() - self.getX(), p.getZ() - self.getZ(), sinY, cosY);
        KitItem main = KitItem.of(p.getMainHandStack());
        boolean using = p.isUsingItem();
        JsonObject r = new JsonObject();
        r.addProperty("present", 1.0);
        r.addProperty("hp", p.getHealth() + p.getAbsorptionAmount());
        r.addProperty("rel_x", rel[0]);
        r.addProperty("rel_y", p.getY() - self.getY());
        r.addProperty("rel_z", rel[1]);
        r.addProperty("vel_x", p.getVelocity().x);
        r.addProperty("vel_y", p.getVelocity().y);
        r.addProperty("vel_z", p.getVelocity().z);
        r.addProperty("ground_height", surfaceY(world, p.getX(), p.getZ(), p.getY()));
        r.addProperty("blocking", using && (main == KitItem.EMPTY || main.isShieldish()) ? 1.0 : 0.0);
        r.addProperty("eating", using && main.isFood() ? 1.0 : 0.0);
        r.addProperty("held_ranged", main.isRanged() ? 1.0 : 0.0);
        r.addProperty("sneaking", p.isSneaking() ? 1.0 : 0.0);
        return r;
    }

    private static void padPresentRows(JsonArray arr, int target) {
        while (arr.size() < target) {
            JsonObject r = new JsonObject();
            r.addProperty("present", 0.0);
            arr.add(r);
        }
    }

    /** world offset = ox*right(yaw) + oz*forward(yaw); right=(cos,sin), forward=(-sin,cos). */
    private static double[] rot(double dx, double dz, double sinY, double cosY) {
        return new double[] { dx * cosY - dz * sinY, dx * sinY + dz * cosY };
    }

    /** y of the topmost full-collision block at/below {@code feetY} near (x,z). */
    private static double surfaceY(ClientWorld world, double x, double z, double feetY) {
        int bx = MathHelper.floor(x);
        int bz = MathHelper.floor(z);
        int fy = MathHelper.floor(feetY);
        BlockPos.Mutable m = new BlockPos.Mutable();
        for (int y = fy + 4; y >= fy - 8; y--) {
            m.set(bx, y, bz);
            BlockState bs = world.getBlockState(m);
            if (Block.isShapeFullCube(bs.getCollisionShape(world, m))) {
                return y + 1;
            }
        }
        return fy - 8;
    }

    private static JsonArray blockColumnView(ClientWorld world, Vec3d pos, double sinY, double cosY,
                                             int size, double maxAmp) {
        JsonArray out = new JsonArray();
        if (size == 0) return out;
        int half = size / 2;
        double invAmp = 1.0 / Math.max(maxAmp, 1e-3);
        int feetY = MathHelper.floor(pos.y);
        BlockPos.Mutable m = new BlockPos.Mutable();
        for (int row = 0; row < size; row++) {
            for (int col = 0; col < size; col++) {
                double ox = col - half;
                double oz = row - half;
                double wx = pos.x + ox * cosY - oz * sinY;
                double wz = pos.z + ox * sinY + oz * cosY;
                int bx = MathHelper.floor(wx);
                int bz = MathHelper.floor(wz);

                int surfaceY = feetY - 8;
                for (int y = feetY + 4; y >= feetY - 8; y--) {
                    m.set(bx, y, bz);
                    if (Block.isShapeFullCube(world.getBlockState(m).getCollisionShape(world, m))) {
                        surfaceY = y + 1;
                        break;
                    }
                }
                double water = 0, lava = 0, cobweb = 0;
                for (int y = feetY - 1; y <= feetY + 3; y++) {
                    m.set(bx, y, bz);
                    var fluid = world.getFluidState(m);
                    if (fluid.isIn(FluidTags.WATER)) water = 1;
                    if (fluid.isIn(FluidTags.LAVA)) lava = 1;
                    if (world.getBlockState(m).isOf(Blocks.COBWEB)) cobweb = 1;
                }
                double topRel = MathHelper.clamp((surfaceY - pos.y) * invAmp, -3.0, 3.0);
                JsonObject c = new JsonObject();
                c.addProperty("top_rel", topRel);
                c.addProperty("water", water);
                c.addProperty("lava", lava);
                c.addProperty("cobweb", cobweb);
                out.add(c);
            }
        }
        return out;
    }

    private static double pingMs(ClientPlayerEntity self) {
        try {
            var entry = self.networkHandler.getPlayerListEntry(self.getUuid());
            return entry != null ? Math.max(0, entry.getLatency()) : 0.0;
        } catch (RuntimeException e) {
            return 0.0;
        }
    }

    private static double clamp01(double v) {
        return MathHelper.clamp(v, 0.0, 1.0);
    }

    /** {@code amplifier + 1} if {@code effect} is active on the player, else 0. */
    private static double effLevel(ClientPlayerEntity self, RegistryEntry<StatusEffect> effect) {
        StatusEffectInstance i = self.getStatusEffect(effect);
        return i == null ? 0.0 : i.getAmplifier() + 1.0;
    }
}
