package rl.minecraft.ai.client.debug;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import rl.minecraft.ai.client.RlConfig;
import rl.minecraft.ai.client.RlMinecraftAiClient;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;

/**
 * A throwaway stand-in for {@code azalea-bot/inference_server.py}, used by
 * {@link SelfTest} so the full observation -> {@code POST /act} -> action ->
 * {@code ClientGuard} pipeline can be exercised without exporting a checkpoint
 * or starting the Python server. Dev-only (see {@link DebugHarness}).
 *
 * <p>It is a deliberately dumb "policy": stand still, hold attack, and turn to
 * face the nearest enemy in the observation (so the look pipeline actually
 * converges - {@code aim_converges} - and every action field is exercised). If
 * the configured inference port is already taken - a real server is up -
 * {@link #startIfFree} returns {@code null} and the selftest uses that instead.
 */
final class MockInferenceServer {
    private static final Gson GSON = new Gson();

    private final HttpServer server;
    private volatile long ticks;

    private MockInferenceServer(HttpServer server) {
        this.server = server;
    }

    /** Bind the mock to the host:port in {@code cfg.inferenceUrl}, or return null if that's taken. */
    static MockInferenceServer startIfFree(RlConfig cfg) {
        URI act;
        try {
            act = URI.create(cfg.inferenceUrl);
        } catch (RuntimeException e) {
            RlMinecraftAiClient.LOGGER.warn("mock inference: bad inference_url {}", cfg.inferenceUrl);
            return null;
        }
        String host = act.getHost() == null ? "127.0.0.1" : act.getHost();
        int port = act.getPort() < 0 ? 80 : act.getPort();
        try {
            HttpServer s = HttpServer.create(new InetSocketAddress(host, port), 0);
            MockInferenceServer mock = new MockInferenceServer(s);
            s.createContext("/spec", mock::handleSpec);
            s.createContext("/act", mock::handleAct);
            s.setExecutor(null);
            s.start();
            RlMinecraftAiClient.LOGGER.info("mock inference server up on {}:{}", host, port);
            return mock;
        } catch (IOException e) {
            RlMinecraftAiClient.LOGGER.info("mock inference: {}:{} already in use - using the real server",
                host, port);
            return null;
        }
    }

    void stop() {
        server.stop(0);
        RlMinecraftAiClient.LOGGER.info("mock inference server stopped ({} acts)", ticks);
    }

    private void handleSpec(HttpExchange ex) throws IOException {
        reply(ex, "{}");
    }

    private void handleAct(HttpExchange ex) throws IOException {
        JsonObject obs;
        try (InputStream in = ex.getRequestBody()) {
            obs = GSON.fromJson(new String(in.readAllBytes(), StandardCharsets.UTF_8), JsonObject.class);
        } catch (RuntimeException e) {
            obs = null;
        }
        ticks++;

        JsonObject a = new JsonObject();
        a.addProperty("move_x", 0.0);
        a.addProperty("move_z", 0.0);
        a.addProperty("yaw_delta", faceEnemyYaw(obs));
        a.addProperty("pitch_delta", 0.0);
        a.addProperty("jump", false);
        a.addProperty("attack", true);
        a.addProperty("sprint", false);
        a.addProperty("use_item", false);
        a.addProperty("sneak", false);
        a.addProperty("held_slot", 0);
        reply(ex, GSON.toJson(a));
    }

    /**
     * Radians to add to yaw this tick to point at the nearest enemy. The obs
     * gives each enemy in the self-frame ({@code rot()} in {@code ObservationBuilder}
     * rotates world offsets by {@code +self_yaw}), so the world bearing is
     * {@code atan2(-rel_x, rel_z) - self_yaw} and the turn to make is that minus
     * the current {@code self_yaw} again. Clamped so the guard, not the mock, is
     * what shapes the motion. Falls back to a harmless wobble with no enemy.
     */
    private double faceEnemyYaw(JsonObject obs) {
        try {
            JsonArray enemies = obs.getAsJsonArray("enemies");
            double selfYaw = obs.get("self_yaw").getAsDouble();
            for (int i = 0; i < enemies.size(); i++) {
                JsonObject e = enemies.get(i).getAsJsonObject();
                if (e.get("present").getAsDouble() < 0.5) continue;
                double relX = e.get("rel_x").getAsDouble();
                double relZ = e.get("rel_z").getAsDouble();
                double delta = Math.atan2(-relX, relZ) - 2.0 * selfYaw;
                delta = Math.atan2(Math.sin(delta), Math.cos(delta));   // wrap to (-pi, pi]
                return Math.max(-0.15, Math.min(0.15, delta));
            }
        } catch (RuntimeException ignored) {
            // fall through to the wobble
        }
        return 0.03 * Math.sin(ticks / 9.0);
    }

    private static void reply(HttpExchange ex, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json");
        ex.sendResponseHeaders(200, bytes.length);
        ex.getResponseBody().write(bytes);
        ex.close();
    }
}
