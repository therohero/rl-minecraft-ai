package rl.minecraft.ai.client.net;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import rl.minecraft.ai.client.RlMinecraftAiClient;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Talks to {@code azalea-bot/inference_server.py} over plain HTTP+JSON - the
 * same seam the Rust {@code azalea_bot} uses. Requests run off the game
 * thread: {@link #requestAsync} fires one {@code POST /act} at a time
 * ("latest observation wins", stale ones are dropped) and the freshest reply
 * is read back with {@link #latestAction()}. So the client keeps acting every
 * tick regardless of model latency, and a dead server just means "no action".
 */
public final class InferenceClient {
    private static final Gson GSON = new Gson();

    private final String actUrl;
    private final String specUrl;
    private final HttpClient http = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(2))
        .build();

    private final AtomicBoolean inFlight = new AtomicBoolean(false);
    private final AtomicReference<Action> latest = new AtomicReference<>(null);
    private volatile long lastOkNanos = 0;
    /** null = no reply seen yet; "" = last reply was 200; otherwise the failure text. */
    private volatile String lastError = null;
    /** EWMA of the {@code POST /act} round-trip, ms; -1 before the first success. */
    private volatile double latencyMsEwma = -1.0;

    public InferenceClient(String actUrl, String specUrl) {
        this.actUrl = actUrl;
        this.specUrl = specUrl;
    }

    /** Blocking {@code GET /spec}; caller runs this off the game thread. */
    public Spec fetchSpec() throws Exception {
        HttpRequest req = HttpRequest.newBuilder(URI.create(specUrl))
            .timeout(Duration.ofSeconds(3))
            .GET()
            .build();
        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() != 200) {
            throw new IllegalStateException("GET /spec -> HTTP " + resp.statusCode());
        }
        return new Spec(GSON.fromJson(resp.body(), JsonObject.class));
    }

    /**
     * Submit an observation for inference unless a request is already
     * outstanding. The result lands in {@link #latestAction()}.
     */
    public void requestAsync(JsonObject observation) {
        if (!inFlight.compareAndSet(false, true)) return;
        String body = GSON.toJson(observation);
        HttpRequest req = HttpRequest.newBuilder(URI.create(actUrl))
            .timeout(Duration.ofSeconds(2))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .build();
        long startNanos = System.nanoTime();
        http.sendAsync(req, HttpResponse.BodyHandlers.ofString())
            .whenComplete((resp, err) -> {
                try {
                    if (err != null) {
                        lastError = "cannot reach " + actUrl + " (" + err.getMessage() + ")";
                        maybeWarn("inference request failed: " + err.getMessage());
                        return;
                    }
                    if (resp.statusCode() != 200) {
                        lastError = "server replied HTTP " + resp.statusCode() + ": " + trim(resp.body());
                        maybeWarn("POST /act -> " + lastError);
                        return;
                    }
                    latest.set(Action.fromJson(GSON.fromJson(resp.body(), JsonObject.class)));
                    lastError = "";
                    long now = System.nanoTime();
                    lastOkNanos = now;
                    double ms = (now - startNanos) / 1_000_000.0;
                    double prev = latencyMsEwma;
                    latencyMsEwma = prev < 0 ? ms : prev * 0.8 + ms * 0.2;
                } finally {
                    inFlight.set(false);
                }
            });
    }

    /** Most recent action the server returned, or {@code null} if none yet. */
    public Action latestAction() {
        return latest.get();
    }

    public boolean seenServer() {
        return lastOkNanos != 0;
    }

    /** Smoothed {@code POST /act} round-trip in ms, or -1 before the first
     *  successful reply. Wall-clock latency of the async call, not model time. */
    public double latencyMs() {
        return latencyMsEwma;
    }

    /** null before any reply, "" when the last /act call succeeded, else the failure. */
    public String lastError() {
        return lastError;
    }

    public void reset() {
        latest.set(null);
        inFlight.set(false);
        lastError = null;
        lastOkNanos = 0;
        latencyMsEwma = -1.0;
    }

    private static String trim(String s) {
        if (s == null) return "";
        s = s.strip();
        return s.length() > 180 ? s.substring(0, 180) + "…" : s;
    }

    private long lastWarnNanos = 0;

    private void maybeWarn(String msg) {
        long now = System.nanoTime();
        if (now - lastWarnNanos > 5_000_000_000L) {
            lastWarnNanos = now;
            RlMinecraftAiClient.LOGGER.warn(msg);
        }
    }
}
