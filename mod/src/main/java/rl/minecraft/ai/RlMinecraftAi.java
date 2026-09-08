package rl.minecraft.ai;

import net.fabricmc.api.ModInitializer;

import net.minecraft.util.Identifier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RlMinecraftAi implements ModInitializer {
	public static final String MOD_ID = "rl-minecraft-ai";

	// This logger is used to write text to the console and the log file.
	// It is considered best practice to use your mod id as the logger's name.
	// That way, it's clear which mod wrote info, warnings, and errors.
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	@Override
	public void onInitialize() {
		// Everything this mod does is client-side (see the `client` source set
		// and RlMinecraftAiClient); nothing to set up on the common/server side.
		LOGGER.info("rl-minecraft-ai loaded (client features only)");
	}

	public static Identifier id(String path) {
		return Identifier.of(MOD_ID, path);
	}
}
