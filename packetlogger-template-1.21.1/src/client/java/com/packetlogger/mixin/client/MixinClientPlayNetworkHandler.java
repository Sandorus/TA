package com.packetlogger.mixin.client;

import net.minecraft.client.network.ClientPlayNetworkHandler;
import net.minecraft.network.packet.CustomPayload;
import net.minecraft.network.packet.s2c.play.GameJoinS2CPacket;
import net.minecraft.network.packet.s2c.common.CustomPayloadS2CPacket;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ClientPlayNetworkHandler.class)
public class MixinClientPlayNetworkHandler {

	private static final Logger logger = LogManager.getLogger("PacketLogger");

	@Inject(method = "onGameJoin", at = @At("HEAD"), require = 1)
	private void onGameJoin(GameJoinS2CPacket packet, CallbackInfo ci) {
		logger.info("Received GameJoin packet: {}", packet);
	}

	@Inject(method = "onCustomPayload", at = @At("HEAD"))
	private void onCustomPayload(CustomPayload payload, CallbackInfo ci) {
		logger.info("Received CustomPayload: {}", payload.getId());
	}
}
