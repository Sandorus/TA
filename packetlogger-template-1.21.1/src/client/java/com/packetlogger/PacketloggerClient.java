package com.packetlogger;

import net.fabricmc.api.ClientModInitializer;

public class PacketloggerClient implements ClientModInitializer {
	@Override
	public void onInitializeClient() {
		System.out.println("[PacketLogger] Mod initialized on client.");
	}
}