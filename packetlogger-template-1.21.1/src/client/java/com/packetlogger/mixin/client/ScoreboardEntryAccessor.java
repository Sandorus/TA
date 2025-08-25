package com.packetlogger.mixin.client;

import net.minecraft.scoreboard.ScoreboardEntry;
import net.minecraft.text.Text;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(ScoreboardEntry.class)
public interface ScoreboardEntryAccessor {
    @Accessor("owner")
    String getOwner();

    @Accessor("value")
    int getValue();

    @Accessor("display")
    Text getDisplay(); // Might be nullable!
}

