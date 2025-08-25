package com.packetlogger.mixin.client;

import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.gui.hud.InGameHud;
import net.minecraft.scoreboard.ScoreboardObjective;
import net.minecraft.text.Text;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import uk.co.cablepost.racing_scoreboard.RacingScoreboardClient;
import uk.co.cablepost.racing_scoreboard.RacingScoreboardEntry;

import java.util.Map;

@Mixin(InGameHud.class)
public class InGameHudMixin {

    @Inject(method = "renderScoreboardSidebar(Lnet/minecraft/client/gui/DrawContext;Lnet/minecraft/scoreboard/ScoreboardObjective;)V",
            at = @At("TAIL"))
    public void logScoreboardSidebar(DrawContext context, ScoreboardObjective objective, CallbackInfo ci) {
        if (objective == null) return;

        String title = objective.getDisplayName().getString();
        System.out.println("[Scoreboard] Title: " + title);

        // Check if Racing Scoreboard is enabled
        if (RacingScoreboardClient.shouldUseCustomScoreboard()) {
            for (Map.Entry<String, RacingScoreboardEntry> entry : RacingScoreboardClient.racingScoreboardEntries.entrySet()) {
                RacingScoreboardEntry e = entry.getValue();

                if (e.player != null && e.time != null && e.current) {
                    String playerName = e.player.getString();
                    String time = e.time.getString();
                    int pos = e.position;

                    System.out.println("[Parsed Scoreboard] " + playerName + " | Pos: " + pos + " | Time: " + time);
                }
            }
        } else {
            System.out.println("[Warning] RacingScoreboard is disabled or not active.");
        }
    }
}
