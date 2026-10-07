import { describe, expect, it } from "vitest";
import { resolveLunaMood } from "@/lib/luna-motion";

describe("Luna's present-tense reactions", () => {
  it("keeps human-support routing above every expressive signal", () => {
    expect(resolveLunaMood({ support: true, error: true, typing: true, hasOffer: true, replyFeelings: ["happy"] })).toBe("support");
  });
  it("listens to typing instead of acting on an earlier interpretation", () => {
    expect(resolveLunaMood({ typing: true, replyFeelings: ["sad"] })).toBe("listening");
    expect(resolveLunaMood({ waiting: true, typing: true })).toBe("thinking");
  });
  it("lets an explicit empty correction clear inferred feelings", () => {
    expect(resolveLunaMood({ confirmedFeelings: [], replyFeelings: ["sad"], hasReply: true })).toBe("reflecting");
    expect(resolveLunaMood({ confirmedFeelings: ["calm"], replyFeelings: ["sad"] })).toBe("grounded");
  });
  it("gives the person's opening mood priority over a conflicting model suggestion", () => {
    expect(resolveLunaMood({ openingMood: 2, replyFeelings: ["happy"] })).toBe("comforting");
    expect(resolveLunaMood({ openingMood: 5, replyFeelings: ["sad"] })).toBe("encouraging");
    expect(resolveLunaMood({ openingMood: 3, replyFeelings: ["happy"] })).toBe("reflecting");
  });
  it("acknowledges mixed feelings without an unqualified celebration", () => {
    expect(resolveLunaMood({ replyFeelings: ["happy", "anxious"] })).toBe("reflecting");
    expect(resolveLunaMood({ replyFeelings: ["overwhelmed"] })).toBe("comforting");
    expect(resolveLunaMood({ replyFeelings: ["hopeful"] })).toBe("encouraging");
  });
  it.each(["not_tried", "stopped", "completed", "partial"])("does not infer benefit from %s", (participation) => {
    expect(resolveLunaMood({ replyFeelings: ["happy"], activity: { participation, stateChange: "same" } })).toBe("reflecting");
  });
  it("uses actual reported change and allows stopping to override it", () => {
    expect(resolveLunaMood({ activity: { participation: "completed", stateChange: "away_from_target" } })).toBe("comforting");
    expect(resolveLunaMood({ activity: { participation: "partial", stateChange: "toward_target" } })).toBe("grounded");
    expect(resolveLunaMood({ move: "pause", hasOffer: true })).toBe("resting");
  });
  it("invites a check-in at expiry without declaring completion", () => {
    expect(resolveLunaMood({ activity: { status: "awaiting_report" } })).toBe("checkin");
    expect(resolveLunaMood({ hasOffer: true })).toBe("offering");
    expect(resolveLunaMood({ hasReply: true, activity: { status: "active" } })).toBe("grounded");
    expect(resolveLunaMood({ hasReply: true, activity: { status: "paused" } })).toBe("resting");
  });
});
