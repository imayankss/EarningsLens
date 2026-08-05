import { ImageResponse } from "next/og";

export const alt = "EarningsLens financial NLP and event-study analytics";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          alignItems: "center",
          background: "linear-gradient(135deg, #050816 0%, #0b1730 58%, #071a19 100%)",
          color: "white",
          display: "flex",
          height: "100%",
          justifyContent: "center",
          padding: "72px",
          width: "100%",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", maxWidth: 1040, width: "100%" }}>
          <div style={{ color: "#67e8f9", display: "flex", fontSize: 28, letterSpacing: 3, textTransform: "uppercase" }}>
            Financial NLP research platform
          </div>
          <div style={{ display: "flex", fontSize: 92, fontWeight: 700, letterSpacing: -4, marginTop: 28 }}>
            EarningsLens
          </div>
          <div style={{ color: "#cbd5e1", display: "flex", fontSize: 38, lineHeight: 1.35, marginTop: 24 }}>
            Verified FinBERT, Loughran–McDonald, transcript, and event-study outputs.
          </div>
          <div style={{ display: "flex", gap: 18, marginTop: 48 }}>
            {["Precomputed artifacts", "Static Next.js", "Deployment safe"].map((item) => (
              <div key={item} style={{ background: "rgba(56,189,248,0.10)", border: "1px solid rgba(103,232,249,0.35)", borderRadius: 12, color: "#cffafe", display: "flex", fontSize: 22, padding: "12px 18px" }}>
                {item}
              </div>
            ))}
          </div>
        </div>
      </div>
    ),
    size,
  );
}
