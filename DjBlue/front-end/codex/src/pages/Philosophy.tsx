import { philosophyStatements } from "../data/content";
import "../styles/philosophy.css";

const designTenets = [
  {
    title: "Calm Surfaces",
    description: "Dark-mode gradients with luminous blues that soothe the eye and let content glow."
  },
  {
    title: "Human Rhythm",
    description:
      "Micro interactions modeled after musical dynamics—crescendo, sustain, release—keep navigation lyrical."
  },
  {
    title: "Sovereign Privacy",
    description:
      "Local-first mindset where users control every waveform, note, and highlight created alongside them."
  }
];

const Philosophy = () => (
  <div className="philosophy">
    <section className="philosophy__hero">
      <p className="section-eyebrow">Philosophy</p>
      <h1 className="section-title">Ambient intelligence with emotional empathy at its core.</h1>
      <p className="section-lede">
        DJ Blue exists to honor the emotional cadence of human conversation. We design with respect,
        craft with intentionality, and build with the belief that technology should feel like a
        trusted companion—not another surface to manage.
      </p>
    </section>

    <section className="philosophy__grid">
      {philosophyStatements.map((statement) => (
        <article key={statement.title} className="philosophy__card">
          <h2>{statement.title}</h2>
          <p>{statement.description}</p>
        </article>
      ))}
    </section>

    <section className="philosophy__design">
      <header className="section-header">
        <p className="section-eyebrow">Design Language</p>
        <h2 className="section-title">Visual rhythm inspired by light, sound, and empathy.</h2>
      </header>
      <div className="design-tenets">
        {designTenets.map((tenet) => (
          <div key={tenet.title} className="design-tenet">
            <h3>{tenet.title}</h3>
            <p>{tenet.description}</p>
          </div>
        ))}
      </div>
      <div className="design-board" aria-hidden>
        <span className="design-board__swatch design-board__swatch--primary" />
        <span className="design-board__swatch design-board__swatch--secondary" />
        <span className="design-board__swatch design-board__swatch--tertiary" />
      </div>
    </section>
  </div>
);

export default Philosophy;
