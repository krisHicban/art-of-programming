import { featureNarratives, journeyPhases, capabilityMatrix } from "../data/content";
import "../styles/features.css";

const Features = () => (
  <div className="features">
    <section className="features__hero">
      <p className="section-eyebrow">Core Capabilities</p>
      <h1 className="section-title">Crafted systems for emotionally intelligent sessions.</h1>
      <p className="section-lede">
        Every layer of DJ Blue is modular, expressive, and human-aware—from how we sense the room
        to how we playback the story. Explore the companion features that keep you in flow.
      </p>
    </section>

    <section className="features__matrix">
      {featureNarratives.map((feature) => (
        <article key={feature.title} className="features__card">
          <header>
            <h2>{feature.title}</h2>
            <p>{feature.description}</p>
          </header>
          <ul>
            {feature.bullets.map((bullet) => (
              <li key={bullet}>{bullet}</li>
            ))}
          </ul>
          <footer>{feature.metric}</footer>
        </article>
      ))}
    </section>

    <section className="features__canvas">
      <header className="section-header">
        <p className="section-eyebrow">Journey Map</p>
        <h2 className="section-title">A companion before, during, and after every session.</h2>
      </header>
      <div className="journey">
        {journeyPhases.map((phase) => (
          <article key={phase.phase} className="journey__phase">
            <span className="journey__badge">{phase.phase}</span>
            <h3>{phase.title}</h3>
            <p>{phase.description}</p>
            <ul>
              {phase.cues.map((cue) => (
                <li key={cue}>{cue}</li>
              ))}
            </ul>
          </article>
        ))}
      </div>
    </section>

    <section className="features__capabilities">
      <header className="section-header">
        <p className="section-eyebrow">Expanded Modules</p>
        <h2 className="section-title">Plug DJ Blue into every ritual with modular interfaces.</h2>
      </header>
      <div className="capabilities__grid capabilities__grid--expanded">
        {capabilityMatrix.map((capability) => (
          <article key={capability.name} className="capability-card capability-card--minimal">
            <header>
              <h3>{capability.name}</h3>
              <p>{capability.description}</p>
            </header>
            <ul>
              {capability.points.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
          </article>
        ))}
      </div>
    </section>
  </div>
);

export default Features;
