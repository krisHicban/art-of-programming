import { aboutTimeline } from "../data/content";
import "../styles/about.css";

const About = () => (
  <div className="about">
    <section className="about__hero">
      <p className="section-eyebrow">About DJ Blue</p>
      <h1 className="section-title">We compose emotional intelligence with human intention.</h1>
      <p className="section-lede">
        DJ Blue emerged from musicians, facilitators, and engineers dreaming of technology that felt
        less like software and more like a companion. We design for presence, memory, and sound that
        responds to people—not the other way around.
      </p>
    </section>

    <section className="about__timeline">
      {aboutTimeline.map((event) => (
        <article key={event.year} className="timeline-card">
          <span className="timeline-card__year">{event.year}</span>
          <h2>{event.title}</h2>
          <p>{event.description}</p>
        </article>
      ))}
    </section>

    <section className="about__vision">
      <header className="section-header">
        <p className="section-eyebrow">Vision</p>
        <h2 className="section-title">A future where every conversation is emotionally aware.</h2>
      </header>
      <div className="vision-grid">
        <article>
          <h3>Companion for Creators</h3>
          <p>
            Expand the soundboard with modular plugins, community libraries, and visualizers that
            respond to live performance.
          </p>
        </article>
        <article>
          <h3>Trusted in Enterprises</h3>
          <p>
            Deliver secure deployments with compliance-ready auditing, role-based controls, and team
            ritual playbooks.
          </p>
        </article>
        <article>
          <h3>Accessible Everywhere</h3>
          <p>
            Build inclusive interfaces, caption-ready exports, and spirit-preserving archives that
            celebrate every voice.
          </p>
        </article>
      </div>
    </section>
  </div>
);

export default About;
