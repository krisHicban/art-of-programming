import { Link } from "react-router-dom";
import { heroContent, executiveHighlights, capabilityMatrix } from "../data/content";
import "../styles/home.css";

const Home = () => (
  <div className="home">
    <section className="hero">
      <div className="hero__content">
        <p className="hero__eyebrow">{heroContent.eyebrow}</p>
        <h1 className="hero__title">{heroContent.title}</h1>
        <p className="hero__subtitle">{heroContent.subtitle}</p>
        <div className="hero__cta">
          <Link to={heroContent.primaryCta.to} className="button button--primary">
            {heroContent.primaryCta.label}
          </Link>
          <Link to={heroContent.secondaryCta.to} className="button button--ghost">
            {heroContent.secondaryCta.label}
          </Link>
        </div>
        <ul className="hero__highlights">
          {heroContent.highlights.map((highlight) => (
            <li key={highlight}>{highlight}</li>
          ))}
        </ul>
      </div>
      <div className="hero__visual" aria-hidden>
        <div className="hero__ring hero__ring--outer" />
        <div className="hero__ring hero__ring--inner" />
        <div className="hero__waveform">
          {Array.from({ length: 32 }).map((_, index) => (
            <span key={index} style={{ animationDelay: `${index * 40}ms` }} />
          ))}
        </div>
      </div>
    </section>

    <section className="executive">
      <header className="section-header">
        <p className="section-eyebrow">Executive Overview</p>
        <h2 className="section-title">Ambient intelligence orchestrated for every moment.</h2>
        <p className="section-lede">
          DJ Blue brings emotional awareness to every conversation. We listen, adapt, and compose the
          moment with you—balancing precision with poetic resonance.
        </p>
      </header>
      <div className="executive__grid">
        {executiveHighlights.map((highlight) => (
          <article key={highlight.title} className="card">
            <h3>{highlight.title}</h3>
            <p>{highlight.description}</p>
          </article>
        ))}
      </div>
    </section>

    <section className="capabilities">
      <header className="section-header">
        <p className="section-eyebrow">Core Capabilities</p>
        <h2 className="section-title">Where conversation meets adaptive creativity.</h2>
      </header>
      <div className="capabilities__grid">
        {capabilityMatrix.map((capability) => (
          <article key={capability.name} className="capability-card">
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
      <div className="capabilities__cta">
        <Link to="/features" className="button button--primary">
          Explore Feature Stories
        </Link>
        <Link to="/download" className="button button--ghost">
          Download Companion
        </Link>
      </div>
    </section>

    <section className="testimonials">
      <header className="section-header">
        <p className="section-eyebrow">Voices from the Studio</p>
        <h2 className="section-title">Stories of presence, focus, and emotional resonance.</h2>
      </header>
      <div className="testimonials__grid">
        <article className="testimonial">
          <p>
            “Every client session now has a cinematic quality. DJ Blue adapts to the conversation in
            ways I didn’t know I needed.”
          </p>
          <span>Maria Chen · Leadership Coach</span>
        </article>
        <article className="testimonial">
          <p>
            “It feels like an ambient engineer, note-taker, and music director in one. My podcast
            edits itself emotionally.”
          </p>
          <span>Chris Alvarez · Creative Producer</span>
        </article>
        <article className="testimonial">
          <p>
            “Students remember the atmosphere we create. DJ Blue makes learning immersive and deeply
            human.”
          </p>
          <span>Elena Castillo · Educator</span>
        </article>
      </div>
    </section>

    <section className="newsletter">
      <div className="newsletter__inner">
        <header>
          <p className="section-eyebrow">Stay in the Loop</p>
          <h2 className="section-title">Receive new rituals, sound libraries, and release notes.</h2>
          <p className="section-lede">
            A monthly pulse with insights on ambient intelligence and exclusive DJ Blue sessions.
            We respect your inbox and your privacy.
          </p>
        </header>
        <form className="newsletter__form" aria-label="Subscribe to DJ Blue updates">
          <label className="visually-hidden" htmlFor="newsletter-email">
            Email address
          </label>
          <input id="newsletter-email" type="email" placeholder="you@studio.com" required />
          <button type="submit" className="button button--primary">
            Join the Frequency
          </button>
        </form>
      </div>
    </section>

    <section className="cta">
      <div className="cta__inner">
        <div>
          <p className="section-eyebrow">Begin the Companion Era</p>
          <h2 className="section-title">Download DJ Blue and score your next session.</h2>
          <p className="section-lede">
            Sign up, sync your library, and let DJ Blue orchestrate your meetings, lessons, and
            creative rituals with intelligence that feels alive.
          </p>
        </div>
        <Link to="/signup" className="button button--primary">
          Create Your Account
        </Link>
      </div>
    </section>
  </div>
);

export default Home;
