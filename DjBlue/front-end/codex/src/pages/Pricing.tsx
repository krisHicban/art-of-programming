import { pricingPlans } from "../data/content";
import "../styles/pricing.css";

const Pricing = () => (
  <div className="pricing">
    <section className="pricing__hero">
      <p className="section-eyebrow">Pricing</p>
      <h1 className="section-title">Choose the rhythm that matches your practice.</h1>
      <p className="section-lede">
        Start with a 14-day free journey. Upgrade, pause, or evolve your plan whenever the tempo of
        your work shifts. Every subscription keeps your data sovereign and your creativity sovereign.
      </p>
    </section>

    <section className="pricing__plans">
      {pricingPlans.map((plan) => (
        <article key={plan.name} className={`pricing-card${plan.popular ? " pricing-card--popular" : ""}`}>
          {plan.popular && <span className="pricing-card__badge">Most Loved</span>}
          <header>
            <h2>{plan.name}</h2>
            <p>{plan.description}</p>
            <div className="pricing-card__price">
              <span>{plan.price}</span>
              <small>{plan.cadence}</small>
            </div>
          </header>
          <ul>
            {plan.features.map((feature) => (
              <li key={feature}>{feature}</li>
            ))}
          </ul>
          <button type="button" className="button button--primary">
            {plan.cta}
          </button>
        </article>
      ))}
    </section>

    <section className="pricing__cta">
      <div>
        <h2 className="section-title">Need something special?</h2>
        <p className="section-lede">
          From high-touch onboarding to bespoke scoring libraries, we compose enterprise experiences
          tailored to studios, campuses, and communities.
        </p>
      </div>
      <button type="button" className="button button--ghost">
        Talk to a Conductor
      </button>
    </section>
  </div>
);

export default Pricing;
