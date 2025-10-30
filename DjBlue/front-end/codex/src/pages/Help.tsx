import { faqs } from "../data/content";
import "../styles/help.css";

const Help = () => (
  <div className="help">
    <section className="help__hero">
      <p className="section-eyebrow">Support</p>
      <h1 className="section-title">We are here to orchestrate with you.</h1>
      <p className="section-lede">
        Explore frequently asked questions or send us a note. Our team of conductors responds within
        one business day.
      </p>
    </section>

    <section className="help__grid">
      <div className="help__faqs">
        {faqs.map((faq) => (
          <details key={faq.question} className="faq">
            <summary>{faq.question}</summary>
            <p>{faq.answer}</p>
          </details>
        ))}
      </div>
      <form className="help__form" aria-label="Contact support">
        <label>
          Name
          <input type="text" name="name" placeholder="Your name" required />
        </label>
        <label>
          Email
          <input type="email" name="email" placeholder="you@studio.com" required />
        </label>
        <label>
          How can we help?
          <textarea name="message" placeholder="Tell us about your question" rows={4} required />
        </label>
        <button type="submit" className="button button--primary">
          Send Message
        </button>
      </form>
    </section>
  </div>
);

export default Help;
