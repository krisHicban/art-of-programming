import { useCaseGroups } from "../data/content";
import "../styles/use-cases.css";

const UseCases = () => (
  <div className="use-cases">
    <section className="use-cases__hero">
      <p className="section-eyebrow">Use Cases</p>
      <h1 className="section-title">DJ Blue adapts to every room you host.</h1>
      <p className="section-lede">
        Whether you are leading a boardroom, guiding a class, producing a show, or bringing friends
        together, DJ Blue senses the emotional flow and composes the atmosphere. Explore how the
        companion unlocks presence across scenarios.
      </p>
    </section>
    <section className="use-cases__grid">
      {useCaseGroups.map((group) => (
        <article key={group.category} className="use-case">
          <header>
            <span className="use-case__category">{group.category}</span>
            <h2>{group.headline}</h2>
          </header>
          <ul>
            {group.scenarios.map((scenario) => (
              <li key={scenario}>{scenario}</li>
            ))}
          </ul>
        </article>
      ))}
    </section>
  </div>
);

export default UseCases;
