import { dashboardHighlights } from "../../data/content";
import "../../styles/dashboard.css";

const Overview = () => (
  <div className="dashboard-view">
    <header className="dashboard-view__header">
      <h1>Welcome back</h1>
      <p>Your ambient intelligence hub, tuned to today’s rituals.</p>
    </header>
    <div className="dashboard-view__grid">
      {dashboardHighlights.map((item) => (
        <article key={item.title} className="dashboard-card">
          <h2>{item.title}</h2>
          {item.chips && (
            <div className="dashboard-card__chips">
              {item.chips.map((chip) => (
                <span key={chip}>{chip}</span>
              ))}
            </div>
          )}
          {item.description && <p>{item.description}</p>}
          {item.action && (
            <button type="button" className="button button--ghost">
              {item.action}
            </button>
          )}
        </article>
      ))}
    </div>
  </div>
);

export default Overview;
