import { useEffect, useState } from "react";
import { downloadSteps } from "../data/content";
import { downloadService } from "../services/api";
import "../styles/download.css";

type Build = {
  platform: string;
  version: string;
  size: string;
  notes: string;
};

const Download = () => {
  const [builds, setBuilds] = useState<Build[]>([]);

  useEffect(() => {
    downloadService.getBuilds().then(setBuilds);
  }, []);

  return (
    <div className="download">
      <section className="download__hero">
        <p className="section-eyebrow">Download</p>
        <h1 className="section-title">Set up DJ Blue on your studio devices.</h1>
        <p className="section-lede">
          Lightweight installers, local-first syncing, and activation that keeps you in control. The
          companion is optimized for laptops, desktops, and creative rigs.
        </p>
      </section>

      <section className="download__builds">
        {builds.map((build) => (
          <article key={build.platform} className="build-card">
            <header>
              <h2>{build.platform}</h2>
              <span>{build.version}</span>
            </header>
            <p>{build.notes}</p>
            <footer>
              <button type="button" className="button button--primary">
                Download {build.size}
              </button>
            </footer>
          </article>
        ))}
      </section>

      <section className="download__steps">
        {downloadSteps.map((step) => (
          <article key={step.step} className="step-card">
            <span className="step-card__index">{step.step}</span>
            <h3>{step.title}</h3>
            <p>{step.description}</p>
            {step.action && (
              <button type="button" className="button button--ghost">
                {step.action}
              </button>
            )}
            {step.secondaryAction && (
              <button type="button" className="button button--ghost">
                {step.secondaryAction}
              </button>
            )}
          </article>
        ))}
      </section>
    </div>
  );
};

export default Download;
