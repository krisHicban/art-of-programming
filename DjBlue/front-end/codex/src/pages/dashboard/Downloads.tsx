import { useEffect, useState } from "react";
import { downloadService } from "../../services/api";
import "../../styles/dashboard.css";

type Build = {
  platform: string;
  version: string;
  size: string;
  notes: string;
};

const Downloads = () => {
  const [builds, setBuilds] = useState<Build[]>([]);

  useEffect(() => {
    downloadService.getBuilds().then(setBuilds);
  }, []);

  return (
    <div className="dashboard-view">
      <header className="dashboard-view__header">
        <h1>Downloads</h1>
        <p>Stay current with the latest DJ Blue builds and release notes.</p>
      </header>
      <div className="dashboard-view__grid">
        {builds.map((build) => (
          <article key={build.platform} className="dashboard-card">
            <h2>{build.platform}</h2>
            <p>
              Version {build.version} · {build.size}
            </p>
            <p>{build.notes}</p>
            <button type="button" className="button button--primary">
              Download
            </button>
          </article>
        ))}
      </div>
    </div>
  );
};

export default Downloads;
