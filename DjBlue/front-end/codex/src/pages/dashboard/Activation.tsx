import { licenseKeys } from "../../data/content";
import "../../styles/dashboard.css";

const Activation = () => (
  <div className="dashboard-view">
    <header className="dashboard-view__header">
      <h1>Activation Codes</h1>
      <p>Manage your device licenses and revoke access instantly.</p>
    </header>
    <div className="activation-table" role="table" aria-label="Activation codes">
      <div className="activation-row activation-row--head" role="row">
        <span role="columnheader">Device</span>
        <span role="columnheader">Status</span>
        <span role="columnheader">Issued</span>
        <span role="columnheader">License ID</span>
        <span role="columnheader" className="activation-row__actions">
          Actions
        </span>
      </div>
      {licenseKeys.map((license) => (
        <div key={license.id} className="activation-row" role="row">
          <span role="cell">{license.device}</span>
          <span role="cell" className={`activation-status activation-status--${license.status.toLowerCase().replace(" ", "-")}`}>
            {license.status}
          </span>
          <span role="cell">{license.issued}</span>
          <span role="cell">{license.id}</span>
          <span role="cell" className="activation-row__actions">
            <button type="button" className="button button--ghost">
              Manage
            </button>
          </span>
        </div>
      ))}
    </div>
  </div>
);

export default Activation;
