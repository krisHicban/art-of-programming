import { useEffect, useState } from "react";
import { subscriptionService } from "../../services/api";
import "../../styles/dashboard.css";

type SubscriptionStatus = {
  plan: string;
  renewalDate: string;
  paymentMethod: string;
  seats: number;
  addOns: string[];
};

const Subscription = () => {
  const [status, setStatus] = useState<SubscriptionStatus | null>(null);

  useEffect(() => {
    subscriptionService.getStatus().then(setStatus);
  }, []);

  return (
    <div className="dashboard-view">
      <header className="dashboard-view__header">
        <h1>Subscription</h1>
        <p>Manage your plan, payment method, and ambient intelligence add-ons.</p>
      </header>

      {status && (
        <section className="dashboard-card">
          <h2>{status.plan} Plan</h2>
          <p>Renews on {status.renewalDate}</p>
          <div className="dashboard-card__chips">
            <span>{status.paymentMethod}</span>
            <span>{status.seats} seats</span>
          </div>
          <h3>Add-ons</h3>
          <ul>
            {status.addOns.map((addOn) => (
              <li key={addOn}>{addOn}</li>
            ))}
          </ul>
          <div className="dashboard__actions">
            <button type="button" className="button button--ghost">
              Upgrade Plan
            </button>
            <button type="button" className="button button--ghost">
              Pause Subscription
            </button>
          </div>
        </section>
      )}
    </div>
  );
};

export default Subscription;
