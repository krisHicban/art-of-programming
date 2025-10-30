import { NavLink, Outlet } from "react-router-dom";
import AmbientBackground from "../common/AmbientBackground";
import ScrollToTop from "../common/ScrollToTop";
import "./dashboard-layout.css";

const DashboardLayout = () => (
  <div className="dashboard">
    <AmbientBackground />
    <ScrollToTop />
    <aside className="dashboard__sidebar">
      <div className="dashboard__brand">
        <span className="dashboard__mark" />
        <span>DJ Blue</span>
      </div>
      <nav aria-label="Dashboard navigation" className="dashboard__nav">
        <NavLink to="/dashboard" end className={({ isActive }) => (isActive ? "dashboard__link active" : "dashboard__link")}>
          Overview
        </NavLink>
        <NavLink to="/dashboard/subscription" className={({ isActive }) => (isActive ? "dashboard__link active" : "dashboard__link")}>
          Subscription
        </NavLink>
        <NavLink to="/dashboard/downloads" className={({ isActive }) => (isActive ? "dashboard__link active" : "dashboard__link")}>
          Downloads
        </NavLink>
        <NavLink to="/dashboard/activation" className={({ isActive }) => (isActive ? "dashboard__link active" : "dashboard__link")}>
          Activation Codes
        </NavLink>
        <NavLink to="/help" className={({ isActive }) => (isActive ? "dashboard__link active" : "dashboard__link")}>
          Support
        </NavLink>
      </nav>
    </aside>
    <main className="dashboard__content">
      <Outlet />
    </main>
  </div>
);

export default DashboardLayout;
