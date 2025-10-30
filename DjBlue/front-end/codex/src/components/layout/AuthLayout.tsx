import { Outlet, NavLink } from "react-router-dom";
import AmbientBackground from "../common/AmbientBackground";
import ScrollToTop from "../common/ScrollToTop";
import "./auth-layout.css";

const AuthLayout = () => (
  <div className="auth-layout">
    <AmbientBackground />
    <ScrollToTop />
    <header className="auth-layout__header">
      <NavLink to="/" className="auth-layout__brand">
        DJ Blue
      </NavLink>
      <NavLink to="/help" className="auth-layout__support">
        Need Support?
      </NavLink>
    </header>
    <main className="auth-layout__main">
      <Outlet />
    </main>
  </div>
);

export default AuthLayout;
