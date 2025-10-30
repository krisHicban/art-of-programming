import { NavLink } from "react-router-dom";
import { primaryNavigation } from "../../data/navigation";
import "./footer.css";

const Footer = () => (
  <footer className="footer">
    <div className="footer__inner">
      <div className="footer__brand">
        <span className="footer__mark" />
        <div>
          <p className="footer__title">DJ Blue</p>
          <p className="footer__subtitle">Ambient intelligence for every conversation.</p>
        </div>
      </div>
      <nav className="footer__nav" aria-label="Secondary">
        {primaryNavigation.map((link) => (
          <NavLink key={link.to} to={link.to} className="footer__link">
            {link.label}
          </NavLink>
        ))}
      </nav>
      <div className="footer__meta">
        <p className="footer__meta-text">© {new Date().getFullYear()} DJ Blue. All rights reserved.</p>
        <div className="footer__meta-links">
          <NavLink to="/help" className="footer__link">
            Support
          </NavLink>
          <NavLink to="/philosophy" className="footer__link">
            Philosophy
          </NavLink>
        </div>
      </div>
    </div>
  </footer>
);

export default Footer;
