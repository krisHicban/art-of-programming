import { useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { primaryNavigation, ctaNavigation } from "../../data/navigation";
import "../../styles/ambient.css";
import "./navbar.css";

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { pathname } = useLocation();

  const toggleMenu = () => setIsOpen((prev) => !prev);
  const closeMenu = () => setIsOpen(false);

  return (
    <header className="nav-container">
      <div className="nav-inner">
        <NavLink to="/" className="nav-brand" aria-label="DJ Blue Home">
          <span className="nav-brand__mark" aria-hidden />
          <span className="nav-brand__text">DJ Blue</span>
        </NavLink>

        <nav className="nav-links" aria-label="Primary">
          {primaryNavigation.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => (isActive ? "nav-link nav-link--active" : "nav-link")}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="nav-cta">
          {ctaNavigation.map((cta) => (
            <NavLink key={cta.to} to={cta.to} className="button button--ghost">
              {cta.label}
            </NavLink>
          ))}
          <NavLink to="/pricing" className="button button--primary">
            Subscribe
          </NavLink>
        </div>

        <button
          className="nav-toggle"
          onClick={toggleMenu}
          aria-expanded={isOpen}
          aria-controls="nav-drawer"
        >
          <span className="visually-hidden">Toggle menu</span>
          <span className="nav-toggle__bar" />
          <span className="nav-toggle__bar" />
        </button>

        <div className={`nav-drawer${isOpen ? " nav-drawer--open" : ""}`} id="nav-drawer">
          <div className="nav-drawer__header">
            <span className="nav-brand__text nav-brand__text--drawer">DJ Blue</span>
            <button className="nav-drawer__close" onClick={closeMenu} aria-label="Close menu">
              ×
            </button>
          </div>
          <nav className="nav-drawer__links" aria-label="Mobile">
            {primaryNavigation.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={pathname === item.to ? "nav-link nav-link--active" : "nav-link"}
                onClick={closeMenu}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="nav-drawer__cta">
            {ctaNavigation.map((cta) => (
              <NavLink key={cta.to} to={cta.to} className="button button--ghost" onClick={closeMenu}>
                {cta.label}
              </NavLink>
            ))}
            <NavLink to="/pricing" className="button button--primary" onClick={closeMenu}>
              Become a Member
            </NavLink>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
