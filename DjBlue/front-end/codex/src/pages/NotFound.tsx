import { Link } from "react-router-dom";
import "../styles/not-found.css";

const NotFound = () => (
  <div className="not-found">
    <div className="not-found__orb" aria-hidden />
    <h1>Page not found</h1>
    <p>
      The page you are looking for drifted out of this soundscape. Return to the home stage or
      explore our features.
    </p>
    <div className="not-found__actions">
      <Link to="/" className="button button--primary">
        Back to Home
      </Link>
      <Link to="/features" className="button button--ghost">
        Explore Features
      </Link>
    </div>
  </div>
);

export default NotFound;
