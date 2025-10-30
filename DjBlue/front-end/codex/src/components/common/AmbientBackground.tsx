import "../../styles/ambient.css";

const AmbientBackground = () => (
  <div aria-hidden className="ambient-backdrop">
    <span className="ambient-backdrop__wave ambient-backdrop__wave--primary" />
    <span className="ambient-backdrop__wave ambient-backdrop__wave--secondary" />
    <span className="ambient-backdrop__wave ambient-backdrop__wave--tertiary" />
  </div>
);

export default AmbientBackground;
