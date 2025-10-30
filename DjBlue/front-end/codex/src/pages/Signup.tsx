import { FormEvent, useState } from "react";
import { Link } from "react-router-dom";
import { authService } from "../services/api";
import "../styles/auth.css";

const Signup = () => {
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [message, setMessage] = useState("");

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const name = String(form.get("name"));
    const email = String(form.get("email"));
    const password = String(form.get("password"));

    try {
      setStatus("loading");
      const response = await authService.signup(name, email, password);
      setMessage(`Welcome, ${name}! Your ${response.plan} plan is ready to explore.`);
      setStatus("success");
    } catch (error) {
      console.error(error);
      setMessage("We couldn’t create your account. Please try again.");
      setStatus("error");
    }
  };

  return (
    <div className="auth-card">
      <h1>Create your DJ Blue ID</h1>
      <p className="auth-card__subtitle">
        Begin with a 14-day free journey. Customize your rituals, sync your library, and invite
        collaborators.
      </p>
      <form className="auth-card__form" onSubmit={handleSubmit}>
        <label>
          Name
          <input type="text" name="name" placeholder="Your name" required autoComplete="name" />
        </label>
        <label>
          Email
          <input type="email" name="email" placeholder="you@studio.com" required autoComplete="email" />
        </label>
        <label>
          Password
          <input type="password" name="password" placeholder="Create a password" required autoComplete="new-password" />
        </label>
        <button type="submit" className="button button--primary" disabled={status === "loading"}>
          {status === "loading" ? "Creating..." : "Create Account"}
        </button>
      </form>
      {status !== "idle" && <p className={`auth-card__message auth-card__message--${status}`}>{message}</p>}
      <p className="auth-card__footer">
        Already have an account? <Link to="/login">Log in instead</Link>
      </p>
    </div>
  );
};

export default Signup;
