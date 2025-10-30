import { FormEvent, useState } from "react";
import { Link } from "react-router-dom";
import { authService } from "../services/api";
import "../styles/auth.css";

const Login = () => {
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [message, setMessage] = useState("");

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const email = String(form.get("email"));
    const password = String(form.get("password"));

    try {
      setStatus("loading");
      const response = await authService.login(email, password);
      setMessage(`Welcome back, ${response.email}. Your ${response.plan} plan is ready.`);
      setStatus("success");
    } catch (error) {
      console.error(error);
      setMessage("We couldn’t sign you in. Please try again.");
      setStatus("error");
    }
  };

  return (
    <div className="auth-card">
      <h1>Log in to DJ Blue</h1>
      <p className="auth-card__subtitle">
        Continue your sessions, manage subscriptions, and download the latest companion builds.
      </p>
      <form className="auth-card__form" onSubmit={handleSubmit}>
        <label>
          Email
          <input type="email" name="email" placeholder="you@studio.com" required autoComplete="email" />
        </label>
        <label>
          Password
          <input type="password" name="password" placeholder="••••••••" required autoComplete="current-password" />
        </label>
        <button type="submit" className="button button--primary" disabled={status === "loading"}>
          {status === "loading" ? "Signing in..." : "Sign In"}
        </button>
      </form>
      {status !== "idle" && <p className={`auth-card__message auth-card__message--${status}`}>{message}</p>}
      <p className="auth-card__footer">
        Don’t have an account? <Link to="/signup">Create your DJ Blue ID</Link>
      </p>
    </div>
  );
};

export default Login;
