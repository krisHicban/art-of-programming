import { Routes, Route } from "react-router-dom";
import MarketingLayout from "./components/layout/MarketingLayout";
import AuthLayout from "./components/layout/AuthLayout";
import DashboardLayout from "./components/layout/DashboardLayout";
import Home from "./pages/Home";
import Features from "./pages/Features";
import Philosophy from "./pages/Philosophy";
import UseCases from "./pages/UseCases";
import Pricing from "./pages/Pricing";
import Download from "./pages/Download";
import About from "./pages/About";
import Help from "./pages/Help";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Overview from "./pages/dashboard/Overview";
import Subscription from "./pages/dashboard/Subscription";
import Downloads from "./pages/dashboard/Downloads";
import Activation from "./pages/dashboard/Activation";
import NotFound from "./pages/NotFound";

const App = () => (
  <Routes>
    <Route element={<MarketingLayout />}>
      <Route index element={<Home />} />
      <Route path="/features" element={<Features />} />
      <Route path="/philosophy" element={<Philosophy />} />
      <Route path="/use-cases" element={<UseCases />} />
      <Route path="/pricing" element={<Pricing />} />
      <Route path="/download" element={<Download />} />
      <Route path="/about" element={<About />} />
      <Route path="/help" element={<Help />} />
    </Route>

    <Route element={<AuthLayout />}>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
    </Route>

    <Route path="/dashboard" element={<DashboardLayout />}>
      <Route index element={<Overview />} />
      <Route path="subscription" element={<Subscription />} />
      <Route path="downloads" element={<Downloads />} />
      <Route path="activation" element={<Activation />} />
    </Route>

    <Route path="*" element={<NotFound />} />
  </Routes>
);

export default App;
