import { Outlet } from "react-router-dom";
import AmbientBackground from "../common/AmbientBackground";
import ScrollToTop from "../common/ScrollToTop";
import Navbar from "./Navbar";
import Footer from "./Footer";
import "./layout.css";

const MarketingLayout = () => (
  <div className="layout">
    <AmbientBackground />
    <ScrollToTop />
    <Navbar />
    <main className="layout__main">
      <Outlet />
    </main>
    <Footer />
  </div>
);

export default MarketingLayout;
